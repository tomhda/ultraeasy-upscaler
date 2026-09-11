#!/usr/bin/env python
"""NPU 2段推論の役割別汎用テンソル実行器 (前半 F / 後半 G)。

B2 の worker_f.py / worker_g.py を置き換える製品版。1プロセスに1セッションだけ
持ち、B1 §3.4 の同一プロセス内 G→F 汚染が届かない構成にする。2つの役割は
起動引数 --role で区別し、本体は役割に依存しない汎用実装
(NpuSession は流用しない。出力÷入力の倍率推定が F/G に合わないため)。

起動:
  python npu_worker.py --role front|back --model <onnx> --cache-dir <dir>
      --cache-key <key> [--require-cache | --allow-compile]
      [--compile-timeout <s>] [--generation <n>]

  - stdin/stdout は O_BINARY。stdout はプロトコル専用、ログは stderr のみ。
    セッション生成中は fd レベルでも stdout を退避する (AIE コンパイラ等の
    ネイティブ stdout 汚染からプロトコルを守る。詳細は _guard_stdout_during_build)。
  - セッション生成後、READY で入出力の name/dtype/shape/byte 長を JSON で返す。
  - --require-cache (既定): 生成が 60 秒を超えたらキャッシュ未ヒット疑いで
    自ら終了する (開発・検証時の安全弁。再コンパイル開始の保証ではない)。
  - --allow-compile: 製品の初回準備用。上限は --compile-timeout (既定 4 時間)。

プロトコルは npu_proto (UW2P) に共通化。B2 の不整合
(QUIT 8B vs 16B、write 戻り値未確認、read_exact 無期限待ち) は持ち込まない。
DATA 応答では出力の全テンソルについて非有限値率を計算し、異常時は
nan_to_num で隠さず ERROR 応答に段階・テンソル名・非有限値率を含める。

終了コード: 0=正常終了、1=セッション生成失敗、2=プロトコル異常、
98=コンパイル期限超過、99=require-cache 60秒超過 (未ヒット疑い)。

試験用フック (本番では使わない):
  UEU_WORKER_SLEEP_S=<秒>: 各 DATA 応答の前に sleep (応答停止の障害注入用)。
  UEU_WORKER_SLEEP_FILE=<パス>: 要求ごとにファイル内容 (秒) を読み直して sleep。
    0/空/欠落は sleep なし。セッション開始後の注入 (ファイル書換え) 用。
"""
from __future__ import annotations

import argparse
import contextlib
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import npu_proto as proto

try:
    import numpy as np
except ImportError:  # pragma: no cover - NPU 環境では常に存在
    np = None  # type: ignore[assignment]

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover - NPU 環境では常に存在
    ort = None  # type: ignore[assignment]

REQUIRE_CACHE_TIMEOUT = 60.0

_ORT_DTYPE_MAP = {
    "tensor(float)": ("float32", 4),
    "tensor(float16)": ("float16", 2),
    "tensor(double)": ("float64", 8),
    "tensor(int64)": ("int64", 8),
    "tensor(int32)": ("int32", 4),
    "tensor(uint8)": ("uint8", 1),
}


def _log(message: str) -> None:
    print(f"[worker pid={os.getpid()}] {message}", file=sys.stderr, flush=True)


def _describe_tensor(value_info) -> dict:
    otype = str(value_info.type)
    dtype, itemsize = _ORT_DTYPE_MAP.get(otype, ("float32", 4))
    shape: list[int] = []
    try:
        for dim in value_info.shape:
            shape.append(int(dim) if isinstance(dim, int) and dim > 0 else -1)
    except (TypeError, ValueError):
        shape = [-1, -1, -1, -1]
    nbytes = -1
    if all(dim > 0 for dim in shape):
        count = 1
        for dim in shape:
            count *= dim
        nbytes = count * itemsize
    return {"name": value_info.name, "dtype": dtype, "shape": shape, "nbytes": nbytes}


def _reshape(desc: dict, blob: bytes):
    arr = np.frombuffer(blob, dtype=np.dtype(desc["dtype"]))
    shape = list(desc["shape"])
    if all(dim > 0 for dim in shape):
        if arr.size != int(np.prod(shape)):
            raise proto.ProtocolError(
                f"tensor {desc['name']!r}: need {int(np.prod(shape))} elements, "
                f"got {arr.size}"
            )
        return arr.reshape(shape).copy()
    # 動的次元 (-1) はバイト数から推定する。
    known = 1
    unknown = 0
    for dim in shape:
        if dim == -1:
            unknown += 1
        else:
            known *= dim
    if unknown != 1 or arr.size % known != 0:
        raise proto.ProtocolError(
            f"tensor {desc['name']!r}: cannot infer dynamic shape {shape} "
            f"from {arr.size} elements"
        )
    resolved = [arr.size // known if dim == -1 else dim for dim in shape]
    return arr.reshape(resolved).copy()


def _nonfinite_rate(arr) -> float:
    try:
        finite = np.isfinite(arr)
    except TypeError:
        return 0.0
    return float((~finite).mean()) if finite.size else 0.0


def _set_binary_1() -> None:
    if os.name == "nt":
        try:
            import msvcrt

            msvcrt.setmode(1, os.O_BINARY)
        except Exception:
            pass


@contextlib.contextmanager
def _guard_stdout_during_build(target_fd: int = 2):
    """セッション生成中だけ fd1 (stdout) を target_fd へ向ける。

    stdout は UW2P プロトコル専用だが、VAIML コンパイル中の AIE コンパイラ等
    ネイティブ側は "Old buffers:" のような行を fd1 へ書く (b1 ログで確認)。
    アプリ経路ではそれがプロトコル混線→親の desync 終了を起こすため、
    生成中は fd レベルで退避し、終了後に復元する (Python の print 規律だけでは
    ネイティブ/孫プロセスの書込みを防げない)。
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:
            pass
    saved = None
    try:
        saved = os.dup(1)
        os.dup2(target_fd, 1)
        _set_binary_1()
    except OSError as exc:
        _log(f"stdout guard unavailable ({exc}); continuing unguarded")
        saved = None
    try:
        yield
    finally:
        try:
            sys.stdout.flush()
        except Exception:
            pass
        if saved is not None:
            try:
                os.dup2(saved, 1)
            finally:
                os.close(saved)
            _set_binary_1()


def _build_session(args: argparse.Namespace):
    """ORT セッションを別スレッドで生成し、期限で打ち切る。"""
    if ort is None:
        raise RuntimeError("onnxruntime is not installed")
    try:
        ort.set_default_logger_severity(3)
    except Exception:
        pass
    allow_compile = bool(args.allow_compile)
    deadline = float(args.compile_timeout) if allow_compile else REQUIRE_CACHE_TIMEOUT
    box: dict = {}

    def _build() -> None:
        started = time.perf_counter()
        try:
            options = [{
                "cache_dir": str(args.cache_dir),
                "cache_key": str(args.cache_key),
                "enable_cache_file_io_in_mem": 0,
            }]
            session = ort.InferenceSession(
                str(args.model),
                providers=["VitisAIExecutionProvider"],
                provider_options=options,
            )
            box["session"] = session
            box["build_s"] = time.perf_counter() - started
        except BaseException as exc:  # noqa: BLE001 - 起動失敗の全文を親へ
            import traceback

            box["error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            box["build_s"] = time.perf_counter() - started

    thread = threading.Thread(target=_build, name="ort-build", daemon=True)
    with _guard_stdout_during_build():
        thread.start()
        thread.join(timeout=deadline)
    if thread.is_alive():
        mode = "compile" if allow_compile else "require-cache"
        _log(f"WATCHDOG exceeded ({mode} {deadline:g}s) -> exit")
        # ビルドスレッドは VAIML コンパイル中の可能性があり join 不能のため即死する。
        os._exit(98 if allow_compile else 99)
    if "error" in box:
        _log(f"SESSION CREATE FAILED build={box['build_s']:.1f}s\n{box['error']}")
        raise RuntimeError(box["error"])
    session = box["session"]
    build_s = float(box["build_s"])
    _log(f"session_build={build_s:.1f}s providers={session.get_providers()}")
    if not allow_compile and build_s > REQUIRE_CACHE_TIMEOUT:
        _log("build exceeded 60s (possible recompile) -> exit(99)")
        raise SystemExit(99)
    return session, build_s


def _serve(args: argparse.Namespace) -> int:
    if np is None:
        raise RuntimeError("numpy is not installed")
    model_path = str(args.model)
    if not os.path.isfile(model_path):
        _log(f"model not found: {model_path}")
        return 1
    os.makedirs(str(args.cache_dir), exist_ok=True)

    try:
        session, build_s = _build_session(args)
    except SystemExit as exc:
        return int(exc.code or 1)
    except Exception:
        return 1

    try:
        input_descs = [_describe_tensor(v) for v in session.get_inputs()]
        output_descs = [_describe_tensor(v) for v in session.get_outputs()]
    except Exception as exc:
        _log(f"failed to describe tensors: {exc}")
        return 1
    if not input_descs or not output_descs:
        _log("model has no inputs or no outputs")
        return 1

    try:
        ort_version = str(ort.__version__)
    except Exception:
        ort_version = "unknown"
    ready = {
        "role": args.role,
        "generation": int(args.generation),
        "ort_version": ort_version,
        "providers": list(session.get_providers()),
        "pid": os.getpid(),
        "inputs": input_descs,
        "outputs": output_descs,
        "cache_key": str(args.cache_key),
        "model_name": os.path.basename(model_path),
        "model_bytes": os.path.getsize(model_path),
        "build_s": build_s,
    }
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer
    proto.send_message(stdout, proto.T_READY, int(args.generation), 0, proto.pack_json(ready))
    _log(f"ready sent (role={args.role} generation={args.generation})")

    inject_sleep = float(os.environ.get("UEU_WORKER_SLEEP_S", "0") or 0)
    sleep_file = os.environ.get("UEU_WORKER_SLEEP_FILE", "")

    def _inject_seconds() -> float:
        if sleep_file:
            try:
                with open(sleep_file, "r", encoding="utf-8") as handle:
                    return max(0.0, float((handle.read() or "0").strip() or 0))
            except (OSError, ValueError):
                return 0.0
        return inject_sleep

    if inject_sleep > 0 or sleep_file:
        _log(f"TEST HOOK armed (static={inject_sleep:g}s file={sleep_file or '-'})")
    n_req = 0
    input_names = [desc["name"] for desc in input_descs]
    while True:
        try:
            header = proto.recv_header(stdin)
        except proto.CleanEOF:
            _log(f"EOF after {n_req} reqs, exiting")
            return 0
        except proto.TruncatedError as exc:
            _log(f"truncated header after {n_req} reqs ({exc}), exiting(2)")
            return 2
        except proto.ProtocolError as exc:
            _log(f"bad header after {n_req} reqs ({exc}), exiting(2)")
            return 2
        mtype, generation, request_id, payload_len = header
        if mtype == proto.T_QUIT:
            if payload_len:
                try:
                    proto.recv_payload(stdin, payload_len, "quit")
                except EOFError:
                    pass
            _log(f"QUIT after {n_req} reqs, exiting")
            return 0
        if mtype == proto.T_PING:
            if payload_len:
                try:
                    proto.recv_payload(stdin, payload_len, "ping")
                except EOFError:
                    pass
            proto.send_message(stdout, proto.T_PING, int(args.generation), request_id, b"")
            continue
        if mtype != proto.T_DATA:
            _log(f"unexpected {proto.type_name(mtype)} after {n_req} reqs, exiting(2)")
            return 2
        try:
            payload = proto.recv_payload(stdin, payload_len, "data")
        except proto.TruncatedError as exc:
            _log(f"truncated DATA body ({exc}), exiting(2)")
            return 2
        try:
            blobs = proto.unpack_tensors(payload)
        except proto.ProtocolError as exc:
            _log(f"bad DATA payload ({exc}), exiting(2)")
            return 2
        if len(blobs) != len(input_descs):
            _log(f"DATA tensor count {len(blobs)} != inputs {len(input_descs)}, exiting(2)")
            return 2
        stalled = _inject_seconds()
        if stalled > 0:
            time.sleep(stalled)
        started = time.perf_counter()
        try:
            feed = {}
            for name, desc, blob in zip(input_names, input_descs, blobs):
                feed[name] = _reshape(desc, blob)
            outputs = session.run(None, feed)
        except Exception as exc:
            err = {
                "stage": args.role,
                "tensor": "",
                "nonfinite_rate": -1.0,
                "generation": int(args.generation),
                "request_id": request_id,
                "message": f"{type(exc).__name__}: {exc}",
            }
            try:
                proto.send_message(
                    stdout, proto.T_ERROR, int(args.generation), request_id,
                    proto.pack_json(err),
                )
            except OSError:
                return 0
            continue
        infer_s = time.perf_counter() - started
        bad_name = ""
        bad_rate = 0.0
        out_blobs: list[bytes] = []
        for desc, out in zip(output_descs, outputs):
            arr = np.ascontiguousarray(out)
            rate = _nonfinite_rate(arr)
            if rate > 0.0 and (bad_rate == 0.0 or rate >= bad_rate):
                bad_name = str(desc["name"])
                bad_rate = rate
            out_blobs.append(arr.tobytes())
        n_req += 1
        try:
            if bad_rate > 0.0:
                # nan_to_num で隠さない。親が復旧手順に入る。
                err = {
                    "stage": args.role,
                    "tensor": bad_name,
                    "nonfinite_rate": bad_rate,
                    "generation": int(args.generation),
                    "request_id": request_id,
                    "infer_s": infer_s,
                    "message": (
                        f"non-finite output in {args.role}: tensor={bad_name} "
                        f"rate={bad_rate:.6f}"
                    ),
                }
                proto.send_message(
                    stdout, proto.T_ERROR, int(args.generation), request_id,
                    proto.pack_json(err),
                )
            else:
                proto.send_message(
                    stdout, proto.T_DATA, int(args.generation), request_id,
                    proto.pack_tensors(out_blobs),
                )
        except (BrokenPipeError, OSError):
            _log(f"parent gone after {n_req} reqs, exiting")
            return 0
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", required=True, choices=("front", "back"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--cache-key", required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--require-cache", action="store_true", default=False)
    mode.add_argument("--allow-compile", action="store_true", default=False)
    parser.add_argument("--compile-timeout", type=float, default=4 * 3600.0)
    parser.add_argument("--generation", type=int, default=0)
    return parser


def main() -> int:
    if os.name == "nt":
        import msvcrt

        msvcrt.setmode(sys.stdin.fileno(), os.O_BINARY)
        msvcrt.setmode(sys.stdout.fileno(), os.O_BINARY)
    try:
        return _serve(_parser().parse_args())
    except SystemExit:
        raise
    except Exception as exc:
        _log(f"fatal: {exc.__class__.__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

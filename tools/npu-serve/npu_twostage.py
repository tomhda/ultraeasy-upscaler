#!/usr/bin/env python
"""AdcSR 前半/後半の2プロセス中継 (TwoStageSession)。

B1 §3.4 の同一プロセス内 G→F 汚染 (G を1回実行すると F が恒久 NaN) を、
F/G 別プロセス＋親経由の境界テンソル受け渡しで回避する (B2 で 100/100 正常、
2.05 s/タイルを確認)。重ね実行は逐次より遅いため並列化しない
(未完了タイル 1 件に固定)。

構成:
  アプリ (ServeClient, UEU) <-> 本モジュール (親。NPU セッションは作らない)
      <-> F ワーカー (npu_worker.py --role front。別プロセス)
      <-> G ワーカー (npu_worker.py --role back。別プロセス)

oracle 必須の反映:
  - 復旧中の内部 ERROR はアプリへ出さない。画像 1 枚に応答 1 件を厳守し、
    復旧成功後の DATA か、断念後の TWO_STAGE_FATAL の ERROR を 1 件だけ返す。
  - F の全 3 出力・G 出力・合成後の有限値を検査する (nan_to_num で隠さない)。
  - 復旧は「両ワーカー停止・終了確認 → 新世代で起動 → セルフテスト →
    元入力から同じタイルを再実行」。1 画像につき 1 回まで。
  - タイムアウトは送信+flush+受信の 1 往復に適用し、監視は別スレッドで行う。
    期限切れは TerminateProcess → パイプと読取スレッドを回収 → 復旧手順へ。
  - NpuSession は使わない (出力÷入力の倍率推定が F/G に合わない)。
    倍率は F の画像入力寸法と G の画像出力寸法から求め 4 を確認する。
  - 合成は逐次加算 (全タイル出力をリスト保持しない)。
  - 1 モデル経路 (npu_serve.py の NpuSession・_merge_tiles・reflect padding・
    clip(...).astype(uint8)) は本モジュールから呼ばない。round に変えない。

診断用環境変数 (独立):
  UEU_NPU_SEAMFIX=0    seam-fix を無効化 (既定は有効・要テンプレート一致)。
  UEU_NPU_CROSSFADE=0  クロスフェードを無効化しハードカットにする (既定は有効)。

試験用フック (本番では使わない):
  UEU_TS_KILL_ROLE=front|back + UEU_TS_KILL_AT_TILE=<1始まり>:
    指定タイルの直前に対象ワーカーを TerminateProcess で落とす
    (外部 kill と同じ死に方。復旧経路の決定性試験用)。
  (ワーカー側: UEU_WORKER_SLEEP_S / UEU_WORKER_SLEEP_FILE は npu_worker.py 参照)
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import npu_job
import npu_proto as proto

try:
    import numpy as np
except ImportError:  # pragma: no cover - NPU 環境・アプリ環境では常に存在
    np = None  # type: ignore[assignment]

_HERE = Path(__file__).resolve().parent
WORKER_PATH = _HERE / "npu_worker.py"
SELFTEST_DIR = _HERE / "selftest"
SELFTEST_TILE = SELFTEST_DIR / "adcsr_tile.npy"
SELFTEST_REF = SELFTEST_DIR / "adcsr_ref_stats.json"

REQUIRE_CACHE_TIMEOUT = 60.0
DEFAULT_WORKER_TIMEOUT = 60.0
DEFAULT_COMPILE_TIMEOUT = 4 * 3600.0
READY_MARGIN = 30.0
QUIT_TIMEOUT = 5.0
KILL_WAIT_TIMEOUT = 10.0

TWO_STAGE_FATAL = "TWO_STAGE_FATAL"
#: 外側 float 合成バッファの上限 (tos 5120x2136 相当 131MiB に対する余裕)。
MAX_OUT_FLOAT_BYTES = 2 * 1024 * 1024 * 1024


class TwoStageError(RuntimeError):
    """2 段モード内部の異常 (画像単位の復旧対象になりうる)。"""


class WorkerReported(TwoStageError):
    """ワーカーが ERROR 応答を返した (非有限値など)。"""

    def __init__(self, stage: str, tensor: str, rate: float, message: str) -> None:
        super().__init__(message)
        self.stage = stage
        self.tensor = tensor
        self.rate = rate


class WorkerTimeout(TwoStageError):
    """ワーカー 1 往復が --worker-timeout を超過した。"""

    def __init__(self, stage: str, timeout: float, elapsed: float) -> None:
        super().__init__(f"{stage} transaction timed out ({elapsed:.1f}s > {timeout:g}s)")
        self.stage = stage
        self.timeout = timeout
        self.elapsed = elapsed


class WorkerGone(TwoStageError):
    """ワーカーの応答前にプロセスが消えた・同期が破綻した。"""


class StartupFailed(TwoStageError):
    """ワーカー起動・READY・照合・セルフテストの失敗 (復旧不能として扱う)。"""


class SelftestFailed(StartupFailed):
    """セルフテスト不合格。"""


class TwoStageFatal(TwoStageError):
    """アプリへ返す 1 件の ERROR の元になる致命的異常。

    str() は ``TWO_STAGE_FATAL ...`` で始まり、ServeClient が
    HelperOutputInvalid へ変換する。
    """

    def __init__(
        self, *, stage: str, tensor: str, rate: float, recoveries: int, detail: str
    ) -> None:
        super().__init__(
            f"{TWO_STAGE_FATAL} stage={stage} tensor={tensor} "
            f"nonfinite_rate={rate:.6f} recoveries={recoveries} {detail}"
        )
        self.stage = stage
        self.tensor = tensor
        self.rate = rate
        self.recoveries = recoveries


# ------------------------------------------------------------ ユーティリティ


def _log(message: str) -> None:
    print(f"[twostage pid={os.getpid()}] {message}", file=sys.stderr, flush=True)


def _env_disabled(name: str) -> bool:
    return os.environ.get(name, "1") == "0"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path | str) -> dict:
    """配布マニフェストを読み、必須キーを検証する。"""
    raw = Path(path).read_bytes()
    try:
        manifest = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise StartupFailed(f"manifest is not valid JSON: {path} ({exc})") from exc
    if not isinstance(manifest, dict):
        raise StartupFailed(f"manifest must be an object: {path}")
    for key in ("model_family", "front", "back", "boundary"):
        if key not in manifest:
            raise StartupFailed(f"manifest lacks {key!r}: {path}")
    for side in ("front", "back"):
        for key in ("file", "sha256", "cache_key"):
            if key not in manifest[side]:
                raise StartupFailed(f"manifest[{side!r}] lacks {key!r}: {path}")
    boundary = manifest["boundary"]
    if not isinstance(boundary, list) or len(boundary) != 3:
        raise StartupFailed(f"manifest boundary must list 3 tensor names: {path}")
    manifest["_path"] = str(path)
    manifest["_sha256"] = hashlib.sha256(raw).hexdigest()
    return manifest


def _ort_info() -> tuple[str, str]:
    try:
        import onnxruntime as ort

        return str(getattr(ort, "__version__", "unknown")), str(getattr(ort, "__file__", "?"))
    except Exception as exc:  # noqa: BLE001 - 情報行のため失敗を許す
        return f"unavailable({exc})", "?"


def _xrt_version() -> str:
    exe = shutil.which("xrt-smi")
    if exe is None:
        return "xrt-smi not found"
    try:
        proc = subprocess.run(
            [exe, "--version"], capture_output=True, text=True, timeout=15,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        first = (proc.stdout or proc.stderr or "").strip().splitlines()
        return first[0][:200] if first else f"xrt-smi exit={proc.returncode}"
    except Exception as exc:  # noqa: BLE001 - 情報行のため失敗を許す
        return f"xrt-smi query failed: {type(exc).__name__}"


def _npu_device() -> str:
    exe = shutil.which("xrt-smi")
    if exe is None:
        return "unknown (xrt-smi not found)"
    try:
        proc = subprocess.run(
            [exe, "examine", "--report", "platform", "--format", "JSON"],
            capture_output=True, text=True, timeout=20,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        payload = (proc.stdout or "").strip()
        if not payload:
            return "unknown (empty platform report)"
        info = json.loads(payload)
        text = json.dumps(info)
        return text[:300].replace("\n", " ")
    except Exception as exc:  # noqa: BLE001 - 情報行のため失敗を許す
        return f"unknown ({type(exc).__name__})"


# ------------------------------------------------------------ ワーカー接続


class _WorkerConn:
    """ワーカー 1 台への接続 (起動・READY・往復・終了)。

    監視スレッド (daemon) で 1 往復 (送信+flush+受信) を包み、
    --worker-timeout 超過で TerminateProcess → パイプ回収する。
    """

    def __init__(
        self,
        *,
        role: str,
        model: Path,
        cache_dir: Path,
        cache_key: str,
        generation: int,
        require_cache: bool,
        compile_timeout: float,
    ) -> None:
        self.role = role
        self.generation = generation
        argv = [
            sys.executable,
            "-u",
            str(WORKER_PATH),
            "--role",
            role,
            "--model",
            str(model),
            "--cache-dir",
            str(cache_dir),
            "--cache-key",
            cache_key,
            "--generation",
            str(generation),
        ]
        if require_cache:
            argv.append("--require-cache")
        else:
            argv += ["--allow-compile", "--compile-timeout", str(compile_timeout)]
        self._argv = argv
        self._proc = npu_job.spawn_in_job(argv)
        self.pid = getattr(self._proc, "pid", -1)
        self._request_id = 0
        self._closed = False
        self.ready: dict = {}
        self._stderr_tail: list[str] = []
        self._stderr_forwarded = 0
        self._start_stderr_forward()

    # -- low level -------------------------------------------------

    def _run_watched(self, func, timeout: float, what: str):
        box: dict = {}

        def _target() -> None:
            try:
                box["value"] = func()
            except BaseException as exc:  # noqa: BLE001 - 呼び出し側へ引き渡す
                box["error"] = exc

        thread = threading.Thread(target=_target, name=f"uw2p-{self.role}-{what}", daemon=True)
        thread.start()
        thread.join(timeout=timeout)
        if thread.is_alive():
            raise WorkerTimeout(self.role, timeout, timeout)
        if "error" in box:
            raise box["error"]
        return box.get("value")

    def _terminate(self) -> None:
        proc = self._proc
        try:
            if proc is not None and proc.poll() is None:
                proc.terminate()
        except Exception:
            pass

    def _kill(self) -> None:
        proc = self._proc
        try:
            if proc is not None and proc.poll() is None:
                proc.kill()
        except Exception:
            pass

    def abort(self) -> None:
        """期限切れ用: 強制終了 → パイプ回収 (読取スレッドは daemon のため残置可)。"""
        self._kill()
        self._close_pipes()

    def _close_pipes(self) -> None:
        proc = self._proc
        if proc is None:
            return
        for name in ("stdin", "stdout", "stderr"):
            try:
                stream = getattr(proc, name, None)
                if stream is not None:
                    stream.close()
            except Exception:
                pass

    def _start_stderr_forward(self) -> None:
        """ワーカーの stderr を親ログへ転送 ([F]/[G] 付き)。死因の証拠用。

        ワーカーは起動行と異常時しか書かない想定だが、ORT ネイティブログの
        大量排出に備えて live 転送は 2000 行で打ち切る (tail は保持する)。
        """

        def _drain() -> None:
            try:
                stream = self._proc.stderr
                if stream is None:
                    return
                for raw in iter(stream.readline, b""):
                    try:
                        line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                    except Exception:
                        continue
                    if not line:
                        continue
                    self._stderr_tail.append(line)
                    if len(self._stderr_tail) > 200:
                        del self._stderr_tail[: len(self._stderr_tail) - 200]
                    if self._stderr_forwarded < 2000:
                        self._stderr_forwarded += 1
                        _log(f"[{self.role} pid={self.pid}] {line}")
                    elif self._stderr_forwarded == 2000:
                        self._stderr_forwarded += 1
                        _log(f"[{self.role} pid={self.pid}] (further stderr suppressed)")
            except Exception:
                pass

        thread = threading.Thread(target=_drain, name=f"uw2p-{self.role}-stderr", daemon=True)
        thread.start()

    # -- startup ----------------------------------------------------

    def wait_ready(self, timeout: float) -> dict:
        try:
            info = self._run_watched(self._read_ready_blocking, timeout, "ready")
        except WorkerTimeout:
            self.abort()
            raise StartupFailed(
                f"{self.role} READY timeout after {timeout:g}s "
                f"(pid={self.pid}; recompile suspected, see worker stderr)"
            )
        assert isinstance(info, dict)
        self.ready = info
        return info

    def _read_ready_blocking(self) -> dict:
        proc = self._proc
        try:
            mtype, generation, request_id, payload_len = proto.recv_header(proc.stdout)
        except proto.CleanEOF as exc:
            raise StartupFailed(f"{self.role} exited before READY (pid={self.pid})") from exc
        except proto.TruncatedError as exc:
            raise StartupFailed(f"{self.role} READY truncated ({exc}; pid={self.pid})") from exc
        if mtype != proto.T_READY:
            raise StartupFailed(
                f"{self.role} expected READY, got {proto.type_name(mtype)} "
                f"(pid={self.pid}; ORT stdout pollution suspected)"
            )
        if generation != self.generation:
            raise StartupFailed(
                f"{self.role} generation mismatch: got {generation}, want {self.generation}"
            )
        if request_id != 0:
            raise StartupFailed(f"{self.role} READY request_id must be 0, got {request_id}")
        try:
            payload = proto.recv_payload(proc.stdout, payload_len, "ready")
        except proto.TruncatedError as exc:
            raise StartupFailed(f"{self.role} READY body truncated ({exc})") from exc
        try:
            info = proto.unpack_json(payload)
        except proto.ProtocolError as exc:
            raise StartupFailed(f"{self.role} READY is not JSON ({exc})") from exc
        for key in ("inputs", "outputs"):
            if not isinstance(info.get(key), list) or not info[key]:
                raise StartupFailed(f"{self.role} READY lacks {key!r}")
        return info

    # -- transaction -------------------------------------------------

    def transact(self, blobs: list[bytes], timeout: float) -> list[bytes]:
        self._request_id += 1
        request_id = self._request_id
        started = time.perf_counter()
        try:
            result = self._run_watched(
                lambda: self._roundtrip_blocking(blobs, request_id), timeout, f"req{request_id}"
            )
        except WorkerTimeout:
            elapsed = time.perf_counter() - started
            self.abort()
            raise WorkerTimeout(self.role, timeout, elapsed)
        assert isinstance(result, list)
        return result

    def _roundtrip_blocking(self, blobs: list[bytes], request_id: int) -> list[bytes]:
        proc = self._proc
        payload = proto.pack_tensors(blobs)
        try:
            proto.send_message(proc.stdin, proto.T_DATA, self.generation, request_id, payload)
        except (BrokenPipeError, OSError) as exc:
            raise WorkerGone(f"{self.role} send failed (pid={self.pid}): {exc}") from exc
        try:
            mtype, generation, rid, payload_len = proto.recv_header(proc.stdout)
        except proto.CleanEOF as exc:
            raise WorkerGone(f"{self.role} exited mid-request {request_id}") from exc
        except proto.TruncatedError as exc:
            raise WorkerGone(f"{self.role} truncated reply ({exc})") from exc
        if generation != self.generation:
            raise WorkerGone(
                f"{self.role} stale generation {generation} != {self.generation} "
                f"(request {request_id})"
            )
        if rid != request_id:
            raise WorkerGone(
                f"{self.role} request_id mismatch: got {rid}, want {request_id}"
            )
        try:
            body = proto.recv_payload(proc.stdout, payload_len, "reply")
        except proto.TruncatedError as exc:
            raise WorkerGone(f"{self.role} truncated reply body ({exc})") from exc
        if mtype == proto.T_ERROR:
            try:
                err = proto.unpack_json(body)
            except proto.ProtocolError:
                err = {"stage": self.role, "tensor": "", "nonfinite_rate": -1.0,
                       "message": body[:200].decode("utf-8", "replace")}
            raise WorkerReported(
                str(err.get("stage", self.role)),
                str(err.get("tensor", "")),
                float(err.get("nonfinite_rate", -1.0)),
                str(err.get("message", "worker ERROR")),
            )
        if mtype != proto.T_DATA:
            raise WorkerGone(f"{self.role} unexpected {proto.type_name(mtype)} in request {request_id}")
        try:
            return proto.unpack_tensors(body)
        except proto.ProtocolError as exc:
            raise WorkerGone(f"{self.role} bad DATA body ({exc})") from exc

    # -- shutdown -----------------------------------------------------

    def quit_and_wait(self) -> int | None:
        """QUIT→EOF の通常終了を試み、終了コードを返す (確認不能なら None)。"""
        proc = self._proc
        if proc is None:
            return None
        try:
            if proc.poll() is None and proc.stdin is not None:
                try:
                    proto.send_message(proc.stdin, proto.T_QUIT, self.generation, 0, b"")
                except (BrokenPipeError, OSError):
                    pass
                try:
                    proc.stdin.close()
                except (BrokenPipeError, OSError):
                    pass
            try:
                return proc.wait(timeout=QUIT_TIMEOUT)
            except subprocess.TimeoutExpired:
                pass
        except Exception:
            pass
        try:
            if proc.poll() is None:
                proc.kill()
            return proc.wait(timeout=KILL_WAIT_TIMEOUT)
        except Exception:
            return None if proc.poll() is None else proc.returncode

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.quit_and_wait()
        finally:
            self._close_pipes()
            proc = self._proc
            if proc is not None and hasattr(proc, "close"):
                try:
                    proc.close()
                except Exception:
                    pass


# ------------------------------------------------------------ 逐次加算合成


class _Canvas:
    """2 段モード専用の逐次加算合成 (全タイル出力を保持しない)。

    crossfade 有効時は winml-sr MergeTile (blend) と同じ線形テーパー＋重み正規化、
    無効時 (UEU_NPU_CROSSFADE=0) はコアのハードカット配置。
    単位は CHW float 0-1。
    """

    def __init__(self, out_w: int, out_h: int, *, crossfade: bool) -> None:
        if np is None:
            raise RuntimeError("numpy is not installed")
        self.out_w = out_w
        self.out_h = out_h
        self.crossfade = crossfade
        self.accum = np.zeros((3, out_h, out_w), dtype=np.float32)
        self.weights = np.zeros((out_h, out_w), dtype=np.float32) if crossfade else None

    def add_tile(
        self,
        tile: np.ndarray,
        tile_w: int,
        tile_h: int,
        overlap: int,
        ntx: int,
        index: int,
    ) -> None:
        if self.crossfade:
            self._add_blend(tile, tile_w, tile_h, overlap, ntx, index)
        else:
            self._add_hardcut(tile, tile_w, tile_h, overlap, ntx, index)

    def _add_hardcut(
        self, tile: np.ndarray, tile_w: int, tile_h: int, overlap: int, ntx: int, index: int
    ) -> None:
        core_w = tile_w - 2 * overlap
        core_h = tile_h - 2 * overlap
        iy, ix = divmod(index, ntx)
        y0, x0 = iy * core_h, ix * core_w
        copy_h = min(core_h, self.out_h - y0)
        copy_w = min(core_w, self.out_w - x0)
        if copy_h <= 0 or copy_w <= 0:
            return
        core = tile[:, overlap:overlap + copy_h, overlap:overlap + copy_w]
        self.accum[:, y0:y0 + copy_h, x0:x0 + copy_w] = core

    def _add_blend(
        self, tile: np.ndarray, tile_w: int, tile_h: int, overlap: int, ntx: int, index: int
    ) -> None:
        # MergeTile (blend) と同じ式。重みは (tx+1)/overlap 等の線形テーパー。
        assert self.weights is not None
        core_w = tile_w - 2 * overlap
        core_h = tile_h - 2 * overlap
        iy, ix = divmod(index, ntx)
        y0, x0 = iy * core_h, ix * core_w
        start_x = max(0, x0 - overlap)
        end_x = min(self.out_w, x0 + core_w + overlap)
        start_y = max(0, y0 - overlap)
        end_y = min(self.out_h, y0 + core_h + overlap)
        if end_x <= start_x or end_y <= start_y:
            return
        ty = np.arange(start_y, end_y, dtype=np.float32) - float(y0 - overlap)
        tx = np.arange(start_x, end_x, dtype=np.float32) - float(x0 - overlap)
        wy = np.where(ty < overlap, (ty + 1.0) / overlap,
                      np.where(ty >= overlap + core_h, (tile_h - ty) / overlap, 1.0))
        wx = np.where(tx < overlap, (tx + 1.0) / overlap,
                      np.where(tx >= overlap + core_w, (tile_w - tx) / overlap, 1.0))
        weight = np.maximum(0.0001, wy[:, None] * wx[None, :]).astype(np.float32)
        ty0 = start_y - (y0 - overlap)
        tx0 = start_x - (x0 - overlap)
        region = tile[:, ty0:ty0 + (end_y - start_y), tx0:tx0 + (end_x - start_x)]
        old = self.weights[start_y:end_y, start_x:end_x]
        total = old + weight
        acc = self.accum[:, start_y:end_y, start_x:end_x]
        acc *= old
        acc += region * weight
        acc /= total
        self.weights[start_y:end_y, start_x:end_x] = total


# ------------------------------------------------------------ seam-fix (Python 移植)


def _box_blur_1pass(img: np.ndarray, r: int, axis: int) -> np.ndarray:
    """flatlib.box_blur_1pass と同じ式 (一様窓・reflect・cumsum)。uint8 化前 float 用。"""
    if r <= 0:
        return img.copy()
    w = 2 * r + 1
    pad = [(0, 0), (0, 0)]
    pad[axis] = (r, r)
    p = np.pad(img, pad, mode="reflect")
    cs = np.cumsum(p, axis=axis)
    head = [slice(None), slice(None)]
    head[axis] = slice(0, 1)
    cs0 = np.concatenate([np.zeros_like(cs[tuple(head)]), cs], axis=axis)
    a = [slice(None), slice(None)]
    a[axis] = slice(0, cs0.shape[axis] - w)
    b = [slice(None), slice(None)]
    b[axis] = slice(w, cs0.shape[axis])
    return (cs0[tuple(b)] - cs0[tuple(a)]) / w


def _gauss_blur(img: np.ndarray, sigma: float, passes: int = 3) -> np.ndarray:
    r = int(round(sigma))
    out = np.ascontiguousarray(img)
    for _ in range(passes):
        out = _box_blur_1pass(out, r, 1)
        out = _box_blur_1pass(out, r, 0)
    return out


def _local_std(y: np.ndarray, r: int = 16) -> np.ndarray:
    m = _box_blur_1pass(_box_blur_1pass(y, r, 1), r, 0)
    m2 = _box_blur_1pass(_box_blur_1pass(y * y, r, 1), r, 0)
    return np.sqrt(np.maximum(m2 - m * m, 0.0))


def load_seam_template(path: Path | str) -> dict:
    """seam テンプレート JSON を読み、既定値 (SeamFix.cs と同じ) で補う。"""
    with open(path, "r", encoding="utf-8") as handle:
        root = json.load(handle)
    pitch = int(root["pitch"])
    gx = [float(v) for v in root["gx"]]
    gy = [float(v) for v in root["gy"]]
    if len(gx) != pitch or len(gy) != pitch:
        raise ValueError(
            f"seam template の長さが pitch と不一致: gx={len(gx)} gy={len(gy)} "
            f"pitch={pitch} ({path})"
        )
    return {
        "pitch": pitch,
        "gx": gx,
        "gy": gy,
        "tau": float(root.get("tau", 4.0)),
        "clip": float(root.get("clip", 2.0)),
        "feather_sigma": float(root.get("feather_sigma", 8.0)),
        "std_window": int(root.get("std_window", 33)),
    }


def apply_seam_fix_inplace(merged_chw01: np.ndarray, template: dict) -> float:
    """merged (CHW float 0-1) を in-place 補正し、処理 ms を返す。

    数値仕様: SeamFix.cs・flatlib/methods.apply_template (gate='none') と同じ式。
    33x33 局所 std → hard マスク (<= tau) → σ8 フェザー (box×3。Gaussian に置換しない)
    → -(gx+gy)*mask を ±clip で RGB 全 ch に (÷255 で) 加算。
    """
    started = time.perf_counter()
    out_h, out_w = merged_chw01.shape[1], merged_chw01.shape[2]
    merged_f = np.ascontiguousarray(merged_chw01, dtype=np.float32)
    y = (
        0.299 * merged_f[0] + 0.587 * merged_f[1] + 0.114 * merged_f[2]
    ).astype(np.float32) * 255.0
    r = int(template["std_window"]) // 2
    std = _local_std(y, r)
    mask = np.where(std <= float(template["tau"]), 1.0, 0.0).astype(np.float32)
    mask = _gauss_blur(mask, float(template["feather_sigma"])).astype(np.float32)
    mask[mask < 1e-3] = 0.0
    pitch = int(template["pitch"])
    half = pitch // 2
    gx = np.asarray(template["gx"], dtype=np.float64)
    gy = np.asarray(template["gy"], dtype=np.float64)
    tmpx = np.empty(pitch, dtype=np.float64)
    tmpy = np.empty(pitch, dtype=np.float64)
    for k in range(pitch):
        d = k if k < pitch - half else k - pitch
        tmpx[k] = gx[d + half]
        tmpy[k] = gy[d + half]
    ph = pitch
    kx = ((np.arange(out_w) - ph) % pitch)
    ky = ((np.arange(out_h) - ph) % pitch)
    field = tmpx[kx][None, :] + tmpy[ky][:, None]
    clip = float(template["clip"])
    corr = np.clip(-field * mask.astype(np.float64), -clip, clip)
    add = (corr / 255.0).astype(np.float32)
    merged_chw01[0] += add
    merged_chw01[1] += add
    merged_chw01[2] += add
    return (time.perf_counter() - started) * 1000.0


def _nonfinite_rate(arr: np.ndarray) -> float:
    finite = np.isfinite(arr)
    return float((~finite).mean()) if finite.size else 0.0


# ------------------------------------------------------------ タイル分割
# npu_serve._split_tiles と同じ reflect パディング＋タイル分割。
# 循環 import を避けるため同形の実装を持つ。同等性は tests で検証する。


def _split_tiles(img_chw, patch_hw: tuple[int, int], overlap: int):
    import math as _math

    _channels, height, width = img_chw.shape
    patch_h, patch_w = patch_hw
    core_h = patch_h - 2 * overlap
    core_w = patch_w - 2 * overlap
    if core_h <= 0 or core_w <= 0:
        raise ValueError("tile overlap is too large")

    n_tiles_h = _math.ceil(height / core_h)
    n_tiles_w = _math.ceil(width / core_w)
    padded_h = n_tiles_h * core_h
    padded_w = n_tiles_w * core_w

    img_pad = np.pad(
        img_chw,
        pad_width=((0, 0), (0, padded_h - height), (0, padded_w - width)),
        mode="reflect",
    )
    big_pad = np.pad(
        img_pad,
        pad_width=((0, 0), (overlap, overlap), (overlap, overlap)),
        mode="reflect",
    )

    tiles = []
    for iy in range(n_tiles_h):
        for ix in range(n_tiles_w):
            y0 = iy * core_h
            x0 = ix * core_w
            tiles.append(big_pad[:, y0:y0 + patch_h, x0:x0 + patch_w])
    return tiles, (height, width), (padded_h, padded_w)


# ------------------------------------------------------------ TwoStageSession


def _image_tensor(desc_list: list[dict]) -> dict:
    """画像テンソル ([?,3,H,W] / [?,H,W,3]) を選ぶ。無ければ ValueError。"""
    for desc in desc_list:
        shape = list(desc.get("shape", []))
        if len(shape) == 4 and (shape[1] == 3 or shape[3] == 3):
            return desc
    raise ValueError(f"no image tensor in {[d.get('name') for d in desc_list]}")


def _spatial_of(desc: dict) -> tuple[int, int]:
    shape = list(desc["shape"])
    if shape[1] == 3:
        return int(shape[2]), int(shape[3])
    return int(shape[1]), int(shape[2])


class TwoStageSession:
    """F ワーカー→G ワーカーの 2 段セッション。親は NPU セッションを作らない。

    使い方:
      session = TwoStageSession(...)
      session.start()          # 起動・照合・セルフテスト・READY 前の全検査
      srgb, ntiles = session.upscale_image_rgb(rgb)
      session.close()
    """

    def __init__(
        self,
        *,
        front_model: Path,
        front_cache_key: str,
        back_model: Path,
        back_cache_key: str,
        cache_dir: Path,
        boundary_names: list[str] | None,
        overlap: int,
        worker_timeout: float = DEFAULT_WORKER_TIMEOUT,
        require_cache: bool = True,
        compile_timeout: float = DEFAULT_COMPILE_TIMEOUT,
        seam_template: dict | None = None,
        model_family: str | None = None,
    ) -> None:
        if np is None:
            raise RuntimeError("numpy is not installed")
        self.front_model = Path(front_model)
        self.back_model = Path(back_model)
        self.front_cache_key = front_cache_key
        self.back_cache_key = back_cache_key
        self.cache_dir = Path(cache_dir)
        self.boundary_names = list(boundary_names) if boundary_names else None
        self.overlap = overlap
        self.worker_timeout = worker_timeout
        self.require_cache = require_cache
        self.compile_timeout = compile_timeout
        self.seam_template = seam_template
        self.model_family = model_family
        self.generation = 0
        self.front: _WorkerConn | None = None
        self.back: _WorkerConn | None = None
        self.tile_h = 0
        self.tile_w = 0
        self.scale = 0
        self.back_order: list[str] = []
        self.g_out_names: list[str] = []
        self.g_out_shape: tuple[int, ...] = ()
        self._f_out: dict = {}
        self.dead = False

    # -- startup ----------------------------------------------------

    def _startup_timeout(self) -> float:
        if self.require_cache:
            return REQUIRE_CACHE_TIMEOUT + READY_MARGIN
        return self.compile_timeout + READY_MARGIN

    def _start_workers(self) -> None:
        timeout = self._startup_timeout()
        front = _WorkerConn(
            role="front", model=self.front_model, cache_dir=self.cache_dir,
            cache_key=self.front_cache_key, generation=self.generation,
            require_cache=self.require_cache, compile_timeout=self.compile_timeout,
        )
        try:
            front_info = front.wait_ready(timeout)
        except BaseException:
            front.abort()
            front._close_pipes()
            raise
        # 前半 READY〜後半ビルド開始を宣言する (後半コンパイル中も「前半 1/2」と
        # 出たままになるのを防ぐ。start() 側の同タグは冗長のため除去)。
        _log("[stage] back-compile")
        back = _WorkerConn(
            role="back", model=self.back_model, cache_dir=self.cache_dir,
            cache_key=self.back_cache_key, generation=self.generation,
            require_cache=self.require_cache, compile_timeout=self.compile_timeout,
        )
        try:
            back_info = back.wait_ready(timeout)
        except BaseException:
            # 部分起動の巻き戻し: 先に起動した F を確実に終了する。
            back.abort()
            back._close_pipes()
            try:
                front.close()
            except Exception:
                front.abort()
            raise
        self.front = front
        self.back = back
        _log(f"[stage] workers-ready generation={self.generation} "
             f"front_pid={front.pid} front_build={front_info.get('build_s', -1):.1f}s "
             f"back_pid={back.pid} back_build={back_info.get('build_s', -1):.1f}s")
        self._match_and_check(front_info, back_info)

    def _match_and_check(self, front_info: dict, back_info: dict) -> None:
        f_out = {desc["name"]: desc for desc in front_info["outputs"]}
        g_in = {desc["name"]: desc for desc in back_info["inputs"]}
        if self.boundary_names is not None:
            if set(f_out) != set(self.boundary_names) or set(g_in) != set(self.boundary_names):
                raise StartupFailed(
                    f"boundary mismatch: manifest={self.boundary_names} "
                    f"front={sorted(f_out)} back={sorted(g_in)}"
                )
        if set(f_out) != set(g_in):
            raise StartupFailed(
                f"F outputs != G inputs by name: "
                f"front_only={sorted(set(f_out) - set(g_in))} "
                f"back_only={sorted(set(g_in) - set(f_out))}"
            )
        for name in f_out:
            fdesc, gdesc = f_out[name], g_in[name]
            if list(fdesc["shape"]) != list(gdesc["shape"]):
                raise StartupFailed(
                    f"boundary {name!r} shape mismatch: front={fdesc['shape']} back={gdesc['shape']}"
                )
            if fdesc["dtype"] != gdesc["dtype"]:
                raise StartupFailed(
                    f"boundary {name!r} dtype mismatch: front={fdesc['dtype']} back={gdesc['dtype']}"
                )
        self.back_order = [desc["name"] for desc in back_info["inputs"]]
        self.g_out_names = [desc["name"] for desc in back_info["outputs"]]
        if len(self.g_out_names) != 1:
            raise StartupFailed(f"back must have exactly 1 output, got {self.g_out_names}")
        try:
            self.g_out_shape = tuple(int(d) for d in back_info["outputs"][0]["shape"])
        except (KeyError, TypeError, ValueError) as exc:
            raise StartupFailed(f"back output shape unreadable ({exc})") from exc
        self._f_out = f_out
        f_img = _image_tensor(front_info["inputs"])
        g_img = _image_tensor(back_info["outputs"])
        f_h, f_w = _spatial_of(f_img)
        g_h, g_w = _spatial_of(g_img)
        if g_h % f_h != 0 or g_w % f_w != 0 or (g_h // f_h) != (g_w // f_w):
            raise StartupFailed(
                f"invalid two-stage scale: front_input={f_img['shape']} back_output={g_img['shape']}"
            )
        self.scale = g_h // f_h
        if self.scale != 4:
            raise StartupFailed(f"two-stage scale must be 4, got {self.scale}")
        self.tile_h, self.tile_w = f_h, f_w
        _log(f"[worker-info] boundary={sorted(f_out)} scale={self.scale} "
             f"tile={self.tile_w}x{self.tile_h}")

    def _log_startup_block(self) -> None:
        ort_version, ort_file = _ort_info()
        _log(f"[worker-info] ort={ort_version} file={ort_file} ep=VitisAIExecutionProvider")
        _log(f"[worker-info] xrt={_xrt_version()}")
        _log(f"[worker-info] npu-device={_npu_device()}")
        for side, model, key in (
            ("front", self.front_model, self.front_cache_key),
            ("back", self.back_model, self.back_cache_key),
        ):
            try:
                digest = _sha256_file(model)
                size = model.stat().st_size
            except OSError as exc:
                raise StartupFailed(f"{side} model unreadable: {model} ({exc})") from exc
            _log(f"[worker-info] {side} model={model.name} sha256={digest} bytes={size} "
                 f"cache_key={key}")
        _log(f"[worker-info] provider_options={{cache_dir={self.cache_dir}}} "
             f"pid={os.getpid()} generation={self.generation}")

    def _selftest(self) -> None:
        _log("[stage] selftest (2 rounds, independent of --warmup)")
        try:
            tile = np.load(str(SELFTEST_TILE))
        except Exception as exc:
            raise StartupFailed(f"selftest tile unreadable: {SELFTEST_TILE} ({exc})") from exc
        try:
            with open(str(SELFTEST_REF), "r", encoding="utf-8") as handle:
                ref = json.load(handle)
        except Exception as exc:
            raise StartupFailed(f"selftest ref unreadable: {SELFTEST_REF} ({exc})") from exc
        try:
            exp = ref["output"]
            tol = ref["tolerance"]
            exp_mean, exp_std, exp_c8 = float(exp["mean"]), float(exp["std"]), float(exp["center8x8_mean"])
            tol_mean, tol_std, tol_c8 = float(tol["mean"]), float(tol["std"]), float(tol["center8x8_mean"])
        except (KeyError, TypeError, ValueError) as exc:
            raise StartupFailed(f"selftest ref has bad fields ({exc})") from exc
        tile_bytes = np.ascontiguousarray(tile, dtype=np.float32).tobytes()
        for round_no in (1, 2):
            started = time.perf_counter()
            try:
                f_blobs = self._front_transact([tile_bytes])
                f_arrs = self._decode_and_check(f_blobs, self.front_out_order(), "front")
                g_blobs = self._back_transact([f_arrs[name].tobytes() for name in self.back_order])
                g_arrs = self._decode_and_check(g_blobs, self.g_out_names, "back")
                (g_flat,) = [g_arrs[name] for name in self.g_out_names[:1]]
                g_out = np.ascontiguousarray(g_flat.reshape(self.g_out_shape))
            except TwoStageError as exc:
                raise SelftestFailed(f"selftest round {round_no}/2 failed: {exc}") from exc
            mean = float(g_out.mean())
            std = float(g_out.std())
            c8 = float(g_out[0, :, 248:256, 248:256].mean()) if g_out.shape[2] >= 512 else float(g_out.mean())
            ok = (abs(mean - exp_mean) <= tol_mean and abs(std - exp_std) <= tol_std
                  and abs(c8 - exp_c8) <= tol_c8)
            _log(f"[selftest] round {round_no}/2 finite "
                 f"mean={mean:.4f}(exp {exp_mean:.4f} tol {tol_mean:g}) "
                 f"std={std:.4f}(exp {exp_std:.4f} tol {tol_std:g}) "
                 f"c8={c8:.4f}(exp {exp_c8:.4f} tol {tol_c8:g}) "
                 f"{(time.perf_counter() - started):.1f}s -> {'PASS' if ok else 'FAIL'}")
            if not ok:
                raise SelftestFailed(
                    f"selftest round {round_no}/2 out of tolerance: "
                    f"mean={mean:.4f} std={std:.4f} c8={c8:.4f}"
                )
        _log("[stage] selftest PASS")

    def front_out_order(self) -> list[str]:
        return [desc["name"] for desc in self.front.ready["outputs"]] if self.front else []

    def start(self) -> None:
        _log("[stage] front-compile")
        self._log_startup_block()
        self._start_workers()
        self._selftest()
        _log(f"[stage] ready (scale x{self.scale}, tile {self.tile_w}x{self.tile_h}, "
             f"overlap={self.overlap})")

    # -- per-tile path ------------------------------------------------

    def _front_transact(self, blobs: list[bytes]) -> list[bytes]:
        assert self.front is not None
        return self.front.transact(blobs, self.worker_timeout)

    def _back_transact(self, blobs: list[bytes]) -> list[bytes]:
        assert self.back is not None
        return self.back.transact(blobs, self.worker_timeout)

    def _decode_and_check(
        self, blobs: list[bytes], names: list[str], stage: str
    ) -> dict[str, np.ndarray]:
        if len(blobs) != len(names):
            raise WorkerGone(f"{stage} blob count {len(blobs)} != {len(names)}")
        out: dict[str, np.ndarray] = {}
        for name, blob in zip(names, blobs):
            arr = np.frombuffer(blob, dtype=np.float32).copy()
            rate = _nonfinite_rate(arr)
            if rate > 0.0:
                raise WorkerReported(stage, name, rate,
                                     f"non-finite {stage} tensor {name}: rate={rate:.6f}")
            out[name] = arr
        return out

    def run_tile_tensor(self, tile_chw01: np.ndarray) -> np.ndarray:
        """1 タイル (CHW float 0-1) → G 出力 ([1,3,512,512] float) を返す。

        F の 3 テンソル検査 → 正常時のみ G へ → G 出力検査の順。
        """
        if self.dead or self.front is None or self.back is None:
            raise TwoStageFatal(stage="session", tensor="", rate=-1.0,
                                recoveries=0, detail="session is dead")
        tile_bytes = np.ascontiguousarray(tile_chw01, dtype=np.float32).tobytes()
        f_blobs = self._front_transact([tile_bytes])
        f_arrs = self._decode_and_check(f_blobs, self.front_out_order(), "front")
        expected = {name for name in self.back_order}
        if set(f_arrs) != expected:
            raise WorkerGone(
                f"front output names {sorted(f_arrs)} != back inputs {sorted(expected)}"
            )
        g_blobs = self._back_transact(
            [np.ascontiguousarray(f_arrs[name]).tobytes() for name in self.back_order]
        )
        g_arrs = self._decode_and_check(g_blobs, self.g_out_names, "back")
        (g_flat,) = [g_arrs[name] for name in self.g_out_names[:1]]
        return np.ascontiguousarray(g_flat.reshape(self.g_out_shape))

    # -- recovery ------------------------------------------------------

    def _recover(self, reason: str) -> None:
        """両ワーカー停止・終了確認 → 新世代で起動 → セルフテスト。"""
        _log(f"[recover] {reason}")
        codes = []
        for conn, role in ((self.front, "front"), (self.back, "back")):
            if conn is None:
                codes.append(None)
                continue
            code = conn.quit_and_wait()
            try:
                alive = conn._proc.poll() is None if conn._proc is not None else False
            except Exception:
                alive = False
            _log(f"[recover] {role} pid={conn.pid} exit={code} alive={alive}")
            for tail_line in conn._stderr_tail[-8:]:
                _log(f"[recover] {role} stderr: {tail_line}")
            codes.append(None if alive else code)
            conn._close_pipes()
        self.front = None
        self.back = None
        if any(code is None for code in codes):
            self.dead = True
            raise TwoStageFatal(stage="recover", tensor="", rate=-1.0,
                                recoveries=1,
                                detail=f"worker exit unconfirmed (exits={codes})")
        self.generation += 1
        _log(f"[recover] restarting generation={self.generation}")
        self._start_workers()
        self._selftest()
        _log(f"[recover] generation={self.generation} restored")

    def close(self) -> None:
        for conn in (self.front, self.back):
            if conn is None:
                continue
            try:
                conn.close()
            except Exception:
                pass
        self.front = None
        self.back = None

    # -- image path -----------------------------------------------------

    def _seam_eligible(self) -> bool:
        if self.seam_template is None:
            return False
        if self.model_family != "AdcSR":
            return False
        pitch = int(self.seam_template.get("pitch", -1))
        core_w = self.tile_w - 2 * self.overlap
        return pitch == core_w * self.scale

    def upscale_image_rgb(self, rgb: np.ndarray) -> tuple[np.ndarray, int]:
        """RGB uint8 HWC → (4x RGB uint8, タイル数)。復旧は 1 画像 1 回まで。"""
        if self.dead:
            raise TwoStageFatal(stage="session", tensor="", rate=-1.0,
                                recoveries=0, detail="session is dead")
        img_chw = np.ascontiguousarray(np.transpose(rgb, (2, 0, 1)), dtype=np.float32) / 255.0
        tiles, orig_hw, padded_hw = _split_tiles(
            img_chw, (self.tile_h, self.tile_w), self.overlap
        )
        height, width = orig_hw
        out_h, out_w = height * self.scale, width * self.scale
        if out_w * out_h * 3 * 4 > MAX_OUT_FLOAT_BYTES:
            raise TwoStageFatal(
                stage="frame", tensor="", rate=-1.0, recoveries=0,
                detail=f"frame too large for float buffer: {width}x{height}x{self.scale}",
            )
        crossfade = not _env_disabled("UEU_NPU_CROSSFADE")
        canvas = _Canvas(padded_hw[1] * self.scale, padded_hw[0] * self.scale,
                         crossfade=crossfade)
        # 試験用フック (本番では使わない): 指定タイルの直前に対象ワーカーを
        # TerminateProcess で落とす (外部 kill と同じ死に方)。復旧経路の決定性試験用。
        kill_role = os.environ.get("UEU_TS_KILL_ROLE", "")
        try:
            kill_at = int(os.environ.get("UEU_TS_KILL_AT_TILE", "0") or 0)
        except ValueError:
            kill_at = 0
        kill_done = False
        core_w = self.tile_w - 2 * self.overlap
        ntx = (padded_hw[1] // core_w) if core_w > 0 else 1
        out_overlap = self.overlap * self.scale
        recoveries = 0
        tile_index = 0
        for tile_index, tile in enumerate(tiles):
            if (not kill_done and kill_role in ("front", "back")
                    and tile_index + 1 == kill_at):
                target = self.front if kill_role == "front" else self.back
                if target is not None:
                    _log(f"[fault-inject] killing {kill_role} pid={target.pid} "
                         f"before tile {tile_index + 1}/{len(tiles)}")
                    target._kill()
                kill_done = True
            attempt = 0
            while True:
                try:
                    g_out = self.run_tile_tensor(tile)
                    break
                except TwoStageFatal:
                    raise
                except TwoStageError as exc:
                    if recoveries >= 1:
                        if isinstance(exc, WorkerReported):
                            raise TwoStageFatal(
                                stage=exc.stage, tensor=exc.tensor, rate=exc.rate,
                                recoveries=recoveries,
                                detail=f"tile {tile_index + 1}/{len(tiles)} re-failed: {exc}",
                            )
                        raise TwoStageFatal(
                            stage="worker", tensor="", rate=-1.0,
                            recoveries=recoveries,
                            detail=f"tile {tile_index + 1}/{len(tiles)} re-failed: "
                                   f"{type(exc).__name__}: {exc}",
                        )
                    recoveries += 1
                    attempt += 1
                    try:
                        self._recover(
                            f"tile {tile_index + 1}/{len(tiles)} attempt {attempt}: "
                            f"{type(exc).__name__}: {exc}"
                        )
                    except StartupFailed as start_exc:
                        raise TwoStageFatal(
                            stage="recover", tensor="", rate=-1.0,
                            recoveries=recoveries, detail=f"restart failed: {start_exc}",
                        )
            tile_3chw = g_out[0]
            canvas.add_tile(tile_3chw, self.tile_w * self.scale, self.tile_h * self.scale,
                            out_overlap, ntx, tile_index)
            _log(f"[progress] tile {tile_index + 1}/{len(tiles)}")
        merged = canvas.accum[:, :out_h, :out_w]
        rate = _nonfinite_rate(merged)
        if rate > 0.0:
            raise TwoStageFatal(stage="merge", tensor="merged", rate=rate,
                                recoveries=recoveries,
                                detail="non-finite merged canvas")
        if self._seam_eligible() and not _env_disabled("UEU_NPU_SEAMFIX"):
            ms = apply_seam_fix_inplace(merged, self.seam_template)
            _log(f"[timing] seam-fix total={ms:.1f} ms "
                 f"(pitch={self.seam_template['pitch']}, size={out_w}x{out_h})")
            rate = _nonfinite_rate(merged)
            if rate > 0.0:
                raise TwoStageFatal(stage="seamfix", tensor="merged", rate=rate,
                                    recoveries=recoveries,
                                    detail="non-finite after seam-fix")
        elif self.seam_template is not None:
            _log("[worker-info] seam-fix skipped (model/pitch mismatch or UEU_NPU_SEAMFIX=0)")
        sr_rgb = np.transpose(
            np.clip(merged * 255.0, 0.0, 255.0).astype(np.uint8), (1, 2, 0)
        )
        return np.ascontiguousarray(sr_rgb), len(tiles)


# ------------------------------------------------------------ 外側 UEU ループ

_OUT_MAGIC_READY = b"UEUH"
_OUT_MAGIC_FRAME = b"UEUF"
_OUT_MAGIC_DATA = b"UEUD"
_OUT_MAGIC_ERROR = b"UEUE"
_OUT_MAX_FRAME_DIM = 16384


def _read_exact(stream, size: int) -> bytes:
    buf = bytearray()
    while len(buf) < size:
        chunk = stream.read(size - len(buf))
        if not chunk:
            raise EOFError(f"unexpected EOF (need={size}, got={len(buf)})")
        buf += chunk
    return bytes(buf)


def _write_error(stream, message: str) -> None:
    raw = message.encode("utf-8", errors="replace")
    stream.write(_OUT_MAGIC_ERROR + struct.pack("<i", len(raw)) + raw)
    stream.flush()


def serve_two_stage(args) -> int:
    """npu_serve.py の 2 段モード入口。--model-back 指定時のみ呼ばれる。"""
    front_model = Path(args.model).resolve()
    back_model = Path(args.model_back).resolve()
    for path in (front_model, back_model):
        if not path.is_file():
            print(f"fatal: model not found: {path}", file=sys.stderr, flush=True)
            return 1
    manifest = None
    front_key = f"modelcachekey_{front_model.stem}"
    back_key = f"modelcachekey_{back_model.stem}"
    boundary: list[str] | None = None
    model_family: str | None = None
    if getattr(args, "manifest", None):
        try:
            manifest = load_manifest(args.manifest)
        except StartupFailed as exc:
            print(f"fatal: {exc}", file=sys.stderr, flush=True)
            return 1
        model_family = manifest.get("model_family")
        for side, path, default_key in (
            ("front", front_model, front_key),
            ("back", back_model, back_key),
        ):
            entry = manifest[side]
            if Path(entry["file"]).name != path.name:
                print(f"fatal: manifest {side} file {entry['file']!r} != model {path.name!r}",
                      file=sys.stderr, flush=True)
                return 1
            try:
                digest = _sha256_file(path)
            except OSError as exc:
                print(f"fatal: cannot hash {side} model: {exc}", file=sys.stderr, flush=True)
                return 1
            if digest != entry["sha256"]:
                print(f"fatal: manifest {side} sha256 mismatch: model={digest} "
                      f"manifest={entry['sha256']}", file=sys.stderr, flush=True)
                return 1
            if side == "front":
                front_key = entry["cache_key"]
            else:
                back_key = entry["cache_key"]
        boundary = list(manifest["boundary"])
        _log(f"[worker-info] manifest={args.manifest} sha256={manifest['_sha256']}")
    seam_template = None
    if getattr(args, "seam_template", None):
        try:
            seam_template = load_seam_template(args.seam_template)
        except (OSError, ValueError) as exc:
            print(f"fatal: bad seam template: {exc}", file=sys.stderr, flush=True)
            return 1
    session = TwoStageSession(
        front_model=front_model,
        front_cache_key=front_key,
        back_model=back_model,
        back_cache_key=back_key,
        cache_dir=Path(args.cache_dir).resolve(),
        boundary_names=boundary,
        overlap=args.overlap,
        worker_timeout=float(getattr(args, "worker_timeout", DEFAULT_WORKER_TIMEOUT)),
        require_cache=not bool(getattr(args, "allow_compile", False)),
        compile_timeout=float(getattr(args, "compile_timeout", DEFAULT_COMPILE_TIMEOUT)),
        seam_template=seam_template,
        model_family=model_family,
    )
    try:
        session.start()
    except TwoStageError as exc:
        print(f"fatal: startup failed: {exc}", file=sys.stderr, flush=True)
        session.close()
        return 1

    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer
    stdout.write(
        _OUT_MAGIC_READY + struct.pack("<iii", session.scale, session.tile_w, session.tile_h)
    )
    stdout.flush()
    _log(f"[serve] ready (two-stage scale x{session.scale}, "
         f"tile {session.tile_w}x{session.tile_h}, overlap={args.overlap})")

    frame_no = 0
    try:
        while True:
            magic = stdin.read(4)
            if not magic:
                _log(f"[serve] stdin EOF, exiting after {frame_no} frames")
                return 0
            if len(magic) != 4:
                _log("[serve] partial frame magic, exiting(1)")
                return 1
            if magic != _OUT_MAGIC_FRAME:
                _log(f"[serve] bad magic: {magic!r} (protocol desync), exiting(1)")
                return 1
            width, height = struct.unpack("<ii", _read_exact(stdin, 8))
            if width <= 0 or height <= 0 or width > _OUT_MAX_FRAME_DIM or height > _OUT_MAX_FRAME_DIM:
                _write_error(stdout, f"ValueError: invalid frame size {width}x{height}")
                continue
            try:
                payload = _read_exact(stdin, width * height * 3)
            except EOFError as exc:
                _log(f"[serve] truncated frame body ({exc}), exiting(1)")
                return 1
            frame_no += 1
            started = time.perf_counter()
            try:
                rgb = np.frombuffer(payload, dtype=np.uint8).reshape(height, width, 3)
                output, tile_count = session.upscale_image_rgb(rgb)
                out_h, out_w = output.shape[:2]
                stdout.write(_OUT_MAGIC_DATA + struct.pack("<ii", out_w, out_h))
                stdout.write(output.tobytes())
                stdout.flush()
                _log(f"frame {frame_no}: {(time.perf_counter() - started) * 1000:.0f} ms "
                     f"({tile_count} tiles)")
            except TwoStageFatal as exc:
                _write_error(stdout, str(exc))
                _log(f"frame {frame_no}: TWO_STAGE_FATAL "
                     f"({(time.perf_counter() - started):.1f}s) {exc}")
            except Exception as exc:  # noqa: BLE001 - 1 モデル経路と同様に継続
                _write_error(stdout, f"{type(exc).__name__}: {exc}")
                _log(f"frame {frame_no}: ERROR {(time.perf_counter() - started) * 1000:.0f} ms "
                     f"({type(exc).__name__}: {exc})")
    finally:
        session.close()

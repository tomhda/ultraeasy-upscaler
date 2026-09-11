"""NPU 2段モード (AdcSR 前半/後半) の単体試験。

Linux でも実行できる範囲 (プロトコル・合成・seam-fix・解決・復旧) を扱う。
NPU 実機が必要な検証は tmp/adcsr-npu/round5/impl/RESULT.md §3 に記録する。
"""
from __future__ import annotations

import io
import json
import os
import struct
import sys
import types
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools" / "npu-serve"
sys.path.insert(0, str(TOOLS))

import npu_proto as proto


# ---------------------------------------------------------- ヘッダ・送受信


def test_header_roundtrip() -> None:
    raw = proto.encode_header(proto.T_DATA, 3, 41, 123456)
    assert len(raw) == proto.HEADER_LEN
    assert proto.decode_header(raw) == (proto.T_DATA, 3, 41, 123456)


def test_header_rejects_bad_magic_version_type() -> None:
    good = proto.encode_header(proto.T_DATA, 0, 0, 0)
    with pytest.raises(proto.ProtocolError):
        proto.decode_header(b"XXXX" + good[4:])
    bad_version = proto._HEADER_STRUCT.pack(proto.MAGIC, 999, proto.T_DATA, 0, 0, 0)
    with pytest.raises(proto.ProtocolError):
        proto.decode_header(bad_version)
    bad_type = proto._HEADER_STRUCT.pack(proto.MAGIC, proto.VERSION, 99, 0, 0, 0)
    with pytest.raises(proto.ProtocolError):
        proto.decode_header(bad_type)
    with pytest.raises(proto.ProtocolError):
        proto.decode_header(b"short")


def test_write_all_handles_short_writes() -> None:
    chunks: list[bytes] = []

    def short_write(data: bytes) -> int:
        piece = bytes(data[:3])
        chunks.append(piece)
        return len(piece)

    data = bytes(range(100))
    assert proto.write_all(short_write, data) == 100
    assert b"".join(chunks) == data


def test_read_exact_distinguishes_clean_and_truncated_eof() -> None:
    with pytest.raises(proto.CleanEOF):
        proto.read_exact(lambda _n: b"", 8, "header")
    with pytest.raises(proto.CleanEOF):
        proto.read_exact(lambda _n: None, 8, "header")
    state = {"calls": 0}

    def partial(_n: int) -> bytes:
        state["calls"] += 1
        return b"abc" if state["calls"] == 1 else b""

    with pytest.raises(proto.TruncatedError) as excinfo:
        proto.read_exact(partial, 8, "data")
    assert excinfo.value.need == 8 and excinfo.value.got == 3


def test_tensors_roundtrip_and_trailing_rejected() -> None:
    blobs = [np.arange(6, dtype=np.float32).tobytes(), b"xy"]
    assert proto.unpack_tensors(proto.pack_tensors(blobs)) == blobs
    with pytest.raises(proto.ProtocolError):
        proto.unpack_tensors(proto.pack_tensors(blobs) + b"trail")
    with pytest.raises(proto.ProtocolError):
        proto.unpack_tensors(struct.pack("<I", 99) + b"\0" * 8)


def test_json_roundtrip() -> None:
    obj = {"stage": "front", "tensor": "main", "nonfinite_rate": 1.0}
    assert proto.unpack_json(proto.pack_json(obj)) == obj


def test_send_recv_message_roundtrip() -> None:
    stream = io.BytesIO()
    proto.send_message(stream, proto.T_ERROR, 2, 7, proto.pack_json({"a": 1}))
    stream.seek(0)
    mtype, generation, request_id, payload_len = proto.recv_header(stream)
    assert (mtype, generation, request_id) == (proto.T_ERROR, 2, 7)
    assert proto.unpack_json(proto.recv_payload(stream, payload_len)) == {"a": 1}


# ---------------------------------------------------------- ワーカー往復 (fake ORT)


def _install_fake_worker(monkeypatch, outputs: list[np.ndarray]):
    """npu_worker の ort/セッションを fake に差し替える。"""
    import npu_worker

    class _ValueInfo:
        def __init__(self, name: str, shape: tuple[int, ...]) -> None:
            self.name = name
            self.type = "tensor(float)"
            self.shape = shape

    in_descs = [
        _ValueInfo("in_main", (1, 256, 64, 64)),
        _ValueInfo("in_mean", (1, 3, 1, 1)),
        _ValueInfo("in_std", (1, 3, 1, 1)),
    ]
    out_descs = [_ValueInfo("output", (1, 3, 512, 512))]

    class _Session:
        def get_inputs(self):
            return in_descs

        def get_outputs(self):
            return out_descs

        def get_providers(self):
            return ["VitisAIExecutionProvider"]

        def run(self, _names, feed):
            assert set(feed) == {"in_main", "in_mean", "in_std"}
            return [np.ascontiguousarray(out) for out in outputs]

    fake_ort = types.SimpleNamespace(
        InferenceSession=lambda *a, **k: _Session(),
        __version__="fake-1.8.0",
        set_default_logger_severity=lambda _level: None,
    )
    monkeypatch.setattr(npu_worker, "ort", fake_ort)
    monkeypatch.setattr(npu_worker, "np", np)
    return npu_worker


def _run_worker(npu_worker, args, stdin_bytes: bytes, expected_rc: int = 0) -> bytes:
    stdin = io.BytesIO(stdin_bytes)
    stdout = io.BytesIO()

    class _Stdin:
        buffer = stdin

    class _Stdout:
        buffer = stdout

    monkeypatch_stdin = _Stdin()
    monkeypatch_stdout = _Stdout()
    old_in, old_out = sys.stdin, sys.stdout
    sys.stdin, sys.stdout = monkeypatch_stdin, monkeypatch_stdout
    try:
        rc = npu_worker._serve(args)
    finally:
        sys.stdin, sys.stdout = old_in, old_out
    assert rc == expected_rc
    return stdout.getvalue()


def _worker_args(tmp_path: Path, role: str = "back"):
    return types.SimpleNamespace(
        role=role,
        model=str(tmp_path / "model.onnx"),
        cache_dir=str(tmp_path / "cache"),
        cache_key="modelcachekey_test",
        allow_compile=False,
        require_cache=True,
        compile_timeout=60.0,
        generation=5,
    )


def test_worker_data_roundtrip_and_quit(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "model.onnx").write_bytes(b"fake")
    expected = np.full((1, 3, 512, 512), 0.25, dtype=np.float32)
    npu_worker = _install_fake_worker(monkeypatch, [expected])

    stdin = io.BytesIO()
    blobs = [
        np.zeros((1, 256, 64, 64), dtype=np.float32).tobytes(),
        np.zeros((1, 3, 1, 1), dtype=np.float32).tobytes(),
        np.zeros((1, 3, 1, 1), dtype=np.float32).tobytes(),
    ]
    proto.send_message(stdin, proto.T_DATA, 5, 11, proto.pack_tensors(blobs))
    proto.send_message(stdin, proto.T_QUIT, 5, 0, b"")
    out = _run_worker(npu_worker, _worker_args(tmp_path), stdin.getvalue())

    stream = io.BytesIO(out)
    mtype, generation, request_id, length = proto.recv_header(stream)
    assert mtype == proto.T_READY and generation == 5 and request_id == 0
    ready = proto.unpack_json(proto.recv_payload(stream, length))
    assert [v["name"] for v in ready["inputs"]] == ["in_main", "in_mean", "in_std"]
    assert ready["outputs"][0]["nbytes"] == 1 * 3 * 512 * 512 * 4

    mtype, generation, request_id, length = proto.recv_header(stream)
    assert (mtype, generation, request_id) == (proto.T_DATA, 5, 11)
    (blob,) = proto.unpack_tensors(proto.recv_payload(stream, length))
    assert np.array_equal(np.frombuffer(blob, dtype=np.float32).reshape(1, 3, 512, 512), expected)
    assert stream.read() == b""


def test_worker_reports_nonfinite_as_error(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "model.onnx").write_bytes(b"fake")
    bad = np.full((1, 3, 512, 512), np.nan, dtype=np.float32)
    npu_worker = _install_fake_worker(monkeypatch, [bad])

    stdin = io.BytesIO()
    blobs = [
        np.zeros((1, 256, 64, 64), dtype=np.float32).tobytes(),
        np.zeros((1, 3, 1, 1), dtype=np.float32).tobytes(),
        np.zeros((1, 3, 1, 1), dtype=np.float32).tobytes(),
    ]
    proto.send_message(stdin, proto.T_DATA, 5, 12, proto.pack_tensors(blobs))
    proto.send_message(stdin, proto.T_QUIT, 5, 0, b"")
    out = _run_worker(npu_worker, _worker_args(tmp_path), stdin.getvalue())

    stream = io.BytesIO(out)
    # READY を読み飛ばす。
    _, _, _, length = proto.recv_header(stream)
    stream.read(length)
    mtype, generation, request_id, length = proto.recv_header(stream)
    assert mtype == proto.T_ERROR
    assert (generation, request_id) == (5, 12)
    err = proto.unpack_json(proto.recv_payload(stream, length))
    assert err["stage"] == "back" and err["tensor"] == "output"
    assert err["nonfinite_rate"] == 1.0


def test_worker_rejects_tensor_count_mismatch(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "model.onnx").write_bytes(b"fake")
    npu_worker = _install_fake_worker(
        monkeypatch, [np.zeros((1, 3, 512, 512), dtype=np.float32)]
    )
    stdin = io.BytesIO()
    proto.send_message(stdin, proto.T_DATA, 5, 13, proto.pack_tensors([b"only-one"]))
    out = _run_worker(npu_worker, _worker_args(tmp_path), stdin.getvalue(), expected_rc=2)
    stream = io.BytesIO(out)
    _, _, _, length = proto.recv_header(stream)
    stream.read(length)
    assert stream.read() == b""  # ERROR 応答ではなく即終了 (rc=2 は _run_worker が確認)


# ---------------------------------------------------------- タイル分割の同等性


def _import_npu_serve():
    """onnxruntime を stub して npu_serve を import する (Linux 用)。"""
    import npu_serve

    return npu_serve


def test_split_tiles_matches_legacy_path() -> None:
    pytest.importorskip("numpy")
    fake_ort = types.SimpleNamespace(get_available_providers=lambda: [])
    sys.modules.setdefault("onnxruntime", fake_ort)
    npu_serve = _import_npu_serve()
    import npu_twostage

    rng = np.random.default_rng(7)
    for shape, patch, overlap in [
        ((3, 100, 200), (128, 128), 32),
        ((3, 64, 64), (128, 128), 32),
        ((3, 65, 129), (128, 128), 32),
        ((3, 300, 200), (512, 512), 16),
    ]:
        img = rng.random(shape, dtype=np.float32)
        legacy = npu_serve._split_tiles(img, patch, overlap)
        staged = npu_twostage._split_tiles(img, patch, overlap)
        assert legacy[1] == staged[1] and legacy[2] == staged[2]
        assert len(legacy[0]) == len(staged[0])
        for a, b in zip(legacy[0], staged[0]):
            assert np.array_equal(a, b)


# ---------------------------------------------------------- 逐次加算合成


def test_canvas_single_tile_blend_equals_hardcut() -> None:
    import npu_twostage

    rng = np.random.default_rng(11)
    tile = rng.random((3, 512, 512), dtype=np.float32)
    core = tile[:, 128:128 + 256, 128:128 + 256]
    hard = npu_twostage._Canvas(256, 256, crossfade=False)
    hard.add_tile(tile, 512, 512, 128, 1, 0)
    blend = npu_twostage._Canvas(256, 256, crossfade=True)
    blend.add_tile(tile, 512, 512, 128, 1, 0)
    # 単一タイルは重み正規化でタイル値そのものになる。
    assert np.allclose(blend.accum, core, atol=1e-5)
    assert np.array_equal(hard.accum, core)


def test_canvas_blends_seam_linearly() -> None:
    import npu_twostage

    left = np.full((3, 320, 320), 0.2, dtype=np.float32)
    right = np.full((3, 320, 320), 0.8, dtype=np.float32)
    canvas = npu_twostage._Canvas(512, 256, crossfade=True)
    canvas.add_tile(left, 320, 320, 32, 2, 0)
    canvas.add_tile(right, 320, 320, 32, 2, 1)
    seam = canvas.accum[:, :, 256]
    assert np.allclose(seam, 0.5, atol=1e-5)
    assert np.allclose(canvas.accum[:, :, 100], 0.2, atol=1e-5)
    assert np.allclose(canvas.accum[:, :, 400], 0.8, atol=1e-5)

    hard = npu_twostage._Canvas(512, 256, crossfade=False)
    hard.add_tile(left, 320, 320, 32, 2, 0)
    hard.add_tile(right, 320, 320, 32, 2, 1)
    assert np.allclose(hard.accum[:, :, 255], 0.2)
    assert np.allclose(hard.accum[:, :, 256], 0.8)


# ---------------------------------------------------------- seam-fix 移植の一致


def _flatlib_methods():
    flat_dir = Path(__file__).resolve().parents[1] / "tmp" / "adcsr-seam" / "flat"
    if str(flat_dir) not in sys.path:
        sys.path.insert(0, str(flat_dir))
    import methods

    return methods


def test_seam_fix_matches_flatlib_reference() -> None:
    import npu_twostage

    methods = _flatlib_methods()
    template_path = (
        Path(__file__).resolve().parents[1]
        / "tools" / "winml-sr" / "seam_templates" / "adcsr_ov32_p256.json"
    )
    template = npu_twostage.load_seam_template(template_path)
    assert template["pitch"] == 256
    assert template["tau"] == 4.0 and template["clip"] == 2.0

    rng = np.random.default_rng(23)
    yy, xx = np.mgrid[0:96, 0:128]
    base = 120.0 + 0.05 * xx + 0.03 * yy + rng.random((96, 128))
    img = np.stack([base, base + 1.0, base - 1.0], axis=2)  # HWC 0-255 float64
    ref_out, _info = methods.apply_template(
        img, np.asarray(template["gx"]), np.asarray(template["gy"]),
        template["pitch"], gate="none",
    )
    merged = np.ascontiguousarray(
        np.transpose(img / 255.0, (2, 0, 1)), dtype=np.float32
    )
    ms = npu_twostage.apply_seam_fix_inplace(merged, template)
    assert ms >= 0.0
    actual = np.transpose(merged, (1, 2, 0)) * 255.0
    assert np.allclose(actual, ref_out, atol=0.05)


def test_seam_template_rejects_pitch_mismatch(tmp_path: Path) -> None:
    import npu_twostage

    bad = tmp_path / "bad.json"
    bad.write_text('{"pitch": 256, "gx": [0.0], "gy": [0.0]}', encoding="utf-8")
    with pytest.raises(ValueError, match="pitch"):
        npu_twostage.load_seam_template(bad)


# ---------------------------------------------------------- TwoStageSession (fake ワーカー)


def _desc(name: str, shape: tuple[int, ...]) -> dict:
    return {"name": name, "dtype": "float32", "shape": list(shape),
            "nbytes": int(np.prod(shape)) * 4}


def _front_ready() -> dict:
    return {
        "role": "front", "generation": 0,
        "inputs": [_desc("input", (1, 3, 128, 128))],
        "outputs": [_desc("main", (1, 256, 64, 64)),
                     _desc("mean", (1, 3, 1, 1)),
                     _desc("std", (1, 3, 1, 1))],
    }


def _back_ready() -> dict:
    return {
        "role": "back", "generation": 0,
        "inputs": [_desc("main", (1, 256, 64, 64)),
                    _desc("mean", (1, 3, 1, 1)),
                    _desc("std", (1, 3, 1, 1))],
        "outputs": [_desc("output", (1, 3, 512, 512))],
    }


class _FakeConn:
    """_WorkerConn の差し替え。実プロセスを作らない。"""

    instances: list["_FakeConn"] = []

    def __init__(self, *, role, model, cache_dir, cache_key, generation,
                 require_cache, compile_timeout) -> None:
        type(self).instances.append(self)
        self.role = role
        self.generation = generation
        self.pid = 1000 + len(type(self).instances)
        self.calls: list[tuple[list[bytes], float]] = []
        self.ready: dict = {}
        self.front_fn = None
        self.back_fn = None
        self.quit_code: int | None = 0
        self.aborted = False
        self.killed = False
        self._proc = None

    def wait_ready(self, timeout: float) -> dict:
        assert self.ready, f"{self.role} ready not staged"
        return self.ready

    def transact(self, blobs: list[bytes], timeout: float) -> list[bytes]:
        self.calls.append((blobs, timeout))
        if self.role == "front":
            assert self.front_fn is not None
            return self.front_fn(blobs)
        assert self.back_fn is not None
        return self.back_fn(blobs)

    def quit_and_wait(self):
        return self.quit_code

    def abort(self) -> None:
        self.aborted = True

    def _kill(self) -> None:
        self.killed = True

    def _close_pipes(self) -> None:
        return None

    def close(self) -> None:
        return None


def _const_blobs(value: float, shapes: list[tuple[int, ...]]) -> list[bytes]:
    return [np.full(s, value, dtype=np.float32).tobytes() for s in shapes]


def _make_session(monkeypatch, tmp_path: Path, *, front_fn=None, back_fn=None,
                  boundary=None, back_ready=None, manifest_family=None):
    import npu_twostage

    _FakeConn.instances.clear()
    front_model = tmp_path / "front.onnx"
    back_model = tmp_path / "back.onnx"
    front_model.write_bytes(b"front")
    back_model.write_bytes(b"back")
    session = npu_twostage.TwoStageSession(
        front_model=front_model, front_cache_key="ck_f",
        back_model=back_model, back_cache_key="ck_b",
        cache_dir=tmp_path / "cache", boundary_names=boundary, overlap=32,
        worker_timeout=5.0, require_cache=True, model_family=manifest_family,
    )
    front = _FakeConn(role="front", model=front_model, cache_dir=tmp_path,
                      cache_key="ck_f", generation=0, require_cache=True,
                      compile_timeout=60.0)
    back = _FakeConn(role="back", model=back_model, cache_dir=tmp_path,
                     cache_key="ck_b", generation=0, require_cache=True,
                     compile_timeout=60.0)
    front.ready = _front_ready()
    back.ready = back_ready or _back_ready()
    front.front_fn = front_fn or (lambda blobs: _const_blobs(
        0.1, [(1, 256, 64, 64), (1, 3, 1, 1), (1, 3, 1, 1)]))
    back.back_fn = back_fn or (lambda blobs: _const_blobs(0.5, [(1, 3, 512, 512)]))
    session.front = front
    session.back = back
    session._match_and_check(front.ready, back.ready)
    return session, front, back


def test_match_names_scale_and_tile(monkeypatch, tmp_path: Path) -> None:
    session, _front, _back = _make_session(
        monkeypatch, tmp_path,
        boundary=["main", "mean", "std"], manifest_family="AdcSR")
    assert session.scale == 4
    assert (session.tile_w, session.tile_h) == (128, 128)
    assert session.back_order == ["main", "mean", "std"]
    assert session._seam_eligible() is False  # テンプレート未指定
    session.seam_template = {"pitch": 256}
    assert session._seam_eligible() is True
    session.seam_template = {"pitch": 999}
    assert session._seam_eligible() is False


def test_match_rejects_name_and_scale_mismatch(monkeypatch, tmp_path: Path) -> None:
    import npu_twostage

    _session, _front, _back = _make_session(monkeypatch, tmp_path)
    bad_back = _back_ready()
    bad_back["inputs"][0] = _desc("other", (1, 256, 64, 64))
    with pytest.raises(npu_twostage.StartupFailed):
        _make_session(monkeypatch, tmp_path, back_ready=bad_back)
    bad_scale = _back_ready()
    bad_scale["outputs"] = [_desc("output", (1, 3, 256, 256))]
    with pytest.raises(npu_twostage.StartupFailed):
        _make_session(monkeypatch, tmp_path, back_ready=bad_scale)


def test_upscale_small_image_both_merge_modes(monkeypatch, tmp_path: Path) -> None:
    import npu_twostage

    for crossfade in ("1", "0"):
        monkeypatch.setenv("UEU_NPU_CROSSFADE", crossfade)
        session, _front, _back = _make_session(monkeypatch, tmp_path)
        rgb = np.full((64, 64, 3), 128, dtype=np.uint8)
        out, ntiles = session.upscale_image_rgb(rgb)
        assert ntiles == 1
        assert out.shape == (256, 256, 3)
        # fake G は定数 0.5 → uint8 化は 127 (clip().astype と同じ切捨て)。
        assert np.all(out == 127)


def test_recovery_retries_once_then_fatal(monkeypatch, tmp_path: Path) -> None:
    import npu_twostage

    calls = {"front": 0}

    def flaky_front(blobs):
        calls["front"] += 1
        if calls["front"] == 1:
            raise npu_twostage.WorkerReported("front", "main", 1.0, "injected NaN")
        return _const_blobs(0.1, [(1, 256, 64, 64), (1, 3, 1, 1), (1, 3, 1, 1)])

    session, _front, _back = _make_session(monkeypatch, tmp_path, front_fn=flaky_front)
    recovered = []
    monkeypatch.setattr(session, "_recover", lambda reason: recovered.append(reason))
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    out, _n = session.upscale_image_rgb(rgb)
    assert out.shape == (256, 256, 3)
    assert len(recovered) == 1  # 1 画像 1 回まで

    def dead_front(blobs):
        raise npu_twostage.WorkerReported("front", "main", 1.0, "always NaN")

    session2, _f2, _b2 = _make_session(monkeypatch, tmp_path, front_fn=dead_front)
    recovered2 = []
    monkeypatch.setattr(session2, "_recover", lambda reason: recovered2.append(reason))
    with pytest.raises(npu_twostage.TwoStageFatal) as excinfo:
        session2.upscale_image_rgb(rgb)
    assert str(excinfo.value).startswith(npu_twostage.TWO_STAGE_FATAL)
    assert excinfo.value.recoveries == 1
    assert len(recovered2) == 1


def test_kill_hook_triggers_recovery(monkeypatch, tmp_path: Path) -> None:
    import npu_twostage

    calls = {"front": 0}

    def flaky_front(blobs):
        calls["front"] += 1
        if calls["front"] == 1:
            # kill 直後の初回 transact は死んだふりをする。
            raise npu_twostage.WorkerGone("front exited mid-request 1")
        return _const_blobs(0.1, [(1, 256, 64, 64), (1, 3, 1, 1), (1, 3, 1, 1)])

    monkeypatch.setenv("UEU_TS_KILL_ROLE", "front")
    monkeypatch.setenv("UEU_TS_KILL_AT_TILE", "1")
    session, front, _back = _make_session(monkeypatch, tmp_path, front_fn=flaky_front)
    recovered = []
    monkeypatch.setattr(session, "_recover", lambda reason: recovered.append(reason))
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    out, _n = session.upscale_image_rgb(rgb)
    assert out.shape == (256, 256, 3)
    assert front.killed is True
    assert len(recovered) == 1


def test_selftest_pass_and_tolerance_fail(monkeypatch, tmp_path: Path) -> None:
    import npu_twostage

    tile = np.zeros((1, 3, 128, 128), dtype=np.float32)
    tile_path = tmp_path / "tile.npy"
    np.save(str(tile_path), tile)
    ref = {"output": {"mean": 0.5, "std": 0.0, "center8x8_mean": 0.5},
           "tolerance": {"mean": 0.01, "std": 0.01, "center8x8_mean": 0.02}}
    ref_path = tmp_path / "ref.json"
    ref_path.write_text(json.dumps(ref), encoding="utf-8")
    monkeypatch.setattr(npu_twostage, "SELFTEST_TILE", tile_path)
    monkeypatch.setattr(npu_twostage, "SELFTEST_REF", ref_path)
    session, _front, _back = _make_session(monkeypatch, tmp_path)
    session._selftest()  # 2 回とも PASS すること
    assert _front.calls and _back.calls

    ref["output"]["mean"] = 0.9
    ref_path.write_text(json.dumps(ref), encoding="utf-8")
    with pytest.raises(npu_twostage.SelftestFailed):
        session._selftest()


def test_manifest_validation(monkeypatch, tmp_path: Path) -> None:
    import npu_twostage

    good = {"model_family": "AdcSR",
            "front": {"file": "f.onnx", "sha256": "a" * 64, "cache_key": "ck_f"},
            "back": {"file": "b.onnx", "sha256": "b" * 64, "cache_key": "ck_b"},
            "boundary": ["main", "mean", "std"]}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(good), encoding="utf-8")
    assert npu_twostage.load_manifest(path)["model_family"] == "AdcSR"
    bad = dict(good, boundary=["main", "mean"])
    path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(npu_twostage.StartupFailed):
        npu_twostage.load_manifest(path)
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(npu_twostage.StartupFailed):
        npu_twostage.load_manifest(path)


def test_serve_two_stage_rejects_sha_mismatch(monkeypatch, tmp_path: Path, capsys) -> None:
    import npu_twostage

    front = tmp_path / "front.onnx"
    back = tmp_path / "back.onnx"
    front.write_bytes(b"front-bytes")
    back.write_bytes(b"back-bytes")
    manifest = {"model_family": "AdcSR",
                "front": {"file": "front.onnx", "sha256": "0" * 64, "cache_key": "ck_f"},
                "back": {"file": "back.onnx", "sha256": "1" * 64, "cache_key": "ck_b"},
                "boundary": ["main", "mean", "std"]}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    args = types.SimpleNamespace(
        model=str(front), model_back=str(back), manifest=str(manifest_path),
        seam_template=None, cache_dir=str(tmp_path), overlap=32,
        worker_timeout=5.0, allow_compile=False, compile_timeout=60.0, warmup=1,
    )
    assert npu_twostage.serve_two_stage(args) == 1
    assert "sha256 mismatch" in capsys.readouterr().err


# ---------------------------------------------------------- 往復の照合 (stub 輸送層)


def _stub_conn(reply_bytes: bytes, generation: int = 5):
    import npu_twostage

    conn = npu_twostage._WorkerConn.__new__(npu_twostage._WorkerConn)
    conn.role = "front"
    conn.generation = generation
    conn._request_id = 0
    conn._proc = types.SimpleNamespace(stdin=io.BytesIO(), stdout=io.BytesIO(reply_bytes))
    return conn


def _reply(mtype: int, generation: int, request_id: int, payload: bytes) -> bytes:
    stream = io.BytesIO()
    proto.send_message(stream, mtype, generation, request_id, payload)
    return stream.getvalue()


def test_roundtrip_rejects_request_id_mismatch() -> None:
    import npu_twostage

    blob = np.zeros((4,), dtype=np.float32).tobytes()
    reply = _reply(proto.T_DATA, 5, 999, proto.pack_tensors([blob]))
    conn = _stub_conn(reply)
    with pytest.raises(npu_twostage.WorkerGone, match="request_id mismatch"):
        conn._roundtrip_blocking([b"in"], 11)


def test_roundtrip_rejects_stale_generation() -> None:
    import npu_twostage

    blob = np.zeros((4,), dtype=np.float32).tobytes()
    reply = _reply(proto.T_DATA, 4, 11, proto.pack_tensors([blob]))
    conn = _stub_conn(reply)
    with pytest.raises(npu_twostage.WorkerGone, match="stale generation"):
        conn._roundtrip_blocking([b"in"], 11)


def test_roundtrip_propagates_worker_error() -> None:
    import npu_twostage

    err = {"stage": "front", "tensor": "main", "nonfinite_rate": 0.5,
           "generation": 5, "request_id": 11, "message": "non-finite output"}
    reply = _reply(proto.T_ERROR, 5, 11, proto.pack_json(err))
    conn = _stub_conn(reply)
    with pytest.raises(npu_twostage.WorkerReported) as excinfo:
        conn._roundtrip_blocking([b"in"], 11)
    assert excinfo.value.stage == "front" and excinfo.value.rate == 0.5


def test_roundtrip_truncated_body_is_gone() -> None:
    import npu_twostage

    stream = io.BytesIO()
    proto.send_message(stream, proto.T_DATA, 5, 11, b"short")
    conn = _stub_conn(stream.getvalue()[:-2])
    with pytest.raises(npu_twostage.WorkerGone, match="truncated"):
        conn._roundtrip_blocking([b"in"], 11)


# ---------------------------------------------------------- 配布スクリプト (onnx 不要部)


def _load_script_module(name: str):
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "scripts" / "adcsr" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_split_script_help_works_without_onnx() -> None:
    import subprocess

    repo = Path(__file__).resolve().parents[1]
    for script in ("scripts/adcsr/split_adcsr_npu.py", "scripts/adcsr/rewrite_in_to_n5.py"):
        proc = subprocess.run(
            [sys.executable, str(repo / script), "--help"],
            capture_output=True, text=True, timeout=60,
        )
        assert proc.returncode == 0, proc.stderr
        assert "--help" in proc.stdout or "usage" in proc.stdout.lower()


def test_build_guard_redirects_fd1_and_restores(tmp_path: Path, capfd) -> None:
    """セッション生成ガード: fd1 への書込みを退避し、終了後に復元する。"""
    import npu_worker

    sink_path = tmp_path / "sink.bin"
    with open(sink_path, "wb") as sink:
        with npu_worker._guard_stdout_during_build(target_fd=sink.fileno()):
            os.write(1, b"Old buffers:\n")
    # 退避中は sink へ (バイナリのため \r\n 化なし)
    assert sink_path.read_bytes() == b"Old buffers:\n"
    # 復元後は fd1 が元に戻り、sink へは追記されない
    os.write(1, b"guard-restored\n")
    assert "guard-restored" in capfd.readouterr().out
    assert sink_path.read_bytes() == b"Old buffers:\n"


def test_build_chatter_does_not_pollute_protocol(monkeypatch, tmp_path: Path,
                                                 capfd) -> None:
    """AIE コンパイラ相当の stdout 汚染があっても READY/DATA が壊れない。"""
    import npu_proto as proto_check

    (tmp_path / "model.onnx").write_bytes(b"fake")
    expected = np.zeros((1, 3, 512, 512), dtype=np.float32)
    npu_worker = _install_fake_worker(monkeypatch, [expected])

    real_inference_session = npu_worker.ort.InferenceSession

    def noisy_build(*args, **kwargs):
        os.write(1, b"Old buffers:\nL3_OFM_Buffer_spill_layer_47\n")
        return real_inference_session(*args, **kwargs)

    monkeypatch.setattr(npu_worker.ort, "InferenceSession", noisy_build)
    stdin = io.BytesIO()
    proto_check.send_message(stdin, proto_check.T_DATA, 5, 7, proto_check.pack_tensors(
        [np.zeros((1, 256, 64, 64), dtype=np.float32).tobytes(),
         np.zeros((1, 3, 1, 1), dtype=np.float32).tobytes(),
         np.zeros((1, 3, 1, 1), dtype=np.float32).tobytes()]))
    proto_check.send_message(stdin, proto_check.T_QUIT, 5, 0, b"")
    out = _run_worker(npu_worker, _worker_args(tmp_path), stdin.getvalue())
    assert b"Old buffers" not in out
    captured = capfd.readouterr()
    assert "Old buffers" in captured.err
    stream = io.BytesIO(out)
    mtype, _gen, _rid, length = proto_check.recv_header(stream)
    assert mtype == proto_check.T_READY


def test_start_workers_announces_back_compile_between_readies(monkeypatch,
                                                              tmp_path: Path) -> None:
    """後半ビルド開始時点で [stage] back-compile を出す (前半表示のままにしない)。"""
    import npu_twostage

    order: list[str] = []

    class _RecConn(_FakeConn):
        def wait_ready(self, timeout: float) -> dict:
            order.append(f"{self.role}-ready")
            return _front_ready() if self.role == "front" else _back_ready()

    _FakeConn.instances.clear()
    monkeypatch.setattr(npu_twostage, "_WorkerConn", _RecConn)
    monkeypatch.setattr(npu_twostage, "_log", lambda msg: order.append(f"log:{msg}"))
    (tmp_path / "front.onnx").write_bytes(b"front")
    (tmp_path / "back.onnx").write_bytes(b"back")
    session = npu_twostage.TwoStageSession(
        front_model=tmp_path / "front.onnx", front_cache_key="ck_f",
        back_model=tmp_path / "back.onnx", back_cache_key="ck_b",
        cache_dir=tmp_path / "cache", boundary_names=["main", "mean", "std"],
        overlap=32, worker_timeout=5.0, require_cache=True, model_family="AdcSR")
    session._start_workers()
    assert order[0] == "front-ready"
    assert order[1] == "log:[stage] back-compile"
    assert order[2] == "back-ready"
    assert order[3].startswith("log:[stage] workers-ready")


def test_split_manifest_satisfies_loader(tmp_path: Path) -> None:
    import npu_twostage

    split = _load_script_module("split_adcsr_npu")
    manifest = split.build_manifest(
        front_name="adcsr_front_nchw_128x128_bf16cast.onnx",
        front_sha256="a" * 64,
        front_cache_key="modelcachekey_test_front",
        back_name="adcsr_back_nchw_128x128_bf16cast.onnx",
        back_sha256="b" * 64,
        back_cache_key="modelcachekey_test_back",
        boundary=list(split.BOUNDARY_B),
        tool_version="1",
        source_model="adcsr_norm_bf16cast.onnx",
    )
    assert len(manifest["boundary"]) == 3
    path = tmp_path / "adcsr_npu_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    loaded = npu_twostage.load_manifest(path)
    assert loaded["model_family"] == "AdcSR"
    assert loaded["front"]["cache_key"] == "modelcachekey_test_front"

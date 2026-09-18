"""NPU tail-cut（DepthToSpace 以降の CPU 実行）のテスト。"""
from __future__ import annotations

import importlib.util
import json
import sys
import threading
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "tools" / "npu-serve"))

import npu_tail
from npu_tail import (
    load_tail_manifest,
    nearest_upsample,
    pixel_shuffle,
    tail_postprocess,
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


split_tail = _load_module("split_tail", _REPO / "scripts" / "npu" / "split_tail.py")
npu_serve = _load_module("npu_serve", _REPO / "tools" / "npu-serve" / "npu_serve.py")


def _run_cpu_ort(path: Path, feed: dict) -> np.ndarray:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return session.run(None, feed)[0]


def _make_plain(path: Path) -> Path:
    """input [1,8,2,2] → Relu → DepthToSpace(r=2, CRD) → output [1,2,4,4]。"""
    node_relu = helper.make_node("Relu", ["input"], ["mid"])
    node_dts = helper.make_node(
        "DepthToSpace", ["mid"], ["output"], blocksize=2, mode="CRD"
    )
    graph = helper.make_graph(
        [node_relu, node_dts],
        "plain",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 8, 2, 2])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2, 4, 4])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, str(path))
    return path


def _make_add(path: Path) -> Path:
    """input → Conv(2→8, 1x1) → DTS(r=2) と input → Resize(nearest 2x) の Add 形。"""
    weight = helper.make_tensor("w", TensorProto.FLOAT, [8, 2, 1, 1],
                                np.full((8, 2, 1, 1), 0.5, np.float32).ravel().tolist())
    bias = helper.make_tensor("b", TensorProto.FLOAT, [8],
                              np.zeros(8, np.float32).tolist())
    node_conv = helper.make_node("Conv", ["input", "w", "b"], ["cut"])
    node_dts = helper.make_node(
        "DepthToSpace", ["cut"], ["dts_out"], blocksize=2, mode="CRD"
    )
    scales = helper.make_tensor("scales", TensorProto.FLOAT, [4], [1.0, 1.0, 2.0, 2.0])
    node_resize = helper.make_node(
        "Resize", ["input", "", "scales"], ["rs_out"],
        mode="nearest", coordinate_transformation_mode="asymmetric",
        nearest_mode="floor",
    )
    node_add = helper.make_node("Add", ["dts_out", "rs_out"], ["output"])
    graph = helper.make_graph(
        [node_conv, node_dts, node_resize, node_add],
        "addform",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2, 2, 2])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2, 4, 4])],
        initializer=[weight, bias, scales],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, str(path))
    return path


def _make_clip(path: Path) -> Path:
    """input → Relu → DTS(r=2) → Clip → output（非対応形）。"""
    node_relu = helper.make_node("Relu", ["input"], ["mid"])
    node_dts = helper.make_node(
        "DepthToSpace", ["mid"], ["dts_out"], blocksize=2, mode="CRD"
    )
    node_clip = helper.make_node("Clip", ["dts_out"], ["output"])
    graph = helper.make_graph(
        [node_relu, node_dts, node_clip],
        "clipform",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 8, 2, 2])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2, 4, 4])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, str(path))
    return path


def test_split_plain_tail(tmp_path: Path) -> None:
    src = _make_plain(tmp_path / "plain.onnx")
    body = tmp_path / "plain_body.onnx"
    manifest_path = tmp_path / "plain_body.tail.json"
    assert split_tail.main(["--input", str(src), "--body", str(body),
                            "--manifest", str(manifest_path)]) == 0
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["version"] == 1
    assert manifest["kind"] == "depth_to_space"
    assert manifest["blocksize"] == 2
    assert manifest["mode"] == "CRD"
    assert manifest["add_nearest_input"] is False
    assert manifest["scale"] == 2
    assert manifest["body_output"] == "mid"
    assert len(manifest["source_sha256"]) == 64
    body_model = onnx.load(str(body))
    assert [o.name for o in body_model.graph.output] == ["mid"]


def test_split_add_tail(tmp_path: Path) -> None:
    src = _make_add(tmp_path / "add.onnx")
    body = tmp_path / "add_body.onnx"
    manifest_path = tmp_path / "add_body.tail.json"
    assert split_tail.main(["--input", str(src), "--body", str(body),
                            "--manifest", str(manifest_path)]) == 0
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["add_nearest_input"] is True
    assert manifest["body_output"] == "cut"


def test_split_clip_tail_is_unsupported(tmp_path: Path) -> None:
    src = _make_clip(tmp_path / "clip.onnx")
    assert split_tail.main(["--input", str(src),
                            "--body", str(tmp_path / "clip_body.onnx"),
                            "--manifest", str(tmp_path / "clip_body.tail.json")]) == 2


def test_pixel_shuffle_matches_ort_depth_to_space(tmp_path: Path) -> None:
    rng = np.random.RandomState(1)
    for mode in ("CRD", "DCR"):
        node = helper.make_node("DepthToSpace", ["input"], ["output"],
                                blocksize=2, mode=mode)
        graph = helper.make_graph(
            [node], "dts",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 8, 3, 3])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2, 6, 6])],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model.ir_version = 8
        path = tmp_path / f"dts_{mode}.onnx"
        onnx.save(model, str(path))
        x = rng.uniform(-1.0, 1.0, size=(1, 8, 3, 3)).astype(np.float32)
        expected = _run_cpu_ort(path, {"input": x})
        actual = pixel_shuffle(x, 2, mode)
        assert actual.shape == expected.shape
        assert float(np.abs(actual - expected).max()) < 1e-5


def test_nearest_upsample_matches_ort_resize(tmp_path: Path) -> None:
    rng = np.random.RandomState(2)
    scales = helper.make_tensor("scales", TensorProto.FLOAT, [4], [1.0, 1.0, 4.0, 4.0])
    node = helper.make_node(
        "Resize", ["input", "", "scales"], ["output"], mode="nearest",
        coordinate_transformation_mode="asymmetric", nearest_mode="floor",
    )
    graph = helper.make_graph(
        [node], "rs",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2, 3, 3])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2, 12, 12])],
        initializer=[scales],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    path = tmp_path / "resize.onnx"
    onnx.save(model, str(path))
    x = rng.uniform(0.0, 1.0, size=(1, 2, 3, 3)).astype(np.float32)
    expected = _run_cpu_ort(path, {"input": x})
    actual = nearest_upsample(x, 4)
    assert actual.shape == expected.shape
    assert float(np.abs(actual - expected).max()) == 0.0


def test_tail_postprocess_add_matches_ort(tmp_path: Path) -> None:
    src = _make_add(tmp_path / "add2.onnx")
    rng = np.random.RandomState(3)
    x = rng.uniform(0.0, 1.0, size=(1, 2, 2, 2)).astype(np.float32)
    expected = _run_cpu_ort(src, {"input": x})
    # body（Conv）の出力を ORT で求め、後処理だけ numpy で行う。
    conv_only = helper.make_node("Conv", ["input", "w", "b"], ["cut"])
    graph = helper.make_graph(
        [conv_only], "convonly",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2, 2, 2])],
        [helper.make_tensor_value_info("cut", TensorProto.FLOAT, [1, 8, 2, 2])],
        initializer=list(onnx.load(str(src)).graph.initializer)[:2],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    conv_path = tmp_path / "convonly.onnx"
    onnx.save(model, str(conv_path))
    bout = _run_cpu_ort(conv_path, {"input": x})
    manifest = {"blocksize": 2, "mode": "CRD", "add_nearest_input": True, "scale": 2}
    actual = tail_postprocess(bout, x, manifest)
    assert actual.shape == expected.shape
    assert float(np.abs(actual - expected).max()) < 1e-5


def test_load_tail_manifest_rejects_bad_fields(tmp_path: Path) -> None:
    good = {"version": 1, "kind": "depth_to_space", "blocksize": 4, "mode": "CRD",
            "add_nearest_input": False, "scale": 4, "body_output": "t",
            "source_sha256": "0" * 64}
    path = tmp_path / "good.tail.json"
    path.write_text(json.dumps(good), encoding="utf-8")
    assert load_tail_manifest(path)["blocksize"] == 4
    for key, bad in [("version", 2), ("kind", "other"), ("blocksize", 0),
                     ("mode", "XX"), ("add_nearest_input", 1), ("scale", -1),
                     ("body_output", "")]:
        broken = dict(good)
        broken[key] = bad
        path.write_text(json.dumps(broken), encoding="utf-8")
        with pytest.raises(ValueError):
            load_tail_manifest(path)


def _tail_threads() -> list[threading.Thread]:
    return [t for t in threading.enumerate()
            if t.name == "npu-tail-post" and t.is_alive()]


def test_run_tail_tiles_keeps_order_and_reuses_thread() -> None:
    manifest = {"blocksize": 2, "mode": "CRD", "add_nearest_input": False, "scale": 2}
    rng = np.random.RandomState(4)
    inputs = [rng.uniform(0, 1, size=(1, 8, 2, 2)).astype(np.float32) for _ in range(5)]
    calls: list[int] = []

    def run_tile(index: int, batch: np.ndarray) -> np.ndarray:
        calls.append(index)
        return batch * 2.0

    before = _tail_threads()
    results, post_ms = npu_serve.run_tail_tiles(run_tile, inputs, manifest)
    assert _tail_threads() == before
    assert len(results) == 5
    for index, (result, source) in enumerate(zip(results, inputs)):
        assert result.shape == (2, 4, 4)
        np.testing.assert_allclose(result, pixel_shuffle(source * 2.0, 2, "CRD")[0],
                                   rtol=0, atol=0)
    assert sorted(calls) == [0, 1, 2, 3, 4]
    assert post_ms >= 0.0


def test_run_tail_tiles_propagates_runner_error() -> None:
    manifest = {"blocksize": 2, "mode": "CRD", "add_nearest_input": False, "scale": 2}
    rng = np.random.RandomState(5)
    inputs = [rng.uniform(0, 1, size=(1, 8, 2, 2)).astype(np.float32) for _ in range(4)]

    def run_tile(index: int, batch: np.ndarray) -> np.ndarray:
        if index == 2:
            raise RuntimeError("npu boom")
        return batch

    before = _tail_threads()
    with pytest.raises(RuntimeError, match="npu boom"):
        npu_serve.run_tail_tiles(run_tile, inputs, manifest)
    assert _tail_threads() == before


def test_run_tail_tiles_propagates_postprocess_error() -> None:
    manifest = {"blocksize": 2, "mode": "CRD", "add_nearest_input": False, "scale": 2}
    good = np.zeros((1, 8, 2, 2), np.float32)
    bad = np.zeros((1, 7, 2, 2), np.float32)  # 7 は r^2=4 で割り切れない
    before = _tail_threads()
    with pytest.raises(ValueError):
        npu_serve.run_tail_tiles(lambda _i, _b: bad, [good], manifest)
    assert _tail_threads() == before


def test_upscale_tail_path_uses_merge_and_quantize() -> None:
    """NpuSession.upscale の tail 分岐が結合・量子化まで通る（フェイク body）。"""
    manifest = {"version": 1, "kind": "depth_to_space", "blocksize": 2, "mode": "CRD",
                "add_nearest_input": False, "scale": 2, "body_output": "cut",
                "source_sha256": "0" * 64}
    rng = np.random.RandomState(6)
    body_out = rng.uniform(0.0, 1.0, size=(1, 12, 4, 4)).astype(np.float32)

    session = npu_serve.NpuSession.__new__(npu_serve.NpuSession)
    session.tile_h = 4
    session.tile_w = 4
    session.overlap = 0
    session.scale = 2
    session.input_format = "nchw"
    session.tail = manifest
    session._run_body_tile = lambda batch: np.ascontiguousarray(body_out)  # noqa: E731

    before = _tail_threads()
    rgb = np.zeros((4, 4, 3), np.uint8)
    output, tile_count = session.upscale(rgb)
    assert _tail_threads() == before
    assert tile_count == 1
    assert output.shape == (8, 8, 3)
    assert output.dtype == np.uint8
    expected_chw = pixel_shuffle(body_out, 2, "CRD")[0]
    expected = np.transpose(np.clip(expected_chw * 255.0, 0.0, 255.0).astype(np.uint8),
                            (1, 2, 0))
    np.testing.assert_array_equal(output, np.ascontiguousarray(expected))


def test_run_tail_tiles_empty() -> None:
    manifest = {"blocksize": 2, "mode": "CRD", "add_nearest_input": False, "scale": 2}
    results, post_ms = npu_serve.run_tail_tiles(
        lambda _i, _b: (_ for _ in ()).throw(AssertionError("呼ばれない")), [], manifest)
    assert results == []
    assert post_ms == 0.0


class _CapturingServeClient:
    last_command: list = []

    def __init__(self, command, workdir, env=None, log=None) -> None:
        type(self).last_command = list(command)

    def connect(self, **kwargs) -> None:
        return None


def _setup_npu_session(monkeypatch, tmp_path: Path, filenames: list[str]) -> None:
    from app.core import helper_backend

    for name in filenames:
        (tmp_path / name).touch()
    monkeypatch.setenv(helper_backend.MODELS_DIR_ENV, str(tmp_path))
    monkeypatch.setattr(helper_backend, "_npu_python", lambda: Path("python.exe"))
    monkeypatch.setattr(helper_backend, "_npu_script", lambda: Path("npu_serve.py"))
    monkeypatch.setattr(helper_backend, "_cache_hit", lambda _path: True)
    monkeypatch.setattr(helper_backend, "ServeClient", _CapturingServeClient)


def _npu_span_command(monkeypatch, tmp_path: Path):
    from app.core import helper_backend
    from app.core.settings import HELPER_MODEL_SPAN, UpscaleBackend, UpscaleSettings

    settings = UpscaleSettings(backend=UpscaleBackend.NPU_NATIVE, model=HELPER_MODEL_SPAN)
    session = helper_backend.open_session(settings, 854, 480)
    assert session.backend == UpscaleBackend.NPU_NATIVE
    return _CapturingServeClient.last_command


def test_npu_serve_uses_tail_when_body_and_manifest_exist(monkeypatch, tmp_path: Path) -> None:
    from app.core.settings import HELPER_MODEL_NPU_TAIL, HELPER_MODEL_SPAN

    body_name, manifest_name = HELPER_MODEL_NPU_TAIL[HELPER_MODEL_SPAN][512]
    _setup_npu_session(monkeypatch, tmp_path, [
        "purephoto_nchw_512x512_bf16cast.onnx", body_name, manifest_name,
    ])
    command = _npu_span_command(monkeypatch, tmp_path)
    assert command[command.index("--model") + 1].endswith(body_name)
    assert command[command.index("--tail") + 1].endswith(manifest_name)


def test_npu_serve_falls_back_without_body(monkeypatch, tmp_path: Path) -> None:
    _setup_npu_session(monkeypatch, tmp_path, ["purephoto_nchw_512x512_bf16cast.onnx"])
    command = _npu_span_command(monkeypatch, tmp_path)
    assert command[command.index("--model") + 1].endswith(
        "purephoto_nchw_512x512_bf16cast.onnx")
    assert "--tail" not in command


def test_npu_serve_falls_back_without_manifest(monkeypatch, tmp_path: Path) -> None:
    from app.core.settings import HELPER_MODEL_NPU_TAIL, HELPER_MODEL_SPAN

    body_name, _manifest_name = HELPER_MODEL_NPU_TAIL[HELPER_MODEL_SPAN][512]
    _setup_npu_session(monkeypatch, tmp_path, [
        "purephoto_nchw_512x512_bf16cast.onnx", body_name,
    ])
    command = _npu_span_command(monkeypatch, tmp_path)
    assert "--tail" not in command


def test_npu_tailcut_env_disables_tail(monkeypatch, tmp_path: Path) -> None:
    from app.core import helper_backend
    from app.core.settings import (
        HELPER_MODEL_NPU_TAIL,
        HELPER_MODEL_SPAN,
        NPU_TAILCUT_ENV,
    )

    body_name, manifest_name = HELPER_MODEL_NPU_TAIL[HELPER_MODEL_SPAN][512]
    _setup_npu_session(monkeypatch, tmp_path, [
        "purephoto_nchw_512x512_bf16cast.onnx", body_name, manifest_name,
    ])
    assert helper_backend.npu_tailcut_enabled() is True
    monkeypatch.setenv(NPU_TAILCUT_ENV, "0")
    assert helper_backend.npu_tailcut_enabled() is False
    command = _npu_span_command(monkeypatch, tmp_path)
    assert "--tail" not in command
    assert command[command.index("--model") + 1].endswith(
        "purephoto_nchw_512x512_bf16cast.onnx")

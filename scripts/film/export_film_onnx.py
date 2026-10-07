"""Export the Apache-2.0 dajes FILM TorchScript checkpoint to ONNX.

The dajes checkpoint is a faithful PyTorch inference port of Google's FILM
Style network.  The checkpoint itself is intentionally kept outside the
repository (normally under ``tmp/film-poc``).

The upstream PyTorch port builds the ``grid_sample`` base grid with Python
shape values.  A normal ``dynamic_axes`` export therefore captures the trace
shape in the grid and produces badly wrong output at another resolution.  The
``dynamic_warp`` replacement below uses ``Shape`` + ``Range`` in the exported
graph, so the same FP32 model can accept any height/width divisible by 64.

Example::

    .venv\\Scripts\\python.exe scripts/film/export_film_onnx.py \\
        --checkpoint tmp/film-poc-dajes-src/film_net_fp32.pt \\
        --source-dir tmp/film-poc-dajes-src \\
        --output tmp/film-poc/film_net_fp32_dynamic.onnx \\
        --check-size 128x192 --dml

The source checkout must contain ``interpolator.py`` and its sibling modules
from https://github.com/dajes/frame-interpolation-pytorch.  No model weight
or generated ONNX file is written to a tracked path by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Iterable

import numpy as np


REPO = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_DIR = REPO / "tmp" / "film-poc-dajes-src"
DEFAULT_OUTPUT = REPO / "tmp" / "film-poc" / "film_net_fp32_dynamic.onnx"
MIN_DIM = 64


def parse_size(value: str) -> tuple[int, int]:
    """Parse a WxH string and enforce FILM's seven-level pyramid alignment."""

    try:
        width_text, height_text = value.lower().split("x", 1)
        width, height = int(width_text), int(height_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("サイズは WxH 形式で指定してください") from exc
    if width < MIN_DIM or height < MIN_DIM:
        raise argparse.ArgumentTypeError("FILM の幅・高さは64以上で指定してください")
    if width % MIN_DIM or height % MIN_DIM:
        raise argparse.ArgumentTypeError("FILM の幅・高さは64の倍数で指定してください")
    return width, height


def dynamic_warp(image, flow):
    """A dynamic-shape equivalent of the dajes ``util.warp`` implementation.

    FILM's flow is NCHW with (dx, dy) channels.  The original helper creates
    ``linspace`` endpoints from ``flow.shape`` and converts those dimensions to
    Python scalars during tracing.  This implementation keeps the dimensions
    as ONNX tensors, making the grid resolution follow the runtime input.
    The arithmetic is intentionally the same as the upstream helper, including
    ``align_corners=False`` and border padding.
    """

    import torch
    import torch.nn.functional as F

    flow = -flow.flip(1)
    dtype = flow.dtype
    device = flow.device

    shape = torch._shape_as_tensor(flow)
    height = shape[2]
    width = shape[3]
    height_f = height.to(dtype=dtype)
    width_f = width.to(dtype=dtype)

    normalized_flow = flow.permute(0, 2, 3, 1) / torch.stack(
        (height_f * 0.5, width_f * 0.5)
    )
    # For align_corners=False, the center of pixel i is (2*i+1)/size - 1.
    x = (2.0 * torch.arange(width, dtype=dtype, device=device) + 1.0) / width_f - 1.0
    y = (2.0 * torch.arange(height, dtype=dtype, device=device) + 1.0) / height_f - 1.0
    grid = torch.stack(
        (
            x[None, None, :] - normalized_flow[..., 1],
            y[None, :, None] - normalized_flow[..., 0],
        ),
        dim=3,
    )
    return F.grid_sample(
        image,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    ).reshape(image.shape)


def load_eager_model(checkpoint: Path, source_dir: Path):
    """Load the TorchScript weights into the eager dajes architecture."""

    import torch

    required = (
        "interpolator.py",
        "feature_extractor.py",
        "fusion.py",
        "pyramid_flow_estimator.py",
        "util.py",
    )
    missing = [name for name in required if not (source_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"dajes source is missing {', '.join(missing)}: {source_dir}"
        )

    # The modules use absolute sibling imports (``import util``), so put the
    # source checkout first and remove it after model construction only if it
    # was not already on sys.path.  The model keeps references to util.warp.
    source_text = str(source_dir)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    try:
        import util as film_util
        from interpolator import Interpolator

        # Patch the function in the module object used by Interpolator.
        film_util.warp = dynamic_warp
        scripted = torch.jit.load(str(checkpoint), map_location="cpu").eval()
        model = Interpolator()
        model.load_state_dict(scripted.state_dict())
        model.eval()
        return scripted, model
    except Exception:
        raise


def _tensor_output(model, x0, x1, time_tensor):
    import torch

    with torch.inference_mode():
        return model(x0, x1, time_tensor).detach().cpu().numpy()


def compare_arrays(reference: np.ndarray, got: np.ndarray) -> tuple[float, float]:
    diff = np.asarray(reference, dtype=np.float32) - np.asarray(got, dtype=np.float32)
    max_diff = float(np.max(np.abs(diff)))
    mse = float(np.mean(diff * diff))
    psnr = 99.0 if mse == 0 else float(10.0 * np.log10(1.0 / mse))
    return max_diff, psnr


def make_input(width: int, height: int, seed: int, time_value: float):
    import torch

    rng = np.random.default_rng(seed)
    x0 = rng.random((1, 3, height, width), dtype=np.float32)
    x1 = rng.random((1, 3, height, width), dtype=np.float32)
    dt = np.array([[time_value]], dtype=np.float32)
    return x0, x1, dt, torch.from_numpy(x0), torch.from_numpy(x1), torch.from_numpy(dt)


def export_model(
    model,
    output: Path,
    width: int,
    height: int,
    opset: int,
) -> None:
    import torch

    x0_np, x1_np, dt_np, x0, x1, dt = make_input(width, height, 12345, 0.5)
    del x0_np, x1_np, dt_np
    output.parent.mkdir(parents=True, exist_ok=True)
    dynamic_axes = {
        "x0": {2: "height", 3: "width"},
        "x1": {2: "height", 3: "width"},
        "image": {2: "height", 3: "width"},
    }
    torch.onnx.export(
        model,
        (x0, x1, dt),
        str(output),
        input_names=["x0", "x1", "time"],
        output_names=["image"],
        opset_version=opset,
        dynamic_axes=dynamic_axes,
        dynamo=False,
        do_constant_folding=True,
        verbose=False,
    )


def inspect_graph(path: Path) -> dict[str, object]:
    import onnx

    model = onnx.load(str(path), load_external_data=False)
    onnx.checker.check_model(model)
    counts: dict[str, int] = {}
    for node in model.graph.node:
        counts[node.op_type] = counts.get(node.op_type, 0) + 1
    return {
        "ir_version": model.ir_version,
        "opset": max((int(op.version) for op in model.opset_import), default=0),
        "nodes": len(model.graph.node),
        "ops": dict(sorted(counts.items())),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }


def check_ort(
    path: Path,
    model,
    sizes: Iterable[tuple[int, int]],
    dml: bool,
) -> list[dict[str, object]]:
    import onnxruntime as ort

    providers = ["CPUExecutionProvider"]
    if dml:
        available = ort.get_available_providers()
        if "DmlExecutionProvider" not in available:
            print("DML: onnxruntime-directml が未インストールのためスキップ")
        else:
            providers = ["DmlExecutionProvider", "CPUExecutionProvider"]

    session = ort.InferenceSession(str(path), providers=providers)
    print(f"ORT providers requested={providers} actual={session.get_providers()}")
    print(
        "ORT inputs="
        + json.dumps(
            [(item.name, item.shape, item.type) for item in session.get_inputs()],
            ensure_ascii=False,
        )
    )

    results: list[dict[str, object]] = []
    for index, (width, height) in enumerate(sizes):
        x0_np, x1_np, dt_np, x0, x1, dt = make_input(width, height, 1000 + index, 0.5)
        reference = _tensor_output(model, x0, x1, dt)
        started = time.perf_counter()
        output = session.run(["image"], {"x0": x0_np, "x1": x1_np, "time": dt_np})[0]
        elapsed = time.perf_counter() - started
        max_diff, psnr = compare_arrays(reference, output)
        result = {
            "size": f"{width}x{height}",
            "seconds": round(elapsed, 4),
            "shape": list(output.shape),
            "max_abs_diff_vs_torch": max_diff,
            "psnr_vs_torch_db": psnr,
            "range": [float(output.min()), float(output.max())],
        }
        results.append(result)
        print(json.dumps(result, ensure_ascii=False))
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--trace-size", type=parse_size, default=(64, 64))
    parser.add_argument("--check-size", type=parse_size, action="append", default=[])
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--dml", action="store_true", help="DirectML EPでも推論を試行")
    parser.add_argument(
        "--skip-check",
        action="store_true",
        help="ONNX Runtime のCPU/DML数値確認をスキップ",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else REPO / args.checkpoint
    source_dir = args.source_dir if args.source_dir.is_absolute() else REPO / args.source_dir
    output = args.output if args.output.is_absolute() else REPO / args.output
    checkpoint, source_dir, output = checkpoint.resolve(), source_dir.resolve(), output.resolve()
    if not checkpoint.exists():
        print(f"checkpoint がありません: {checkpoint}")
        return 2

    print(f"checkpoint={checkpoint}")
    print(f"checkpoint_sha256={hashlib.sha256(checkpoint.read_bytes()).hexdigest()}")
    print(f"source_dir={source_dir}")
    print(f"output={output}")

    try:
        scripted, model = load_eager_model(checkpoint, source_dir)
        width, height = args.trace_size
        _, _, _, x0, x1, dt = make_input(width, height, 12345, 0.5)
        scripted_output = _tensor_output(scripted, x0, x1, dt)
        eager_output = _tensor_output(model, x0, x1, dt)
        max_diff, psnr = compare_arrays(scripted_output, eager_output)
        print(f"eager_vs_torchscript size={width}x{height} max_abs_diff={max_diff:.3e} psnr={psnr:.2f}dB")
        if max_diff > 2e-4:
            print("eager model conversion check failed")
            return 1

        export_model(model, output, width, height, args.opset)
        info = inspect_graph(output)
        print("onnx=" + json.dumps(info, ensure_ascii=False, sort_keys=True))
        if "GridSample" not in info["ops"]:
            print("warning: ONNX graph has no GridSample node")

        if not args.skip_check:
            sizes = [args.trace_size] + list(args.check_size)
            # Preserve order while avoiding duplicate work.
            sizes = list(dict.fromkeys(sizes))
            results = check_ort(output, model, sizes, args.dml)
            for result in results:
                if float(result["psnr_vs_torch_db"]) < 80.0:
                    print("numeric check failed: " + json.dumps(result, ensure_ascii=False))
                    return 1
    except Exception as exc:  # noqa: BLE001 - CLI reports concrete conversion failure
        print(f"FILM ONNX export/check failed: {type(exc).__name__}: {exc}")
        return 1

    print("FILM ONNX export/check OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

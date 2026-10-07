"""Run an ONNX FILM model on one image pair through CPU or DirectML.

This is a small diagnostic runner for the ``tools/winml-film`` helper.  It
pads both RGB images to 64-pixel alignment, runs ``x0, x1, time -> image``,
clamps FILM's output to [0, 1], crops to the input size, and writes a PNG.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, PngImagePlugin


REPO = Path(__file__).resolve().parents[2]
MIN_DIM = 64


def read_rgb(path: Path) -> np.ndarray:
    # FILM release example PNGs contain a large zTXt metadata block.  Pillow's
    # safety limit is raised only for this local diagnostic reader.
    PngImagePlugin.MAX_TEXT_CHUNK = 64 * 1024 * 1024
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def pad_to_64(
    image: np.ndarray,
) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    height, width = image.shape[:2]
    padded_height = ((height + MIN_DIM - 1) // MIN_DIM) * MIN_DIM
    padded_width = ((width + MIN_DIM - 1) // MIN_DIM) * MIN_DIM
    pad_height = padded_height - height
    pad_width = padded_width - width
    # Match tools/winml-film/Program.cs: zero pad symmetrically, with the
    # extra pixel on the bottom/right when the difference is odd.  This keeps
    # the diagnostic runner numerically comparable to the production helper.
    padded = np.pad(
        image,
        (
            (pad_height // 2, pad_height - pad_height // 2),
            (pad_width // 2, pad_width - pad_width // 2),
            (0, 0),
        ),
        mode="constant",
        constant_values=0.0,
    )
    return padded, (width, height), (pad_width // 2, pad_height // 2)


def write_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8), "RGB").save(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input0", type=Path, required=True)
    parser.add_argument("--input1", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--ep-name",
        choices=("CPUExecutionProvider", "DmlExecutionProvider"),
        default="CPUExecutionProvider",
    )
    parser.add_argument("--time", type=float, default=0.5)
    parser.add_argument("--verify-cpu", action="store_true")
    return parser


def run_session(model: Path, x0: np.ndarray, x1: np.ndarray, time_value: float, providers: list[str]):
    import onnxruntime as ort

    session = ort.InferenceSession(str(model), providers=providers)
    dt = np.array([[time_value]], dtype=np.float32)
    started = time.perf_counter()
    output = session.run(["image"], {"x0": x0, "x1": x1, "time": dt})[0]
    elapsed = time.perf_counter() - started
    return output, elapsed, session.get_providers()


def main() -> int:
    args = build_parser().parse_args()
    model = args.model if args.model.is_absolute() else REPO / args.model
    input0 = args.input0 if args.input0.is_absolute() else REPO / args.input0
    input1 = args.input1 if args.input1.is_absolute() else REPO / args.input1
    output_path = args.output if args.output.is_absolute() else REPO / args.output
    model, input0, input1, output_path = (
        model.resolve(), input0.resolve(), input1.resolve(), output_path.resolve()
    )
    if not model.exists() or not input0.exists() or not input1.exists():
        print("model/input file is missing")
        return 2

    image0, image1 = read_rgb(input0), read_rgb(input1)
    if image0.shape != image1.shape:
        print(f"input sizes differ: {image0.shape[:2]} vs {image1.shape[:2]}")
        return 2
    padded0, original_size, crop_offset = pad_to_64(image0)
    padded1, _, _ = pad_to_64(image1)
    x0 = np.transpose(padded0, (2, 0, 1))[None].astype(np.float32)
    x1 = np.transpose(padded1, (2, 0, 1))[None].astype(np.float32)
    if not 0.0 <= args.time <= 1.0:
        print("--time は0から1の範囲で指定してください")
        return 2

    providers = [args.ep_name]
    if args.ep_name == "DmlExecutionProvider":
        providers.append("CPUExecutionProvider")
    try:
        output, elapsed, actual = run_session(model, x0, x1, args.time, providers)
    except Exception as exc:  # noqa: BLE001 - diagnostic CLI reports backend error
        print(f"inference failed: {type(exc).__name__}: {exc}")
        return 1

    width, height = original_size
    left, top = crop_offset
    output_rgb = np.transpose(output[0], (1, 2, 0))[top : top + height, left : left + width]
    output_rgb = np.clip(output_rgb, 0.0, 1.0)
    write_rgb(output_path, output_rgb)
    result = {
        "model": str(model),
        "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "requested_providers": providers,
        "actual_providers": actual,
        "input_size": [width, height],
        "padded_size": [x0.shape[3], x0.shape[2]],
        "time": args.time,
        "seconds": round(elapsed, 4),
        "raw_range": [float(output.min()), float(output.max())],
        "output": str(output_path),
    }
    print(json.dumps(result, ensure_ascii=False))

    if args.verify_cpu and args.ep_name != "CPUExecutionProvider":
        cpu, cpu_elapsed, _ = run_session(
            model, x0, x1, args.time, ["CPUExecutionProvider"]
        )
        diff = cpu.astype(np.float32) - output.astype(np.float32)
        mse = float(np.mean(diff * diff))
        psnr = 99.0 if mse == 0 else float(10 * np.log10(1.0 / mse))
        print(
            json.dumps(
                {
                    "cpu_seconds": round(cpu_elapsed, 4),
                    "max_abs_diff_vs_cpu": float(np.max(np.abs(diff))),
                    "psnr_vs_cpu_db": psnr,
                },
                ensure_ascii=False,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

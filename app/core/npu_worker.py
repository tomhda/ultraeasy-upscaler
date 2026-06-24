"""Subprocess entrypoint for Ryzen AI NPU image upscaling."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import media
from .binaries import repo_root
def _emit_progress(fraction: float, message: str) -> None:
    fraction = max(0.0, min(1.0, fraction))
    print(f"UEU_PROGRESS\t{fraction:.6f}\t{message}", flush=True)


def _npu_root() -> Path:
    return repo_root() / "vendor" / "amd-npu"


def _model_path() -> Path:
    return _npu_root() / "onnx-models" / "realesrgan_nchw_256x256_u8s8.onnx"


def _cache_dir() -> Path:
    return _npu_root()


def _write_image(path: Path, img_bgr) -> None:
    import cv2

    ext = path.suffix or ".png"
    ok, encoded = cv2.imencode(ext, img_bgr)
    if not ok:
        raise RuntimeError(f"画像をエンコードできませんでした: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        encoded.tofile(str(path))
    except OSError as exc:
        raise RuntimeError(f"画像を書き出せませんでした: {path}") from exc
    except Exception as exc:
        raise RuntimeError(f"画像を書き出せませんでした: {path}") from exc


def _read_image(path: Path):
    import cv2
    import numpy as np

    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError as exc:
        raise RuntimeError(f"画像を読めませんでした: {path}") from exc

    img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError(f"画像を読めませんでした: {path}")
    return img_bgr


def _build_runner():
    from .npu_runner import NpuRealEsrganRunner

    model = _model_path()
    if not model.exists():
        raise RuntimeError(f"NPU ONNXモデルが見つかりません: {model}")
    return NpuRealEsrganRunner(model, _cache_dir(), sr_scale=4, tile_overlap=16)


def _run_image(input_path: Path, output_path: Path) -> None:
    runner = _build_runner()
    _emit_progress(0.01, "画像を読み込み中…")
    img_bgr = _read_image(input_path)
    sr_bgr = runner.run(img_bgr, _emit_progress)
    _emit_progress(0.98, "画像を書き出し中…")
    _write_image(output_path, sr_bgr)
    _emit_progress(1.0, "完了")


def _run_folder(input_dir: Path, output_dir: Path, image_format: str) -> None:
    runner = _build_runner()
    images = [
        p for p in sorted(input_dir.iterdir())
        if p.is_file() and p.suffix.lower() in media.IMAGE_EXTS
    ]
    if not images:
        output_dir.mkdir(parents=True, exist_ok=True)
        _emit_progress(1.0, "0/0 枚")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    total = len(images)
    for index, src in enumerate(images, start=1):
        base = (index - 1) / total
        span = 1.0 / total

        def file_progress(frac: float, msg: str) -> None:
            _emit_progress(base + span * frac, f"{index}/{total} 枚: {msg}")

        file_progress(0.01, f"{src.name} を読み込み中…")
        img_bgr = _read_image(src)
        sr_bgr = runner.run(img_bgr, file_progress)
        out = output_dir / f"{src.stem}.{image_format}"
        file_progress(0.98, f"{out.name} を書き出し中…")
        _write_image(out, sr_bgr)

    _emit_progress(1.0, f"{total}/{total} 枚")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["image", "folder"], required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--image-format", default="png", choices=["png", "jpg", "webp"])
    args = parser.parse_args(argv)

    try:
        input_path = Path(args.input)
        output_path = Path(args.output)
        if args.mode == "image":
            _run_image(input_path, output_path)
        else:
            _run_folder(input_path, output_path, args.image_format)
    except Exception as exc:
        print(f"UEU_ERROR\t{exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

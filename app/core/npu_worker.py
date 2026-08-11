"""Subprocess entrypoint for Ryzen AI NPU image upscaling."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import binaries, media


def _emit_progress(fraction: float, message: str) -> None:
    fraction = max(0.0, min(1.0, fraction))
    print(f"UEU_PROGRESS\t{fraction:.6f}\t{message}", flush=True)


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


def _build_runner(model_name: str | None):
    from .npu_runner import NpuRealEsrganRunner

    model, sr_scale = binaries.npu_model_spec(model_name)
    if not model.exists():
        raise RuntimeError(
            f"NPU ONNXモデルが見つかりません ({model_name or 'default'}): {model}"
        )
    cache = binaries.npu_cache_dir() / f"modelcachekey_{model.stem}"
    if not cache.is_dir():
        # キャッシュ不在でも動くが、初回はVitisAI EPの再コンパイルで数分かかる
        print(f"NPU compile cache not found: {cache} (first run will be slow)",
              flush=True)
    return NpuRealEsrganRunner(
        model, binaries.npu_cache_dir(), sr_scale=sr_scale, tile_overlap=16
    )


def _run_image(input_path: Path, output_path: Path, model_name: str | None) -> None:
    runner = _build_runner(model_name)
    _emit_progress(0.01, "画像を読み込み中…")
    img_bgr = _read_image(input_path)
    sr_bgr = runner.run(img_bgr, _emit_progress)
    _emit_progress(0.98, "画像を書き出し中…")
    _write_image(output_path, sr_bgr)
    _emit_progress(1.0, "完了")


def _run_folder(input_dir: Path, output_dir: Path, image_format: str,
                model_name: str | None) -> None:
    runner = _build_runner(model_name)
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

        file_progress(0.01, "読み込み中…")
        img_bgr = _read_image(src)
        sr_bgr = runner.run(img_bgr, file_progress)
        out = output_dir / f"{src.stem}.{image_format}"
        file_progress(0.98, "書き出し中…")
        _write_image(out, sr_bgr)

    _emit_progress(1.0, f"{total}/{total} 枚")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["image", "folder"], required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--image-format", default="png", choices=["png", "jpg", "webp"])
    # choices は付けない: 未対応名は binaries.npu_model_spec の日本語エラーで返す
    parser.add_argument("--model", default=None)
    args = parser.parse_args(argv)

    try:
        input_path = Path(args.input)
        output_path = Path(args.output)
        if args.mode == "image":
            _run_image(input_path, output_path, args.model)
        else:
            _run_folder(input_path, output_path, args.image_format, args.model)
    except Exception as exc:
        print(f"UEU_ERROR\t{exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

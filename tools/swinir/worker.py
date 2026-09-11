"""Experimental SwinIR-M real-world x4 worker.

The worker intentionally lives outside the main application dependency set.
It supports a one-shot image command and the same UEU raw-RGB serve protocol
used by the WinML helper, so short-video experiments can reuse the existing
ffmpeg pipeline without starting PyTorch once per frame.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import struct
import sys
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from network_swinir import SwinIR

SCALE = 4
WINDOW_SIZE = 8
MAGIC_READY = b"UEUH"
MAGIC_FRAME = b"UEUF"
MAGIC_DATA = b"UEUD"
MAGIC_ERROR = b"UEUE"
MAX_FRAME_DIM = 16384
AMP_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}


def _configure_utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


_configure_utf8()


def build_model(model_kind: str) -> tuple[SwinIR, str]:
    if model_kind != "real_sr_m":
        raise ValueError(f"unsupported SwinIR model: {model_kind}")
    model = SwinIR(
        upscale=SCALE,
        in_chans=3,
        img_size=64,
        window_size=WINDOW_SIZE,
        img_range=1.0,
        depths=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler="nearest+conv",
        resi_connection="1conv",
    )
    return model, "params_ema"


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDAが利用できません。SwinIR実験環境を確認してください。")
    return device


def load_model(
    model_path: Path,
    model_kind: str,
    device: torch.device,
    precision: str,
) -> torch.nn.Module:
    model, preferred_key = build_model(model_kind)
    checkpoint = torch.load(model_path, map_location="cpu")
    if isinstance(checkpoint, Mapping):
        state = checkpoint.get(preferred_key) or checkpoint.get("params") or checkpoint
    else:
        state = checkpoint
    model.load_state_dict(state, strict=True)
    model.eval()
    model.to(device)
    if precision in AMP_DTYPES:
        if device.type != "cuda":
            raise ValueError(f"{precision}はCUDA実験時のみ使用できます。")
    return model


def _pad_to_window(image: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    _, _, height, width = image.shape
    pad_h = (-height) % WINDOW_SIZE
    pad_w = (-width) % WINDOW_SIZE
    if pad_h:
        image = torch.cat((image, torch.flip(image, dims=[2])), dim=2)[:, :, :height + pad_h, :]
    if pad_w:
        image = torch.cat((image, torch.flip(image, dims=[3])), dim=3)[:, :, :, :width + pad_w]
    return image, height, width


def _positions(length: int, tile: int, stride: int) -> list[int]:
    if length <= tile:
        return [0]
    return [*range(0, length - tile, stride), length - tile]


def infer_tensor(
    image: torch.Tensor,
    model: torch.nn.Module,
    device: torch.device,
    tile: int,
    tile_overlap: int,
) -> tuple[torch.Tensor, int]:
    image, old_height, old_width = _pad_to_window(image)
    _, channels, height, width = image.shape
    tile = min(tile, height, width)
    if tile <= 0 or tile % WINDOW_SIZE != 0:
        raise ValueError(f"tileは{WINDOW_SIZE}の倍数で指定してください: {tile}")
    if not 0 <= tile_overlap < tile:
        raise ValueError(f"tile-overlapは0以上、tile未満で指定してください: {tile_overlap}")

    scale = SCALE
    if tile == height and tile == width:
        positions = [(0, 0)]
    else:
        stride = tile - tile_overlap
        positions = [(y, x) for y in _positions(height, tile, stride) for x in _positions(width, tile, stride)]

    output = torch.zeros(
        (1, channels, height * scale, width * scale),
        dtype=torch.float32,
        device=device,
    )
    weight = torch.zeros_like(output)
    for y, x in positions:
        patch = image[:, :, y:y + tile, x:x + tile]
        patch_output = model(patch).float()
        output[:, :, y * scale:(y + tile) * scale, x * scale:(x + tile) * scale].add_(patch_output)
        weight[:, :, y * scale:(y + tile) * scale, x * scale:(x + tile) * scale].add_(1.0)

    output = output.div_(weight)
    return output[:, :, :old_height * scale, :old_width * scale], len(positions)


def upscale_array(
    image: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    precision: str,
    tile: int,
    tile_overlap: int,
) -> tuple[np.ndarray, int]:
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"RGB uint8のHxWx3画像が必要です: {image.dtype} {image.shape}")
    tensor = torch.from_numpy(np.ascontiguousarray(image).copy()).permute(2, 0, 1).unsqueeze(0)
    tensor = tensor.to(device=device, dtype=torch.float32)
    autocast = (
        torch.autocast(device_type="cuda", dtype=AMP_DTYPES[precision])
        if precision in AMP_DTYPES
        else nullcontext()
    )
    with torch.inference_mode(), autocast:
        output, tile_count = infer_tensor(tensor / 255.0, model, device, tile, tile_overlap)
    if not torch.isfinite(output).all():
        raise RuntimeError(
            f"SwinIR出力に非有限値があります（precision={precision}）。"
            "このモデルではbf16またはfp32を使用してください。"
        )
    array = output.squeeze(0).clamp(0.0, 1.0).cpu().numpy()
    array = np.rint(np.transpose(array, (1, 2, 0)) * 255.0).astype(np.uint8)
    return array, tile_count


def read_exact(stream, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = stream.read(size - len(chunks))
        if not chunk:
            raise EOFError(f"stdin EOF (need={size}, got={len(chunks)})")
        chunks.extend(chunk)
    return bytes(chunks)


def serve(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    model_path = Path(args.model).resolve()
    model = load_model(model_path, args.model_kind, device, args.precision)
    stderr = sys.stderr
    print(
        f"[swinir] model={args.model_kind} device={device} precision={args.precision} "
        f"tile={args.tile} overlap={args.tile_overlap}",
        file=stderr,
        flush=True,
    )
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer
    stdout.write(struct.pack("<4siii", MAGIC_READY, SCALE, args.tile, args.tile))
    stdout.flush()
    frame_count = 0
    tile_count = 0
    started = time.perf_counter()
    while True:
        magic = stdin.read(4)
        if not magic:
            break
        if magic != MAGIC_FRAME:
            raise ValueError(f"bad frame magic: {magic!r}")
        width, height = struct.unpack("<ii", read_exact(stdin, 8))
        if not 0 < width <= MAX_FRAME_DIM or not 0 < height <= MAX_FRAME_DIM:
            raise ValueError(f"invalid frame size: {width}x{height}")
        raw = read_exact(stdin, width * height * 3)
        image = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
        try:
            output, count = upscale_array(
                image, model, device, args.precision, args.tile, args.tile_overlap
            )
        except (RuntimeError, ValueError) as exc:
            message = str(exc).encode("utf-8", errors="replace")[:16 * 1024 * 1024]
            stdout.write(struct.pack("<4si", MAGIC_ERROR, len(message)))
            stdout.write(message)
            stdout.flush()
            print(f"[swinir] frame error: {exc}", file=stderr, flush=True)
            continue
        frame_count += 1
        tile_count += count
        out_h, out_w = output.shape[:2]
        stdout.write(struct.pack("<4sii", MAGIC_DATA, out_w, out_h))
        stdout.write(np.ascontiguousarray(output).tobytes())
        stdout.flush()
    elapsed = time.perf_counter() - started
    print(f"[swinir] frames={frame_count} tiles={tile_count} elapsed={elapsed:.3f}s", file=stderr, flush=True)


def image(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    model_path = Path(args.model).resolve()
    model = load_model(model_path, args.model_kind, device, args.precision)
    with Image.open(args.input) as source:
        source_rgb = np.asarray(source.convert("RGB"), dtype=np.uint8)
    started = time.perf_counter()
    output, tile_count = upscale_array(
        source_rgb, model, device, args.precision, args.tile, args.tile_overlap
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(output, mode="RGB").save(args.output)
    print(
        f"[swinir] device={device} precision={args.precision} input={source_rgb.shape[1]}x{source_rgb.shape[0]} "
        f"output={output.shape[1]}x{output.shape[0]} tiles={tile_count} elapsed={time.perf_counter() - started:.3f}s"
    )


def parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model", required=True, help="SwinIR .pth weight")
    common.add_argument("--model-kind", default="real_sr_m", choices=["real_sr_m"])
    common.add_argument("--device", default="auto", help="auto, cuda, or cpu")
    common.add_argument("--precision", default="bf16", choices=["bf16", "fp16", "fp32"])
    common.add_argument("--tile", type=int, default=256)
    common.add_argument("--tile-overlap", type=int, default=32)
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    image_parser = sub.add_parser("image", parents=[common])
    image_parser.add_argument("--input", required=True)
    image_parser.add_argument("--output", required=True)
    image_parser.set_defaults(handler=image)
    serve_parser = sub.add_parser("serve", parents=[common])
    serve_parser.set_defaults(handler=serve)
    return root


if __name__ == "__main__":
    arguments = parser().parse_args()
    arguments.handler(arguments)

#!/usr/bin/env python
"""Ryzen AI VitisAI EPをUEUプロトコルで公開する常駐ワーカー。"""
from __future__ import annotations

import argparse
import math
import os
import struct
import sys
import time
from pathlib import Path

os.environ.setdefault("XLNX_ENABLE_CACHE", "1")

import numpy as np
import onnxruntime as ort

MAGIC_READY = b"UEUH"
MAGIC_FRAME = b"UEUF"
MAGIC_DATA = b"UEUD"
MAGIC_ERROR = b"UEUE"
MAX_FRAME_DIM = 16384


def _split_tiles(img_chw: np.ndarray, patch_hw: tuple[int, int], overlap: int):
    """本体app/core/npu_runner.pyと同じreflectパディング＋タイル分割。"""
    _channels, height, width = img_chw.shape
    patch_h, patch_w = patch_hw
    core_h = patch_h - 2 * overlap
    core_w = patch_w - 2 * overlap
    if core_h <= 0 or core_w <= 0:
        raise ValueError("tile overlap is too large")

    n_tiles_h = math.ceil(height / core_h)
    n_tiles_w = math.ceil(width / core_w)
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


def _merge_tiles(
    tiles_chw: list[np.ndarray],
    orig_hw: tuple[int, int],
    padded_hw: tuple[int, int],
    overlap: int,
) -> np.ndarray:
    """本体app/core/npu_runner.pyと同じoverlap除去＋コア合成。"""
    channels, patch_h, patch_w = tiles_chw[0].shape
    height, width = orig_hw
    padded_h, padded_w = padded_hw
    core_h = patch_h - 2 * overlap
    core_w = patch_w - 2 * overlap
    n_h = padded_h // core_h
    n_w = padded_w // core_w

    img_pad = np.zeros((channels, padded_h, padded_w), dtype=tiles_chw[0].dtype)
    idx = 0
    for iy in range(n_h):
        for ix in range(n_w):
            y0 = iy * core_h
            x0 = ix * core_w
            tile = tiles_chw[idx]
            img_pad[:, y0:y0 + core_h, x0:x0 + core_w] = tile[
                :, overlap:overlap + core_h, overlap:overlap + core_w
            ]
            idx += 1
    return np.ascontiguousarray(img_pad[:, :height, :width])


def _read_exact(stream, size: int) -> bytes:
    buf = bytearray()
    while len(buf) < size:
        chunk = stream.read(size - len(buf))
        if not chunk:
            raise EOFError(f"unexpected EOF (need={size}, got={len(buf)})")
        buf += chunk
    return bytes(buf)


def _write_error(stream, exc: BaseException) -> None:
    message = f"{exc.__class__.__name__}: {exc}".encode("utf-8", errors="replace")
    stream.write(MAGIC_ERROR + struct.pack("<i", len(message)) + message)
    stream.flush()


class NpuSession:
    def __init__(self, model_path: Path, cache_dir: Path, overlap: int, warmup: int) -> None:
        providers = ort.get_available_providers()
        if "VitisAIExecutionProvider" not in providers:
            raise RuntimeError(f"VitisAIExecutionProvider is unavailable: {providers}")

        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_key = f"modelcachekey_{model_path.stem}"
        options = [{
            "cache_dir": str(cache_dir),
            "cache_key": cache_key,
            "enable_cache_file_io_in_mem": 0,
        }]
        started = time.perf_counter()
        print(f"[session] creating from {model_path.name} ...", file=sys.stderr, flush=True)
        self.session = ort.InferenceSession(
            str(model_path),
            providers=["VitisAIExecutionProvider"],
            provider_options=options,
        )
        print(f"[session] created in {time.perf_counter() - started:.1f}s", file=sys.stderr, flush=True)

        input0 = self.session.get_inputs()[0]
        output0 = self.session.get_outputs()[0]
        self.input_name = input0.name
        self.output_name = output0.name
        shape = tuple(input0.shape)
        out_shape = tuple(output0.shape)
        if len(shape) != 4 or len(out_shape) != 4:
            raise ValueError(f"expected 4D model tensors, got input={shape}, output={out_shape}")
        if isinstance(shape[1], int) and shape[1] == 3:
            self.input_format = "nchw"
            self.tile_h, self.tile_w = int(shape[2]), int(shape[3])
            out_h, out_w = int(out_shape[2]), int(out_shape[3])
        elif isinstance(shape[3], int) and shape[3] == 3:
            self.input_format = "nhwc"
            self.tile_h, self.tile_w = int(shape[1]), int(shape[2])
            out_h, out_w = int(out_shape[1]), int(out_shape[2])
        else:
            raise ValueError(f"cannot determine input layout: {shape}")
        scale_h = out_h // self.tile_h
        scale_w = out_w // self.tile_w
        if scale_h <= 0 or scale_h != scale_w:
            raise ValueError(f"invalid model scale: input={shape}, output={out_shape}")
        self.scale = scale_h
        self.overlap = overlap

        dummy = np.zeros((1, 3, self.tile_h, self.tile_w), dtype=np.float32)
        if self.input_format == "nhwc":
            dummy = np.transpose(dummy, (0, 2, 3, 1))
        for index in range(warmup):
            warm_started = time.perf_counter()
            self.session.run([self.output_name], {self.input_name: dummy})
            print(
                f"[warmup {index + 1}/{warmup}] {(time.perf_counter() - warm_started) * 1000:.0f} ms",
                file=sys.stderr,
                flush=True,
            )

    def upscale(self, rgb: np.ndarray) -> tuple[np.ndarray, int]:
        img_chw = np.ascontiguousarray(np.transpose(rgb, (2, 0, 1)), dtype=np.float32) / 255.0
        tiles, orig_hw, padded_hw = _split_tiles(
            img_chw, (self.tile_h, self.tile_w), self.overlap
        )
        sr_tiles: list[np.ndarray] = []
        for tile in tiles:
            input_3d = np.transpose(tile, (1, 2, 0)) if self.input_format == "nhwc" else tile
            output = self.session.run(
                [self.output_name], {self.input_name: input_3d[None, ...]}
            )[0][0]
            if self.input_format == "nhwc":
                output = np.transpose(output, (2, 0, 1))
            sr_tiles.append(output)

        sr_chw = _merge_tiles(
            sr_tiles,
            (orig_hw[0] * self.scale, orig_hw[1] * self.scale),
            (padded_hw[0] * self.scale, padded_hw[1] * self.scale),
            self.overlap * self.scale,
        )
        sr_rgb = np.transpose(
            np.clip(sr_chw * 255.0, 0.0, 255.0).astype(np.uint8), (1, 2, 0)
        )
        return np.ascontiguousarray(sr_rgb), len(tiles)


def serve(args: argparse.Namespace) -> int:
    model_path = Path(args.model).resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"model not found: {model_path}")
    session = NpuSession(
        model_path=model_path,
        cache_dir=Path(args.cache_dir).resolve(),
        overlap=args.overlap,
        warmup=args.warmup,
    )

    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer
    stdout.write(
        MAGIC_READY
        + struct.pack("<iii", session.scale, session.tile_w, session.tile_h)
    )
    stdout.flush()
    print(
        f"[serve] ready (scale x{session.scale}, tile {session.tile_w}x{session.tile_h}, overlap={args.overlap})",
        file=sys.stderr,
        flush=True,
    )

    frame_no = 0
    while True:
        magic = stdin.read(4)
        if not magic:
            print(f"[serve] stdin EOF, exiting after {frame_no} frames", file=sys.stderr, flush=True)
            return 0
        if len(magic) != 4:
            raise EOFError("partial frame magic")
        if magic != MAGIC_FRAME:
            raise ValueError(f"bad magic: {magic!r} (protocol desync)")
        width, height = struct.unpack("<ii", _read_exact(stdin, 8))
        if width <= 0 or height <= 0 or width > MAX_FRAME_DIM or height > MAX_FRAME_DIM:
            raise ValueError(f"invalid frame size {width}x{height}")
        payload = _read_exact(stdin, width * height * 3)
        frame_no += 1
        started = time.perf_counter()
        try:
            rgb = np.frombuffer(payload, dtype=np.uint8).reshape(height, width, 3)
            output, tile_count = session.upscale(rgb)
            out_h, out_w = output.shape[:2]
            stdout.write(MAGIC_DATA + struct.pack("<ii", out_w, out_h))
            stdout.write(output.tobytes())
            stdout.flush()
            print(
                f"frame {frame_no}: {(time.perf_counter() - started) * 1000:.0f} ms ({tile_count} tiles)",
                file=sys.stderr,
                flush=True,
            )
        except Exception as exc:
            _write_error(stdout, exc)
            print(
                f"frame {frame_no}: ERROR {(time.perf_counter() - started) * 1000:.0f} ms "
                f"({exc.__class__.__name__}: {exc})",
                file=sys.stderr,
                flush=True,
            )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--overlap", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=1)
    return parser


def main() -> int:
    if os.name == "nt":
        import msvcrt

        msvcrt.setmode(sys.stdin.fileno(), os.O_BINARY)
        msvcrt.setmode(sys.stdout.fileno(), os.O_BINARY)
    try:
        return serve(_parser().parse_args())
    except Exception as exc:
        print(f"fatal: {exc.__class__.__name__}: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


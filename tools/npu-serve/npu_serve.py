#!/usr/bin/env python
"""Ryzen AI VitisAI EPをUEUプロトコルで公開する常駐ワーカー。"""
from __future__ import annotations

import argparse
import math
import os
import queue
import struct
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("XLNX_ENABLE_CACHE", "1")

import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from npu_tail import (
    TAIL_QUEUE_DEPTH,
    load_tail_manifest,
    quantize_chw_into,
    tail_postprocess,
    tail_quantize_into,
)

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


def _tile_boxes(orig_hw: tuple[int, int], patch_hw: tuple[int, int], overlap: int):
    """_split_tiles と同じ順で、各タイルの有効コア（タイル座標）と画像上の位置を返す。

    戻り値は ``((ty0, ty1, tx0, tx1), (iy0, iy1, ix0, ix1))`` の列。パディングで
    足した領域（_merge_tiles が最後に切り落とす部分）は含めない。
    """
    height, width = orig_hw
    core_h = patch_hw[0] - 2 * overlap
    core_w = patch_hw[1] - 2 * overlap
    boxes = []
    for iy in range(math.ceil(height / core_h)):
        for ix in range(math.ceil(width / core_w)):
            y0, x0 = iy * core_h, ix * core_w
            y1, x1 = min(y0 + core_h, height), min(x0 + core_w, width)
            boxes.append((
                (overlap, overlap + (y1 - y0), overlap, overlap + (x1 - x0)),
                (y0, y1, x0, x1),
            ))
    return boxes


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


def run_tail_tiles(run_tile, input_batches, manifest, post=None) -> tuple[list, float]:
    """各タイルの body 実行と numpy 後処理を重ね、順序を保って返す。

    ``run_tile(index, input_batch)`` は NPU body を実行して ``[1, C, H, W]``
    を返す callable。``input_batches`` はモデル論理入力（NCHW）のバッチ列。
    戻り値は（後処理済み CHW タイル列, 後処理合計 ms）。
    ``post(index, body_out, model_in)`` を渡すと既定の後処理の代わりに呼び、
    その戻り値を結果列に入れる（出力バッファへ直接書く用途）。

    後処理は別スレッド（キュー深さ :data:`TAIL_QUEUE_DEPTH`）で行い、
    次タイルの NPU 実行と重ねる。実行側・後処理側どちらの例外も呼び出し側へ
    伝播し、ワーカーは必ず回収する（スレッドを残さない）。
    """
    count = len(input_batches)
    results: list = [None] * count
    errors: list[BaseException] = []
    post_ms = [0.0]
    work: queue.Queue = queue.Queue(maxsize=TAIL_QUEUE_DEPTH)
    done = object()

    def _worker() -> None:
        while True:
            item = work.get()
            if item is done:
                return
            index, body_out, model_in = item
            try:
                started = time.perf_counter()
                if post is not None:
                    output = post(index, body_out, model_in)
                else:
                    output = tail_postprocess(body_out, model_in, manifest)[0]
                post_ms[0] += (time.perf_counter() - started) * 1000.0
                results[index] = output
            except Exception as exc:  # noqa: BLE001 - 記録して排出し、呼び出し側で送出
                if not errors:
                    errors.append(exc)

    thread = threading.Thread(target=_worker, name="npu-tail-post", daemon=True)
    thread.start()
    try:
        for index, model_in in enumerate(input_batches):
            body_out = run_tile(index, model_in)
            work.put((index, body_out, model_in))
    finally:
        # ワーカーは DONE でのみ終了する（例外時は排出し続けて必ず到達）。
        # BaseException でワーカーが死んだ場合に備え daemon=True にしてある。
        work.put(done)
        thread.join()
    if errors:
        raise errors[0]
    return results, post_ms[0]


class NpuSession:
    def __init__(
        self,
        model_path: Path,
        cache_dir: Path,
        overlap: int,
        warmup: int,
        tail: dict | None = None,
    ) -> None:
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
        self.tail = tail
        if tail is not None:
            if self.input_format != "nchw":
                raise ValueError(f"tail-cut needs an nchw body, got input={shape}")
            if self.output_name != tail["body_output"]:
                raise ValueError(
                    f"tail manifest body_output={tail['body_output']!r} "
                    f"!= model output {self.output_name!r}"
                )
            blocksize = int(tail["blocksize"])
            out_c = int(out_shape[1])
            if out_c % (blocksize * blocksize) != 0:
                raise ValueError(
                    f"tail-cut body channels {out_c} not divisible "
                    f"by blocksize^2 ({blocksize * blocksize})"
                )
            if int(out_shape[2]) != self.tile_h or int(out_shape[3]) != self.tile_w:
                raise ValueError(
                    f"tail-cut body output {tuple(out_shape)} must be "
                    f"tile-sized [1, C, {self.tile_h}, {self.tile_w}]"
                )
            # body の出力はタイルと同寸なので、倍率はマニフェストから取る。
            self.scale = int(tail["scale"])

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

    def _run_body_tile(self, tile_batch_nchw: np.ndarray) -> np.ndarray:
        """1 タイルの body を実行して ``[1, C, H, W]``（NCHW 論理）を返す。"""
        feed = (
            np.transpose(tile_batch_nchw, (0, 2, 3, 1))
            if self.input_format == "nhwc"
            else tile_batch_nchw
        )
        output = self.session.run(
            [self.output_name], {self.input_name: np.ascontiguousarray(feed)}
        )[0]
        if self.input_format == "nhwc":
            output = np.transpose(output, (0, 3, 1, 2))
        return np.ascontiguousarray(output)

    def upscale(self, rgb: np.ndarray) -> tuple[np.ndarray, int]:
        img_chw = np.ascontiguousarray(np.transpose(rgb, (2, 0, 1)), dtype=np.float32) / 255.0
        tiles, orig_hw, _padded_hw = _split_tiles(
            img_chw, (self.tile_h, self.tile_w), self.overlap
        )
        # 量子化までをタイル単位で行い、有効コアだけを最終バッファへ直接書く。
        # 結合してから clip(x * 255).astype(uint8) する手順と要素ごとに同じ演算。
        scale = self.scale
        boxes = _tile_boxes(orig_hw, (self.tile_h, self.tile_w), self.overlap)
        out = np.empty((orig_hw[0] * scale, orig_hw[1] * scale, 3), dtype=np.uint8)

        def _dst(index: int) -> np.ndarray:
            y0, y1, x0, x1 = boxes[index][1]
            return out[y0 * scale:y1 * scale, x0 * scale:x1 * scale]

        if self.tail is not None:
            tail = self.tail

            def _post(index: int, body_out: np.ndarray, model_in: np.ndarray) -> None:
                tail_quantize_into(_dst(index), body_out, model_in, tail, boxes[index][0])

            _results, post_ms = run_tail_tiles(
                lambda _index, tile_batch: self._run_body_tile(tile_batch),
                [np.ascontiguousarray(tile[None, ...]) for tile in tiles],
                tail,
                post=_post,
            )
            print(
                f"[timing] tail-post total={post_ms:.0f} ms",
                file=sys.stderr,
                flush=True,
            )
        else:
            for index, tile in enumerate(tiles):
                input_3d = np.transpose(tile, (1, 2, 0)) if self.input_format == "nhwc" else tile
                output = self.session.run(
                    [self.output_name], {self.input_name: input_3d[None, ...]}
                )[0][0]
                if self.input_format == "nhwc":
                    output = np.transpose(output, (2, 0, 1))
                ty0, ty1, tx0, tx1 = boxes[index][0]
                quantize_chw_into(
                    _dst(index),
                    output[:, ty0 * scale:ty1 * scale, tx0 * scale:tx1 * scale],
                )
        return out, len(tiles)


def serve(args: argparse.Namespace) -> int:
    if getattr(args, "model_back", None):
        import npu_twostage

        return npu_twostage.serve_two_stage(args)
    model_path = Path(args.model).resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"model not found: {model_path}")
    tail = None
    if getattr(args, "tail", None):
        if getattr(args, "model_back", None):
            raise ValueError("--tail は 1 モデル経路専用（2 段モードとは併用不可）")
        tail = load_tail_manifest(args.tail)
    session = NpuSession(
        model_path=model_path,
        cache_dir=Path(args.cache_dir).resolve(),
        overlap=args.overlap,
        warmup=args.warmup,
        tail=tail,
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
            stdout.write(memoryview(output).cast("B"))
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
    # 2 段モード (AdcSR 前半/後半)。--model-back 指定時のみ有効。
    # 1 モデル経路の既存オプションの意味は変えない。
    parser.add_argument("--model-back", default=None)
    parser.add_argument("--manifest", default=None)
    # tail-cut モード (DepthToSpace 以降を CPU で実行)。--tail 指定時のみ有効。
    # 未指定時の出力は従来と同一。
    parser.add_argument("--tail", default=None)
    parser.add_argument("--seam-template", default=None)
    parser.add_argument("--worker-timeout", type=float, default=60.0)
    parser.add_argument("--compile-timeout", type=float, default=4 * 3600.0)
    two_stage_mode = parser.add_mutually_exclusive_group()
    two_stage_mode.add_argument("--require-cache", action="store_true", default=False)
    two_stage_mode.add_argument("--allow-compile", action="store_true", default=False)
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


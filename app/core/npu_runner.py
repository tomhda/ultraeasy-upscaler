"""Ryzen AI NPU ONNX runner for the AMD Real-ESRGAN model.

Portions are adapted from AMD's `realesrgan-256x256-tiles-amdnpu`
sample repository. The app keeps this isolated so normal Vulkan usage
does not require importing ONNX Runtime or OpenCV.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Callable

import numpy as np
import onnxruntime as ort

os.environ.setdefault("XLNX_ENABLE_CACHE", "1")

ProgressFn = Callable[[float, str], None]


def get_npu_info() -> str:
    process = subprocess.Popen(
        r"pnputil /enum-devices /bus PCI /deviceids",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, _stderr = process.communicate()
    text = stdout.decode(errors="ignore")
    if "PCI\\VEN_1022&DEV_1502&REV_00" in text:
        return "PHX/HPT"
    if "PCI\\VEN_1022&DEV_17F0&REV_00" in text:
        return "STX"
    if "PCI\\VEN_1022&DEV_17F0&REV_10" in text:
        return "STX"
    if "PCI\\VEN_1022&DEV_17F0&REV_11" in text:
        return "STX"
    if "PCI\\VEN_1022&DEV_17F0&REV_20" in text:
        return "KRK"
    return ""


def _cache_key(model_path: Path) -> str:
    return f"modelcachekey_{model_path.stem}"


def _split_tiles(img_chw: np.ndarray, patch_hw: tuple[int, int], overlap: int):
    import math

    channels, height, width = img_chw.shape
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


def _parse_input_format(input_shape) -> str:
    c1, c2, c3 = input_shape[1:]
    if c1 < min(c2, c3):
        return "nchw"
    if c3 < min(c1, c2):
        return "nhwc"
    raise ValueError(f"cannot parse model input format: {input_shape}")


def _preprocess(img_bgr: np.ndarray) -> np.ndarray:
    img_rgb = img_bgr[..., ::-1]
    img_chw = np.transpose(img_rgb, [2, 0, 1])
    return np.ascontiguousarray(np.float32(img_chw) / 255.0)


def _postprocess(pred_chw: np.ndarray) -> np.ndarray:
    uint8_chw = (pred_chw * 255).clip(0, 255).astype(np.uint8)
    img_rgb = np.transpose(uint8_chw, [1, 2, 0])
    return np.ascontiguousarray(img_rgb[..., ::-1])


class NpuRealEsrganRunner:
    """Run the AMD 256x256 tiled Real-ESRGAN ONNX model."""

    def __init__(
        self,
        onnx_path: Path,
        cache_dir: Path,
        *,
        sr_scale: int = 4,
        tile_overlap: int = 16,
        device: str = "npu",
    ) -> None:
        if device != "npu":
            raise ValueError("only device='npu' is supported by this app runner")

        providers = ort.get_available_providers()
        if "VitisAIExecutionProvider" not in providers:
            raise RuntimeError(
                "VitisAIExecutionProvider が利用できません。"
                f" available={providers}"
            )

        npu_type = get_npu_info()
        if not npu_type:
            raise RuntimeError("対応するRyzen AI NPUが見つかりません。")

        print(f"Using NPU type: {npu_type}", flush=True)
        print("Running inference with providers: ['VitisAIExecutionProvider']", flush=True)

        cache_key = _cache_key(onnx_path)
        provider_options = [{
            "cache_dir": str(cache_dir),
            "cache_key": cache_key,
            "enable_cache_file_io_in_mem": 0,
        }]

        session = ort.InferenceSession(
            str(onnx_path),
            providers=["VitisAIExecutionProvider"],
            provider_options=provider_options,
        )
        input0 = session.get_inputs()[0]
        self.input_name = input0.name
        self.input_shape = tuple(input0.shape)
        self.input_format = _parse_input_format(self.input_shape)
        self.session = session
        self.sr_scale = sr_scale
        self.tile_overlap = max(tile_overlap, 0)

        if self.input_format == "nchw":
            self._in_h, self._in_w = self.input_shape[2:]
        else:
            self._in_h, self._in_w = self.input_shape[1:3]

    def run(self, img_bgr: np.ndarray, progress: ProgressFn | None = None) -> np.ndarray:
        progress = progress or (lambda _f, _m: None)
        if img_bgr.dtype != np.uint8:
            raise ValueError(f"expected uint8 image, got {img_bgr.dtype}")
        if img_bgr.ndim != 3:
            raise ValueError("expected BGR image with 3 channels")

        progress(0.05, "NPU前処理中…")
        img_chw = _preprocess(img_bgr)
        tiles_chw, orig_hw, padded_hw = _split_tiles(
            img_chw, (self._in_h, self._in_w), self.tile_overlap
        )

        sr_tiles_chw: list[np.ndarray] = []
        total = len(tiles_chw)
        for idx, tile_chw in enumerate(tiles_chw, start=1):
            if self.input_format == "nhwc":
                input_3d = np.transpose(tile_chw, [1, 2, 0])
            else:
                input_3d = tile_chw

            outputs = self.session.run(None, {self.input_name: input_3d[None, ...]})
            sr_tile = outputs[0][0]
            if self.input_format == "nhwc":
                sr_tile = np.transpose(sr_tile, [2, 0, 1])
            sr_tiles_chw.append(sr_tile)

            progress(0.08 + 0.84 * (idx / total), f"{idx}/{total} タイル")

        progress(0.94, "NPU後処理中…")
        sr_orig_hw = (orig_hw[0] * self.sr_scale, orig_hw[1] * self.sr_scale)
        sr_padded_hw = (padded_hw[0] * self.sr_scale, padded_hw[1] * self.sr_scale)
        sr_overlap = self.tile_overlap * self.sr_scale
        sr_chw = _merge_tiles(sr_tiles_chw, sr_orig_hw, sr_padded_hw, sr_overlap)
        return _postprocess(sr_chw)

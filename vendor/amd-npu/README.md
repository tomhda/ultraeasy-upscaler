# AMD Ryzen AI NPU assets

This folder contains the NPU models used by ultraeasy-upscaler's optional
NPU backend.

## realesrgan (from AMD fp32, BF16-converted 2026-08)
- Source model: `amd/realesrgan-256x256-tiles-amdnpu` fp32 (HuggingFace)
- Converted with `quark.onnx.tools.convert_fp32_to_bf16 --format with_cast`
- Model: `onnx-models/realesrgan_nchw_256x256_bf16cast.onnx`
- Cache: `modelcachekey_realesrgan_nchw_256x256_bf16cast/` (VAIML flow)
- License: see `LICENSE` (Research-only RAIL-MS)

## animevideov3 (self-quantized, 2026-08)
- Source weights: xinntao/Real-ESRGAN `realesr-animevideov3.pth` (BSD-3-Clause)
- Pipeline: `scripts/npu/export_animevideov3.py` (SRVGGNetCompact -> fixed
  256x256 fp32 ONNX) -> `scripts/npu/quantize_animevideov3.py` (Quark XINT8
  u8s8, calibrated on local frames) -> first VitisAI session compiles into
  `modelcachekey_animevideov3_nchw_256x256_u8s8/`
- Model: `onnx-models/animevideov3dp_nchw_256x256_bf16cast.onnx`
- "dp" = decomposed PReLU (ReLU(x) - w*ReLU(-x)): the VAIML BF16 flow
  silently miscompiles trained PReLU weights (negative slopes); the
  equivalent decomposition avoids the PReLU kernel entirely.
  Export with `scripts/npu/export_animevideov3.py --decompose-prelu`.
- Cache: `modelcachekey_animevideov3dp_nchw_256x256_bf16cast/` (VAIML flow).
  This one exceeds GitHub's 100MB file limit and is NOT committed; the
  first NPU run after a fresh clone recompiles it (about 5 minutes).

## x4plus_anime (self-converted BF16, 2026-08)
- Source weights: xinntao/Real-ESRGAN `RealESRGAN_x4plus_anime_6B.pth`
  (BSD-3-Clause), exported via `scripts/npu/export_x4plus_anime.py`
- Model: `onnx-models/x4plus_anime_nchw_256x256_bf16cast.onnx`
- Cache: `modelcachekey_x4plus_anime_nchw_256x256_bf16cast/` (VAIML flow)

Note: BF16 (VAIML) turned out both faster and closer to fp32 than the
int8/XIR flow on this machine; int8 builds were removed once the BF16
variants were verified.

The cache directory is intentionally kept next to the ONNX model. Ryzen AI's
VitisAI Execution Provider uses this path through explicit `cache_dir` and
`cache_key` options, avoiding first-run recompilation on the tested machine.

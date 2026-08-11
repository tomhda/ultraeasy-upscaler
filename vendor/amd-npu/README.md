# AMD Ryzen AI NPU assets

This folder contains the NPU models used by ultraeasy-upscaler's optional
NPU backend.

## realesrgan (AMD ready-to-run)
- Source model: `amd/realesrgan-256x256-tiles-amdnpu` (HuggingFace)
- Model: `onnx-models/realesrgan_nchw_256x256_u8s8.onnx`
- Cache: `modelcachekey_realesrgan_nchw_256x256_u8s8/`
- License: see `LICENSE` (Research-only RAIL-MS)

## animevideov3 (self-quantized, 2026-08)
- Source weights: xinntao/Real-ESRGAN `realesr-animevideov3.pth` (BSD-3-Clause)
- Pipeline: `scripts/npu/export_animevideov3.py` (SRVGGNetCompact -> fixed
  256x256 fp32 ONNX) -> `scripts/npu/quantize_animevideov3.py` (Quark XINT8
  u8s8, calibrated on local frames) -> first VitisAI session compiles into
  `modelcachekey_animevideov3_nchw_256x256_u8s8/`
- Model: `onnx-models/animevideov3_nchw_256x256_u8s8.onnx`
- Note: NPU throughput on this machine is bandwidth-bound (~4.2 MP/s
  regardless of model or tile size); a 512x512 variant showed no gain.

The cache directory is intentionally kept next to the ONNX model. Ryzen AI's
VitisAI Execution Provider uses this path through explicit `cache_dir` and
`cache_key` options, avoiding first-run recompilation on the tested machine.

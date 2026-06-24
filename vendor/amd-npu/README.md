# AMD Ryzen AI NPU assets

This folder contains the AMD ready-to-run Real-ESRGAN NPU assets used by
ultraeasy-upscaler's optional NPU backend.

- Source model: `amd/realesrgan-256x256-tiles-amdnpu`
- Model: `onnx-models/realesrgan_nchw_256x256_u8s8.onnx`
- Cache: `modelcachekey_realesrgan_nchw_256x256_u8s8/`
- License: see `LICENSE`

The cache directory is intentionally kept next to the ONNX model. Ryzen AI's
VitisAI Execution Provider uses this path through explicit `cache_dir` and
`cache_key` options, avoiding first-run recompilation on the tested machine.

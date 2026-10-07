# FILM ONNX / DirectML model and conversion tools

The production FILM model is `models/film/film_style_fp32.onnx`.  It is an
FP32 ONNX graph for Windows ML / DirectML and does not require CUDA or a
PyTorch runtime during normal application execution.  The model is kept
outside Git under the repository's ignored model directory.  Its SHA-256 is:

```text
AFB6804F0446C92C7DE642EC3268F83D821A838FDD7474AC737E4F9975AC34A3
```

Build the production Windows ML helper with .NET 8:

```powershell
dotnet build tools/winml-film/WinMLFilm.csproj -c Release
```

The resulting executable is
`tools/winml-film/bin/Release/net8.0-windows10.0.22621.0/win-x64/winml-film.exe`.
It accepts the production model with the `x0`, `x1`, `time` → `image`
interface described below.  Select AMD or NVIDIA DirectML with
`--device-index` when more than one GPU is present.

The model source used for this PoC is
[dajes/frame-interpolation-pytorch](https://github.com/dajes/frame-interpolation-pytorch),
which is Apache-2.0 licensed and implements the Google Research FILM Style
network.  The tested release is `v1.0.2` (commit
`33bf477ea2c65a2ed1b5f2280ee1f02be06910e8`) with the FP32 checkpoint
`film_net_fp32.pt`.  The checkpoint is downloaded into `tmp/` and is ignored by
Git.  Its SHA-256 in the feasibility run was:

```text
F810CADA26D0C288E50A27EAC43AF74446EB84B857CCBC77A22BB006F4D27240
```

The dajes source checkout and checkpoint can be prepared with:

```powershell
git clone --depth 1 --branch v1.0.2 `
  https://github.com/dajes/frame-interpolation-pytorch.git tmp/film-poc-dajes-src
Invoke-WebRequest `
  https://github.com/dajes/frame-interpolation-pytorch/releases/download/v1.0.2/film_net_fp32.pt `
  -OutFile tmp/film-poc-dajes-src/film_net_fp32.pt
```

The exporter loads this TorchScript checkpoint into the eager dajes model and
exports one FP32 ONNX graph with this contract:

| name | shape | meaning |
| --- | --- | --- |
| `x0` | `1x3xHxW` | RGB image at t=0, `[0,1]` |
| `x1` | `1x3xHxW` | RGB image at t=1, `[0,1]` |
| `time` | `1x1` | interpolation time, normally `0.5` |
| `image` | `1x3xHxW` | RGB result, clamp to `[0,1]` at the caller |

`H` and `W` are dynamic and must be multiples of 64.  The C# helper and the
diagnostic runner pad arbitrary input images with symmetric zero padding,
run the model, then crop back to the source dimensions.

```powershell
.venv\Scripts\python.exe scripts/film/export_film_onnx.py `
  --checkpoint tmp/film-poc-dajes-src/film_net_fp32.pt `
  --source-dir tmp/film-poc-dajes-src `
  --output tmp/film-poc/film_net_fp32_dynamic.onnx `
  --check-size 128x192 --check-size 256x256 --dml
New-Item -ItemType Directory -Force models/film | Out-Null
Copy-Item tmp/film-poc/film_net_fp32_dynamic.onnx models/film/film_style_fp32.onnx
```

The usual `dynamic_axes` export of the upstream `util.warp` function is not
valid: Python shape extraction makes the normalized sampling grid depend on the
trace shape.  `export_film_onnx.py` replaces only that grid construction with
`Shape` + `Range` operations.  The model weights and all other computation stay
unchanged.  A 64x64/128x128/128x192/192x256 check produced CPU-vs-eager PSNR of
about 103–120dB; the largest absolute error was below `7e-5` in that run.
An independent comparison against the unmodified dajes FP32 TorchScript
checkpoint gave the following results for the production ONNX:

| input | CPU max error / PSNR | DirectML max error / PSNR |
| --- | --- | --- |
| 64x64 | `1.07e-5` / `118.56dB` | `2.05e-5` / `116.60dB` |
| 128x192 | `5.12e-5` / `108.08dB` | `6.63e-5` / `106.13dB` |

The table uses width-by-height notation.  With the same symmetric zero padding
and center crop, the 63x65 Python DirectML runner and C# helper middle PNG
matched exactly after 8-bit PNG encoding (max difference 0, PSNR 99dB).

DirectML was available on the test Windows machine (`onnxruntime-directml`
1.24.4).  For the 768x1024 pair shipped in the dajes repository, the dynamic
model took about 5.11s on CPU and 0.52s on DirectML after warmup.  The DML
output compared with CPU at small resolutions had max absolute error below
`1.2e-4` and RMSE below `8e-6`.  ONNX Runtime assigned the convolution and
`GridSample` nodes to `DmlExecutionProvider`; shape/index bookkeeping nodes
remained on `CPUExecutionProvider`, which is expected for this EP fallback.

The Windows ML C# helper also ran the final ONNX on both GPUs in the test
machine.  `--device-index 1` selected the AMD Radeon(TM) Graphics adapter and
`--device-index 0` selected the NVIDIA GeForce RTX 5060 Ti adapter.  Both
completed a 63x65 two-frame pair (padded to 64x128) and a 1024x768 pair with
the `DmlExecutionProvider`; the helper logged only the expected shape/index CPU
assignment warning.  The C# middle PNG and the Python DirectML PNG from the
1024x768 pair were within one 8-bit level (PSNR 88.49dB after PNG rounding).

For a direct pair test with the production model:

```powershell
.venv\Scripts\python.exe scripts/film/run_film_poc.py `
  --model models/film/film_style_fp32.onnx `
  --input0 tmp/film-poc-dajes-src/photos/one.png `
  --input1 tmp/film-poc-dajes-src/photos/two.png `
  --output tmp/film-poc/film-dml.png `
  --ep-name DmlExecutionProvider --verify-cpu
```

The Python runner is only for conversion and numerical diagnostics.  Keep the
source checkout, checkpoint, generated ONNX copies, and profiling JSON under
`tmp/` or another user-selected model directory.  Do not add them to Git.

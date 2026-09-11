# SwinIR CUDA worker

This optional NVIDIA-focused worker uses the official SwinIR-M real-world x4
network and a separate PyTorch CUDA environment. The app exposes it as
`SwinIR-M (CUDA, very slow)` without adding PyTorch to the default environment.

## Setup

From the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_swinir.ps1
```

The script puts the large CUDA/PyTorch environment and the `.pth` weight under
ignored `tmp/` paths. The main `.venv` and `requirements.txt` remain unchanged.

## Image test

```powershell
tmp\swinir-venv\Scripts\python.exe tools\swinir\worker.py image `
  --model tmp\swinir-models\003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth `
  --input tmp\bench-854x480.png `
  --output tmp\bench-swinir-m.png `
  --tile 256 --tile-overlap 32 --precision bf16
```

BF16 is the default CUDA mixed-precision mode on the tested NVIDIA GPU. Use
`--precision fp32` for a slower reference run. FP16 is exposed for further
experiments but is not the default because this model produced non-finite
values on the tested real image. `--tile` must be a multiple of 8, matching
SwinIR's window size.

## Video use

The `serve` subcommand speaks the same `UEUH/UEUF/UEUD` raw-RGB protocol as
the WinML helper. The app keeps one worker alive, commits final-quality H.264
chunks every 150 frames, and resumes from completed chunks after cancellation
or a crash. The checkpoint directory is removed only after the final video is
successfully published. Source, model, worker, and completed chunks are checked
with SHA-256; VFR input is counted under the same CFR conversion used by the
main decoder. Audio is muxed from the source at the end.

Select `SwinIR-M (CUDA, very slow)` in the app after setup. SwinIR CUDA video
cannot currently be combined with RIFE, and HDR input is rejected. The default
BF16/256-tile configuration is the measured fast path; the app's 128- and
64-tile memory-saving choices are also passed to this worker.

`network_swinir.py` is derived from the official Apache-2.0 SwinIR project;
see `NOTICE.md` and `SWINIR-LICENSE.txt`. The pretrained weights are downloaded
separately and are not committed.

# AGENTS.md

How to use ultraeasy-upscaler from the command line, without opening the window.
Written for AI agents and scripts. The README covers the window.

## What this is

A Windows app that enlarges images and videos 4x with super-resolution models, and
interpolates video frames. Everything runs on the local PC (GPU, or the NPU on some PCs).

## Setup

- Requires: Windows 11 x64 and a DirectX 12 GPU.
- Portable build: extract `ultraeasy-upscaler-portable-win64.zip`. The command is
  `ultraeasy-upscaler-cli.exe`, in the same folder as `ultraeasy-upscaler.exe`. Nothing else to install.
- From source: Python 3.13, `pip install -r requirements.txt`, `ffmpeg` and `ffprobe` on `PATH`,
  then run `powershell -ExecutionPolicy Bypass -File setup.ps1` once to fetch the helper and models.
  The command is `python -m app.cli`, run from the repository root.
- Working directory: any. Paths in arguments can be relative or absolute.

The examples below use `ultraeasy-upscaler-cli`. From source, replace it with `python -m app.cli`.

Check that it works before anything else:

```bash
ultraeasy-upscaler-cli status --json
```

Success: exit code 0, `"ok": true`, `"ffmpeg": {"found": true}`, and at least one of
`backends.gpu.available` / `backends.vulkan.available` is `true`.

## Common tasks

### See which models can be used

```bash
ultraeasy-upscaler-cli models --json
```

- Output: `upscale[]` and `interpolation[]`. Each entry has `key`, `name`, `available`, and `note`.
  Upscaling entries also have `image` and `video` (what the model can process).
- Use `key` as the value of `--model` / `--interpolation`. Only entries with `"available": true` work.
- Starting points: `animevideov3` for anime and CG, `4xNomosUni` for live action.

### Upscale an image

```bash
ultraeasy-upscaler-cli run photo.png --model 4xNomosUni --out-dir out --json
```

- Input: one or more images, videos, or folders of images.
- Output: `out\photo_x4.png`. Without `--out-dir`, files go to an `upscaled` folder next to each input.
- Check: exit code 0, `"ok": true`, `results[0].status` is `"done"`, and `results[0].width` /
  `height` are 4 times the input's (compare with `info photo.png --json`). These values are read
  back from the written file. `results[0].output` is the full path of the file.
- Failure: exit code 1 and `results[i].status` is `"failed"` with an `error` object. Other inputs
  still run.

### Upscale a video, or interpolate its frames

```bash
ultraeasy-upscaler-cli run clip.mp4 --model animevideov3 --json
ultraeasy-upscaler-cli run clip.mp4 --model none --interpolation rife-v4.6 --json
```

- Output: an H.264 `.mp4`. Audio is kept unless `--no-audio` is given.
- `--interpolation` doubles the frame rate. `--factor 4` or `8` multiplies it further;
  `--fps 60` sets a fixed rate instead (RIFE only).
- Check: `results[0].fps` and `results[0].frames` are read back from the written file.
  With `--interpolation` and no `--fps`, `fps` is the input's fps times the factor.
- Time: videos take long, because every frame is processed. Try `quick-check` or a short clip
  first. FILM (Style) is about 30 times slower than RIFE v4.6. Results larger than 3840x2160
  are reduced to that size.

### Check the result on one frame first

```bash
ultraeasy-upscaler-cli quick-check clip.mp4 --time 5 --model animevideov3 --out check.png --json
```

- Enlarges one image (for a video, the frame at `--time` seconds; default 2) and saves it as a PNG.
  `--rect x,y,w,h` limits it to part of that image, in source pixels.
- Use it to compare models before a long video run.

### Plan without processing

```bash
ultraeasy-upscaler-cli run a.png b.mp4 --model animevideov3 --dry-run --json
```

- Validates every input and prints the settings and the exact output path each one would get.
  Writes nothing. `results[i].status` is `"planned"`.

## Running without prompts

The command never asks for input.

- `--json` prints exactly one JSON object to stdout. Progress goes to stderr; `--quiet` hides it.
- Existing output files are not overwritten: the new file gets a number such as `(1)`.
  `--overwrite` replaces the existing file instead.
- All inputs are validated before anything starts. If one is wrong, nothing is processed.
- Exit codes: `0` success, `1` a processing failure, `2` a usage error (bad argument, missing file,
  model that can't be used for this input).

Errors look like this, on stdout with `--json` (otherwise as `error:` and `fix:` lines on stderr):

```json
{"ok": false, "error": {"code": "model_unknown", "message": "Unknown model: foo.", "fix": "Check available models with: ultraeasy-upscaler-cli models --json"}}
```

`code` is a fixed name, and `fix` says what to do next. Common codes: `input_not_found`,
`unsupported_input`, `model_unknown`, `model_not_for_video` (AdcSR is for still images),
`kit_missing` (an add-on is not installed), `npu_not_converted`, `fps_not_multiple`
(FILM takes `--factor`, not `--fps`), `fps_too_low`, `output_exists`, `process_failed`.

## Side effects

- Writes: output files, in `--out-dir` or an `upscaled` folder next to the input. `quick-check`
  writes the `--out` file, and for a video also keeps the extracted frame in a temp folder
  (its path is `source_frame` in the result).
- Overwrites or deletes: nothing, unless `--overwrite` is given. Input files are never changed.
- Network: none. Nothing is sent or downloaded.
- Settings: reads and writes no settings file. The same command gives the same result.
- Safe to run again: yes. A rerun without `--overwrite` adds a numbered file next to the first one.
- `npu-convert <model>` converts a model for the NPU. It takes from 14 minutes to 2 hours and
  writes a cache on the PC; run it only when asked to use the NPU.

## More help

- `ultraeasy-upscaler-cli --help`, and `--help` after any command, list the arguments, defaults,
  and examples.
- The window, the models, and the add-on kits: [README.md](README.md) (Japanese),
  [README.en.md](README.en.md) (English).
- Internals and running from source: [docs/technical.md](docs/technical.md) (Japanese).

## agent-friendly check record

Not checked yet.

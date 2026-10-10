# AGENTS.md

How to use TOGU SCALER from the command line, without opening the window.
Written for AI agents and scripts. The README covers the window.

## What this is

A Windows app that enlarges images and videos 4x with super-resolution models, and
interpolates video frames. Everything runs on the local PC (GPU, or the NPU on some PCs).

## Setup

- Requires: Windows 11 x64 and a DirectX 12 GPU.
- Portable build: extract `togu-scaler-portable-win64.zip`. The command is
  `togu-scaler-cli.exe`, in the same folder as `togu-scaler.exe`. Nothing else to install.
- From source: Python 3.13, `pip install -r requirements.txt`, `ffmpeg` and `ffprobe` on `PATH`,
  then run `powershell -ExecutionPolicy Bypass -File setup.ps1` once to fetch the helper and models.
  The command is `python -m app.cli`, run from the repository root.
- Working directory: any. Paths in arguments can be relative or absolute.

The examples below use `togu-scaler-cli`. From source, replace it with `python -m app.cli`.

Check that it works before anything else:

```bash
togu-scaler-cli status --json
```

Success: exit code 0, `"ok": true`, `"ffmpeg": {"found": true}`, and at least one of
`backends.gpu.available` / `backends.vulkan.available` is `true`.

## Common tasks

### See which models can be used

```bash
togu-scaler-cli models --json
```

- Output: `upscale[]` and `interpolation[]`. Each entry has `key`, `name`, `available`, and `note`.
  Upscaling entries also have `image` and `video` (what the model can process).
- Use `key` as the value of `--model` / `--interpolation`. Only entries with `"available": true` work;
  the others need an add-on kit (see "Add-on kits" below).
- Starting points: `animevideov3` for anime and CG, `4xNomosUni` for live action. Both are fast.
  `SwinIR` and `AdcSR` take about a minute or more per image, so use them for single stills.
  `AdcSR` can't process videos (`"video": false`).

### Upscale an image

```bash
togu-scaler-cli run photo.png --model 4xNomosUni --out-dir out --json
```

- Input: one or more images, videos, or folders of images.
- Output: `out\photo_x4.png`. Without `--out-dir`, files go to an `upscaled` folder next to each input.
- **Without `--model`, the default model is used and the input is enlarged 4x.** Pass
  `--model none` when you don't want upscaling.
- Check: exit code 0, `"ok": true`, `results[0].status` is `"done"`, and `results[0].width` /
  `height` are 4 times `results[0].source.width` / `height`. `width` and `height` are read back
  from the written file; `source` is the input. `results[0].output` is the full path of the file.
- Failure: exit code 1 and `results[i].status` is `"failed"` with an `error` object. Other inputs
  still run.

### Upscale a video, or interpolate its frames

```bash
togu-scaler-cli run clip.mp4 --model animevideov3 --json
togu-scaler-cli run clip.mp4 --model none --interpolation rife-v4.6 --json
```

- Output: an H.264 `.mp4`. Audio is kept unless `--no-audio` is given.
- `--interpolation` doubles the frame rate. `--factor 4` or `8` multiplies it further;
  `--fps 60` sets a fixed rate instead (RIFE only).
- Check: `results[0].fps` and `results[0].frames` are read back from the written file, and
  `results[0].source` has the input's. With `--interpolation` and no `--fps`, `fps` and `frames`
  are the source's times the factor. For a check that doesn't go through this tool, `status --json`
  gives the path of the bundled `ffprobe`.
- Time: videos take long, because every frame is processed. Try `quick-check` or a short clip
  first. FILM (Style) is about 30 times slower than RIFE v4.6. Results larger than 3840x2160
  are reduced to that size.

### Check the result on one frame first

```bash
togu-scaler-cli quick-check clip.mp4 --time 5 --model animevideov3 --out check.png --json
```

- Enlarges one image (for a video, the frame at `--time` seconds; default 2) and saves it as a PNG.
  `--rect x,y,w,h` limits it to part of that image, in source pixels.
- Use it to compare models before a long video run.

### Plan without processing

```bash
togu-scaler-cli run a.png b.mp4 --model animevideov3 --dry-run --json
```

- Validates every input and prints the settings and the exact output path each one would get.
  Writes nothing. `results[i].status` is `"planned"`.
- There is no time estimate and no time-limit option. `seconds` of a `quick-check` includes
  several seconds of model start-up, so multiplying it by the frame count overestimates a video
  run many times over. For a closer figure, run a short piece of the video first.

### Output file names

The name is the input's name plus what was done to it:

| Command | Output |
|---|---|
| `run photo.png --model 4xNomosUni` | `photo_x4.png` |
| `run clip.mp4 --model animevideov3` | `clip_x4.mp4` |
| `run clip.mp4 --model none --interpolation rife-v4.6` | `clip_RIFE-v4.6_2xfps.mp4` |
| `run clip.mp4 --model animevideov3 --interpolation film-style` | `clip_x4_FILM-Style_2xfps.mp4` |

Don't build the path yourself: read `results[i].output`, or get it beforehand with `--dry-run`.

## Running without prompts

The command never asks for input.

- `--json` prints exactly one JSON object to stdout. Progress goes to stderr; `--quiet` hides it.
- Existing output files are not overwritten: the new file gets a number such as `(1)`.
  `--overwrite` replaces the existing file instead.
- All inputs are validated before anything starts. If one is wrong, nothing is processed.
- Exit codes: `0` success, `1` a processing failure, `2` a usage error (bad argument, missing file,
  model that can't be used for this input). In PowerShell, read `$LASTEXITCODE` right after the
  command; a wrapper such as `pwsh -Command` reports every non-zero code as 1. The `code` in the
  JSON error is the same either way.

Errors look like this, on stdout with `--json` (otherwise as `error:` and `fix:` lines on stderr):

```json
{"ok": false, "error": {"code": "model_unknown", "message": "Unknown model: foo.", "fix": "Check available models with: togu-scaler-cli models --json"}}
```

`code` is a fixed name, and `fix` says what to do next. Common codes: `input_not_found`,
`unsupported_input`, `model_unknown`, `model_not_for_video` (AdcSR is for still images),
`kit_missing` (an add-on is not installed), `npu_not_converted`, `fps_not_multiple`
(FILM takes `--factor`, not `--fps`), `fps_too_low`, `output_exists`, `process_failed`.

## Side effects

- Writes: output files, in `--out-dir` or an `upscaled` folder next to the input. `quick-check`
  writes the `--out` file, and for a video also keeps the extracted frame in a temp folder
  (its path is `source_frame` in the result).
- Temporary files: video runs keep intermediate frames in the system temp folder (`TEMP`) and
  remove them when done. Set `TEMP` and `TMP` to move them.
- Overwrites or deletes: nothing, unless `--overwrite` is given. Input files are never changed.
- Network: none. Nothing is sent or downloaded.
- Settings: reads and writes no settings file. The same command gives the same result.
- Safe to run again: yes. A rerun without `--overwrite` adds a numbered file next to the first one.
- `npu-convert <model>` converts a model for the NPU. It takes from 14 minutes to 2 hours and
  writes a cache on the PC; run it only when asked to use the NPU.

## Add-on kits

Some models are separate downloads. `status --json` shows which are installed under `kits`:

| `kits` key | Adds | File |
|---|---|---|
| `film` | `--interpolation film-style` | `togu-scaler-film-kit.zip` |
| `adcsr_gpu` | `--model AdcSR` on the GPU | `togu-scaler-adcsr-kit.zip` |
| `npu` | `--backend npu` (AMD Ryzen AI PCs with Ryzen AI Software) | `togu-scaler-npu-kit.zip` |
| `adcsr_npu` | `--model AdcSR` on the NPU | `togu-scaler-npu-kit-adcsr.zip` |

The files are on the [releases page](https://github.com/tomhda/togu-scaler/releases/latest).
To install one, extract the zip into the folder that contains `togu-scaler-cli.exe`.
The command itself downloads nothing; installing a kit is a decision for the person you work for.

## More help

- `togu-scaler-cli --help`, and `--help` after any command, list the arguments, defaults,
  and examples.
- The window, the models, and the add-on kits: [README.md](README.md) (Japanese),
  [README.en.md](README.en.md) (English). The README uses display names such as "4xNomosUni SPAN";
  the value for `--model` is the `key` from `models --json`.
- Internals and running from source:
  [docs/technical.md](https://github.com/tomhda/togu-scaler/blob/main/docs/technical.md) (Japanese).

## agent-friendly check record

2026-10-10 — Claude Opus 5.5 and Codex gpt-6.1-sol (high), each in a new conversation with only the
portable folder. Tasks: enlarge a still 4x, double a clip's frame rate, and enlarge a clip with
AdcSR (a planted failure: stills only). Both finished all three without help, without guessing a
command that doesn't exist, and recovered from the failure by choosing another model. Their reports
led to: rejecting uninstalled models before a run, `source` in results, `ffprobe` in `status`,
and the notes above on the default model, output names, timing, and exit codes.

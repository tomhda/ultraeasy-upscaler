"""Command line entry: run the same pipeline as the GUI without a screen.

Source: ``python -m app.cli <command> ...``. Frozen: ``ultraeasy-upscaler-cli.exe``.
The processing itself calls into ``app/core/`` (engine/trial/npu_prepare);
nothing is reimplemented for the CLI. This module never imports Qt.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from app import __version__


# Exit codes: 0 success, 1 processing failure, 2 usage error.
EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2

# Fixed error codes (lowercase with underscores).
# エラーの「直し方」やヘルプの例に出す、このコマンドの呼び方
PROG = (
    "ultraeasy-upscaler-cli" if getattr(sys, "frozen", False)
    else "python -m app.cli"
)

CODE_INPUT_NOT_FOUND = "input_not_found"
CODE_UNSUPPORTED_INPUT = "unsupported_input"
CODE_MODEL_UNKNOWN = "model_unknown"
CODE_INTERPOLATION_UNKNOWN = "interpolation_unknown"
CODE_MODEL_NOT_FOR_VIDEO = "model_not_for_video"
CODE_SCALE_NOT_SUPPORTED = "scale_not_supported"
CODE_KIT_MISSING = "kit_missing"
CODE_FFMPEG_MISSING = "ffmpeg_missing"
CODE_FPS_NOT_MULTIPLE = "fps_not_multiple"
CODE_FPS_TOO_LOW = "fps_too_low"
CODE_FACTOR_FPS_CONFLICT = "factor_fps_conflict"
CODE_NPU_NOT_CONVERTED = "npu_not_converted"
CODE_NPU_UNAVAILABLE = "npu_unavailable"
CODE_OUTPUT_EXISTS = "output_exists"
CODE_INVALID_ARGUMENT = "invalid_argument"
CODE_PROCESS_FAILED = "process_failed"

# CLI backend names and the fixed choices.
BACKEND_NAMES = ("auto", "gpu", "npu", "vulkan")
VIDEO_CONTAINER = "mp4"
QUICK_CHECK_DEFAULT_SECONDS = 2.0


class UsageError(Exception):
    """Usage error: report without processing anything (exit 2)."""

    def __init__(self, code: str, message: str, fix: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.fix = fix


class CliParser(argparse.ArgumentParser):
    """ArgumentParser that follows the CLI error shape on bad arguments."""

    def __init__(self, *args, **kwargs) -> None:
        self._json_requested = False
        super().__init__(*args, **kwargs)

    def error(self, message: str) -> None:  # noqa: N802
        fix = "Run with --help to see arguments, defaults, and examples."
        if self._json_requested:
            print(json.dumps(
                {"ok": False, "error": {
                    "code": CODE_INVALID_ARGUMENT,
                    "message": message,
                    "fix": fix,
                }},
                ensure_ascii=False,
            ))
        else:
            print(f"error: {message}", file=sys.stderr)
            print(f"fix: {fix}", file=sys.stderr)
        raise SystemExit(EXIT_USAGE)


def _emit_error(code: str, message: str, fix: str, as_json: bool) -> None:
    if as_json:
        print(json.dumps(
            {"ok": False, "error": {
                "code": code, "message": message, "fix": fix,
            }},
            ensure_ascii=False,
        ))
    else:
        print(f"error: {message}", file=sys.stderr)
        print(f"fix: {fix}", file=sys.stderr)


def _emit_json(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False))


def _apply_language(args: argparse.Namespace) -> str:
    """CLI messages are English by default; core wording follows the setting."""
    from app import i18n

    lang = args.lang or os.environ.get(i18n.LANG_ENV) or "en"
    if lang not in ("ja", "en"):
        lang = "en"
    i18n.set_language(lang)
    return lang


def _abs(path: str) -> str:
    return str(Path(path).expanduser().resolve())


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--json", action="store_true",
        help="Print only one JSON object to stdout.",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Hide progress lines on stderr.",
    )
    parser.add_argument(
        "--lang", choices=("ja", "en"), default=None,
        help="Message language. Default: UEU_LANG or English.",
    )


def build_parser(argv: list[str] | None = None) -> CliParser:
    common = argparse.ArgumentParser(add_help=False)
    _add_common(common)

    parser = CliParser(
        prog=PROG,
        description=(
            "Run the same upscaling pipeline as the GUI without a screen. "
            "Stdout carries only the result (one JSON object with --json). "
            "Progress and warnings go to stderr. Nothing is asked interactively."
        ),
    )
    parser._json_requested = "--json" in (argv if argv is not None else sys.argv[1:])
    parser.add_argument(
        "--version", action="version", version=__version__,
        help="Print the version and exit.",
    )
    _add_common(parser)
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    p = sub.add_parser(
        "status", parents=[common],
        help="Show state: version, ffmpeg, backends, add-on kits.",
        description="Show the current state. No processing is started.",
        epilog=f"Example: {PROG} status --json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.set_defaults(func=_cmd_status)

    p = sub.add_parser(
        "models", parents=[common],
        help="List usable models for a backend.",
        description=(
            "List upscaling and interpolation models. Keys can be passed "
            "to run --model / --interpolation as is."
        ),
        epilog=f"Example: {PROG} models --backend gpu --json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--backend", choices=BACKEND_NAMES, default="auto",
        help="AI backend. Default: auto.",
    )
    p.set_defaults(func=_cmd_models)

    p = sub.add_parser(
        "info", parents=[common],
        help="Show input info (kind, size, fps).",
        description="Probe one image, video, or folder. Missing values are null.",
        epilog=f"Example: {PROG} info clip.mp4 --json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("input", help="Image, video, or folder to probe.")
    p.set_defaults(func=_cmd_info)

    p = sub.add_parser(
        "run", parents=[common],
        help="Upscale images, videos, or folders.",
        description=(
            "Process inputs one by one, in order. All inputs are validated "
            "before anything starts; one usage error stops everything with "
            "exit 2 and nothing is processed."
        ),
        epilog=(
            "Examples:\n"
            f"  {PROG} run a.png --model animevideov3 --json\n"
            f"  {PROG} run clip.mp4 --interpolation rife-v4.6 "
            "--factor 4 --out-dir out"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("inputs", nargs="+", help="Images, videos, or folders.")
    p.add_argument(
        "--model", default=None,
        help="Upscaling model key (see models). 'none' skips upscaling. "
        "Default: first model of the backend.",
    )
    p.add_argument(
        "--interpolation", default="none",
        help="Frame interpolation model for videos "
        "(none, rife-v4.6, film-style). Ignored for images. Default: none.",
    )
    p.add_argument(
        "--factor", type=int, choices=(2, 4, 8), default=None,
        help="Interpolation factor. Cannot be used with --fps. Default: 2.",
    )
    p.add_argument(
        "--fps", type=float, default=None,
        help="Interpolated fps as a number. Cannot be used with --factor "
        "or with FILM.",
    )
    p.add_argument(
        "--scale", type=int, choices=(2, 4), default=4,
        help="Upscaling factor. 2 works only for some Vulkan models. "
        "Default: 4.",
    )
    p.add_argument(
        "--backend", choices=BACKEND_NAMES, default="auto",
        help="AI backend. Default: auto.",
    )
    p.add_argument(
        "--out-dir", default=None,
        help="Save directly into DIR (no subfolder). "
        "Default: an 'upscaled' folder next to each input.",
    )
    p.add_argument(
        "--image-format", choices=("png", "jpg", "webp"), default="png",
        help="Image output format. Default: png.",
    )
    p.add_argument(
        "--video-quality", type=int, default=18,
        help="Video quality (CRF/QP, smaller is better). Default: 18.",
    )
    p.add_argument(
        "--no-audio", action="store_true",
        help="Drop the audio of videos.",
    )
    p.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite same-named outputs. Without it, (1), (2), ... "
        "are appended.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Only validate and show the plan. No files or folders are made.",
    )
    p.set_defaults(func=_cmd_run)

    p = sub.add_parser(
        "quick-check", parents=[common],
        help="Upscale one frame to a PNG (same as the GUI quick check).",
        description=(
            "Upscale one image (or one video frame) and save it as PNG. "
            "Uses the same path as trial.run_trial."
        ),
        epilog=(
            "Examples:\n"
            f"  {PROG} quick-check a.png --out check.png --json\n"
            f"  {PROG} quick-check clip.mp4 --out check.png "
            "--time 5 --rect 0,0,640,360"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("input", help="Image or video.")
    p.add_argument("--out", required=True, help="Output PNG path.")
    p.add_argument(
        "--time", type=float, default=QUICK_CHECK_DEFAULT_SECONDS,
        help="Video frame position in seconds. "
        "Default: 2 (same as the GUI).",
    )
    p.add_argument(
        "--rect", default=None,
        help="Crop range in source pixels: x,y,w,h.",
    )
    p.add_argument(
        "--model", default=None,
        help="Upscaling model key (see models). Default: first model "
        "of the backend.",
    )
    p.add_argument(
        "--scale", type=int, choices=(2, 4), default=4,
        help="Upscaling factor. Default: 4.",
    )
    p.add_argument(
        "--backend", choices=BACKEND_NAMES, default="auto",
        help="AI backend. Default: auto.",
    )
    p.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite --out when it already exists.",
    )
    p.set_defaults(func=_cmd_quick_check)

    p = sub.add_parser(
        "npu-convert", parents=[common],
        help="Convert a model for the NPU (same as the GUI conversion).",
        description=(
            "Run npu_prepare.convert for one model. Takes a long time on "
            "first run; an estimate is printed to stderr first. "
            "Already-converted models succeed without doing anything."
        ),
        epilog=f"Example: {PROG} npu-convert animevideov3 --json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("model", help="Model key (see models --backend npu).")
    p.set_defaults(func=_cmd_npu_convert)
    return parser


def _use_utf8_streams() -> None:
    """コンソールの既定の文字コード（cp932 など）で出せない文字があっても落ちないようにする。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def main(argv: list[str] | None = None) -> int:
    _use_utf8_streams()
    args_list = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser(args_list)
    try:
        args = parser.parse_args(args_list)
    except SystemExit as exc:
        return int(exc.code or 0)
    if not getattr(args, "command", None):
        parser.print_usage(sys.stderr)
        _emit_error(
            CODE_INVALID_ARGUMENT, "No command given.",
            "Run with --help to see commands and examples.", False,
        )
        return EXIT_USAGE
    _apply_language(args)
    try:
        return int(args.func(args))
    except UsageError as exc:
        _emit_error(exc.code, exc.message, exc.fix, bool(args.json))
        return EXIT_USAGE
    except KeyboardInterrupt:
        print("error: Interrupted.", file=sys.stderr)
        print("fix: Rerun the remaining inputs when ready.", file=sys.stderr)
        return EXIT_FAILED


# ------------------------------------------------------------ shared lookups

def _cli_to_upscale_backend(name: str):
    """CLI backend name to UpscaleBackend (auto is resolved by the caller)."""
    from app.core.settings import UpscaleBackend

    return {
        "gpu": UpscaleBackend.WINML_GPU,
        "npu": UpscaleBackend.NPU_NATIVE,
        "vulkan": UpscaleBackend.VULKAN,
    }[name]


def _gpu_available() -> bool:
    from app.core import helper_backend

    try:
        helper_backend._winml_helper()
    except Exception:
        return False
    return True


def _vulkan_available() -> bool:
    from app.core import binaries

    try:
        binaries.realesrgan_exe()
    except Exception:
        return False
    return True


def _npu_state() -> tuple[bool, str | None]:
    """(available, reason). Reason is None when available."""
    from app.core import binaries, helper_backend, npu_prepare

    if not (binaries.repo_root() / "tools" / "npu-serve" / "npu_serve.py").is_file():
        return False, "NPU kit is not installed."
    try:
        helper_backend._npu_python()
    except Exception:
        return False, "Ryzen AI Python is not installed."
    if not npu_prepare.npu_available():
        return False, "NPU is not available on this PC."
    return True, None


def _resolve_backend(name: str) -> str:
    """Resolve auto to gpu or vulkan (the GUI auto means GPU first)."""
    if name != "auto":
        return name
    return "gpu" if _gpu_available() else "vulkan"


def _upscale_keys(backend: str) -> list[str]:
    """Model keys in display order (same source as the GUI lists)."""
    from app.core import binaries
    from app.core.settings import (
        HELPER_MODEL_ADCSR,
        HELPER_MODEL_AMD_RRDB,
        HELPER_MODEL_ANIME,
        HELPER_MODEL_SPAN,
        HELPER_MODEL_SWINIR,
    )

    if backend in ("gpu", "npu"):
        return [
            HELPER_MODEL_ANIME,
            HELPER_MODEL_SPAN,
            HELPER_MODEL_AMD_RRDB,
            HELPER_MODEL_SWINIR,
            HELPER_MODEL_ADCSR,
        ]
    return list(binaries.available_models())


def _normalize_model_key(backend: str, key: str) -> str:
    """Accept GUI aliases (old keys) and return the concrete model key."""
    from app.core.settings import canonical_helper_model

    if backend == "vulkan":
        return key
    return canonical_helper_model(key) or key


def _check_model_key(backend: str, key: str | None) -> str | None:
    """Validate --model. Returns the concrete key (None stays None)."""
    if key is None or key == "none":
        return None
    valid = set(_upscale_keys(backend))
    concrete = _normalize_model_key(backend, key)
    if key in valid:
        return key
    if concrete in valid:
        return concrete
    raise UsageError(
        CODE_MODEL_UNKNOWN,
        f"Unknown model: {key}.",
        f"Check available models with: {PROG} models --json",
    )


def _default_model_key(backend: str) -> str | None:
    """First model of the backend (same default as the GUI)."""
    from app.core.settings import DEFAULT_HELPER_MODEL, DEFAULT_MODEL

    keys = _upscale_keys(backend)
    if backend in ("gpu", "npu"):
        return DEFAULT_HELPER_MODEL
    if DEFAULT_MODEL in keys:
        return DEFAULT_MODEL
    return keys[0] if keys else None


def _check_interpolation_key(key: str) -> str | None:
    from app.core import binaries

    if key in (None, "none"):
        return None
    if key in ("rife-v4.6", binaries.FILM_MODEL):
        return key
    raise UsageError(
        CODE_INTERPOLATION_UNKNOWN,
        f"Unknown interpolation model: {key}.",
        f"Check available models with: {PROG} models --json",
    )


def _check_interpolation_kit(key: str) -> None:
    from app.core import binaries

    if key in binaries.available_interpolation_models():
        return
    if key == "rife-v4.6":
        raise UsageError(
            CODE_KIT_MISSING,
            "RIFE v4.6 files are missing.",
            "Run the model download script (scripts/get_models.py), "
            "or choose another interpolation model.",
        )
    raise UsageError(
        CODE_KIT_MISSING,
        "FILM (Style) files are missing.",
        "Extract the FILM add-on kit next to the app, "
        "or choose another interpolation model.",
    )


def _model_available(backend: str, key: str) -> bool:
    from app.core import helper_backend
    from app.core.settings import HELPER_MODEL_FILES, UpscaleBackend

    backend_enum = _cli_to_upscale_backend(backend)
    try:
        filenames = HELPER_MODEL_FILES[backend_enum][key]
    except KeyError:
        return False
    if backend == "npu":
        return helper_backend.npu_compiled(key) is not None
    return any(
        helper_backend._search_model_file(name) is not None
        for name in filenames.values()
    )


def _upscale_entries(backend: str) -> list[dict]:
    """Shared body of models --json upscale list (GUI tables are the source)."""
    from app.core import binaries
    from app.core.catalog import MODEL_HINT, MODEL_LABELS
    from app.core import helper_backend
    from app.i18n import t as translate

    entries: list[dict] = []
    for key in _upscale_keys(backend):
        name = translate(MODEL_LABELS.get(key, key))
        hint = MODEL_HINT.get(key, "")
        note = translate(hint) if hint else ""
        if backend == "vulkan":
            scales = [s for s in (2, 4) if binaries.model_supports_scale(key, s)]
            entry: dict = {
                "key": key, "name": name, "image": True, "video": True,
                "scales": scales or [4], "available": True, "note": note,
            }
        else:
            from app.core.settings import HELPER_MODEL_ADCSR

            entry = {
                "key": key, "name": name, "image": True,
                "video": key != HELPER_MODEL_ADCSR,
                "scales": [4],
                "available": _model_available(backend, key),
                "note": note,
            }
            if backend == "npu":
                entry["converted"] = bool(helper_backend.npu_compiled(key))
        entries.append(entry)
    return entries


def _interpolation_entries() -> list[dict]:
    from app.core import binaries
    from app.core.catalog import INTERPOLATION_HINT, INTERPOLATION_LABELS
    from app.i18n import t as translate

    available = binaries.available_interpolation_models()
    return [
        {
            "key": key,
            "name": translate(INTERPOLATION_LABELS[key]),
            "available": key in available,
            "note": translate(INTERPOLATION_HINT[key]),
        }
        for key in ("rife-v4.6", binaries.FILM_MODEL)
    ]


# ------------------------------------------------------------------ commands

def _cmd_status(args: argparse.Namespace) -> int:
    from app.core import addon_kits, binaries

    try:
        ffmpeg = binaries.ffmpeg_exe()
        ffmpeg_state = {"found": True, "path": ffmpeg}
    except Exception:
        ffmpeg_state = {"found": False, "path": None}
    npu_ok, npu_reason = _npu_state()
    backends: dict = {
        "gpu": {"available": _gpu_available()},
        "npu": {"available": npu_ok, "reason": npu_reason},
        "vulkan": {"available": _vulkan_available()},
    }
    kits = {
        "film": addon_kits.film_kit_installed(),
        "npu": addon_kits.npu_kit_installed(),
        "adcsr_gpu": addon_kits.adcsr_gpu_installed(),
        "adcsr_npu": addon_kits.adcsr_npu_installed(),
    }
    payload = {
        "ok": True,
        "version": __version__,
        "app_root": str(_app_root()),
        "ffmpeg": ffmpeg_state,
        "backends": backends,
        "kits": kits,
    }
    if args.json:
        _emit_json(payload)
    else:
        print(f"ultraeasy-upscaler {__version__}")
        print(f"app root: {payload['app_root']}")
        if ffmpeg_state["found"]:
            print(f"ffmpeg: found ({ffmpeg_state['path']})")
        else:
            print("ffmpeg: not found")
        for name in ("gpu", "npu", "vulkan"):
            state = backends[name]
            line = f"backend {name}: "
            line += "available" if state["available"] else "unavailable"
            if not state["available"] and state.get("reason"):
                line += f" ({state['reason']})"
            print(line)
        for key, installed in kits.items():
            print(f"kit {key}: {'installed' if installed else 'not installed'}")
    return EXIT_OK


def _app_root() -> Path:
    from app.core.settings import _app_root as root

    return root()


def _cmd_models(args: argparse.Namespace) -> int:
    backend = _resolve_backend(args.backend)
    payload = {
        "ok": True,
        "backend": backend,
        "upscale": _upscale_entries(backend),
        "interpolation": _interpolation_entries(),
    }
    if args.json:
        _emit_json(payload)
    else:
        print(f"backend: {backend}")
        print("upscale models:")
        for entry in payload["upscale"]:
            kinds = "+".join(
                k for k, flag in (
                    ("image", entry["image"]), ("video", entry["video"]))
                if flag
            )
            scales = "x".join(str(s) for s in entry["scales"])
            state = "available" if entry["available"] else "missing files"
            extra = ""
            if "converted" in entry:
                extra = ", converted" if entry["converted"] else ", not converted"
            print(f"  {entry['key']} - {entry['name']} "
                  f"({kinds}, {scales}{extra}) [{state}] - {entry['note']}")
        print("interpolation models:")
        for entry in payload["interpolation"]:
            state = "available" if entry["available"] else "not installed"
            print(f"  {entry['key']} - {entry['name']} "
                  f"[{state}] - {entry['note']}")
    return EXIT_OK


def _cmd_info(args: argparse.Namespace) -> int:
    from app.core import media
    from app.core.jobs import JobKind

    raw = args.input
    path = Path(raw)
    if not path.exists():
        raise UsageError(
            CODE_INPUT_NOT_FOUND,
            f"Input not found: {raw}.",
            "Check the path and try again.",
        )
    try:
        kind = media.classify(str(path))
    except ValueError:
        raise UsageError(
            CODE_UNSUPPORTED_INPUT,
            f"Unsupported input: {Path(raw).name}.",
            "Use a supported image, video, or folder input.",
        ) from None
    payload: dict = {
        "ok": True, "path": _abs(raw),
        "kind": ("folder" if kind == JobKind.FOLDER
                 else "video" if kind == JobKind.VIDEO else "image"),
        "width": None, "height": None, "fps": None,
        "frames": None, "duration": None, "audio": False,
    }
    if kind != JobKind.FOLDER:
        info = media.probe(str(path))
        payload["width"] = info.width or None
        payload["height"] = info.height or None
        payload["fps"] = info.fps
        payload["frames"] = info.frame_count
        payload["duration"] = info.duration
        payload["audio"] = bool(info.has_audio)
    if args.json:
        _emit_json(payload)
    else:
        line = f"{payload['path']}: {payload['kind']}"
        if payload["width"] and payload["height"]:
            line += f" {payload['width']}x{payload['height']}"
        if payload["fps"]:
            line += f" {payload['fps']:g}fps"
        if payload["frames"]:
            line += f" {payload['frames']} frames"
        if payload["audio"]:
            line += " with audio"
        print(line)
    return EXIT_OK


# ------------------------------------------------------------------- run

def _run_options(args: argparse.Namespace) -> dict:
    """Validate run/quick-check option combinations (no input checks yet)."""
    from app.core import binaries

    factor = args.factor if args.factor is not None else 2
    fps = args.fps
    if args.factor is not None and fps is not None:
        raise UsageError(
            CODE_FACTOR_FPS_CONFLICT,
            "Use either --factor or --fps, not both.",
            "Drop --factor to fix the fps, or drop --fps to use the factor.",
        )
    interpolation = _check_interpolation_key(args.interpolation)
    if interpolation == binaries.FILM_MODEL and fps is not None:
        raise UsageError(
            CODE_FPS_NOT_MULTIPLE,
            "FILM (Style) does not take --fps.",
            "Use --factor 2, 4, or 8 instead of --fps for FILM (Style).",
        )
    return {"factor": factor, "fps": fps, "interpolation": interpolation}


def _check_scale(backend: str, model: str | None, scale: int) -> None:
    from app.core import binaries

    if model is None:
        return
    if backend in ("gpu", "npu"):
        if scale != 4:
            raise UsageError(
                CODE_SCALE_NOT_SUPPORTED,
                f"Scale x{scale} is not supported on {backend} (4x models only).",
                "Use --scale 4, or switch to the Vulkan backend "
                "for 2x on some models.",
            )
        return
    if not binaries.model_supports_scale(model, scale):
        raise UsageError(
            CODE_SCALE_NOT_SUPPORTED,
            f"Model {model} does not support scale x{scale}.",
            "Use --scale 4, or choose another model. "
            f"Check available models with: {PROG} models --json",
        )


def _check_ffmpeg(inputs_kinds: list[str]) -> None:
    from app.core import binaries

    if "video" not in inputs_kinds:
        return
    try:
        binaries.ffmpeg_exe()
        binaries.ffprobe_exe()
    except Exception:
        raise UsageError(
            CODE_FFMPEG_MISSING,
            "ffmpeg or ffprobe was not found, but a video input needs it.",
            "Install ffmpeg and ffprobe and add them to PATH.",
        ) from None


def _probe_dims(path: Path) -> tuple[int, int]:
    """Image/video dimensions (0, 0 when unknown). No inference involved."""
    from app.core import media

    try:
        info = media.probe(str(path))
    except Exception:
        return (0, 0)
    return (int(info.width or 0), int(info.height or 0))


def _check_npu(backend: str, model: str | None, kinds: list[str],
               paths: list[Path]) -> None:
    """NPU pre-checks (same rules as the GUI start button)."""
    from app.core import npu_prepare

    if backend != "npu" or model is None:
        return
    npu_ok, _reason = _npu_state()
    if not npu_ok:
        raise UsageError(
            CODE_NPU_UNAVAILABLE,
            "NPU is not available on this PC.",
            "The NPU needs an AMD Ryzen AI PC with Ryzen AI Software "
            "and the NPU kit.",
        )
    for kind, path in zip(kinds, paths):
        if kind == "folder":
            dims = [
                _probe_dims(p) for p in sorted(path.iterdir())
                if p.is_file()
            ]
            small = all(
                (w and h and min(w, h) < 480) or (not w and not h)
                for w, h in dims
            ) if dims else True
            check_needed = not small
        elif kind in ("image", "video"):
            w, h = _probe_dims(path)
            if w and h and min(w, h) < 480:
                continue  # Auto-switched to the GPU, like effective_backend.
            check_needed = True
        else:
            check_needed = True
        if check_needed and not npu_prepare.is_converted(model):
            label = str(model)
            raise UsageError(
                CODE_NPU_NOT_CONVERTED,
                f"Model {label} is not converted for the NPU yet.",
                f"Convert it first with: {PROG} npu-convert {label}",
            )


def _check_rife_fps(interpolation: str | None, fps: float | None,
                    kinds: list[str], paths: list[Path]) -> None:
    """RIFE --fps must exceed the source fps (needs a probe, like the GUI)."""
    if interpolation != "rife-v4.6" or fps is None:
        return
    from app.core import media

    for kind, path in zip(kinds, paths):
        if kind != "video":
            continue
        try:
            source = media.probe(str(path)).fps
        except Exception:
            continue
        if source and fps <= source:
            raise UsageError(
                CODE_FPS_TOO_LOW,
                f"--fps {fps:g} is not greater than the source "
                f"({source:.3f} fps).",
                "Choose an --fps value greater than the source frame rate.",
            )


def _classify_inputs(raw_inputs: list[str]) -> tuple[list[Path], list[str]]:
    from app.core import media

    paths: list[Path] = []
    kinds: list[str] = []
    for raw in raw_inputs:
        path = Path(raw)
        if not path.exists():
            raise UsageError(
                CODE_INPUT_NOT_FOUND,
                f"Input not found: {raw}.",
                "Check the path and try again.",
            )
        try:
            kind = media.classify(str(path))
        except ValueError:
            raise UsageError(
                CODE_UNSUPPORTED_INPUT,
                f"Unsupported input: {path.name}.",
                "Use a supported image, video, or folder input.",
            ) from None
        paths.append(path)
        kinds.append(kind.value)
    return paths, kinds


def _build_run_settings(args: argparse.Namespace, backend: str,
                        model: str | None, interpolation: str | None,
                        factor: int, fps: float | None):
    from app.core.settings import OutputLocation, UpscaleSettings

    if args.out_dir:
        location, out_dir, subfolder = (
            OutputLocation.CUSTOM, _abs(args.out_dir), False)
    else:
        location, out_dir, subfolder = OutputLocation.SAME, None, True
    return UpscaleSettings(
        backend=_cli_to_upscale_backend(backend),
        scale=int(args.scale),
        model=model,
        image_format=args.image_format,
        output_location=location,
        output_dir=out_dir,
        create_subfolder=subfolder,
        subfolder_name="upscaled",
        overwrite=bool(args.overwrite),
        video_quality=int(args.video_quality),
        keep_audio=not bool(args.no_audio),
        interpolation_model=interpolation,
        target_fps=float(fps) if fps is not None else None,
        interpolation_factor=int(factor),
    )


def _unique_planned(path: Path, overwrite: bool,
                    reserved: set[Path]) -> Path:
    """engine._unique without touching the filesystem (dry-run planning)."""
    if overwrite or (not path.exists() and path not in reserved):
        return path
    i = 1
    while True:
        cand = path.parent / f"{path.stem}({i}){path.suffix}"
        if not cand.exists() and cand not in reserved:
            return cand
        i += 1


def _plan_output(input_path: Path, kind: str, settings,
                 reserved: set[Path]):
    """Output path the real engine run would produce (numbering included)."""
    from app.core.settings import OutputLocation

    # Mirror engine._output_base exactly (without creating folders).
    if (settings.output_location == OutputLocation.CUSTOM
            and settings.output_dir):
        base = Path(settings.output_dir)
        if settings.create_subfolder:
            base = base / settings.subfolder_name
    else:
        base = input_path.parent
        if settings.create_subfolder:
            base = base / settings.subfolder_name
    if kind == "folder":
        return base / f"{input_path.name}{settings.output_suffix()}"
    if kind == "video":
        name = f"{input_path.stem}{settings.output_suffix()}.{VIDEO_CONTAINER}"
    else:
        name = (f"{input_path.stem}{settings.output_suffix()}"
                f".{settings.image_format}")
    return _unique_planned(base / name, settings.overwrite, reserved)


def _result_settings(backend: str, model: str | None, scale: int,
                     interpolation: str | None, factor: int | None,
                     fps: float | None) -> dict:
    return {
        "backend": backend,
        "model": model,
        "scale": scale,
        "interpolation": interpolation,
        "factor": factor,
        "fps": fps,
    }


def _validate_run(args: argparse.Namespace):
    """Validate everything before processing (exit 2, nothing processed)."""
    from app.core import binaries
    from app.core.settings import HELPER_MODEL_ADCSR

    backend = _resolve_backend(args.backend)
    options = _run_options(args)
    model = _check_model_key(
        backend, args.model
        if args.model is not None else _default_model_key(backend))
    interpolation = options["interpolation"]
    factor, fps = options["factor"], options["fps"]
    _check_scale(backend, model, int(args.scale))
    paths, kinds = _classify_inputs(args.inputs)
    _check_ffmpeg(kinds)
    if model == HELPER_MODEL_ADCSR and "video" in kinds:
        raise UsageError(
            CODE_MODEL_NOT_FOR_VIDEO,
            "AdcSR is for still images only. It cannot be used for videos.",
            "Choose another model for videos. Check available models with: "
            f"{PROG} models --json",
        )
    if interpolation is not None:
        _check_interpolation_kit(interpolation)
    _check_npu(backend, model, kinds, paths)
    _check_rife_fps(interpolation, fps, kinds, paths)
    if interpolation == binaries.FILM_MODEL:
        # FILM validates the factor against the source fps at runtime
        # (film.resolve_factor); unsupported fps fails that input (exit 1).
        pass
    return backend, model, interpolation, factor, fps, paths, kinds


def _cmd_run(args: argparse.Namespace) -> int:
    from dataclasses import replace

    (backend, model, interpolation, factor, fps,
     paths, kinds) = _validate_run(args)
    base_settings = _build_run_settings(
        args, backend, model, interpolation, factor, fps)
    use_fps = fps if interpolation is not None else None
    use_factor = factor if interpolation is not None else None

    # Per-input settings: images (and folders) ignore interpolation
    # so video and image inputs can be mixed in one command.
    planned: list[dict] = []
    reserved: set[Path] = set()
    for path, kind in zip(paths, kinds):
        if kind in ("image", "folder"):
            settings = replace(base_settings, interpolation_model=None,
                               target_fps=None)
            note = (None if interpolation is None
                    else "Interpolation is ignored for images.")
            entry_interp, entry_factor, entry_fps = None, None, None
        else:
            settings = base_settings
            note = None
            entry_interp, entry_factor, entry_fps = (
                interpolation, use_factor, use_fps)
        if args.dry_run:
            out = _plan_output(path, kind, settings, reserved)
            reserved.add(out)
        else:
            out = None
        planned.append({
            "path": path, "kind": kind, "settings": settings,
            "result_settings": _result_settings(
                backend, model, int(args.scale),
                entry_interp, entry_factor, entry_fps),
            "note": note, "planned": out,
        })

    if args.dry_run:
        results = [
            {
                "input": _abs(str(item["path"])),
                "kind": item["kind"],
                "status": "planned",
                "output": str(item["planned"]),
                "settings": item["result_settings"],
                **({"note": item["note"]} if item["note"] else {}),
            }
            for item in planned
        ]
        payload = {"ok": True, "dry_run": True, "results": results}
        if args.json:
            _emit_json(payload)
        else:
            for item, result in zip(planned, results):
                line = f"plan: {item['path'].name} -> {result['output']}"
                print(line)
        return EXIT_OK
    return _execute_run(args, planned)


def _execute_run(args: argparse.Namespace, planned: list[dict]) -> int:
    from app.core import engine
    from app.core.jobs import Job, JobKind

    total = len(planned)
    results: list[dict] = []
    failed = False
    interrupted = False
    last_line: list[str] = []

    def progress(index: int, name: str):
        def _cb(fraction: float, message: str) -> None:
            if args.quiet:
                return
            line = (f"[{index + 1}/{total}] {name} "
                    f"{int(max(0.0, min(1.0, fraction)) * 100)}% {message}")
            if last_line and last_line[0] == line:
                return
            last_line.clear()
            last_line.append(line)
            print(line, file=sys.stderr)
        return _cb

    for index, item in enumerate(planned):
        path: Path = item["path"]
        kind: str = item["kind"]
        settings = item["settings"]
        job = Job(input_path=path, kind=JobKind(kind))
        start = time.monotonic()
        try:
            out = engine.process_job(
                job, settings, progress=progress(index, path.name))
        except KeyboardInterrupt:
            interrupted = True
            break
        except Exception as exc:  # One failure never stops the rest.
            failed = True
            results.append(_failed_result(
                path, kind, item, CODE_PROCESS_FAILED, str(exc) or repr(exc),
                "Run with --dry-run to check the plan, then retry. "
                "If it keeps failing, try another backend with --backend."))
            if not args.json:
                print(f"failed: {path.name}: {exc}", file=sys.stderr)
            continue
        seconds = round(time.monotonic() - start, 2)
        results.append(_done_result(
            path, kind, item, out, seconds, args, item["note"]))
        if not results[-1]["_failed"]:
            if not args.json:
                print(f"done: {path.name} -> {results[-1]['output']} "
                      f"({seconds}s)")
        else:
            failed = True
    ok = not failed and not interrupted
    payload = {"ok": ok, "dry_run": False,
               "results": [{k: v for k, v in r.items()
                             if not k.startswith("_")} for r in results]}
    if args.json:
        _emit_json(payload)
    if interrupted:
        print("error: Interrupted.", file=sys.stderr)
        print("fix: Rerun the remaining inputs when ready.", file=sys.stderr)
    return EXIT_FAILED if (failed or interrupted) else EXIT_OK


def _failed_result(path: Path, kind: str, item: dict,
                   code: str, message: str, fix: str) -> dict:
    result = {
        "input": _abs(str(path)),
        "kind": kind,
        "status": "failed",
        "output": None,
        "settings": item["result_settings"],
        "error": {"code": code, "message": message, "fix": fix},
    }
    if item.get("note"):
        result["note"] = item["note"]
    return result


def _done_result(path: Path, kind: str, item: dict, out: Path,
                 seconds: float, args: argparse.Namespace,
                 note: str | None) -> dict:
    """Verify the output file exists and re-probe it (exit-1 on mismatch)."""
    ok_file = False
    try:
        if kind == "folder":
            ok_file = Path(out).is_dir()
        else:
            ok_file = (Path(out).is_file()
                       and Path(out).stat().st_size > 0)
    except OSError:
        ok_file = False
    if not ok_file:
        result = _failed_result(
            path, kind, item, CODE_PROCESS_FAILED,
            f"Output was not written: {out}.",
            "Run with --dry-run to check the plan, then retry.")
        result["_failed"] = True
        return result
    result: dict = {
        "input": _abs(str(path)),
        "kind": kind,
        "status": "done",
        "output": str(Path(out).resolve()),
        "seconds": seconds,
        "settings": item["result_settings"],
        "width": None,
        "height": None,
        "_failed": False,
    }
    if kind in ("image", "video"):
        try:
            from app.core import media

            probed = media.probe(str(out))
            result["width"] = probed.width or None
            result["height"] = probed.height or None
            if kind == "video":
                result["fps"] = probed.fps
                result["frames"] = probed.frame_count
        except Exception:
            pass
    if note:
        result["note"] = note
    if not args.json and note:
        print(f"note: {path.name}: {note}", file=sys.stderr)
    return result


# ---------------------------------------------------------- quick-check

def _parse_rect(value: str | None) -> tuple[int, int, int, int] | None:
    if value is None:
        return None
    try:
        parts = [int(v) for v in value.split(",")]
    except ValueError:
        raise UsageError(
            CODE_INVALID_ARGUMENT,
            f"Bad --rect: {value}. Use x,y,w,h in source pixels.",
            "Example: --rect 0,0,640,360.",
        ) from None
    if len(parts) != 4 or parts[2] <= 0 or parts[3] <= 0:
        raise UsageError(
            CODE_INVALID_ARGUMENT,
            f"Bad --rect: {value}. Use x,y,w,h in source pixels.",
            "Example: --rect 0,0,640,360.",
        )
    return (parts[0], parts[1], parts[2], parts[3])


def _cmd_quick_check(args: argparse.Namespace) -> int:
    import tempfile

    from app.core import trial as trial_core

    backend = _resolve_backend(args.backend)
    # quick-check never interpolates (same as trial.run_trial).
    model = _check_model_key(
        backend, args.model
        if args.model is not None else _default_model_key(backend))
    _check_scale(backend, model, int(args.scale))
    raw = args.input
    path = Path(raw)
    if not path.exists():
        raise UsageError(
            CODE_INPUT_NOT_FOUND,
            f"Input not found: {raw}.",
            "Check the path and try again.",
        )
    from app.core import media
    from app.core.jobs import JobKind

    try:
        kind = media.classify(str(path))
    except ValueError:
        raise UsageError(
            CODE_UNSUPPORTED_INPUT,
            f"Unsupported input: {path.name}.",
            "Quick check needs an image or a video.",
        ) from None
    if kind == JobKind.FOLDER:
        raise UsageError(
            CODE_UNSUPPORTED_INPUT,
            f"Unsupported input: {path.name}.",
            "Quick check needs an image or a video.",
        )
    if kind == JobKind.VIDEO:
        _check_ffmpeg(["video"])
    out = Path(args.out)
    if out.exists() and not args.overwrite:
        raise UsageError(
            CODE_OUTPUT_EXISTS,
            f"Output already exists: {args.out}.",
            "Add --overwrite, or choose another path with --out.",
        )
    if args.time < 0:
        raise UsageError(
            CODE_INVALID_ARGUMENT,
            f"Bad --time: {args.time}. Use 0 or more seconds.",
            "Example: --time 5.",
        )
    rect = _parse_rect(args.rect)
    from app.core.settings import UpscaleSettings

    settings = UpscaleSettings(
        backend=_cli_to_upscale_backend(backend),
        scale=int(args.scale),
        model=model,
        image_format="png",
        overwrite=bool(args.overwrite),
    )

    workdir = Path(tempfile.mkdtemp(prefix="ueu-quick-"))
    if kind == JobKind.VIDEO:
        frame = workdir / "source_frame.png"
        source_frame = str(frame)
    else:
        source_frame = _abs(raw)
    trial_source = source_frame
    if rect is not None:
        trial_source = str(workdir / "cropped.png")
    start = time.monotonic()
    try:
        if kind == JobKind.VIDEO:
            trial_core.extract_frame(raw, float(args.time), source_frame)
        if rect is not None:
            trial_core.crop_image(source_frame, rect, trial_source)
        out.parent.mkdir(parents=True, exist_ok=True)
        trial_core.run_trial(trial_source, settings, str(out))
    except UsageError:
        raise
    except ValueError as exc:
        # Out-of-range rect and similar input problems are usage errors.
        raise UsageError(
            CODE_INVALID_ARGUMENT, str(exc) or repr(exc),
            "Example: --rect 0,0,640,360.",
        ) from None
    except Exception as exc:
        _emit_error(CODE_PROCESS_FAILED, str(exc) or repr(exc),
                    "Run with --dry-run on run to check the plan, then retry. "
                    "If it keeps failing, try another backend with --backend.",
                    bool(args.json))
        return EXIT_FAILED
    seconds = round(time.monotonic() - start, 2)
    width, height = None, None
    try:
        probed = media.probe(str(out))
        width, height = probed.width or None, probed.height or None
    except Exception:
        pass
    payload = {
        "ok": True,
        "output": str(out.resolve()),
        "source_frame": (_abs(source_frame) if kind == JobKind.VIDEO
                         else source_frame),
        "width": width,
        "height": height,
        "seconds": seconds,
        "settings": _result_settings(
            backend, model, int(args.scale), None, None, None),
    }
    if args.json:
        _emit_json(payload)
    else:
        print(f"quick check done: {payload['output']} "
              f"({width}x{height}, {seconds}s)")
    return EXIT_OK


# ---------------------------------------------------------- npu-convert

def _cmd_npu_convert(args: argparse.Namespace) -> int:
    from app.core import npu_prepare
    from app.core.settings import canonical_helper_model

    key = canonical_helper_model(args.model) or args.model
    if key not in npu_prepare.NPU_CONVERT_MINUTES:
        raise UsageError(
            CODE_MODEL_UNKNOWN,
            f"Unknown model: {args.model}.",
            "Check available models with: "
            f"{PROG} models --backend npu --json",
        )
    npu_ok, _reason = _npu_state()
    if not npu_ok:
        _emit_error(
            CODE_NPU_UNAVAILABLE,
            "NPU is not available on this PC.",
            "The NPU needs an AMD Ryzen AI PC with Ryzen AI Software "
            "and the NPU kit.", bool(args.json))
        return EXIT_FAILED
    if npu_prepare.is_converted(key):
        payload = {"ok": True, "model": key, "converted": True,
                   "already_converted": True}
        if args.json:
            _emit_json(payload)
        else:
            print(f"Already converted: {key}.")
        return EXIT_OK
    minutes = npu_prepare.NPU_CONVERT_MINUTES[key]
    if not args.quiet:
        print(f"First conversion takes about {minutes} minutes. "
              "The PC stays usable during conversion.",
              file=sys.stderr)
    try:
        npu_prepare.convert(key)
    except Exception as exc:
        _emit_error(CODE_PROCESS_FAILED, str(exc) or repr(exc),
                    "Check the NPU kit and Ryzen AI Software, then retry.",
                    bool(args.json))
        return EXIT_FAILED
    payload = {"ok": True, "model": key, "converted": True,
               "already_converted": False}
    if args.json:
        _emit_json(payload)
    else:
        print(f"Converted: {key}.")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())



"""Parent-process wrapper for Ryzen AI NPU upscaling."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

from . import binaries, jobs, media
from .jobs import ProgressCb
from .settings import UpscaleSettings

CONDA_ENV = "ryzen-ai-1.7.1"
_CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0


def _noop(_f: float, _m: str) -> None:
    pass


def _conda_exe() -> str:
    found = shutil.which("conda")
    if found:
        return found

    candidates = [
        Path.home() / "miniforge3" / "Scripts" / "conda.exe",
        Path.home() / "miniforge3" / "condabin" / "conda.bat",
    ]
    for path in candidates:
        if path.exists():
            return str(path)

    raise RuntimeError(
        "conda が見つかりません。Miniforge / Ryzen AI 環境を確認してください。"
    )


def _check_npu_settings(settings: UpscaleSettings) -> None:
    if int(settings.scale) != 4:
        raise ValueError("NPU backend は現在 x4 のみ対応です。倍率を 4x にしてください。")
    # conda 起動（数秒）前に未対応モデルを弾く。BinaryError を ValueError として上げる
    try:
        binaries.npu_model_spec(settings.model)
    except binaries.BinaryError as exc:
        raise ValueError(str(exc)) from exc


def _fmt(value: str) -> str:
    value = (value or "png").lower()
    if value == "jpeg":
        return "jpg"
    if value not in {"png", "jpg", "webp"}:
        return "png"
    return value


def _run_worker(args: list[str], progress: ProgressCb, cancel=None) -> None:
    cmd = [
        _conda_exe(),
        "run",
        "--no-capture-output",
        "-n",
        CONDA_ENV,
        "python",
        "-m",
        "app.core.npu_worker",
        *args,
    ]
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env["PYTHONIOENCODING"] = "utf-8"

    proc = subprocess.Popen(
        cmd,
        cwd=str(binaries.repo_root()),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
        creationflags=_CREATE_NO_WINDOW,
    )

    lines: list[str] = []
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            if cancel is not None and cancel.is_set():
                _terminate(proc)
                raise jobs.Cancelled()

            line = line.rstrip()
            if not line:
                continue
            lines.append(line)
            if line.startswith("UEU_PROGRESS\t"):
                parts = line.split("\t", 2)
                if len(parts) == 3:
                    try:
                        progress(float(parts[1]), parts[2])
                    except ValueError:
                        pass

        ret = proc.wait()
    finally:
        if proc.poll() is None:
            _terminate(proc)

    if cancel is not None and cancel.is_set():
        raise jobs.Cancelled()

    if ret != 0:
        tail = "\n".join(lines[-12:])
        raise RuntimeError(f"NPU backend が失敗しました (exit={ret})\n{tail}")


def _terminate(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=3)


def upscale_image(
    in_path: str,
    out_path: str,
    settings: UpscaleSettings,
    progress: Optional[ProgressCb] = None,
    cancel=None,
) -> None:
    progress = progress or _noop
    _check_npu_settings(settings)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    _run_worker([
        "--mode", "image",
        "--input", in_path,
        "--output", out_path,
        "--image-format", _fmt(out.suffix.lstrip(".") or settings.image_format),
        "--model", settings.model or binaries.DEFAULT_NPU_MODEL,
    ], progress, cancel)

    if not out.exists():
        raise RuntimeError(f"NPU出力ファイルが生成されませんでした: {out}")


def upscale_folder(
    in_dir: str,
    out_dir: str,
    settings: UpscaleSettings,
    progress: Optional[ProgressCb] = None,
    cancel=None,
) -> None:
    progress = progress or _noop
    _check_npu_settings(settings)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    total = sum(
        1 for p in Path(in_dir).iterdir()
        if p.is_file() and p.suffix.lower() in media.IMAGE_EXTS
    )
    if total == 0:
        progress(1.0, "0/0 枚")
        return

    _run_worker([
        "--mode", "folder",
        "--input", in_dir,
        "--output", out_dir,
        "--image-format", _fmt(settings.image_format),
        "--model", settings.model or binaries.DEFAULT_NPU_MODEL,
    ], progress, cancel)

    done = sum(
        1 for p in out.iterdir()
        if p.is_file() and p.suffix.lower() in media.IMAGE_EXTS
    )
    progress(1.0, f"{done}/{total} 枚")

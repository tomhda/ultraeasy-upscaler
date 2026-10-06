"""試し（1コマ/範囲の事前拡大）のGUI非依存コア。

本処理と同じ `upscaler.upscale_image` 経路を使う薄い層。
フレーム補間は試しでは無視する（静止画1枚の拡大だけ行う）。
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from ..i18n import t
from . import binaries
from .jobs import ProgressCb
from .settings import UpscaleSettings

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

# 作業フォルダ名の接頭辞（仕様どおり）。
WORK_DIR_PREFIX = "ueu-trial-"


def extract_frame(video_path: str, seconds: float, out_png: str) -> None:
    """動画の指定秒の1コマをPNGに書き出す。

    回転メタデータ付き動画でも表示どおりの向きになるよう、
    ffmpeg の既定の自動回転に任せる。
    """
    out = Path(out_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    sec = max(0.0, float(seconds))
    cmd = [
        binaries.ffmpeg_exe(),
        "-hide_banner", "-loglevel", "error",
        "-ss", f"{sec:.3f}",
        "-i", video_path,
        "-frames:v", "1",
        "-y", str(out),
    ]
    try:
        done = subprocess.run(
            cmd, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=30,
            creationflags=_NO_WINDOW,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(
            t("コマを取り出せませんでした: {error}", error=exc)
        ) from exc
    if done.returncode != 0 or not out.exists():
        tail = (done.stderr or "").strip().splitlines()[-1:]
        detail = tail[0] if tail else f"exit={done.returncode}"
        raise RuntimeError(
            t("コマを取り出せませんでした: {error}", error=detail)
        )


def crop_image(src_png: str, rect: tuple[int, int, int, int], out_png: str) -> None:
    """元画像のピクセル座標 rect=(x, y, w, h) で切り出す。

    範囲は画像内に丸める。重ならない範囲は ValueError。
    """
    from PIL import Image

    out = Path(out_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    x, y, w, h = (int(v) for v in rect)
    with Image.open(src_png) as im:
        width, height = im.width, im.height
        # 範囲外の指定を画像内に丸める（負の開始やはみ出しを切り詰める）。
        x0 = min(max(0, x), width)
        y0 = min(max(0, y), height)
        x1 = min(max(0, x + w), width)
        y1 = min(max(0, y + h), height)
        if x1 <= x0 or y1 <= y0:
            raise ValueError(t("範囲が画像の外です"))
        cropped = im.crop((x0, y0, x1, y1))
        cropped.save(out, "PNG")


def run_trial(
    source_png: str,
    settings: UpscaleSettings,
    out_png: str,
    progress: ProgressCb | None = None,
    cancel=None,
) -> None:
    """`upscaler.upscale_image` を本処理と同じ設定・同じ経路で呼ぶ。

    フレーム補間は試しでは無視する（設定に残っていても拡大だけ行う）。
    動画に使えないモデルでも静止画1コマの試しとしては実行する。
    """
    from dataclasses import replace

    from . import upscaler

    # upscale_image自体は補間設定を見ないが、念のため試し設定からは外す。
    # 呼び出し元の設定は変えない（replaceで複製する）。
    trial_settings = replace(settings, interpolation_model=None, target_fps=None)
    upscaler.upscale_image(source_png, out_png, trial_settings,
                           progress=progress, cancel=cancel)


def create_work_dir() -> Path:
    """アプリ起動ごとの試し作業フォルダを1つ作る。終了時に消すこと。"""
    return Path(tempfile.mkdtemp(prefix=WORK_DIR_PREFIX))


def cleanup_work_dir(path: str | Path) -> None:
    """試し作業フォルダを消す。無ければ何もしない。"""
    shutil.rmtree(str(path), ignore_errors=True)


def _format_hms(total_seconds: int, use_hour: bool) -> str:
    """秒数を M:SS または H:MM:SS にする。"""
    total_seconds = max(0, int(total_seconds))
    if use_hour:
        hour, rest = divmod(total_seconds, 3600)
        minute, sec = divmod(rest, 60)
        return f"{hour}:{minute:02d}:{sec:02d}"
    minute, sec = divmod(total_seconds, 60)
    return f"{minute}:{sec:02d}"


def format_time_position(seconds: float, duration: float) -> str:
    """位置スライダーの時刻表示（`0:42 / 2:10`、1時間以上は `1:02:03`）。"""
    use_hour = max(float(seconds), float(duration)) >= 3600
    return f"{_format_hms(seconds, use_hour)} / {_format_hms(duration, use_hour)}"

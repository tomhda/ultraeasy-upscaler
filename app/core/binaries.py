"""外部バイナリ（realesrgan-ncnn-vulkan / ffmpeg / ffprobe）の探索とモデル列挙。"""
from __future__ import annotations

import re
import shutil
from functools import lru_cache
from pathlib import Path


class BinaryError(RuntimeError):
    """必要なバイナリ/モデルが見つからない。"""


def repo_root() -> Path:
    # app/core/binaries.py -> リポジトリルート
    return Path(__file__).resolve().parents[2]


def realesrgan_dir() -> Path:
    return repo_root() / "vendor" / "realesrgan"


def realesrgan_exe() -> str:
    exe = realesrgan_dir() / "realesrgan-ncnn-vulkan.exe"
    if not exe.exists():
        raise BinaryError(
            f"realesrgan-ncnn-vulkan.exe が見つかりません: {exe}\n"
            "vendor/realesrgan/ に展開してください。"
        )
    return str(exe)


def models_dir() -> Path:
    return realesrgan_dir() / "models"


def _which(name: str) -> str:
    found = shutil.which(name)
    if found:
        return found
    raise BinaryError(f"{name} が見つかりません（PATH を確認してください）。")


@lru_cache(maxsize=None)
def ffmpeg_exe() -> str:
    return _which("ffmpeg")


@lru_cache(maxsize=None)
def ffprobe_exe() -> str:
    return _which("ffprobe")


@lru_cache(maxsize=None)
def available_models() -> list[str]:
    """models/*.param から末尾の -x2/-x3/-x4 を除いたモデル名一覧を返す。"""
    d = models_dir()
    if not d.exists():
        return []
    names: set[str] = set()
    for p in d.glob("*.param"):
        names.add(re.sub(r"-x\d+$", "", p.stem))
    return sorted(names)


def model_supports_scale(model: str, scale: int) -> bool:
    """指定モデルが scale をサポートするか（param ファイルの有無で判定）。"""
    d = models_dir()
    return (d / f"{model}.param").exists() or (d / f"{model}-x{scale}.param").exists()

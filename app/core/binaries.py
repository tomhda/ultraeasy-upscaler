"""外部バイナリ（realesrgan-ncnn-vulkan / ffmpeg / ffprobe）の探索とモデル列挙。"""
from __future__ import annotations

import os
import re
import shutil
import sys
from functools import lru_cache
from pathlib import Path

from ..i18n import t


class BinaryError(RuntimeError):
    """必要なバイナリ/モデルが見つからない。"""


def repo_root() -> Path:
    # PyInstaller one-folder 版では vendor/ を exe と同じ階層に置く。
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    # app/core/binaries.py -> リポジトリルート
    return Path(__file__).resolve().parents[2]


def realesrgan_dir() -> Path:
    return repo_root() / "vendor" / "realesrgan"


def realesrgan_exe() -> str:
    exe = realesrgan_dir() / "realesrgan-ncnn-vulkan.exe"
    if not exe.exists():
        raise BinaryError(
            t(
                "realesrgan-ncnn-vulkan.exe が見つかりません: {exe}\n"
                "vendor/realesrgan/ に展開してください。",
                exe=exe,
            )
        )
    return str(exe)


def models_dir() -> Path:
    return realesrgan_dir() / "models"


def npu_dir() -> Path:
    return repo_root() / "vendor" / "amd-npu"


def npu_models_dir() -> Path:
    return npu_dir() / "onnx-models"


def npu_cache_dir() -> Path:
    """VitisAIキャッシュの場所を返す（TOGU_NPU_CACHEで上書き可能）。"""
    override = os.environ.get("TOGU_NPU_CACHE")
    if override:
        return Path(override).expanduser()
    return repo_root() / "vendor" / "amd-npu-1.8"


# NPU対応モデルのレジストリ。キーは Vulkan 側と同じモデル名（settings.model）。
# 値は (ONNXファイル名, sr_scale)。ONNXの stem がそのままキャッシュキーになる
# （modelcachekey_<stem>/ を隣に置くこと）。
NPU_MODELS: dict[str, tuple[str, int]] = {
    "realesrgan-x4plus": ("realesrgan_nchw_256x256_bf16cast.onnx", 4),
    "realesr-animevideov3": ("animevideov3dp_nchw_256x256_bf16cast.onnx", 4),
    "realesrgan-x4plus-anime": ("x4plus_anime_nchw_256x256_bf16cast.onnx", 4),
}
DEFAULT_NPU_MODEL = "realesrgan-x4plus"


@lru_cache(maxsize=None)
def available_npu_models() -> list[str]:
    """ONNX が実在するNPU対応モデル名の一覧。"""
    d = npu_models_dir()
    return [name for name, (fname, _s) in NPU_MODELS.items() if (d / fname).exists()]


def npu_model_spec(model: str | None) -> tuple[Path, int]:
    """モデル名から (ONNXパス, sr_scale) を返す。未対応名は BinaryError。"""
    key = model or DEFAULT_NPU_MODEL
    if key not in NPU_MODELS:
        raise BinaryError(
            t(
                "モデル '{key}' はNPUバックエンドに対応していません。"
                "GPU (Vulkan) を選ぶか、NPU対応モデルに切り替えてください。",
                key=key,
            )
        )
    fname, scale = NPU_MODELS[key]
    return npu_models_dir() / fname, scale


def _which(name: str) -> str:
    exe_name = name if name.lower().endswith(".exe") else f"{name}.exe"
    local_candidates = (
        repo_root() / "vendor" / "ffmpeg" / "bin" / exe_name,
        repo_root() / "vendor" / "ffmpeg" / exe_name,
    )
    for candidate in local_candidates:
        if candidate.exists():
            return str(candidate)
    found = shutil.which(name)
    if found:
        return found
    raise BinaryError(t("{name} が見つかりません（PATH を確認してください）。", name=name))


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


def rife_exe() -> str:
    """同梱された rife-ncnn-vulkan.exe を返す。"""
    base = repo_root() / "vendor" / "rife"
    candidates = (
        base / "rife-ncnn-vulkan.exe",
        base / "rife-ncnn-vulkan-20221029-windows" / "rife-ncnn-vulkan.exe",
    )
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    raise BinaryError(
        t(
            "rife-ncnn-vulkan.exe が見つかりません: {base}\n"
            "モデル取得スクリプトを実行してください。",
            base=base,
        )
    )


def rife_models_dir() -> Path:
    return Path(rife_exe()).parent


FILM_MODEL = "film-style"


def film_helper_exe() -> str:
    """FILM/Windows ML 補間ヘルパーを探す（起動はしない）。"""
    override = os.environ.get("TOGU_FILM_HELPER")
    if override:
        candidate = Path(override).expanduser()
        if candidate.is_file():
            return str(candidate)
        raise BinaryError(
            t("FILM (Style) の実行ファイルが見つかりません: {path}", path=candidate)
        )
    root = repo_root()
    candidates = [root / "vendor" / "winml-film" / "winml-film.exe"]
    candidates.extend((root / "tools" / "winml-film" / "bin").glob(
        "Release/net*/win-x64/winml-film.exe"
    ))
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    raise BinaryError(
        t("FILM (Style) の実行ファイルが見つかりません。FILM キットを追加してください。")
    )


def film_model_path() -> Path:
    """配布版 ONNX または明示指定された FILM モデルを返す（起動はしない）。"""
    override = os.environ.get("TOGU_FILM_MODEL")
    path = (Path(override).expanduser() if override else
            repo_root() / "models" / "film" / "film_style_fp32.onnx")
    if not path.is_file():
        raise BinaryError(
            t("FILM (Style) のモデルが見つかりません: {path}", path=path)
        )
    return path


@lru_cache(maxsize=None)
def available_interpolation_models() -> list[str]:
    """利用可能な補間モデルを列挙する。"""
    models: list[str] = []
    try:
        base = rife_models_dir()
    except BinaryError:
        pass
    else:
        if (base / "rife-v4.6").is_dir():
            models.append("rife-v4.6")
    try:
        film_helper_exe()
        film_model_path()
    except BinaryError:
        pass
    else:
        models.append(FILM_MODEL)
    return models


def interpolation_model_dir(model: str) -> Path:
    path = rife_models_dir() / model
    if not path.is_dir():
        raise BinaryError(t("フレーム補間モデルが見つかりません: {path}", path=path))
    return path


def model_supports_scale(model: str, scale: int) -> bool:
    """指定モデルが scale をサポートするか（param ファイルの有無で判定）。"""
    d = models_dir()
    return (d / f"{model}.param").exists() or (d / f"{model}-x{scale}.param").exists()

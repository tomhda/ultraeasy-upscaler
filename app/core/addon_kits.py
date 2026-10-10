"""追加キット（別配布 zip）の導入状態とダウンロード先。GUI 非依存。

アプリ自身はダウンロードも展開もしない。判定はファイルの有無だけで行い、
実行ファイルの起動やモデルの読み込みはしない。
"""
from __future__ import annotations

from dataclasses import dataclass

from ..i18n import N_
from . import binaries, helper_backend
from .settings import (
    HELPER_MODEL_ADCSR,
    HELPER_MODEL_FILES,
    UpscaleBackend,
)

# GitHub の最新リリースからキット zip を取る。ファイル名は 1 か所で組み立てる。
KIT_RELEASE_BASE = "https://github.com/tomhda/togu-scaler/releases/latest/download"


@dataclass(frozen=True)
class AddonKit:
    """詳細設定の「追加キット」に出す 1 行分。name_key は表示名の t() キー。"""

    key: str
    name_key: str
    size: str  # 訳さない（"178 MB" など）
    filename: str  # ダウンロードする zip 名
    installed: bool


def kit_download_url(filename: str) -> str:
    """キット zip のダウンロード先 URL を組み立てる。"""
    return f"{KIT_RELEASE_BASE}/{filename}"


def film_kit_installed() -> bool:
    """FILM のヘルパーとモデルが両方あるか。"""
    try:
        binaries.film_helper_exe()
        binaries.film_model_path()
    except binaries.BinaryError:
        return False
    return True


def npu_kit_installed() -> bool:
    """NPU 用の資材があるか（自己診断がキットありと見なす条件と同じ）。"""
    return (binaries.repo_root() / "tools" / "npu-serve" / "npu_serve.py").is_file()


def adcsr_gpu_installed() -> bool:
    """GPU 用の AdcSR モデルが見つかるか。"""
    filename = HELPER_MODEL_FILES[UpscaleBackend.WINML_GPU][HELPER_MODEL_ADCSR][128]
    return helper_backend._search_model_file(filename) is not None


def adcsr_npu_installed() -> bool:
    """NPU 用の AdcSR モデル（前半・後半・マニフェスト）が見つかるか。"""
    return helper_backend.adcsr_two_stage_files() is not None


def addon_kits() -> list[AddonKit]:
    """「追加キット」の 4 行を返す（表示順は固定）。"""
    return [
        AddonKit(
            key="film",
            name_key=N_("FILM (Style)"),
            size="178 MB",
            filename="togu-scaler-film-kit.zip",
            installed=film_kit_installed(),
        ),
        AddonKit(
            key="npu",
            name_key=N_("NPU キット"),
            size="56 MB",
            filename="togu-scaler-npu-kit.zip",
            installed=npu_kit_installed(),
        ),
        AddonKit(
            key="adcsr_gpu",
            name_key=N_("AdcSR（GPU 用）"),
            size="1.7 GB",
            filename="togu-scaler-adcsr-kit.zip",
            installed=adcsr_gpu_installed(),
        ),
        AddonKit(
            key="adcsr_npu",
            name_key=N_("AdcSR（NPU 用）"),
            size="1.7 GB",
            filename="togu-scaler-npu-kit-adcsr.zip",
            installed=adcsr_npu_installed(),
        ),
    ]

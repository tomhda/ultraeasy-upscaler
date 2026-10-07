"""モデル・実行先の表示表（GUI と CLI の共通出どころ。Qt 非依存）。

`app/gui/main_window.py` と `app/gui/settings_drawer.py` にあった表のうち、
CLI（`app/cli.py`）でも要るものをここへ移した。GUI はこの表を参照し、
中身を二重に持たない。表示には `t()` を通す（CLI 自身は言語設定に従う）。
"""
from __future__ import annotations

from ..i18n import N_
from .binaries import FILM_MODEL
from .settings import (
    HELPER_MODEL_ADCSR,
    HELPER_MODEL_AMD_RRDB,
    HELPER_MODEL_ANIME,
    HELPER_MODEL_SPAN,
    HELPER_MODEL_SWINIR,
    UpscaleBackend,
)

MODEL_LABELS = {
    "realesrgan-x4plus": N_("Real-ESRGAN"),
    "realesrgan-x4plus-anime": N_("Real-ESRGAN Anime"),
    "realesr-animevideov3": N_("Anime Video v3"),
    HELPER_MODEL_ANIME: N_("Anime Video v3"),
    "realesr-general-x4v3": N_("General Video v3（ノイズ除去強）"),
    "realesr-general-wdn-x4v3": N_("General Video v3（ノイズ除去弱）"),
    HELPER_MODEL_SPAN: N_("4xNomosUni SPAN"),
    HELPER_MODEL_AMD_RRDB: N_("Real-ESRGAN（AMD縮小版）"),
    HELPER_MODEL_SWINIR: N_("SwinIR-M"),
    HELPER_MODEL_ADCSR: N_("AdcSR"),
}
HELPER_MODEL_OPTIONS = [
    (N_("なし（拡大しない）"), None),
    (MODEL_LABELS[HELPER_MODEL_ANIME], HELPER_MODEL_ANIME),
    (MODEL_LABELS[HELPER_MODEL_SPAN], HELPER_MODEL_SPAN),
    (MODEL_LABELS[HELPER_MODEL_AMD_RRDB], HELPER_MODEL_AMD_RRDB),
    (MODEL_LABELS[HELPER_MODEL_SWINIR], HELPER_MODEL_SWINIR),
    (MODEL_LABELS[HELPER_MODEL_ADCSR], HELPER_MODEL_ADCSR),
]
HELPER_MODEL_VALUES = {
    value for _label, value in HELPER_MODEL_OPTIONS if value is not None
}
SWINIR_CUDA_MODEL_OPTIONS = [
    (N_("なし（拡大しない）"), None),
    (N_("SwinIR-M（real-world x4）"), HELPER_MODEL_SWINIR),
]

# モデルの説明行（1 行）。速さ・実行先の注意は付けない。
MODEL_HINT = {
    HELPER_MODEL_ANIME: N_("アニメ向け・速い"),
    "realesr-animevideov3": N_("アニメ向け・速い"),
    HELPER_MODEL_SPAN: N_("実写向け・速い"),
    HELPER_MODEL_AMD_RRDB: N_("実写向け・くっきり・やや遅い"),
    HELPER_MODEL_SWINIR: N_("実写の静止画向け・高精細・遅い"),
    HELPER_MODEL_ADCSR: N_("実写の静止画向け・最高画質・とても遅い"),
    "realesrgan-x4plus": N_("実写向け・高画質・遅い"),
    "realesrgan-x4plus-anime": N_("アニメ向け・高画質・遅い"),
    "realesr-general-x4v3": N_("実写・アニメ兼用・ノイズ除去強め"),
    "realesr-general-wdn-x4v3": N_("実写向け・ノイズ除去弱め"),
}

# (backend, model) → (速度, 画質, アニメ適性, 実写適性, 推奨タグ or None)
# 速度・画質・適性の印。英語の表示では言葉に置き換える（◎○△✕ は日本語圏の記号のため）。
MODEL_INFO: dict[tuple[UpscaleBackend, str],
                 tuple[str, str, str, str, str | None]] = {
    (UpscaleBackend.VULKAN, "realesr-animevideov3"):
        ("◎", "◎", "◎", "△", N_("アニメ")),
    (UpscaleBackend.VULKAN, "realesr-general-x4v3"):
        ("◎", "○", "○", "○", None),
    (UpscaleBackend.VULKAN, "realesr-general-wdn-x4v3"):
        ("◎", "○", "○", "◎", N_("実写")),
    (UpscaleBackend.VULKAN, "realesrgan-x4plus"):
        ("✕", "◎", "○", "◎", None),
    (UpscaleBackend.VULKAN, "realesrgan-x4plus-anime"):
        ("✕", "◎", "◎", "○", None),
    (UpscaleBackend.NPU, "realesrgan-x4plus"):
        ("◎", "◎", "○", "◎", N_("実写")),
    (UpscaleBackend.NPU, "realesrgan-x4plus-anime"):
        ("○", "◎", "◎", "○", None),
    (UpscaleBackend.NPU, "realesr-animevideov3"):
        ("◎", "◎", "◎", "△", N_("アニメ")),
    (UpscaleBackend.WINML_GPU, HELPER_MODEL_ANIME):
        ("◎", "◎", "◎", "△", N_("アニメ")),
    (UpscaleBackend.WINML_GPU, HELPER_MODEL_SPAN):
        ("◎", "◎", "○", "◎", N_("実写")),
    (UpscaleBackend.WINML_GPU, HELPER_MODEL_AMD_RRDB):
        ("△", "◎", "○", "◎", None),
    (UpscaleBackend.WINML_GPU, HELPER_MODEL_SWINIR):
        ("✕", "◎", "○", "◎", N_("静止画")),
    (UpscaleBackend.WINML_GPU, HELPER_MODEL_ADCSR):
        ("✕", "◎◎", "○", "◎", N_("実写")),
    (UpscaleBackend.NPU_NATIVE, HELPER_MODEL_ADCSR):
        ("✕", "◎◎", "○", "◎", N_("実写")),
    (UpscaleBackend.NPU_NATIVE, HELPER_MODEL_ANIME):
        ("○", "◎", "◎", "△", N_("アニメ")),
    (UpscaleBackend.NPU_NATIVE, HELPER_MODEL_SPAN):
        ("◎", "◎", "○", "◎", N_("実写")),
    (UpscaleBackend.NPU_NATIVE, HELPER_MODEL_AMD_RRDB):
        ("△", "◎", "○", "◎", None),
    (UpscaleBackend.NPU_NATIVE, HELPER_MODEL_SWINIR):
        ("✕", "◎", "○", "◎", N_("静止画")),
    (UpscaleBackend.SWINIR_CUDA, HELPER_MODEL_SWINIR):
        (N_("極遅"), "◎◎", "△", "◎", N_("実写・再開可")),
}

BACKEND_OPTIONS = [
    (N_("自動（GPU優先）"), "auto"),
    (N_("GPU（DirectML）"), UpscaleBackend.WINML_GPU.value),
    (N_("NPU（GPU温存）"), UpscaleBackend.NPU_NATIVE.value),
    (N_("SwinIR-M（CUDA・超低速）"), UpscaleBackend.SWINIR_CUDA.value),
    (N_("Vulkan"), UpscaleBackend.VULKAN.value),
]

# フレーム補間モデルの表示名と説明行。GUI の補間コンボと CLI の `models` が
# 同じ i18n キーを使う（使える条件は `binaries` の実在判定がただ一つの出どころ）。
INTERPOLATION_RIFE = "rife-v4.6"
INTERPOLATION_LABELS = {
    INTERPOLATION_RIFE: N_("RIFE v4.6"),
    FILM_MODEL: N_("FILM (Style)"),
}
INTERPOLATION_HINT = {
    INTERPOLATION_RIFE: N_("高速"),
    FILM_MODEL: N_("低速・高品質"),
}

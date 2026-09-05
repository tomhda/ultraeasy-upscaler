"""アップスケール設定モデル（GUI / エンジン共通）。"""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path


class OutputLocation(str, Enum):
    SAME = "same"      # 入力と同じ場所に出力
    CUSTOM = "custom"  # 指定フォルダに出力


class UpscaleBackend(str, Enum):
    # GUIの「自動（GPU優先）」は、設定へはWINML_GPUとして正規化する。
    WINML_GPU = "winml_gpu"    # UEU helper / DirectML GPU
    NPU_NATIVE = "npu_native"  # UEU helper / Ryzen AI VitisAI EP
    VULKAN = "vulkan"          # realesrgan-ncnn-vulkan fallback
    # 旧GUI/API互換用。新しいGUIのNPU選択はNPU_NATIVEへ寄せる。
    NPU = "npu"                # deprecated: legacy npu_worker route


class ModelFamily(str, Enum):
    """旧UIが保存していたモデル系統（読み込み互換用）。"""

    ANIME = "anime"
    PHOTO = "photo"
    REALESRGAN = "realesrgan"


class ProcessingOrder(str, Enum):
    """動画でアップスケールと補間を両方行う場合の実行順。"""

    UPSCALE_FIRST = "upscale_first"          # アプコン → 補間（速い・既定）
    INTERPOLATE_FIRST = "interpolate_first"  # 補間 → アプコン（省メモリ）


# vendor 同梱の既定モデル
DEFAULT_MODEL = "realesrgan-x4plus"
DEFAULT_INTERPOLATION_MODEL = "rife-v4.6"
DEFAULT_MODEL_FAMILY = ModelFamily.ANIME

# 新AIヘルパーのモデルキー。表示名ではなくこのキーを設定に保存し、
# GPU版/NPU版で実体ファイルだけを差し替える。
HELPER_MODEL_ANIME = "animevideov3"
HELPER_MODEL_SPAN = "4xNomosUni"
HELPER_MODEL_AMD_RRDB = "AMD-RRDB"
HELPER_MODEL_SWINIR = "SwinIR"
HELPER_MODEL_ADCSR = "AdcSR"
DEFAULT_HELPER_MODEL = HELPER_MODEL_ANIME

VULKAN_FALLBACK_MODELS = {
    HELPER_MODEL_ANIME: "realesr-animevideov3",
    HELPER_MODEL_SPAN: "realesrgan-x4plus",
    HELPER_MODEL_AMD_RRDB: "realesrgan-x4plus",
    HELPER_MODEL_SWINIR: "realesrgan-x4plus",
    HELPER_MODEL_ADCSR: "realesrgan-x4plus",
}

# 統合前後で保存された値を壊さないための読み込み互換表。
# 抽象3系統はUIには出さず、ここで具体的なモデルキーへ正規化する。
HELPER_MODEL_ALIASES = {
    HELPER_MODEL_ANIME: HELPER_MODEL_ANIME,
    "realesr-animevideov3": HELPER_MODEL_ANIME,
    "animevideov3dp": HELPER_MODEL_ANIME,
    ModelFamily.ANIME.value: HELPER_MODEL_ANIME,
    HELPER_MODEL_SPAN: HELPER_MODEL_SPAN,
    "4xNomosUni_span_multijpg": HELPER_MODEL_SPAN,
    "purephoto": HELPER_MODEL_SPAN,
    ModelFamily.PHOTO.value: HELPER_MODEL_SPAN,
    HELPER_MODEL_SWINIR: HELPER_MODEL_SWINIR,
    "swinir": HELPER_MODEL_SWINIR,
    "swinir-m": HELPER_MODEL_SWINIR,
    "adcsr": HELPER_MODEL_ADCSR,
    HELPER_MODEL_AMD_RRDB: HELPER_MODEL_AMD_RRDB,
    "amd-rrdb": HELPER_MODEL_AMD_RRDB,
    "realesrgan-amd": HELPER_MODEL_AMD_RRDB,
    "realesrgan-amd-rrdb": HELPER_MODEL_AMD_RRDB,
    ModelFamily.REALESRGAN.value: HELPER_MODEL_AMD_RRDB,
    # 現行統合版がhelper設定へ誤って保存していた既存既定値。
    "realesrgan-x4plus": HELPER_MODEL_AMD_RRDB,
}


def canonical_helper_model(model: str | ModelFamily | None) -> str | None:
    """helperの旧キーを具体的なモデルキーへ正規化する。"""
    if model is None:
        return None
    value = model.value if isinstance(model, ModelFamily) else str(model)
    return HELPER_MODEL_ALIASES.get(value, value)


def helper_model_family(model: str | ModelFamily | None) -> ModelFamily:
    """具体的なhelperモデルに対応する旧系統値を返す。"""
    key = canonical_helper_model(model)
    if key in (HELPER_MODEL_SPAN, HELPER_MODEL_SWINIR, HELPER_MODEL_ADCSR):
        return ModelFamily.PHOTO
    if key == HELPER_MODEL_AMD_RRDB:
        return ModelFamily.REALESRGAN
    return ModelFamily.ANIME


def vulkan_fallback_settings(settings: "UpscaleSettings") -> "UpscaleSettings":
    """新AIヘルパー設定をVulkanで実行できる設定へ変換する。"""
    model = VULKAN_FALLBACK_MODELS.get(canonical_helper_model(settings.model), settings.model)
    return replace(settings, backend=UpscaleBackend.VULKAN, model=model)


# 新AIヘルパー用ONNXモデル。キーは具体的なモデルキー、値はタイルの一辺。
# NPU_NATIVEはアニメ/質感系を512、AMD縮小RRDBを256で実行する。
HELPER_MODEL_FILES = {
    UpscaleBackend.WINML_GPU: {
        HELPER_MODEL_ANIME: {
            256: "animevideov3_nchw_256x256_fp32.onnx",
            512: "animevideov3_nchw_512x512_fp32.onnx",
        },
        HELPER_MODEL_SPAN: {
            256: "purephoto_nchw_256x256_fp32.onnx",
            512: "purephoto_nchw_512x512_fp32.onnx",
        },
        HELPER_MODEL_AMD_RRDB: {
            256: "realesrgan_nchw_256x256_fp32.onnx",
        },
        HELPER_MODEL_SWINIR: {
            256: "swinir_nchw_256x256_fp32.onnx",
        },
        HELPER_MODEL_ADCSR: {
            128: "adcsr_nchw_128x128_fp32.onnx",
        },
    },
    UpscaleBackend.NPU_NATIVE: {
        HELPER_MODEL_ANIME: {
            512: "animevideov3dp_nchw_512x512_bf16cast.onnx",
        },
        HELPER_MODEL_SPAN: {
            512: "purephoto_nchw_512x512_bf16cast.onnx",
        },
        HELPER_MODEL_AMD_RRDB: {
            256: "realesrgan_nchw_256x256_bf16cast.onnx",
        },
        # SwinIRのNPU版はVAIML crash回避のroll書換済みグラフ（研究ノート参照）
        HELPER_MODEL_SWINIR: {
            256: "swinir_nchw_256x256_bf16cast.onnx",
        },
    },
}


_REPO_ROOT = Path(__file__).resolve().parents[2]
# マシン固有の絶対パスは使わず、リポジトリ相対の既定値にする。
# 実運用ではUEU_MODELS_DIR / UEU_NPU_CACHEで上書きできる。
DEFAULT_MODELS_DIR = _REPO_ROOT / "tmp" / "npu-anime"
DEFAULT_VENDOR_MODELS_DIR = _REPO_ROOT / "vendor" / "amd-npu" / "onnx-models"
DEFAULT_NPU_CACHE_DIR = _REPO_ROOT / "vendor" / "amd-npu-1.8"


@dataclass
class UpscaleSettings:
    """1ジョブ分のアップスケール設定。"""

    # --- 基本 ---
    backend: UpscaleBackend = UpscaleBackend.WINML_GPU # 自動（GPU優先）を正規化した値
    scale: int = 4                                   # 倍率: 2 / 3 / 4（新ヘルパーは4固定）
    # None = アップスケールしない（動画で補間だけ行う場合に使う）
    model: str | None = DEFAULT_MODEL                # Vulkan/helper共通のモデルキー
    model_family: ModelFamily = DEFAULT_MODEL_FAMILY # 旧helper設定の読み込み互換値
    image_format: str = "png"                        # 画像出力形式: png / jpg / webp

    # --- 出力先 ---
    output_location: OutputLocation = OutputLocation.SAME
    output_dir: str | None = None                    # CUSTOM のときの出力フォルダ
    create_subfolder: bool = True                    # 出力をサブフォルダにまとめる
    subfolder_name: str = "upscaled"
    overwrite: bool = False                          # 既存ファイルを上書き

    # --- realesrgan 詳細（詳細設定ドロワー） ---
    tile_size: int = 0                               # 0 = 自動（VRAM 不足時に下げる）
    gpu_id: int = -1                                 # -1 = 既定GPU（-g を渡さない）
    tta_mode: bool = False                           # TTA（高品質・低速）
    threads: str = "1:2:2"                           # -j load:proc:save

    # --- 動画 ---
    keep_audio: bool = True                          # 元音声を維持
    hw_encode: bool = True                           # 利用可能なら HW エンコード
    video_format: str = "mp4"                        # 出力コンテナ
    video_quality: int = 18                          # 画質(CRF/QP 相当, 小さいほど高画質)

    # --- フレーム補間 ---
    # None = 補間しない。target_fps=None のままモデルを選ぶと元fpsの2倍。
    interpolation_model: str | None = None
    target_fps: float | None = None
    # 両方有効時の実行順。重いESRGANを補間前の元フレーム数に抑えられる
    # UPSCALE_FIRST が既定。高解像度出力でメモリが厳しい場合は
    # INTERPOLATE_FIRST を選ぶ。
    processing_order: ProcessingOrder = ProcessingOrder.UPSCALE_FIRST

    # 動画エンコード時の最大寸法。None のときは UEU_MAX_VIDEO_DIM、未設定なら
    # video.py の既定値 3840x2160 を使う。値は (幅, 高さ) のタプル。
    # 既存の位置引数互換を保つため、動画・補間設定の末尾に置く。
    max_video_dim: tuple[int, int] | None = None

    def output_suffix(self) -> str:
        """選択した処理を表す出力接尾辞を返す。"""
        parts: list[str] = []
        if self.model is not None:
            parts.append(f"x{self.scale}")
        if self.interpolation_model is not None:
            model = self.interpolation_model.replace("rife-", "RIFE-")
            fps = f"{self.target_fps:g}fps" if self.target_fps else "2xfps"
            parts.extend((model, fps))
        return "_" + "_".join(parts) if parts else "_processed"

    @property
    def upscale_enabled(self) -> bool:
        return self.model is not None

    @property
    def interpolation_enabled(self) -> bool:
        return self.interpolation_model is not None

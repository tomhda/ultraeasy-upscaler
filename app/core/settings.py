"""アップスケール設定モデル（GUI / エンジン共通）。"""
from __future__ import annotations

from dataclasses import dataclass
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
    """新AIヘルパーが提供する3系統のモデル。"""

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

# 新AIヘルパー用ONNXモデル。キーはタイルの一辺。
# NPU_NATIVEはアニメ/質感系を512、Real-ESRGANを256で実行する。
HELPER_MODEL_FILES = {
    UpscaleBackend.WINML_GPU: {
        ModelFamily.ANIME: {
            256: "animevideov3_nchw_256x256_fp32.onnx",
            512: "animevideov3_nchw_512x512_fp32.onnx",
        },
        ModelFamily.PHOTO: {
            256: "purephoto_nchw_256x256_fp32.onnx",
            512: "purephoto_nchw_512x512_fp32.onnx",
        },
        ModelFamily.REALESRGAN: {
            256: "realesrgan_nchw_256x256_fp32.onnx",
        },
    },
    UpscaleBackend.NPU_NATIVE: {
        ModelFamily.ANIME: {
            512: "animevideov3dp_nchw_512x512_bf16cast.onnx",
        },
        ModelFamily.PHOTO: {
            512: "purephoto_nchw_512x512_bf16cast.onnx",
        },
        ModelFamily.REALESRGAN: {
            256: "realesrgan_nchw_256x256_bf16cast.onnx",
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
    model: str | None = DEFAULT_MODEL                # realesrgan モデル名（-n）
    model_family: ModelFamily = DEFAULT_MODEL_FAMILY # 新AIヘルパーのモデル系統
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

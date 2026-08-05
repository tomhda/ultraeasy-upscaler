"""アップスケール設定モデル（GUI / エンジン共通）。"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class OutputLocation(str, Enum):
    SAME = "same"      # 入力と同じ場所に出力
    CUSTOM = "custom"  # 指定フォルダに出力


class UpscaleBackend(str, Enum):
    VULKAN = "vulkan"  # realesrgan-ncnn-vulkan
    NPU = "npu"        # Ryzen AI NPU (VitisAI EP)


class ProcessingOrder(str, Enum):
    """動画でアップスケールと補間を両方行う場合の実行順。"""

    UPSCALE_FIRST = "upscale_first"          # アプコン → 補間（速い・既定）
    INTERPOLATE_FIRST = "interpolate_first"  # 補間 → アプコン（省メモリ）


# vendor 同梱の既定モデル
DEFAULT_MODEL = "realesrgan-x4plus"
DEFAULT_INTERPOLATION_MODEL = "rife-v4.6"


@dataclass
class UpscaleSettings:
    """1ジョブ分のアップスケール設定。"""

    # --- 基本 ---
    backend: UpscaleBackend = UpscaleBackend.VULKAN     # 実行経路
    scale: int = 4                                   # 倍率: 2 / 3 / 4
    # None = アップスケールしない（動画で補間だけ行う場合に使う）
    model: str | None = DEFAULT_MODEL                # realesrgan モデル名（-n）
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

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


# vendor 同梱の既定モデル
DEFAULT_MODEL = "realesrgan-x4plus"


@dataclass
class UpscaleSettings:
    """1ジョブ分のアップスケール設定。"""

    # --- 基本 ---
    backend: UpscaleBackend = UpscaleBackend.VULKAN     # 実行経路
    scale: int = 4                                   # 倍率: 2 / 3 / 4
    model: str = DEFAULT_MODEL                       # realesrgan モデル名（-n）
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

    # --- フレーム補間（フェーズ2の布石・現状未使用） ---
    interpolate: bool = False
    target_fps: float | None = None

    def output_suffix(self) -> str:
        """出力ファイル名に付ける接尾辞（例 "_x4"）。"""
        return f"_x{self.scale}"

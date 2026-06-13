"""詳細設定ドロワー（折りたたみ）。

v1 モックアップ相当の詳細オプションを公開する:
  HWエンコード / 音声維持 / 画質 / 出力形式 / タイルサイズ / GPU /
  TTA / サブフォルダ(有無・名前) / 動画コンテナ。
基本設定（倍率・モデル・出力先・動画結合）は MainWindow のコントロール行が持つ。
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from app.core.settings import UpscaleSettings

_IMAGE_FORMATS = ["png", "jpg", "webp"]
# mp4/mkv/mov は H.264(+AAC) を収容できる。webm は VP9/Opus が必要で
# 現状のエンコーダ選択(H.264系)とは噛み合わないため除外（フェーズ2で対応検討）。
_VIDEO_FORMATS = ["mp4", "mkv", "mov"]


class SettingsDrawer(QFrame):
    """折りたたみ可能な詳細設定パネル。"""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("card")
        self._build()

    def _label(self, text: str) -> QLabel:
        lab = QLabel(text)
        lab.setObjectName("fieldLabel")
        return lab

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 14, 16, 14)
        root.setSpacing(10)

        title = QLabel("詳細設定")
        title.setObjectName("sectionTitle")
        root.addWidget(title)

        grid = QGridLayout()
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(10)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)

        # --- 出力形式（画像） ---
        self.image_format = QComboBox()
        self.image_format.addItems(_IMAGE_FORMATS)
        grid.addWidget(self._label("出力形式（画像）"), 0, 0)
        grid.addWidget(self.image_format, 0, 1)

        # --- 動画コンテナ ---
        self.video_format = QComboBox()
        self.video_format.addItems(_VIDEO_FORMATS)
        grid.addWidget(self._label("動画コンテナ"), 0, 2)
        grid.addWidget(self.video_format, 0, 3)

        # --- 画質（CRF/QP, 小さいほど高画質） ---
        self.video_quality = QSpinBox()
        self.video_quality.setRange(0, 51)
        self.video_quality.setToolTip("小さいほど高画質・大きいほど低容量（CRF/QP 相当）")
        grid.addWidget(self._label("画質 (CRF/QP)"), 1, 0)
        grid.addWidget(self.video_quality, 1, 1)

        # --- タイルサイズ（0=自動） ---
        self.tile_size = QSpinBox()
        self.tile_size.setRange(0, 1024)
        self.tile_size.setSingleStep(32)
        self.tile_size.setSpecialValueText("自動")  # 0 のとき
        self.tile_size.setToolTip("0=自動。VRAM 不足時は値を下げる")
        grid.addWidget(self._label("タイルサイズ"), 1, 2)
        grid.addWidget(self.tile_size, 1, 3)

        # --- GPU ID（-1=既定） ---
        self.gpu_id = QSpinBox()
        self.gpu_id.setRange(-1, 7)
        self.gpu_id.setSpecialValueText("既定")  # -1 のとき
        self.gpu_id.setToolTip("-1=既定GPU。マルチGPU時に番号指定")
        grid.addWidget(self._label("GPU"), 2, 0)
        grid.addWidget(self.gpu_id, 2, 1)

        # --- サブフォルダ名 ---
        self.subfolder_name = QLineEdit()
        self.subfolder_name.setPlaceholderText("upscaled")
        grid.addWidget(self._label("サブフォルダ名"), 2, 2)
        grid.addWidget(self.subfolder_name, 2, 3)

        root.addLayout(grid)

        # --- トグル群（HWエンコード / 音声維持 / TTA / サブフォルダ作成） ---
        toggles = QGridLayout()
        toggles.setHorizontalSpacing(24)
        toggles.setVerticalSpacing(8)
        self.hw_encode = QCheckBox("HWエンコード（GPU）")
        self.hw_encode.setToolTip("対応時にハードウェアエンコードを使用（高速）")
        self.keep_audio = QCheckBox("音声を維持")
        self.tta_mode = QCheckBox("TTA（高品質・低速）")
        self.create_subfolder = QCheckBox("出力をサブフォルダにまとめる")
        toggles.addWidget(self.hw_encode, 0, 0)
        toggles.addWidget(self.keep_audio, 0, 1)
        toggles.addWidget(self.tta_mode, 1, 0)
        toggles.addWidget(self.create_subfolder, 1, 1)
        root.addLayout(toggles)

        # サブフォルダ名はチェック時のみ有効
        self.create_subfolder.toggled.connect(self.subfolder_name.setEnabled)

        self.load_defaults()

    # --- 既定値の読込／設定への反映 ---
    def load_defaults(self, settings: UpscaleSettings | None = None) -> None:
        s = settings or UpscaleSettings()
        self.image_format.setCurrentText(s.image_format)
        self.video_format.setCurrentText(s.video_format)
        self.video_quality.setValue(s.video_quality)
        self.tile_size.setValue(s.tile_size)
        self.gpu_id.setValue(s.gpu_id)
        self.subfolder_name.setText(s.subfolder_name)
        self.hw_encode.setChecked(s.hw_encode)
        self.keep_audio.setChecked(s.keep_audio)
        self.tta_mode.setChecked(s.tta_mode)
        self.create_subfolder.setChecked(s.create_subfolder)
        self.subfolder_name.setEnabled(s.create_subfolder)

    def apply_to(self, s: UpscaleSettings) -> None:
        """ドロワーの値を UpscaleSettings に書き込む。"""
        s.image_format = self.image_format.currentText()
        s.video_format = self.video_format.currentText()
        s.video_quality = self.video_quality.value()
        s.tile_size = self.tile_size.value()
        s.gpu_id = self.gpu_id.value()
        name = self.subfolder_name.text().strip() or "upscaled"
        s.subfolder_name = name
        s.hw_encode = self.hw_encode.isChecked()
        s.keep_audio = self.keep_audio.isChecked()
        s.tta_mode = self.tta_mode.isChecked()
        s.create_subfolder = self.create_subfolder.isChecked()

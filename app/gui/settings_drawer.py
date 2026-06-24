"""詳細設定ドロワー（折りたたみ）。"""
from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QPaintEvent, QPainter, QPen
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from app.core.settings import UpscaleSettings

_IMAGE_FORMATS = ["png", "jpg", "webp"]
# mp4/mkv/mov は H.264(+AAC) を収容できる。webm は VP9/Opus が必要で
# 現状のエンコーダ選択(H.264系)とは噛み合わないため除外（フェーズ2で対応検討）。
_VIDEO_FORMATS = ["mp4", "mkv", "mov"]
_VIDEO_QUALITY_OPTIONS = [
    ("高画質（容量大）", 18),
    ("標準", 23),
    ("軽量（容量小）", 28),
]
_TILE_OPTIONS = [
    ("自動", 0),
    ("メモリ節約", 128),
    ("強めに節約", 64),
]
_GPU_OPTIONS = [
    ("自動", -1),
    ("GPU 0", 0),
    ("GPU 1", 1),
    ("GPU 2", 2),
    ("GPU 3", 3),
]


class ClearCheckBox(QCheckBox):
    """詳細設定用の見やすいチェックボックス。"""

    def __init__(self, label: str, parent=None) -> None:
        super().__init__(parent)
        self.setText(label)
        self.setObjectName("clearCheck")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(34)

    def sizeHint(self) -> QSize:  # noqa: N802
        hint = super().sizeHint()
        return QSize(max(hint.width(), 220), max(hint.height(), 34))

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        checked = self.isChecked()
        enabled = self.isEnabled()
        hovered = self.underMouse() and enabled

        rect = self.rect()
        box_size = 22
        box = QRectF(2, (rect.height() - box_size) / 2, box_size, box_size)

        border = QColor("#21c7d9" if checked else "#4a5664")
        fill = QColor("#21c7d9" if checked else ("#20262e" if hovered else "#151a20"))
        if not enabled:
            border = QColor("#2a323b")
            fill = QColor("#171d24")

        painter.setPen(QPen(border, 2))
        painter.setBrush(fill)
        painter.drawRoundedRect(box, 5, 5)

        if checked:
            pen = QPen(QColor("#041015"), 3)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)
            painter.drawLine(
                QPointF(box.left() + 5.5, box.top() + 11.5),
                QPointF(box.left() + 9.5, box.top() + 15.5),
            )
            painter.drawLine(
                QPointF(box.left() + 9.5, box.top() + 15.5),
                QPointF(box.left() + 16.5, box.top() + 7.0),
            )

        painter.setPen(QColor("#f3f5f7" if enabled else "#67717e"))
        text_rect = QRectF(34, 0, max(0, rect.width() - 34), rect.height())
        painter.drawText(
            text_rect,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            self.text(),
        )


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

    def _combo_with_data(self, options: list[tuple[str, int]]) -> QComboBox:
        combo = QComboBox()
        for label, value in options:
            combo.addItem(label, value)
        return combo

    def _set_combo_value(self, combo: QComboBox, value: int) -> None:
        idx = combo.findData(value)
        combo.setCurrentIndex(idx if idx >= 0 else 0)

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 14, 16, 14)
        root.setSpacing(12)

        title = QLabel("詳細設定")
        title.setObjectName("sectionTitle")
        root.addWidget(title)

        grid = QGridLayout()
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(10)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)

        # --- 画像の保存形式 ---
        self.image_format = QComboBox()
        self.image_format.addItems(_IMAGE_FORMATS)
        grid.addWidget(self._label("画像の保存形式"), 0, 0)
        grid.addWidget(self.image_format, 0, 1)

        # --- 動画の保存形式 ---
        self.video_format = QComboBox()
        self.video_format.addItems(_VIDEO_FORMATS)
        grid.addWidget(self._label("動画の保存形式"), 0, 2)
        grid.addWidget(self.video_format, 0, 3)

        # --- 動画の画質 ---
        self.video_quality = self._combo_with_data(_VIDEO_QUALITY_OPTIONS)
        grid.addWidget(self._label("動画の画質"), 1, 0)
        grid.addWidget(self.video_quality, 1, 1)

        # --- 分割処理 ---
        self.tile_size = self._combo_with_data(_TILE_OPTIONS)
        grid.addWidget(self._label("分割処理"), 1, 2)
        grid.addWidget(self.tile_size, 1, 3)

        # --- 使うGPU ---
        self.gpu_id = self._combo_with_data(_GPU_OPTIONS)
        grid.addWidget(self._label("使うGPU"), 2, 0)
        grid.addWidget(self.gpu_id, 2, 1)

        # --- 出力フォルダ名 ---
        self.subfolder_name = QLineEdit()
        self.subfolder_name.setPlaceholderText("upscaled")
        grid.addWidget(self._label("出力フォルダ名"), 2, 2)
        grid.addWidget(self.subfolder_name, 2, 3)

        root.addLayout(grid)

        # --- トグル群 ---
        toggles = QGridLayout()
        toggles.setHorizontalSpacing(12)
        toggles.setVerticalSpacing(8)
        self.hw_encode = ClearCheckBox("動画の保存を速くする")
        self.keep_audio = ClearCheckBox("動画の音声を残す")
        self.tta_mode = ClearCheckBox("高品質モード（遅い）")
        self.create_subfolder = ClearCheckBox("出力フォルダを作る")
        toggles.addWidget(self.hw_encode, 0, 0)
        toggles.addWidget(self.keep_audio, 0, 1)
        toggles.addWidget(self.tta_mode, 1, 0)
        toggles.addWidget(self.create_subfolder, 1, 1)
        toggle_wrap = QWidget()
        toggle_wrap.setObjectName("toggleWrap")
        toggle_wrap.setLayout(toggles)
        root.addWidget(toggle_wrap)

        # 出力フォルダ名はチェック時のみ有効
        self.create_subfolder.toggled.connect(self.subfolder_name.setEnabled)

        self.load_defaults()

    # --- 既定値の読込／設定への反映 ---
    def load_defaults(self, settings: UpscaleSettings | None = None) -> None:
        s = settings or UpscaleSettings()
        self.image_format.setCurrentText(s.image_format)
        self.video_format.setCurrentText(s.video_format)
        self._set_combo_value(self.video_quality, s.video_quality)
        self._set_combo_value(self.tile_size, s.tile_size)
        self._set_combo_value(self.gpu_id, s.gpu_id)
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
        s.video_quality = int(self.video_quality.currentData())
        s.tile_size = int(self.tile_size.currentData())
        s.gpu_id = int(self.gpu_id.currentData())
        name = self.subfolder_name.text().strip() or "upscaled"
        s.subfolder_name = name
        s.hw_encode = self.hw_encode.isChecked()
        s.keep_audio = self.keep_audio.isChecked()
        s.tta_mode = self.tta_mode.isChecked()
        s.create_subfolder = self.create_subfolder.isChecked()

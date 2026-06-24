"""ドラッグ&ドロップ受け入れゾーン。

ファイル・フォルダ・複数選択を受け付け、ドロップされたローカルパスを
`pathsDropped(list[str])` シグナルで通知する。
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QMouseEvent
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout

from .icons import Icon, apply_icon_font


class DropZone(QFrame):
    """画像・動画・フォルダをドロップで受け取るパネル。"""

    pathsDropped = Signal(list)  # list[str] のローカルパス
    browseRequested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("dropZone")
        self.setAcceptDrops(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(250)
        self._pressed_inside = False
        self._build()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(28, 26, 28, 26)
        root.setSpacing(12)
        root.addStretch(1)

        glyph_row = QHBoxLayout()
        glyph_row.setSpacing(34)
        glyph_row.addStretch(1)
        for ch in (Icon.IMAGE, Icon.VIDEO, Icon.FOLDER):
            g = QLabel(ch)
            g.setObjectName("dropGlyph")
            apply_icon_font(g, 64)
            g.setAlignment(Qt.AlignmentFlag.AlignCenter)
            g.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
            glyph_row.addWidget(g)
        glyph_row.addStretch(1)
        root.addLayout(glyph_row)

        title = QLabel("画像・動画をドロップ")
        title.setObjectName("dropTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        root.addWidget(title)

        hint = QLabel("クリックでファイル選択 / 複数選択OK")
        hint.setObjectName("dropHint")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        root.addWidget(hint)

        root.addStretch(1)

    # --- D&D イベント ---
    def _set_active(self, active: bool) -> None:
        # 動的プロパティを切り替えて QSS を再適用させる
        self.setProperty("dragActive", active)
        self.style().unpolish(self)
        self.style().polish(self)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self._set_active(True)
        else:
            event.ignore()

    def dragLeaveEvent(self, event) -> None:  # noqa: N802
        self._set_active(False)
        super().dragLeaveEvent(event)

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        self._set_active(False)
        urls = event.mimeData().urls()
        paths = [u.toLocalFile() for u in urls if u.isLocalFile()]
        paths = [p for p in paths if p]
        if paths:
            event.acceptProposedAction()
            self.pathsDropped.emit(paths)
        else:
            event.ignore()

    # --- クリックでファイル選択 ---
    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._pressed_inside = True
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self._pressed_inside
            and self.rect().contains(event.position().toPoint())
        ):
            self._pressed_inside = False
            self.browseRequested.emit()
            event.accept()
            return
        self._pressed_inside = False
        super().mouseReleaseEvent(event)

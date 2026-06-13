"""Windows icon-font helpers for the GUI."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import QLabel, QPushButton


ICON_FONT = "Segoe Fluent Icons"


class Icon:
    ADD = "\ue710"
    CLOSE = "\ue8bb"
    SETTINGS = "\ue713"
    VIDEO = "\ue714"
    DELETE = "\ue74d"
    FOLDER = "\ue8b7"
    IMAGE = "\ue8b9"
    PAUSE = "\ue769"
    PLAY = "\ue768"
    PROCESSOR = "\uf158"
    UPLOAD = "\ue74a"


def icon_font(size: int, weight: int = QFont.Weight.Normal) -> QFont:
    font = QFont(ICON_FONT)
    font.setPixelSize(size)
    font.setWeight(weight)
    return font


def apply_icon_font(widget: QLabel | QPushButton, size: int) -> None:
    widget.setFont(icon_font(size))


def make_icon(glyph: str, size: int, color: str) -> QIcon:
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(QColor(color))
    painter.setFont(icon_font(max(12, size - 3)))
    painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, glyph)
    painter.end()

    return QIcon(pixmap)

"""app/assets/logo.svg から、アプリのアイコン（.ico）と PNG を作り直す。

使い方: .venv\\Scripts\\python.exe scripts\\make_icon.py
ロゴを変えたら実行し、できた app.ico / logo.png も一緒にコミットする。
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

from PIL import Image
from PySide6.QtCore import QBuffer, QByteArray, QIODevice, Qt
from PySide6.QtGui import QGuiApplication, QImage, QPainter
from PySide6.QtSvg import QSvgRenderer

ASSETS = Path(__file__).resolve().parents[1] / "app" / "assets"
ICO_SIZES = (16, 20, 24, 32, 40, 48, 64, 128, 256)


def render(renderer: QSvgRenderer, size: int) -> Image.Image:
    image = QImage(size, size, QImage.Format.Format_ARGB32)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    renderer.render(painter)
    painter.end()
    data = QByteArray()
    buffer = QBuffer(data)
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    image.save(buffer, "PNG")
    return Image.open(io.BytesIO(bytes(data))).convert("RGBA")


def main() -> int:
    app = QGuiApplication.instance() or QGuiApplication(sys.argv[:1])  # noqa: F841
    renderer = QSvgRenderer(str(ASSETS / "logo.svg"))
    if not renderer.isValid():
        print("logo.svg を読めません", file=sys.stderr)
        return 1
    # 各サイズを SVG から直接描く（大きい絵を縮めるより小さいサイズの縁がきれい）
    images = [render(renderer, size) for size in ICO_SIZES]
    images[-1].save(
        ASSETS / "app.ico", format="ICO",
        sizes=[(s, s) for s in ICO_SIZES], append_images=images[:-1],
    )
    render(renderer, 256).save(ASSETS / "logo.png")
    print("wrote", ASSETS / "app.ico", ASSETS / "logo.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

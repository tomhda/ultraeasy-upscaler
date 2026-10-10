"""app/assets/logo.png から、アプリのアイコン（.ico）を作り直す。

使い方: .venv\\Scripts\\python.exe scripts\\make_icon.py
ロゴを変えたら実行し、できた app.ico も一緒にコミットする。
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageFilter

ASSETS = Path(__file__).resolve().parents[1] / "app" / "assets"
ICO_SIZES = (16, 20, 24, 32, 40, 48, 64, 128, 256)


def render(source: Image.Image, size: int) -> Image.Image:
    image = source.resize((size, size), Image.LANCZOS)
    if size <= 48:
        # 小さいサイズは縮小でぼけるので、刃の線が残るよう少し締める
        image = image.filter(ImageFilter.UnsharpMask(radius=0.6, percent=80, threshold=0))
    return image


def main() -> int:
    source = Image.open(ASSETS / "logo.png").convert("RGBA")
    images = [render(source, size) for size in ICO_SIZES]
    images[-1].save(
        ASSETS / "app.ico", format="ICO",
        sizes=[(s, s) for s in ICO_SIZES], append_images=images[:-1],
    )
    print("wrote", ASSETS / "app.ico")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

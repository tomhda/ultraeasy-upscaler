"""README の「画質の比較」の画像（docs/benchmarks/model_guide_*.png）を作る。

各素材を、アプリと同じ経路（GPU / DirectML）で 5 つのモデルに通し、同じ範囲を切り出して
オリジナル（Lanczos 4 倍）と並べる。SwinIR-M と AdcSR は 1 枚に数分かかる。

    .venv\\Scripts\\python.exe scripts\\make_model_guide.py --sources <素材のフォルダ>

素材のフォルダには bbb_src.png（Big Buck Bunny, 853x480）、tos_src.png（Tears of Steel, 1280x534）、
superman_src.png（Superman 1941, 320x240）を置く。拡大結果は --work に残し、あれば再利用する。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core import upscaler  # noqa: E402
from app.core.settings import (  # noqa: E402
    HELPER_MODEL_ADCSR,
    HELPER_MODEL_AMD_RRDB,
    HELPER_MODEL_ANIME,
    HELPER_MODEL_SPAN,
    HELPER_MODEL_SWINIR,
    UpscaleBackend,
    UpscaleSettings,
)

# (表示名, モデルキー)。表示名は GUI と同じ。
MODELS = [
    ("Anime Video v3", HELPER_MODEL_ANIME),
    ("4xNomosUni SPAN", HELPER_MODEL_SPAN),
    ("Real-ESRGAN（AMD縮小版）", HELPER_MODEL_AMD_RRDB),
    ("SwinIR-M", HELPER_MODEL_SWINIR),
    ("AdcSR", HELPER_MODEL_ADCSR),
]
ORIGINAL_LABEL = "オリジナル（Lanczos で 4 倍・AI なし）"
# 英語版の README 用のラベル（アプリの英語表示と同じ表記）
EN_LABELS = {
    ORIGINAL_LABEL: "Original (enlarged 4× with Lanczos, no AI)",
    "Real-ESRGAN（AMD縮小版）": "Real-ESRGAN (AMD compact)",
}

# 素材ごとの切り出し範囲（元画像の座標: x, y, w, h）。4 倍後に 752x560 になる。
SOURCES = {
    "bbb": ("bbb_src.png", (230, 130, 188, 140)),
    "sup": ("superman_src.png", (110, 70, 188, 140)),
    "tos": ("tos_src.png", (255, 150, 188, 140)),
}
COLUMNS = 3
LABEL_HEIGHT = 34
GAP = 4
BACKGROUND = (20, 20, 20)
LABEL_COLOR = (235, 235, 235)


def _font(size: int, lang: str = "ja") -> ImageFont.FreeTypeFont:
    names = ("segoeui.ttf",) if lang == "en" else ()
    for name in (*names, "YuGothM.ttc", "meiryo.ttc", "msgothic.ttc"):
        path = Path("C:/Windows/Fonts") / name
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _upscaled(src: Path, model_key: str, work: Path) -> Path:
    out = work / f"{src.stem}_{model_key}.png"
    if out.is_file():
        return out
    settings = UpscaleSettings(backend=UpscaleBackend.WINML_GPU, model=model_key, scale=4)
    started = time.monotonic()
    upscaler.upscale_image(str(src), str(out), settings)
    print(f"{out.name}: {time.monotonic() - started:.0f} 秒", flush=True)
    return out


def make_sheet(name: str, sources: Path, work: Path, dest: Path, lang: str = "ja") -> None:
    filename, (x, y, w, h) = SOURCES[name]
    src = sources / filename
    box = (x * 4, y * 4, (x + w) * 4, (y + h) * 4)
    with Image.open(src) as im:
        original = im.convert("RGB").resize((im.width * 4, im.height * 4), Image.LANCZOS)
    tiles = [(ORIGINAL_LABEL, original.crop(box))]
    for label, key in MODELS:
        with Image.open(_upscaled(src, key, work)) as im:
            tiles.append((label, im.convert("RGB").crop(box)))

    tile_w, tile_h = tiles[0][1].size
    rows = -(-len(tiles) // COLUMNS)
    sheet = Image.new(
        "RGB",
        (COLUMNS * tile_w + (COLUMNS - 1) * GAP, rows * (LABEL_HEIGHT + tile_h) + (rows - 1) * GAP),
        BACKGROUND,
    )
    draw = ImageDraw.Draw(sheet)
    font = _font(20, lang)
    for index, (label, tile) in enumerate(tiles):
        if lang == "en":
            label = EN_LABELS.get(label, label)
        left = (index % COLUMNS) * (tile_w + GAP)
        top = (index // COLUMNS) * (LABEL_HEIGHT + tile_h + GAP)
        draw.text((left + 10, top + 5), label, font=font, fill=LABEL_COLOR)
        sheet.paste(tile, (left, top + LABEL_HEIGHT))
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / (f"model_guide_{name}_en.png" if lang == "en" else f"model_guide_{name}.png")
    sheet.save(out, optimize=True)
    print(f"{out} {sheet.size}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--work", type=Path, default=ROOT / "tmp" / "model-guide")
    parser.add_argument("--dest", type=Path, default=ROOT / "docs" / "benchmarks")
    parser.add_argument("--only", choices=sorted(SOURCES), nargs="*")
    parser.add_argument("--lang", choices=("ja", "en"), default="ja", help="ラベルの言語")
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    for name in args.only or sorted(SOURCES):
        make_sheet(name, args.sources, args.work, args.dest, args.lang)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

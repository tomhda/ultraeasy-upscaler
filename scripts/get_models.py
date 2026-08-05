"""モデル/バイナリ取得スクリプト（再クローン時の復元用）。

`vendor/` は .gitignore 対象のため realesrgan-ncnn-vulkan 本体と各モデルは
git に含まれない。本スクリプトで公式リリースからまとめて取得・復元する。

    .venv\\Scripts\\python.exe scripts\\get_models.py

取得物（いずれも既に存在すればスキップ＝冪等）:
  - realesrgan-ncnn-vulkan 一式（exe + 基本モデル）… xinntao/Real-ESRGAN v0.2.5.0
  - 追加の汎用動画モデル … TransparentLC/realesrgan-gui の additional-models
      realesr-general-x4v3 / realesr-general-wdn-x4v3
"""
from __future__ import annotations

import io
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor" / "realesrgan"
MODELS = VENDOR / "models"
RIFE_VENDOR = ROOT / "vendor" / "rife"

BASE_ZIP_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/"
    "v0.2.5.0/realesrgan-ncnn-vulkan-20220424-windows.zip"
)

EXTRA_MODELS_BASE = (
    "https://github.com/TransparentLC/realesrgan-gui/releases/download/additional-models/"
)
EXTRA_MODELS = [
    "realesr-general-x4v3.param",
    "realesr-general-x4v3.bin",
    "realesr-general-wdn-x4v3.param",
    "realesr-general-wdn-x4v3.bin",
]
RIFE_ZIP_URL = (
    "https://github.com/nihui/rife-ncnn-vulkan/releases/download/"
    "20221029/rife-ncnn-vulkan-20221029-windows.zip"
)


def _download(url: str) -> bytes:
    print(f"  download: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "ultraeasy-upscaler/get_models"})
    with urllib.request.urlopen(req, timeout=180) as resp:  # 302 はデフォルトで追従
        return resp.read()


def fetch_base() -> None:
    exe = VENDOR / "realesrgan-ncnn-vulkan.exe"
    if exe.exists():
        print("base: realesrgan-ncnn-vulkan.exe あり -> スキップ")
        return
    print("base: realesrgan-ncnn-vulkan 一式を取得中…")
    VENDOR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(_download(BASE_ZIP_URL))) as zf:
        zf.extractall(VENDOR)
    print(f"base: 展開完了 -> {VENDOR}")


def fetch_extra_models() -> None:
    MODELS.mkdir(parents=True, exist_ok=True)
    for name in EXTRA_MODELS:
        dst = MODELS / name
        if dst.exists():
            print(f"model: {name} あり -> スキップ")
            continue
        print(f"model: {name} を取得中…")
        dst.write_bytes(_download(EXTRA_MODELS_BASE + name))
    print(f"models: {MODELS}")


def fetch_rife() -> None:
    candidates = list(RIFE_VENDOR.glob("**/rife-ncnn-vulkan.exe"))
    if candidates and (candidates[0].parent / "rife-v4.6").is_dir():
        print("rife: exe + rife-v4.6 あり -> スキップ")
        return
    print("rife: RIFE v4.6 一式を取得中…")
    RIFE_VENDOR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(_download(RIFE_ZIP_URL))) as zf:
        zf.extractall(RIFE_VENDOR)
    print(f"rife: 展開完了 -> {RIFE_VENDOR}")


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    print(f"取得先: {VENDOR}")
    fetch_base()
    fetch_extra_models()
    fetch_rife()
    exe = VENDOR / "realesrgan-ncnn-vulkan.exe"
    params = sorted(p.stem for p in MODELS.glob("*.param")) if MODELS.exists() else []
    print("\n--- 結果 ---")
    print("exe   :", "OK" if exe.exists() else "なし", f"({exe})")
    print("models:", params)
    rife_ok = any(RIFE_VENDOR.glob("**/rife-v4.6/flownet.param"))
    print("rife  :", "OK" if rife_ok else "なし")
    return 0 if exe.exists() and rife_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

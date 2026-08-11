"""量子化キャリブレーション用の 256x256 パッチを既存フレームから切り出す。

Ryzen AI conda 環境で実行:
  %USERPROFILE%\\miniforge3\\envs\\ryzen-ai-1.7.1\\python.exe scripts/npu/make_calib_patches.py

入力: tmp/ 配下の過去評価フレーム（ユーザー自身のコンテンツ＝ドメイン一致）
出力: tmp/npu-anime/calib/*.png （既定 200 枚）
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
SOURCES = [
    REPO / "tmp" / "rife-v4.6-evaluation-20260802",
    REPO / "tmp" / "rife-v4.6-eval-b-20260802",
    REPO / "tmp" / "rife-v4.6-live-action-20260802",
]
PATCH = int(sys.argv[sys.argv.index("--patch") + 1]) if "--patch" in sys.argv else 256
OUT = REPO / "tmp" / "npu-anime" / ("calib" if PATCH == 256 else f"calib{PATCH}")
COUNT = 200
SEED = 20260812


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    files: list[Path] = []
    for src in SOURCES:
        if src.exists():
            files.extend(src.rglob("*.png"))
    if not files:
        print("ソースフレームが見つかりません")
        return 1

    rng = random.Random(SEED)
    rng.shuffle(files)
    OUT.mkdir(parents=True, exist_ok=True)

    saved = 0
    for path in files:
        if saved >= COUNT:
            break
        data = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        if h < PATCH or w < PATCH:
            # フレームがパッチより小さい場合は反射パディングで埋める
            pad_h = max(0, PATCH - h)
            pad_w = max(0, PATCH - w)
            img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
            h, w = img.shape[:2]
        y = rng.randrange(0, h - PATCH + 1)
        x = rng.randrange(0, w - PATCH + 1)
        patch = img[y:y + PATCH, x:x + PATCH]
        ok, encoded = cv2.imencode(".png", patch)
        if not ok:
            continue
        encoded.tofile(str(OUT / f"calib_{saved:04d}.png"))
        saved += 1

    print(f"{saved} 枚のパッチを {OUT} に保存")
    return 0 if saved >= 50 else 1


if __name__ == "__main__":
    raise SystemExit(main())

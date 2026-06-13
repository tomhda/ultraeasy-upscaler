"""手動E2E確認: engine.process_job を実画像 / 短い動画で通す。

GUI を介さずコア統合（engine → upscaler / video）を end-to-end で検証する。
使い方:
    .venv\\Scripts\\python.exe scripts\\e2e_check.py
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

# 日本語Windowsのコンソール(cp932)でも UTF-8 で出力する（絵文字/日本語の文字化け・クラッシュ防止）
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# プロジェクトルートを import パスに追加
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core import binaries, media          # noqa: E402
from app.core.engine import process_job        # noqa: E402
from app.core.jobs import Job                  # noqa: E402
from app.core.settings import OutputLocation, UpscaleSettings  # noqa: E402


def _progress(label: str):
    def cb(frac: float, msg: str) -> None:
        print(f"  [{label}] {frac * 100:5.1f}%  {msg}")
    return cb


def main() -> int:
    vendor = binaries.realesrgan_dir()
    out_root = Path(tempfile.mkdtemp(prefix="ueu_e2e_"))
    print("出力先:", out_root)

    # --- 画像 E2E: input.jpg(220) を 4x → 880 ---
    print("\n=== 画像 E2E: input.jpg を 4x ===")
    img_job = Job.create(str(vendor / "input.jpg"))
    img_settings = UpscaleSettings(
        scale=4, model="realesrgan-x4plus",
        output_location=OutputLocation.CUSTOM, output_dir=str(out_root),
        create_subfolder=False,
    )
    img_out = process_job(img_job, img_settings, _progress("img"))
    from PIL import Image
    with Image.open(img_out) as im:
        iw, ih = im.width, im.height
    print(f"  -> {img_out.name}  {iw}x{ih}")
    assert img_out.exists() and iw == 880, f"画像出力が不正: {iw}x{ih}"

    # --- 動画 E2E: bbb_demo 0.6s を 2x（分解→拡大→結合＋音声維持）---
    print("\n=== 動画 E2E: bbb_demo 0.6s を 2x ===")
    clip = out_root / "clip.mp4"
    subprocess.run(
        [binaries.ffmpeg_exe(), "-y", "-i", str(vendor / "bbb_demo.mp4"),
         "-t", "0.6", str(clip)],
        capture_output=True, check=True,
    )
    src = media.probe(str(clip))
    print(f"  クリップ: {src.width}x{src.height} fps={src.fps} audio={src.has_audio} frames={src.frame_count}")
    vid_job = Job.create(str(clip))
    vid_settings = UpscaleSettings(
        scale=2, model="realesr-animevideov3",
        output_location=OutputLocation.CUSTOM, output_dir=str(out_root),
        create_subfolder=False, keep_audio=True, hw_encode=True,
    )
    vid_out = process_job(vid_job, vid_settings, _progress("vid"))
    info = media.probe(str(vid_out))
    print(f"  -> {vid_out.name}  {info.width}x{info.height}  fps={info.fps}  audio={info.has_audio}  frames={info.frame_count}")
    assert vid_out.exists(), "動画出力なし"
    assert info.width == src.width * 2, f"動画が2xになっていない: {info.width} vs {src.width*2}"
    assert info.has_audio, "音声が維持されていない"

    # --- 動画 E2E（#1 回帰）: 出力形式 jpg でも中間フレームは PNG 固定で成功する ---
    print("\n=== 動画 E2E: image_format=jpg でも動画は成功（#1 回帰）===")
    vid2_settings = UpscaleSettings(
        scale=2, model="realesr-animevideov3",
        output_location=OutputLocation.CUSTOM, output_dir=str(out_root),
        create_subfolder=False, image_format="jpg",
    )
    vid2_out = process_job(Job.create(str(clip)), vid2_settings, _progress("vid-jpg"))
    info2 = media.probe(str(vid2_out))
    print(f"  -> {vid2_out.name}  {info2.width}x{info2.height}  audio={info2.has_audio}")
    assert vid2_out.exists() and info2.width == src.width * 2, "jpg設定で動画が壊れた"

    print("\nE2E OK ✅  ->", out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

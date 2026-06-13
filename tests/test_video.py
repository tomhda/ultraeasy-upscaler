"""video.py のテスト: HWエンコーダ検出 / フレーム抽出 / 再結合ラウンドトリップ。

高速化のため、デモ動画から ~1 秒のクリップを切り出して使う。
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from app.core import binaries, media, video
from app.core.settings import UpscaleSettings

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_DEMO = binaries.realesrgan_dir() / "bbb_demo.mp4"

pytestmark = pytest.mark.skipif(not _DEMO.exists(), reason="デモ動画が無い")


@pytest.fixture(scope="module")
def clip(tmp_path_factory) -> Path:
    """デモ動画から ~1 秒のクリップを切り出して返す（音声付き・再エンコード）。"""
    out = tmp_path_factory.mktemp("clip") / "clip.mp4"
    cmd = [
        binaries.ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(_DEMO), "-t", "1",
        # フレーム数を安定させるため映像は再エンコード、音声も保持
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        str(out),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          creationflags=_NO_WINDOW)
    assert proc.returncode == 0, proc.stderr
    assert out.exists()
    return out


def test_detect_hw_encoder_no_raise():
    # str か None を返し、例外を送出しないこと
    result = video.detect_hw_encoder()
    assert result is None or isinstance(result, str)


def test_extract_frames(clip, tmp_path):
    out_dir = tmp_path / "frames"
    n = video.extract_frames(str(clip), str(out_dir))

    pngs = sorted(out_dir.glob(video.FRAME_GLOB))
    # 返り値が実ファイル数と一致すること
    assert n == len(pngs)
    assert n > 0
    # 連番命名が FRAME_PATTERN と整合（先頭は frame_00000001.png）
    assert pngs[0].name == "frame_00000001.png"

    # probe の frame_count と ±1 で一致すること
    info = media.probe(str(clip))
    assert info.frame_count is not None
    assert abs(n - info.frame_count) <= 1


def test_extract_progress_called(clip, tmp_path):
    out_dir = tmp_path / "frames_p"
    seen: list[float] = []
    video.extract_frames(str(clip), str(out_dir),
                         progress=lambda f, m: seen.append(f))
    assert seen  # 何らかの進捗が報告される
    assert seen[-1] == pytest.approx(1.0)
    assert all(0.0 <= f <= 1.0 for f in seen)


def test_roundtrip_with_audio(clip, tmp_path):
    """抽出 → 再結合 → probe で寸法・フレーム数・音声を検証する。"""
    src = media.probe(str(clip))

    frames = tmp_path / "frames"
    n = video.extract_frames(str(clip), str(frames))
    assert n > 0

    out = tmp_path / "out.mp4"
    settings = UpscaleSettings(keep_audio=True, hw_encode=True, video_quality=23)
    video.reassemble(str(frames), str(clip), str(out), src.fps, settings)

    assert out.exists() and out.stat().st_size > 0

    res = media.probe(str(out))
    assert res.kind == "video"
    # 生フレームをそのまま動画化したので寸法は元と一致する
    assert res.width == src.width
    assert res.height == src.height
    # フレーム数はほぼ一致（エンコーダの端数で ±2 程度許容）
    assert res.frame_count is not None
    assert abs(res.frame_count - n) <= 2
    # クリップは音声を持ち keep_audio=True なので出力にも音声がある
    assert src.has_audio is True
    assert res.has_audio is True


def test_roundtrip_no_audio(clip, tmp_path):
    """keep_audio=False のときは音声なしで出力されること。"""
    src = media.probe(str(clip))

    frames = tmp_path / "frames_na"
    video.extract_frames(str(clip), str(frames))

    out = tmp_path / "out_na.mp4"
    settings = UpscaleSettings(keep_audio=False, hw_encode=False, video_quality=23)
    video.reassemble(str(frames), str(clip), str(out), src.fps, settings)

    res = media.probe(str(out))
    assert res.kind == "video"
    assert res.has_audio is False

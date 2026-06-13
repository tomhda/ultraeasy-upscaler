"""media.probe() のテスト（実アセットを ffprobe / PIL で解析）。"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.core import binaries, media

_VENDOR = binaries.realesrgan_dir()
_DEMO = _VENDOR / "bbb_demo.mp4"
_IMAGE = _VENDOR / "input.jpg"


@pytest.mark.skipif(not _DEMO.exists(), reason="デモ動画が無い")
def test_probe_video():
    info = media.probe(str(_DEMO))
    assert info.kind == "video"
    assert info.width > 0
    assert info.height > 0
    assert info.fps is not None and info.fps > 0
    assert info.frame_count is not None and info.frame_count > 0
    assert info.duration is not None and info.duration > 0
    assert info.size_bytes > 0
    assert info.codec  # 例: "h264"
    # デモ動画は音声トラックを含む
    assert info.has_audio is True


@pytest.mark.skipif(not _IMAGE.exists(), reason="入力画像が無い")
def test_probe_image():
    info = media.probe(str(_IMAGE))
    assert info.kind == "image"
    assert info.width > 0
    assert info.height > 0
    assert info.size_bytes > 0
    # 画像は動画系メタを持たない
    assert info.has_audio is False


def test_parse_fps():
    # "30000/1001" 形式を float 化できること
    assert media._parse_fps("30000/1001") == pytest.approx(29.97, abs=0.01)
    assert media._parse_fps("24") == pytest.approx(24.0)
    assert media._parse_fps("0/0") is None
    assert media._parse_fps("") is None
    assert media._parse_fps(None) is None

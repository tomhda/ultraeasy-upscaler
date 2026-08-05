"""RIFEフレーム補間ラッパの実動スモークテスト。"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from app.core import binaries, interpolator, video
from app.core.settings import UpscaleSettings


def _rife_available() -> bool:
    return "rife-v4.6" in binaries.available_interpolation_models()


@pytest.mark.skipif(not _rife_available(), reason="RIFE v4.6が無い")
def test_rife_v46_doubles_small_frame_sequence(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()
    cmd = [
        binaries.ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", "testsrc2=size=128x96:rate=3:duration=1",
        str(source / video.FRAME_PATTERN),
    ]
    assert subprocess.run(cmd).returncode == 0

    settings = UpscaleSettings(model=None, interpolation_model="rife-v4.6")
    count, fps = interpolator.interpolate_folder(
        str(source), str(output), settings, source_fps=3.0
    )
    assert count == 6
    assert fps == pytest.approx(6.0)
    assert len(list(output.glob(video.FRAME_GLOB))) == 6

"""video.py のテスト: HWエンコーダ検出 / フレーム抽出 / 再結合ラウンドトリップ。

高速化のため、デモ動画から ~1 秒のクリップを切り出して使う。
"""
from __future__ import annotations

import io
import subprocess
from pathlib import Path

import pytest
from PIL import Image

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


def test_detect_hw_encoder_prefers_nvenc_when_nvidia_driver_is_present(monkeypatch):
    video.detect_hw_encoder.cache_clear()
    monkeypatch.setattr(
        video.shutil, "which",
        lambda name: "C:/Windows/System32/nvidia-smi.exe"
        if name in {"nvidia-smi", "nvidia-smi.exe"} else None,
    )
    monkeypatch.setattr(
        video, "_available_encoders",
        lambda: frozenset({"h264_amf", "h264_nvenc"}),
    )
    monkeypatch.setattr(video, "_encoder_works", lambda _encoder: True)

    assert video.detect_hw_encoder("h264") == "h264_nvenc"
    video.detect_hw_encoder.cache_clear()


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ((3840, 2160), (3840, 2160)),  # 上限ちょうどは縮小しない
        ((3412, 1920), (3412, 1920)),  # 480p x4 の既存出力
        ((7680, 4320), (3840, 2160)),  # 1080p x4
        ((3840, 2161), (3838, 2160)),  # 高さ超過側の境界
        ((7680, 4352), (3810, 2160)),  # 比率維持 + 偶数切り下げ
    ],
)
def test_fit_video_dimensions_boundaries(source, expected):
    fitted = video.fit_video_dimensions(*source)

    assert fitted == expected
    assert fitted[0] <= video.DEFAULT_MAX_VIDEO_DIM[0]
    assert fitted[1] <= video.DEFAULT_MAX_VIDEO_DIM[1]
    assert fitted[0] % 2 == 0
    assert fitted[1] % 2 == 0
    assert fitted[0] / fitted[1] == pytest.approx(source[0] / source[1], abs=0.002)


def test_max_video_dim_setting_precedes_environment(monkeypatch):
    monkeypatch.setenv(video.MAX_VIDEO_DIM_ENV, "1280x720")

    assert video.resolve_max_video_dim(UpscaleSettings()) == (1280, 720)
    assert video.resolve_max_video_dim(
        UpscaleSettings(max_video_dim=(1920, 1080))
    ) == (1920, 1080)


def test_reassemble_adds_lanczos_fit_filter(monkeypatch, tmp_path):
    frames = tmp_path / "frames"
    frames.mkdir()
    Image.new("RGB", (800, 600), color=(0, 0, 0)).save(
        frames / "frame_00000001.png"
    )
    commands: list[list[str]] = []
    messages: list[str] = []

    def fake_run(cmd, *_args, **_kwargs):
        commands.append(cmd)

    monkeypatch.setattr(video, "_run_with_progress", fake_run)
    settings = UpscaleSettings(
        keep_audio=False,
        hw_encode=False,
        max_video_dim=(640, 480),
    )
    video.reassemble(
        str(frames), "", str(tmp_path / "out.mp4"), 30.0, settings,
        progress=lambda _fraction, message: messages.append(message),
    )

    vf = commands[0][commands[0].index("-vf") + 1]
    assert vf.startswith("scale=640:480:flags=lanczos,")
    assert any("出力を640x480へ縮小" in message for message in messages)


class _FailingPipe:
    def write(self, _payload):
        raise OSError(22, "Invalid argument")

    def close(self):
        pass


class _FakePipelineProcess:
    def __init__(self, *, decoder: bool):
        self.stdout = io.BytesIO(b"\x00" * 12) if decoder else None
        self.stderr = io.BytesIO(
            b"[h264_amf] invalid resolution: 7680x4320\n"
            if not decoder else b""
        )
        self.stdin = None if decoder else _FailingPipe()
        self.returncode = 0 if decoder else 1

    def poll(self):
        return self.returncode

    def terminate(self):
        self.returncode = 1

    def kill(self):
        self.returncode = 1

    def wait(self, timeout=None):
        del timeout
        return self.returncode


def test_piped_encoder_failure_prefers_encoder_stderr(monkeypatch, tmp_path):
    from app.core import helper_backend

    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    processes: list[_FakePipelineProcess] = []

    monkeypatch.setattr(
        media,
        "probe",
        lambda _path: media.MediaInfo(
            kind="video", width=2, height=2, fps=1.0, frame_count=1
        ),
    )

    class _Session:
        scale = 1

        def upscale(self, image):
            return image

        def close(self, *, force=False):
            del force

    monkeypatch.setattr(helper_backend, "open_session", lambda *_args, **_kwargs: _Session())
    monkeypatch.setattr(video, "detect_hw_encoder", lambda *_args: "h264_amf")

    def fake_popen(cmd, **_kwargs):
        del cmd
        process = _FakePipelineProcess(decoder=len(processes) == 0)
        processes.append(process)
        return process

    monkeypatch.setattr(video.subprocess, "Popen", fake_popen)

    with pytest.raises(RuntimeError) as caught:
        video.upscale_video_piped(
            str(source), str(tmp_path / "out.mp4"),
            UpscaleSettings(hw_encode=True),
        )

    message = str(caught.value)
    assert "動画エンコードに失敗しました" in message
    assert "invalid resolution: 7680x4320" in message
    assert "Errno 22" not in message


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

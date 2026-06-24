"""engine の経路選択テスト。"""
from __future__ import annotations

from pathlib import Path

from app.core import engine, media, upscaler, video
from app.core.jobs import Job, JobKind
from app.core.settings import UpscaleBackend, UpscaleSettings


def test_video_forces_vulkan_backend_when_settings_are_npu(monkeypatch, tmp_path: Path) -> None:
    """動画は当面NPU非対応なので、フレーム拡大時はVulkanへ戻す。"""
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"fake")
    job = Job(input_path=src, kind=JobKind.VIDEO)
    seen_backends: list[UpscaleBackend] = []

    monkeypatch.setattr(
        media,
        "probe",
        lambda _path: media.MediaInfo(kind="video", width=16, height=16, fps=30.0),
    )

    def fake_extract(_src, out_dir, progress=None, cancel=None):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "frame_00000001.png").write_bytes(b"frame")
        if progress:
            progress(1.0, "抽出")
        return 1

    def fake_upscale_folder(_in_dir, out_dir, settings, progress=None, cancel=None):
        seen_backends.append(settings.backend)
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "frame_00000001.png").write_bytes(b"up")
        if progress:
            progress(1.0, "拡大")

    def fake_reassemble(_frames, _src, out_path, _fps, _settings, progress=None, cancel=None):
        Path(out_path).write_bytes(b"video")
        if progress:
            progress(1.0, "結合")

    monkeypatch.setattr(video, "extract_frames", fake_extract)
    monkeypatch.setattr(upscaler, "upscale_folder", fake_upscale_folder)
    monkeypatch.setattr(video, "reassemble", fake_reassemble)

    settings = UpscaleSettings(
        backend=UpscaleBackend.NPU,
        output_location=engine.OutputLocation.CUSTOM,
        output_dir=str(tmp_path),
        create_subfolder=False,
    )
    out = engine.process_job(job, settings)

    assert out.exists()
    assert seen_backends == [UpscaleBackend.VULKAN]

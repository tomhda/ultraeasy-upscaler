"""engine の経路選択テスト。"""
from __future__ import annotations

from pathlib import Path

from app.core import engine, interpolator, media, upscaler, video
from app.core.jobs import Job, JobKind
from app.core.settings import ProcessingOrder, UpscaleBackend, UpscaleSettings


def test_video_keeps_npu_backend_for_frames(monkeypatch, tmp_path: Path) -> None:
    """NPU選択時は動画のフレーム拡大もNPUで処理される。"""
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
    assert seen_backends == [UpscaleBackend.NPU]


def test_video_uses_raw_pipe_for_new_helper_without_rife(monkeypatch, tmp_path: Path) -> None:
    """RIFEなしの新AI動画はPNG抽出を経由せずrawパイプへ進む。"""
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"fake")
    job = Job(input_path=src, kind=JobKind.VIDEO)
    calls: list[str] = []

    monkeypatch.setattr(
        media,
        "probe",
        lambda _path: media.MediaInfo(
            kind="video", width=16, height=16, fps=30.0, has_audio=True
        ),
    )

    def fake_pipe(_src, out_path, _settings, progress=None, cancel=None):
        del cancel
        calls.append("raw")
        Path(out_path).write_bytes(b"video")
        if progress:
            progress(1.0, "パイプ完了")

    monkeypatch.setattr(video, "upscale_video_piped", fake_pipe)
    monkeypatch.setattr(
        video,
        "extract_frames",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("raw path must not extract PNG frames")
        ),
    )

    settings = UpscaleSettings(
        backend=UpscaleBackend.WINML_GPU,
        output_location=engine.OutputLocation.CUSTOM,
        output_dir=str(tmp_path),
        create_subfolder=False,
    )
    out = engine.process_job(job, settings)

    assert out.exists()
    assert calls == ["raw"]


def test_video_with_rife_keeps_png_route_for_new_helper(monkeypatch, tmp_path: Path) -> None:
    """新AIを選んでもRIFE有効時は従来のフレームファイル経路を維持する。"""
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"fake")
    job = Job(input_path=src, kind=JobKind.VIDEO)
    calls: list[str] = []

    monkeypatch.setattr(
        media,
        "probe",
        lambda _path: media.MediaInfo(kind="video", width=16, height=16, fps=30.0),
    )

    def fake_extract(_src, out_dir, progress=None, cancel=None):
        del cancel
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "frame_00000001.png").write_bytes(b"frame")
        if progress:
            progress(1.0, "抽出")
        return 1

    def fake_interpolate(_in_dir, out_dir, *_args, **_kwargs):
        calls.append("rife")
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "frame_00000001.png").write_bytes(b"frame")
        return 1, 30.0

    def fake_upscale(_in_dir, out_dir, settings, **_kwargs):
        calls.append(f"upscale:{settings.backend.value}")
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "frame_00000001.png").write_bytes(b"up")

    def fake_reassemble(_frames, _src, out_path, _fps, _settings, **_kwargs):
        calls.append("reassemble")
        Path(out_path).write_bytes(b"video")

    monkeypatch.setattr(video, "extract_frames", fake_extract)
    monkeypatch.setattr(interpolator, "interpolate_folder", fake_interpolate)
    monkeypatch.setattr(upscaler, "upscale_folder", fake_upscale)
    monkeypatch.setattr(video, "reassemble", fake_reassemble)
    monkeypatch.setattr(
        video,
        "upscale_video_piped",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("RIFE route must not use raw pipe")
        ),
    )

    settings = UpscaleSettings(
        backend=UpscaleBackend.NPU_NATIVE,
        interpolation_model="rife-v4.6",
        output_location=engine.OutputLocation.CUSTOM,
        output_dir=str(tmp_path),
        create_subfolder=False,
    )
    out = engine.process_job(job, settings)

    assert out.exists()
    assert calls == ["upscale:npu_native", "rife", "reassemble"]


def test_video_raw_helper_failure_falls_back_to_vulkan(monkeypatch, tmp_path: Path) -> None:
    """常駐ヘルパー起動失敗時はVulkanのPNG経路へ退避する。"""
    from app.core import helper_backend

    src = tmp_path / "clip.mp4"
    src.write_bytes(b"fake")
    job = Job(input_path=src, kind=JobKind.VIDEO)
    seen: list[UpscaleBackend] = []

    monkeypatch.setattr(
        media,
        "probe",
        lambda _path: media.MediaInfo(kind="video", width=16, height=16, fps=30.0),
    )

    def fake_pipe(*_args, **_kwargs):
        raise helper_backend.HelperBackendUnavailable("helper missing")

    def fake_extract(_src, out_dir, **_kwargs):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "frame_00000001.png").write_bytes(b"frame")
        return 1

    def fake_upscale(_in_dir, out_dir, settings, **_kwargs):
        seen.append(settings.backend)
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "frame_00000001.png").write_bytes(b"up")

    monkeypatch.setattr(video, "upscale_video_piped", fake_pipe)
    monkeypatch.setattr(video, "extract_frames", fake_extract)
    monkeypatch.setattr(upscaler, "upscale_folder", fake_upscale)
    monkeypatch.setattr(
        video,
        "reassemble",
        lambda _frames, _src, out_path, *_args, **_kwargs: Path(out_path).write_bytes(b"video"),
    )

    settings = UpscaleSettings(
        backend=UpscaleBackend.WINML_GPU,
        output_location=engine.OutputLocation.CUSTOM,
        output_dir=str(tmp_path),
        create_subfolder=False,
    )
    out = engine.process_job(job, settings)

    assert out.exists()
    assert seen == [UpscaleBackend.VULKAN]


def test_video_can_interpolate_without_upscaling(monkeypatch, tmp_path: Path) -> None:
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"fake")
    job = Job(input_path=src, kind=JobKind.VIDEO)
    calls: list[str] = []

    monkeypatch.setattr(
        media, "probe",
        lambda _path: media.MediaInfo(kind="video", width=16, height=16, fps=30.0),
    )

    def fake_extract(_src, out_dir, progress=None, cancel=None):
        Path(out_dir, "frame_00000001.png").write_bytes(b"frame")
        Path(out_dir, "frame_00000002.png").write_bytes(b"frame")
        return 2

    def fake_interpolate(in_dir, out_dir, settings, source_fps, progress=None, cancel=None):
        calls.append("interpolate")
        Path(out_dir, "frame_00000001.png").write_bytes(b"frame")
        Path(out_dir, "frame_00000002.png").write_bytes(b"frame")
        Path(out_dir, "frame_00000003.png").write_bytes(b"frame")
        Path(out_dir, "frame_00000004.png").write_bytes(b"frame")
        return 4, 60.0

    def fake_reassemble(frames, _src, out_path, fps, _settings, progress=None, cancel=None):
        calls.append(f"reassemble:{Path(frames).name}:{fps}")
        Path(out_path).write_bytes(b"video")

    monkeypatch.setattr(video, "extract_frames", fake_extract)
    monkeypatch.setattr(interpolator, "interpolate_folder", fake_interpolate)
    monkeypatch.setattr(
        upscaler, "upscale_folder",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("upscaler must be skipped")),
    )
    monkeypatch.setattr(
        video, "upscale_video_piped",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("helper raw pipe must be skipped")
        ),
    )
    monkeypatch.setattr(video, "reassemble", fake_reassemble)

    settings = UpscaleSettings(
        backend=UpscaleBackend.WINML_GPU,
        model=None,
        interpolation_model="rife-v4.6",
        output_location=engine.OutputLocation.CUSTOM,
        output_dir=str(tmp_path),
        create_subfolder=False,
    )
    out = engine.process_job(job, settings)

    assert out.exists()
    assert calls == ["interpolate", "reassemble:interp:60.0"]
    assert "RIFE-v4.6_2xfps" in out.name


def _run_both_stages(monkeypatch, tmp_path: Path, **settings_kwargs) -> list[str]:
    """アプコン+補間の両方を有効にした動画ジョブを実行し、呼び出し順を返す。"""
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"fake")
    job = Job(input_path=src, kind=JobKind.VIDEO)
    calls: list[str] = []
    monkeypatch.setattr(
        media, "probe",
        lambda _path: media.MediaInfo(kind="video", width=16, height=16, fps=30.0),
    )

    def fake_extract(_src, out_dir, progress=None, cancel=None):
        Path(out_dir, "frame_00000001.png").write_bytes(b"frame")
        Path(out_dir, "frame_00000002.png").write_bytes(b"frame")
        return 2

    def fake_interpolate(in_dir, out_dir, *_args, **_kwargs):
        calls.append(f"interpolate:{Path(in_dir).name}")
        Path(out_dir, "frame_00000001.png").write_bytes(b"frame")
        Path(out_dir, "frame_00000002.png").write_bytes(b"frame")
        return 2, 60.0

    def fake_upscale(in_dir, out_dir, *_args, **_kwargs):
        calls.append(f"upscale:{Path(in_dir).name}")
        Path(out_dir, "frame_00000001.png").write_bytes(b"frame")
        Path(out_dir, "frame_00000002.png").write_bytes(b"frame")

    def fake_reassemble(frames, _src, out_path, *_args, **_kwargs):
        calls.append(f"reassemble:{Path(frames).name}")
        Path(out_path).write_bytes(b"video")

    monkeypatch.setattr(video, "extract_frames", fake_extract)
    monkeypatch.setattr(interpolator, "interpolate_folder", fake_interpolate)
    monkeypatch.setattr(upscaler, "upscale_folder", fake_upscale)
    monkeypatch.setattr(video, "reassemble", fake_reassemble)

    settings = UpscaleSettings(
        model="realesr-animevideov3",
        interpolation_model="rife-v4.6",
        output_location=engine.OutputLocation.CUSTOM,
        output_dir=str(tmp_path),
        create_subfolder=False,
        **settings_kwargs,
    )
    engine.process_job(job, settings)
    return calls


def test_video_defaults_to_upscale_then_interpolation(monkeypatch, tmp_path: Path) -> None:
    calls = _run_both_stages(monkeypatch, tmp_path)
    assert calls == ["upscale:src", "interpolate:up", "reassemble:interp"]


def test_video_can_interpolate_before_upscaling(monkeypatch, tmp_path: Path) -> None:
    calls = _run_both_stages(
        monkeypatch, tmp_path,
        processing_order=ProcessingOrder.INTERPOLATE_FIRST,
    )
    assert calls == ["interpolate:src", "upscale:interp", "reassemble:up"]


def test_video_rejects_no_operations(tmp_path: Path) -> None:
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"fake")
    job = Job(input_path=src, kind=JobKind.VIDEO)
    settings = UpscaleSettings(model=None, interpolation_model=None)
    try:
        engine.process_job(job, settings)
    except ValueError as exc:
        assert "モデル" in str(exc)
    else:
        raise AssertionError("no-op video must be rejected")

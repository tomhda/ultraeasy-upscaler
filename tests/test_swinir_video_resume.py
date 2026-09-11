"""SwinIR CUDA動画チャンクの再開契約を、ffmpeg/torchなしで検証する。"""
from __future__ import annotations

import json
import os
from pathlib import Path

from app.core import media, video
from app.core.jobs import Cancelled
from app.core.settings import HELPER_MODEL_SWINIR, UpscaleBackend, UpscaleSettings


def _settings() -> UpscaleSettings:
    return UpscaleSettings(
        backend=UpscaleBackend.SWINIR_CUDA,
        model=HELPER_MODEL_SWINIR,
        create_subfolder=False,
        hw_encode=False,
    )


def test_swinir_chunk_resume_skips_completed_chunks(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "input.mp4"
    source.write_bytes(b"source")
    output = tmp_path / "output.mp4"
    work = tmp_path / "work"
    calls: list[int] = []
    sessions = []
    count_calls: list[str] = []

    monkeypatch.setenv(video.SWINIR_CHUNK_FRAMES_ENV, "100")
    monkeypatch.setattr(
        media,
        "probe",
        lambda _path: media.MediaInfo(
            kind="video", width=4, height=4, fps=24.0, frame_count=250
        ),
    )
    monkeypatch.setattr(
        video,
        "_swinir_paths",
        lambda: (tmp_path / "python.exe", tmp_path / "worker.py", tmp_path / "model.pth"),
    )
    monkeypatch.setattr(
        video,
        "_swinir_count_cfr_frames",
        lambda path, _fps, _cancel: count_calls.append(path) or 250,
    )

    class Session:
        scale = 4

        def close(self, *, force=False):
            del force

    def fake_session(*_args, **_kwargs):
        sessions.append(True)
        return Session()

    def fake_chunk(_input, chunk_path, start, count, *_args, **_kwargs):
        calls.append(start)
        chunk_path.write_bytes(f"chunk-{start}-{count}".encode())
        return count

    def fake_mux(chunks, _source, out_path, _settings, **_kwargs):
        assert all(path.is_file() for path in chunks)
        Path(out_path).write_bytes(b"final")

    monkeypatch.setattr(video, "_swinir_session", fake_session)
    monkeypatch.setattr(video, "_swinir_open_decoder", lambda _input, _fps: None)
    monkeypatch.setattr(video, "_swinir_discard_frames", lambda *_args: None)
    monkeypatch.setattr(video, "_swinir_run_chunk", fake_chunk)
    monkeypatch.setattr(video, "_swinir_concat_mux", fake_mux)

    video.upscale_video_swinir_chunked(
        str(source), str(output), _settings(), work_dir=work
    )
    assert calls == [0, 100, 200]
    assert len(sessions) == 1
    manifest = json.loads((work / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["chunks"]) == 3
    assert manifest["cfr_frame_count"] == 250
    assert len(count_calls) == 1
    assert output.read_bytes() == b"final"

    calls.clear()
    sessions.clear()
    output.unlink()
    video.upscale_video_swinir_chunked(
        str(source), str(output), _settings(), work_dir=work
    )
    assert calls == []
    assert sessions == []
    assert output.read_bytes() == b"final"
    assert len(count_calls) == 1

    # 先頭チャンクを同じサイズの別内容へ壊しても、SHA-256で検出する。
    calls.clear()
    sessions.clear()
    output.unlink()
    first_chunk = work / "chunk_000000.mp4"
    first_chunk.write_bytes(b"x" * first_chunk.stat().st_size)
    video.upscale_video_swinir_chunked(
        str(source), str(output), _settings(), work_dir=work
    )
    assert calls == [0]
    assert len(sessions) == 1
    assert len(count_calls) == 1


def test_swinir_manifest_identity_invalidates_chunks(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "input.mp4"
    source.write_bytes(b"source")
    output = tmp_path / "output.mp4"
    work = tmp_path / "work"
    calls: list[int] = []

    monkeypatch.setenv(video.SWINIR_CHUNK_FRAMES_ENV, "100")
    monkeypatch.setattr(
        media,
        "probe",
        lambda _path: media.MediaInfo(
            kind="video", width=4, height=4, fps=24.0, frame_count=150
        ),
    )
    model = tmp_path / "model.pth"
    model.write_bytes(b"model-v1")
    monkeypatch.setattr(
        video,
        "_swinir_paths",
        lambda: (tmp_path / "python.exe", tmp_path / "worker.py", model),
    )
    monkeypatch.setattr(
        video, "_swinir_count_cfr_frames", lambda _path, _fps, _cancel: 150
    )

    class Session:
        scale = 4

        def close(self, *, force=False):
            del force

    monkeypatch.setattr(video, "_swinir_session", lambda *_args, **_kwargs: Session())
    monkeypatch.setattr(video, "_swinir_open_decoder", lambda _input, _fps: None)
    monkeypatch.setattr(video, "_swinir_discard_frames", lambda *_args: None)

    def fake_chunk(_input, chunk_path, start, count, *_args, **_kwargs):
        calls.append(start)
        chunk_path.write_bytes(b"chunk")
        return count

    monkeypatch.setattr(video, "_swinir_run_chunk", fake_chunk)
    monkeypatch.setattr(
        video,
        "_swinir_concat_mux",
        lambda _chunks, _source, out_path, _settings, **_kwargs: Path(out_path).write_bytes(b"final"),
    )

    settings = _settings()
    settings.model = "swinir-m"  # 旧保存設定のaliasも動画入口で正規化する。
    settings.hw_encode = True
    active_encoder = {"value": "h264_nvenc"}
    monkeypatch.setattr(
        video, "detect_hw_encoder", lambda _codec: active_encoder["value"]
    )
    video.upscale_video_swinir_chunked(str(source), str(output), settings, work_dir=work)
    assert calls == [0, 100]

    # サイズとmtimeが同じでも入力内容が変われば、既存chunkを再利用しない。
    calls.clear()
    output.unlink()
    source_stat = source.stat()
    source.write_bytes(b"change")
    os.utime(source, ns=(source_stat.st_atime_ns, source_stat.st_mtime_ns))
    video.upscale_video_swinir_chunked(str(source), str(output), settings, work_dir=work)
    assert calls == [0, 100]

    # モデルもサイズとmtimeを維持した差し替えをSHA-256で検出する。
    calls.clear()
    output.unlink()
    model_stat = model.stat()
    model.write_bytes(b"model-v2")
    os.utime(model, ns=(model_stat.st_atime_ns, model_stat.st_mtime_ns))
    video.upscale_video_swinir_chunked(str(source), str(output), settings, work_dir=work)
    assert calls == [0, 100]

    # 同じ設定でも実際のencoderが変わった場合は混在させない。
    calls.clear()
    output.unlink()
    active_encoder["value"] = None
    video.upscale_video_swinir_chunked(str(source), str(output), settings, work_dir=work)
    assert calls == [0, 100]


def test_swinir_cancel_keeps_finished_chunks_for_next_run(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "input.mp4"
    source.write_bytes(b"source")
    output = tmp_path / "output.mp4"
    work = tmp_path / "work"
    calls: list[int] = []
    canceled = {"value": True}

    monkeypatch.setenv(video.SWINIR_CHUNK_FRAMES_ENV, "100")
    monkeypatch.setattr(
        media,
        "probe",
        lambda _path: media.MediaInfo(
            kind="video", width=4, height=4, fps=24.0, frame_count=200
        ),
    )
    monkeypatch.setattr(
        video,
        "_swinir_paths",
        lambda: (tmp_path / "python.exe", tmp_path / "worker.py", tmp_path / "model.pth"),
    )
    monkeypatch.setattr(
        video, "_swinir_count_cfr_frames", lambda _path, _fps, _cancel: 200
    )

    class Session:
        scale = 4

        def close(self, *, force=False):
            del force

    monkeypatch.setattr(video, "_swinir_session", lambda *_args, **_kwargs: Session())
    monkeypatch.setattr(video, "_swinir_open_decoder", lambda _input, _fps: None)
    monkeypatch.setattr(video, "_swinir_discard_frames", lambda *_args: None)

    def fake_chunk(_input, chunk_path, start, count, *_args, **_kwargs):
        calls.append(start)
        if start >= 100 and canceled["value"]:
            raise Cancelled()
        chunk_path.write_bytes(b"chunk")

    monkeypatch.setattr(video, "_swinir_run_chunk", fake_chunk)
    monkeypatch.setattr(
        video,
        "_swinir_concat_mux",
        lambda _chunks, _source, out_path, _settings, **_kwargs: Path(out_path).write_bytes(b"final"),
    )

    try:
        video.upscale_video_swinir_chunked(str(source), str(output), _settings(), work_dir=work)
    except Cancelled:
        pass
    else:
        raise AssertionError("cancel must interrupt the chunked job")
    # A chunk is recorded only after _swinir_run_chunk returns successfully.
    assert (work / "chunk_000000.mp4").exists()
    assert [item["index"] for item in json.loads(
        (work / "manifest.json").read_text(encoding="utf-8")
    )["chunks"]] == [0]

    canceled["value"] = False
    output.unlink(missing_ok=True)
    video.upscale_video_swinir_chunked(
        str(source), str(output), _settings(), work_dir=work
    )
    assert calls == [0, 100, 100]

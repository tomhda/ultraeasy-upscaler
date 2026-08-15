"""オーケストレーション: ジョブ種別ごとに upscaler / video を呼び出す。

GUI 非依存。テスト/CLI からも process_job(job, settings, progress, cancel) で実行できる。
進捗配分（動画）: 抽出 2-20% / アップスケール 20-80% / 再結合 80-100%。
"""
from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import replace
from pathlib import Path

from . import interpolator, media, upscaler, video
from .jobs import Cancelled, Job, JobKind, ProgressCb
from .settings import (
    OutputLocation,
    ProcessingOrder,
    UpscaleBackend,
    UpscaleSettings,
)


def _check_cancel(cancel) -> None:
    if cancel is not None and cancel.is_set():
        raise Cancelled()


def _output_base(job: Job, settings: UpscaleSettings) -> Path:
    if settings.output_location == OutputLocation.CUSTOM and settings.output_dir:
        base = Path(settings.output_dir)
    else:
        base = job.input_path.parent
    if settings.create_subfolder:
        base = base / settings.subfolder_name
    base.mkdir(parents=True, exist_ok=True)
    return base


def _unique(path: Path, overwrite: bool) -> Path:
    if overwrite or not path.exists():
        return path
    i = 1
    while True:
        cand = path.parent / f"{path.stem}({i}){path.suffix}"
        if not cand.exists():
            return cand
        i += 1


def _image_output(job: Job, settings: UpscaleSettings) -> Path:
    base = _output_base(job, settings)
    name = f"{job.input_path.stem}{settings.output_suffix()}.{settings.image_format}"
    return _unique(base / name, settings.overwrite)


def _video_output(job: Job, settings: UpscaleSettings) -> Path:
    base = _output_base(job, settings)
    name = f"{job.input_path.stem}{settings.output_suffix()}.{settings.video_format}"
    return _unique(base / name, settings.overwrite)


def _part_path(out: Path) -> Path:
    """最終出力と同フォルダ・同拡張子の一時ファイル名。

    途中で失敗/キャンセルしても壊れたファイルを正式名で残さないため、
    まずこの .part ファイルに書き、成功時だけ os.replace で確定する。
    （拡張子はそのまま保つ＝ffmpeg/realesrgan の形式判定が崩れない）
    """
    return out.with_name(f"{out.stem}.part{out.suffix}")


def process_job(job: Job, settings: UpscaleSettings,
                progress: ProgressCb | None = None, cancel=None) -> Path:
    """ジョブを処理して出力パスを返す。失敗時は例外、キャンセル時は Cancelled。"""
    progress = progress or (lambda f, m: None)
    if job.kind == JobKind.IMAGE:
        return _process_image(job, settings, progress, cancel)
    if job.kind == JobKind.FOLDER:
        return _process_folder(job, settings, progress, cancel)
    if job.kind == JobKind.VIDEO:
        return _process_video(job, settings, progress, cancel)
    raise ValueError(f"未知のジョブ種別: {job.kind}")


def _process_image(job, settings, progress, cancel) -> Path:
    if not settings.upscale_enabled:
        raise ValueError("画像にはアップスケーラーモデルを選択してください。")
    out = _image_output(job, settings)
    tmp = _part_path(out)
    progress(0.0, "アップスケール中…")
    try:
        upscaler.upscale_image(str(job.input_path), str(tmp), settings,
                               progress=progress, cancel=cancel)
        os.replace(tmp, out)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    progress(1.0, "完了")
    return out


def _process_folder(job, settings, progress, cancel) -> Path:
    if not settings.upscale_enabled:
        raise ValueError("画像フォルダにはアップスケーラーモデルを選択してください。")
    base = _output_base(job, settings)
    out_dir = base / f"{job.input_path.name}{settings.output_suffix()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    progress(0.0, "フォルダを一括アップスケール中…")
    upscaler.upscale_folder(str(job.input_path), str(out_dir), settings,
                            progress=progress, cancel=cancel)
    progress(1.0, "完了")
    return out_dir


def _process_video(job, settings, progress, cancel) -> Path:
    if not settings.upscale_enabled and not settings.interpolation_enabled:
        raise ValueError(
            "アップスケーラーモデルまたはフレーム補間モデルを選択してください。"
        )
    tmp = Path(tempfile.mkdtemp(prefix="ueu_"))
    src_frames = tmp / "src"
    interp_frames = tmp / "interp"
    up_frames = tmp / "up"
    src_frames.mkdir()
    try:
        _check_cancel(cancel)
        progress(0.01, "動画を解析中…")
        info = media.probe(str(job.input_path))
        fps = info.fps or job.fps or 30.0

        # RIFEはフレームファイルを前提にするため、補間が有効なジョブでは
        # 必ず従来のPNG経路を使う。RIFEなしの新AIヘルパーだけrawパイプへ進む。
        active_settings = settings
        if (
            settings.upscale_enabled
            and not settings.interpolation_enabled
            and settings.backend in {UpscaleBackend.WINML_GPU, UpscaleBackend.NPU_NATIVE}
        ):
            from . import helper_backend

            out = _video_output(job, settings)
            out_tmp = _part_path(out)
            try:
                video.upscale_video_piped(
                    str(job.input_path), str(out_tmp), settings,
                    progress=lambda f, m: progress(0.02 + 0.98 * f, m),
                    cancel=cancel,
                )
            except helper_backend.HelperBackendUnavailable as exc:
                out_tmp.unlink(missing_ok=True)
                active_settings = replace(settings, backend=UpscaleBackend.VULKAN)
                progress(0.02, f"AIヘルパーを起動できないためVulkanへ切替… ({exc})")
            except BaseException:
                out_tmp.unlink(missing_ok=True)
                raise
            else:
                os.replace(out_tmp, out)
                progress(1.0, "完了")
                return out

        progress(0.02, "フレームを抽出中…")
        video.extract_frames(
            str(job.input_path), str(src_frames),
            progress=lambda f, m: progress(0.02 + 0.18 * f, m or "フレーム抽出中…"),
            cancel=cancel,
        )

        current_frames = src_frames
        output_fps = fps

        def _run_interpolation(start: float, end: float) -> None:
            nonlocal current_frames, output_fps
            interp_frames.mkdir()
            progress(start, "RIFEでフレーム補間中…")
            _count, output_fps = interpolator.interpolate_folder(
                str(current_frames), str(interp_frames), settings, fps,
                progress=lambda f, m: progress(
                    start + (end - start) * f,
                    m or "RIFEでフレーム補間中…",
                ),
                cancel=cancel,
            )
            current_frames = interp_frames

        def _run_upscale(start: float, end: float) -> None:
            nonlocal current_frames
            up_frames.mkdir()
            # 中間フレームは常に PNG に固定する。バックエンドは選択どおり
            # （NPU選択時は動画フレームもNPUで処理する）。
            frame_settings = replace(
                active_settings,
                image_format="png",
            )
            progress(start, "フレームをアップスケール中…")
            upscaler.upscale_folder(
                str(current_frames), str(up_frames), frame_settings,
                progress=lambda f, m: progress(
                    start + (end - start) * f,
                    m or "アップスケール中…",
                ),
                cancel=cancel,
            )
            current_frames = up_frames

        if settings.upscale_enabled and settings.interpolation_enabled:
            # 既定はアプコン→補間: 重いESRGANの対象を補間前の元フレーム数に
            # 抑えられるため合計時間が短い。拡大後の高解像度フレームをRIFEに
            # 渡すとVRAM/RAM消費が増えるため、省メモリ順（補間→アプコン）も
            # processing_order で選択できる。
            if settings.processing_order == ProcessingOrder.INTERPOLATE_FIRST:
                _run_interpolation(0.20, 0.45)
                _run_upscale(0.45, 0.80)
            else:
                _run_upscale(0.20, 0.65)
                _run_interpolation(0.65, 0.80)
        elif settings.interpolation_enabled:
            _run_interpolation(0.20, 0.80)
        elif settings.upscale_enabled:
            _run_upscale(0.20, 0.80)

        out = _video_output(job, settings)
        out_tmp = _part_path(out)
        progress(0.80, "動画を再結合中…")
        try:
            video.reassemble(
                str(current_frames), str(job.input_path), str(out_tmp), output_fps, active_settings,
                progress=lambda f, m: progress(0.80 + 0.20 * f, m or "再結合中…"),
                cancel=cancel,
            )
            os.replace(out_tmp, out)
        except BaseException:
            out_tmp.unlink(missing_ok=True)
            raise
        progress(1.0, "完了")
        return out
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

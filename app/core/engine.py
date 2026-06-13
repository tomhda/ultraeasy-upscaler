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

from . import media, upscaler, video
from .jobs import Cancelled, Job, JobKind, ProgressCb
from .settings import OutputLocation, UpscaleSettings


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
    base = _output_base(job, settings)
    out_dir = base / f"{job.input_path.name}{settings.output_suffix()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    progress(0.0, "フォルダを一括アップスケール中…")
    upscaler.upscale_folder(str(job.input_path), str(out_dir), settings,
                            progress=progress, cancel=cancel)
    progress(1.0, "完了")
    return out_dir


def _process_video(job, settings, progress, cancel) -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="ueu_"))
    src_frames = tmp / "src"
    up_frames = tmp / "up"
    src_frames.mkdir()
    up_frames.mkdir()
    try:
        _check_cancel(cancel)
        progress(0.01, "動画を解析中…")
        info = media.probe(str(job.input_path))
        fps = info.fps or job.fps or 30.0

        progress(0.02, "フレームを抽出中…")
        video.extract_frames(
            str(job.input_path), str(src_frames),
            progress=lambda f, m: progress(0.02 + 0.18 * f, m or "フレーム抽出中…"),
            cancel=cancel,
        )

        # 中間フレームは常に PNG に固定する。
        # 出力形式設定が jpg/webp でも、再結合側は frame_%08d.png 固定で読むため。
        frame_settings = replace(settings, image_format="png")
        progress(0.20, "フレームをアップスケール中…")
        upscaler.upscale_folder(
            str(src_frames), str(up_frames), frame_settings,
            progress=lambda f, m: progress(0.20 + 0.60 * f, m or "アップスケール中…"),
            cancel=cancel,
        )

        out = _video_output(job, settings)
        out_tmp = _part_path(out)
        progress(0.80, "動画を再結合中…")
        try:
            video.reassemble(
                str(up_frames), str(job.input_path), str(out_tmp), fps, settings,
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

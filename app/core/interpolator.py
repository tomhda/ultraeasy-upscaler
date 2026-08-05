"""RIFE NCNN/Vulkan による連番PNGのフレーム補間。"""
from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

from PIL import Image

from . import binaries
from .jobs import Cancelled, ProgressCb
from .settings import UpscaleSettings
from .video import FRAME_GLOB, FRAME_PATTERN

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _terminate(proc: subprocess.Popen) -> None:
    try:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
    except Exception:
        pass


def _is_uhd(frames_dir: Path) -> bool:
    first = next(iter(sorted(frames_dir.glob(FRAME_GLOB))), None)
    if first is None:
        return False
    try:
        with Image.open(first) as image:
            return image.width * image.height >= 3840 * 2160
    except OSError:
        return False


def interpolate_folder(
    input_dir: str,
    output_dir: str,
    settings: UpscaleSettings,
    source_fps: float,
    progress: Optional[ProgressCb] = None,
    cancel=None,
) -> tuple[int, float]:
    """input_dir の連番PNGを補間し、(生成枚数, 実効fps) を返す。"""
    if not settings.interpolation_model:
        raise ValueError("フレーム補間モデルが選択されていません。")
    if source_fps <= 0:
        raise ValueError("元動画のfpsを取得できません。")

    source = Path(input_dir)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    input_count = sum(1 for _ in source.glob(FRAME_GLOB))
    if input_count < 2:
        raise ValueError("フレーム補間には2枚以上のフレームが必要です。")

    requested_fps = settings.target_fps or (source_fps * 2.0)
    if requested_fps <= source_fps:
        raise ValueError(
            f"補間後のfpsは元動画より大きい値にしてください（元: {source_fps:.3f}fps）。"
        )
    target_count = max(input_count + 1, round(input_count * requested_fps / source_fps))
    # 枚数の丸め後も元動画と尺を完全に一致させる。
    effective_fps = source_fps * target_count / input_count

    cmd = [
        binaries.rife_exe(),
        "-i", str(source),
        "-o", str(destination),
        "-n", str(target_count),
        "-m", str(binaries.interpolation_model_dir(settings.interpolation_model)),
        "-f", FRAME_PATTERN,
        "-j", "1:2:2",
    ]
    if settings.gpu_id >= 0:
        cmd += ["-g", str(settings.gpu_id)]
    if _is_uhd(source):
        cmd.append("-u")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=_NO_WINDOW,
    )
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []

    def _drain(stream, bucket: list[str]) -> None:
        try:
            for line in stream:
                bucket.append(line)
        except Exception:
            pass

    threads = [
        threading.Thread(target=_drain, args=(proc.stdout, stdout_lines), daemon=True),
        threading.Thread(target=_drain, args=(proc.stderr, stderr_lines), daemon=True),
    ]
    for thread in threads:
        thread.start()

    if progress:
        progress(0.0, "RIFEでフレーム補間中…")
    try:
        while proc.poll() is None:
            if cancel is not None and cancel.is_set():
                _terminate(proc)
                raise Cancelled()
            produced = sum(1 for _ in destination.glob(FRAME_GLOB))
            if progress:
                progress(min(0.999, produced / target_count), "RIFEでフレーム補間中…")
            time.sleep(0.25)
        ret = proc.wait()
    finally:
        for thread in threads:
            thread.join(timeout=2)

    if cancel is not None and cancel.is_set():
        raise Cancelled()
    if ret != 0:
        tail = (stdout_lines + stderr_lines)[-20:]
        raise RuntimeError("RIFEが失敗しました:\n" + "".join(tail).strip())

    produced = sum(1 for _ in destination.glob(FRAME_GLOB))
    if produced != target_count:
        raise RuntimeError(
            f"RIFEの生成枚数が一致しません（予定 {target_count} / 実際 {produced}）。"
        )
    if progress:
        progress(1.0, "フレーム補間完了")
    return produced, effective_fps

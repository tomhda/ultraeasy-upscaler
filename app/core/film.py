"""Windows ML/DirectML 版 FILM による連番 PNG のフレーム補間。"""
from __future__ import annotations

import math
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

from ..i18n import t
from . import binaries
from .jobs import Cancelled, ProgressCb
from .settings import UpscaleSettings
from .video import FRAME_GLOB


_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def resolve_factor(settings: UpscaleSettings, source_fps: float) -> int:
    """FILM の 0.5 中間フレームを再帰生成できる倍率に限定する。"""
    if source_fps <= 0:
        raise ValueError(t("元動画のfpsを取得できません。"))
    if settings.target_fps is None:
        # 倍率指定（2/4/8）はそのまま使う。範囲外の値は下の検査で落とす。
        if int(settings.interpolation_factor) in (2, 4, 8):
            return int(settings.interpolation_factor)
        requested = source_fps * float(settings.interpolation_factor)
    else:
        requested = float(settings.target_fps)
        for factor in (2, 4, 8):
            if math.isclose(requested, source_fps * factor, rel_tol=0, abs_tol=0.001):
                return factor
    # t() の第 1 引数名と重なるため、差し込みは取り出してから行う。
    template = t(
        "FILM (Style) は元動画の 2 倍・4 倍・8 倍の fps だけに対応します"
        "（元: {source}fps / 指定: {requested}fps）。"
    )
    raise ValueError(
        template.format(source=f"{source_fps:.3f}", requested=f"{requested:.3f}")
    )


def interpolate_folder(
    input_dir: str,
    output_dir: str,
    settings: UpscaleSettings,
    source_fps: float,
    progress: Optional[ProgressCb] = None,
    cancel=None,
) -> tuple[int, float]:
    """FILMヘルパーで補間し、(出力枚数, fps) を返す。"""
    source, destination = Path(input_dir), Path(output_dir)
    input_count = sum(1 for _ in source.glob(FRAME_GLOB))
    if input_count < 2:
        raise ValueError(t("フレーム補間には2枚以上のフレームが必要です。"))
    factor = resolve_factor(settings, source_fps)
    target_count = input_count * factor
    destination.mkdir(parents=True, exist_ok=True)
    cmd = [
        binaries.film_helper_exe(),
        "--model", str(binaries.film_model_path()),
        "--input-dir", str(source),
        "--output-dir", str(destination),
        "--factor", str(factor),
        "--ep-name", "DmlExecutionProvider",
    ]
    if settings.gpu_id >= 0:
        cmd += ["--device-index", str(settings.gpu_id)]
    message = t("FILM (Style) でフレーム補間中…")
    if progress:
        progress(0.0, message)
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace", creationflags=_NO_WINDOW,
    )
    output: list[str] = []

    def drain(stream) -> None:
        try:
            for line in stream:
                output.append(line)
        finally:
            stream.close()

    threads = [threading.Thread(target=drain, args=(stream,), daemon=True)
               for stream in (proc.stdout, proc.stderr)]
    for thread in threads:
        thread.start()
    try:
        while proc.poll() is None:
            if cancel is not None and cancel.is_set():
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                raise Cancelled()
            if progress:
                count = sum(1 for _ in destination.glob(FRAME_GLOB))
                progress(min(0.999, count / target_count), message)
            time.sleep(0.25)
        ret = proc.wait()
    finally:
        for thread in threads:
            thread.join(timeout=2)
    if cancel is not None and cancel.is_set():
        raise Cancelled()
    if ret != 0:
        raise RuntimeError(
            t(
                "FILM (Style) の処理に失敗しました:\n{detail}",
                detail="".join(output[-20:]).strip(),
            )
        )
    produced = sum(1 for _ in destination.glob(FRAME_GLOB))
    if produced != target_count:
        raise RuntimeError(
            t(
                "FILM (Style) が作ったフレームの数が合いません"
                "（予定 {expected} / 実際 {actual}）。",
                expected=target_count,
                actual=produced,
            )
        )
    if progress:
        progress(1.0, t("フレーム補間完了"))
    return produced, source_fps * factor

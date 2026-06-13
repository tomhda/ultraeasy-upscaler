"""動画パイプライン: フレーム抽出 / 再結合 / HWエンコード判定。

実装担当: 動画/ffmpeg エージェント。共通規約:
  - すべて binaries.ffmpeg_exe() を使う。
  - progress は progress(fraction 0..1, message) を呼ぶ。
  - cancel は threading.Event 互換。定期的に cancel.is_set() を確認し、True なら
    subprocess を terminate/kill して jobs.Cancelled を送出する。
"""
from __future__ import annotations

import re
import subprocess
import tempfile
import threading
from functools import lru_cache
from pathlib import Path
from typing import Optional

from . import binaries
from .jobs import Cancelled, ProgressCb
from .settings import UpscaleSettings

# 抽出フレームのファイル名パターン（抽出と再結合で必ず一致させる）
FRAME_PATTERN = "frame_%08d.png"
FRAME_GLOB = "frame_*.png"

# Windows でコンソールウィンドウを出さない
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

# 本機は AMD のため amf を優先。可搬性のため nvenc/qsv も検出候補に含める。
# (key, encoder_name) の順で優先する。
_H264_CANDIDATES = ["h264_amf", "h264_nvenc", "h264_qsv"]
_HEVC_CANDIDATES = ["hevc_amf", "hevc_nvenc", "hevc_qsv"]


@lru_cache(maxsize=None)
def _available_encoders() -> frozenset[str]:
    """`ffmpeg -encoders` を解析し、利用可能なエンコーダ名集合を返す。"""
    cmd = [binaries.ffmpeg_exe(), "-hide_banner", "-encoders"]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            encoding="utf-8", errors="replace",
            creationflags=_NO_WINDOW,
        )
    except OSError:
        return frozenset()
    names: set[str] = set()
    # 各行は " V....D h264_amf  AMD AMF ..." のような形式。2列目がエンコーダ名。
    for line in proc.stdout.splitlines():
        m = re.match(r"\s*[A-Z.]{6}\s+(\S+)", line)
        if m:
            names.add(m.group(1))
    return frozenset(names)


@lru_cache(maxsize=None)
def _encoder_works(encoder: str) -> bool:
    """エンコーダが実際に動作するか、極小クリップで検証する。

    `ffmpeg -encoders` に列挙されていても、対応 GPU やドライバ（nvcuda.dll 等）が
    無ければ実行時に失敗する。実機で数フレーム encode して確かめる。

    注意:
      - AMF は `-f null` 出力では Init に失敗するため、実コンテナ（mp4）へ書く。
      - 解像度が小さすぎると失敗する HW があるため 256x256 を使う。
    """
    tmp = Path(tempfile.gettempdir()) / f"ueu_enc_probe_{encoder}.mp4"
    cmd = [
        binaries.ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", "testsrc=size=256x256:rate=10:duration=0.3",
        "-c:v", encoder, "-pix_fmt", "yuv420p",
        str(tmp),
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            encoding="utf-8", errors="replace",
            creationflags=_NO_WINDOW,
        )
        return proc.returncode == 0
    except OSError:
        return False
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass


@lru_cache(maxsize=None)
def detect_hw_encoder(prefer: str = "auto") -> Optional[str]:
    """利用可能な HW エンコーダ名を返す（'hevc_amf' / 'h264_amf' / 'h264_nvenc' /
    'h264_qsv' 等）。無ければ None。`ffmpeg -encoders` を調べる。
    本機は AMD なので *_amf を優先する。

    prefer: "auto"（既定, h264系→hevc系）/ "h264" / "hevc" / "none"。
    """
    if prefer == "none":
        return None

    available = _available_encoders()

    if prefer == "hevc":
        order = _HEVC_CANDIDATES + _H264_CANDIDATES
    elif prefer == "h264":
        order = _H264_CANDIDATES + _HEVC_CANDIDATES
    else:  # auto: H.264 を優先（互換性が高い）
        order = _H264_CANDIDATES + _HEVC_CANDIDATES

    for enc in order:
        if enc not in available:
            continue
        # 列挙されていても GPU/ドライバ非対応なら実行時に失敗するため、
        # 全候補を実機で検証してから採用する（本機では h264_amf が選ばれる）。
        if not _encoder_works(enc):
            continue
        return enc
    return None


# --- progress 解析用 ---
# -progress pipe:1 は "frame=123" や "out_time_us=456" の key=value 行を出力する。
_PROGRESS_RE = re.compile(r"^(\w+)=(.*)$")


def _terminate(proc: subprocess.Popen) -> None:
    """ffmpeg を穏当に terminate し、効かなければ kill する。"""
    try:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
    except Exception:
        pass


def _run_with_progress(cmd: list[str], total_frames: Optional[int],
                       progress: Optional[ProgressCb], cancel,
                       message: str) -> None:
    """ffmpeg を `-progress pipe:1` 付きで実行し、進捗解析とキャンセルを行う。

    cmd には `-progress pipe:1 -nostats` を含めて渡すこと（stdout を解析する）。
    失敗時は RuntimeError、キャンセル時は jobs.Cancelled を送出する。
    """
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        creationflags=_NO_WINDOW,
    )

    # stderr は別スレッドで吸い出す。stdout(progress) だけを読むと、ffmpeg が
    # stderr に大量出力した際に Windows の pipe バッファが詰まりハングするため。
    stderr_lines: list[str] = []

    def _drain_stderr() -> None:
        try:
            assert proc.stderr is not None
            for el in proc.stderr:
                stderr_lines.append(el)
        except Exception:
            pass

    drainer = threading.Thread(target=_drain_stderr, daemon=True)
    drainer.start()

    last_fraction = 0.0
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            if cancel is not None and cancel.is_set():
                _terminate(proc)
                raise Cancelled()

            m = _PROGRESS_RE.match(line.strip())
            if not m:
                continue
            key, value = m.group(1), m.group(2)
            if key == "frame" and total_frames:
                n = _safe_int(value)
                if n is not None:
                    last_fraction = max(0.0, min(0.999, n / total_frames))
                    if progress:
                        progress(last_fraction, message)
            elif key == "progress" and value == "end":
                if progress:
                    progress(1.0, message)
    finally:
        # プロセスを回収し、stderr ドレインスレッドを合流させる
        ret = proc.wait()
        drainer.join(timeout=2)
        stderr = "".join(stderr_lines)

    if cancel is not None and cancel.is_set():
        raise Cancelled()
    if ret != 0:
        tail = (stderr or "").strip().splitlines()[-15:]
        raise RuntimeError(
            "ffmpeg が失敗しました (exit %d):\n%s" % (ret, "\n".join(tail))
        )


def _safe_int(value: str) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def extract_frames(video_path: str, out_dir: str,
                   progress: Optional[ProgressCb] = None, cancel=None) -> int:
    """動画を out_dir/FRAME_PATTERN へ連番 PNG 展開し、抽出フレーム数を返す。

    ffmpeg -i <video> <out_dir>/frame_%08d.png
    進捗は `-progress pipe:1` の frame= を probe の総フレーム数に対して算出する。
    """
    from . import media  # 遅延 import（循環回避）

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 総フレーム数（進捗算出用）。失敗しても抽出は続行する。
    total_frames: Optional[int] = None
    try:
        info = media.probe(video_path)
        total_frames = info.frame_count
    except Exception:
        total_frames = None

    out_pattern = str(out / FRAME_PATTERN)
    cmd = [
        binaries.ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
        "-i", video_path,
        out_pattern,
        "-progress", "pipe:1", "-nostats",
    ]

    if progress:
        progress(0.0, "フレーム抽出中…")
    _run_with_progress(cmd, total_frames, progress, cancel, "フレーム抽出中…")

    # 実際に書き出されたファイル数を数えて返す（これが信頼できる値）。
    count = sum(1 for _ in out.glob(FRAME_GLOB))
    if progress:
        progress(1.0, "フレーム抽出完了")
    return count


def _encoder_args(encoder: str, quality: int) -> list[str]:
    """エンコーダ種別ごとの -c:v と画質指定（CRF/QP 相当）を返す。"""
    args = ["-c:v", encoder]
    if encoder.endswith("_amf"):
        # AMD AMF: cqp レート制御で I/P/B フレームの QP を指定する（検証済み）。
        args += ["-rc", "cqp",
                 "-qp_i", str(quality), "-qp_p", str(quality),
                 "-qp_b", str(quality),
                 "-quality", "quality"]
    elif encoder.endswith("_nvenc"):
        # NVENC: 一定品質モード（CQ）。
        args += ["-rc", "constqp", "-qp", str(quality)]
    elif encoder.endswith("_qsv"):
        # Intel QSV: グローバル品質。
        args += ["-global_quality", str(quality)]
    else:  # libx264 等
        args += ["-crf", str(quality)]
    return args


def reassemble(frames_dir: str, audio_source: str, out_path: str, fps: float,
               settings: UpscaleSettings,
               progress: Optional[ProgressCb] = None, cancel=None) -> None:
    """frames_dir/FRAME_PATTERN を fps で動画化して out_path に書く。

    - settings.hw_encode かつ detect_hw_encoder() が非 None ならそれを使い、無ければ libx264。
    - settings.keep_audio かつ audio_source に音声があれば map（可能なら -c:a copy、
      不可なら aac 再エンコード）。
    - settings.video_quality を CRF/QP 相当に反映。出力コンテナは settings.video_format。
    - 既定で -pix_fmt yuv420p（再生互換性のため）。
    """
    from . import media  # 遅延 import（循環回避）

    frames = Path(frames_dir)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    in_pattern = str(frames / FRAME_PATTERN)
    total_frames = sum(1 for _ in frames.glob(FRAME_GLOB))

    # エンコーダ選択
    encoder = "libx264"
    if settings.hw_encode:
        hw = detect_hw_encoder()
        if hw:
            encoder = hw

    # 音声を含めるか判定（keep_audio かつ audio_source に音声があるとき）
    want_audio = False
    if settings.keep_audio and audio_source:
        try:
            want_audio = media.probe(audio_source).has_audio
        except Exception:
            want_audio = False

    fps_str = repr(float(fps))

    # 入力: フレーム連番。-framerate は入力フレームレート。
    cmd = [
        binaries.ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
        "-framerate", fps_str,
        "-i", in_pattern,
    ]
    if want_audio:
        cmd += ["-i", audio_source]

    # 映像エンコード設定
    cmd += _encoder_args(encoder, settings.video_quality)
    cmd += ["-pix_fmt", "yuv420p", "-r", fps_str]

    if want_audio:
        # 映像は入力0、音声は入力1からマップ。
        # 注: -shortest は image2 入力＋音声 copy だと末尾フレームを落とすため使わない
        #     （フレーム列を正としたいので映像を切り詰めない。A/V 長差は通常 1ms 未満）。
        cmd += ["-map", "0:v:0", "-map", "1:a:0", "-c:a", "copy"]

    cmd += [str(out), "-progress", "pipe:1", "-nostats"]

    if progress:
        progress(0.0, "動画を再結合中…")

    try:
        _run_with_progress(cmd, total_frames, progress, cancel, "再結合中…")
    except RuntimeError:
        # 音声 copy がコンテナ非互換なら aac 再エンコードで再試行する。
        if want_audio:
            cmd2 = [
                binaries.ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
                "-framerate", fps_str,
                "-i", in_pattern,
                "-i", audio_source,
            ]
            cmd2 += _encoder_args(encoder, settings.video_quality)
            cmd2 += ["-pix_fmt", "yuv420p", "-r", fps_str]
            cmd2 += ["-map", "0:v:0", "-map", "1:a:0",
                     "-c:a", "aac", "-b:a", "192k"]
            cmd2 += [str(out), "-progress", "pipe:1", "-nostats"]
            _run_with_progress(cmd2, total_frames, progress, cancel, "再結合中…")
        else:
            raise

    if progress:
        progress(1.0, "再結合完了")

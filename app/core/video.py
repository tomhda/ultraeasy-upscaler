"""動画パイプライン: フレーム抽出 / 再結合 / HWエンコード判定。

実装担当: 動画/ffmpeg エージェント。共通規約:
  - すべて binaries.ffmpeg_exe() を使う。
  - progress は progress(fraction 0..1, message) を呼ぶ。
  - cancel は threading.Event 互換。定期的に cancel.is_set() を確認し、True なら
    subprocess を terminate/kill して jobs.Cancelled を送出する。
"""
from __future__ import annotations

import hashlib
import json
import os
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np

from . import binaries
from .jobs import Cancelled, ProgressCb
from .settings import (
    HELPER_MODEL_SWINIR,
    UpscaleBackend,
    UpscaleSettings,
    canonical_helper_model,
)

# 抽出フレームのファイル名パターン（抽出と再結合で必ず一致させる）
FRAME_PATTERN = "frame_%08d.png"
FRAME_GLOB = "frame_*.png"

# Windows でコンソールウィンドウを出さない
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

# AMD専用機ではAMFを優先する。NVIDIAドライバが見える環境ではNVENCを優先し、
# ハイブリッド機（NVIDIA dGPU + AMD/Intel iGPU）でもdGPUを選びやすくする。
# いずれも実エンコード検証に通ったものだけ採用する。
_H264_CANDIDATES = ["h264_amf", "h264_nvenc", "h264_qsv"]
_HEVC_CANDIDATES = ["hevc_amf", "hevc_nvenc", "hevc_qsv"]

# H.264 の実用上限。GPUエンコーダだけでなく libx264 等にも同じガードを
# 適用し、入力経路による挙動差をなくす。
DEFAULT_MAX_VIDEO_DIM = (3840, 2160)
MAX_VIDEO_DIM_ENV = "UEU_MAX_VIDEO_DIM"

_VIDEO_DIM_SEPARATOR_RE = re.compile(r"\s*[xX×,:]\s*|\s+")
_VIDEO_COLOR_FILTER = (
    "setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709:range=tv,"
    "pad=ceil(iw/2)*2:ceil(ih/2)*2"
)


def _coerce_video_dim(value) -> tuple[int, int]:
    """設定値/環境変数の最大動画寸法を (幅, 高さ) に正規化する。"""
    if isinstance(value, str):
        parts = [part for part in _VIDEO_DIM_SEPARATOR_RE.split(value.strip()) if part]
        if len(parts) != 2:
            raise ValueError(f"動画の最大寸法は WIDTHxHEIGHT で指定してください: {value!r}")
        value = parts
    try:
        width, height = value
        width, height = int(width), int(height)
    except (TypeError, ValueError, IndexError):
        raise ValueError(f"動画の最大寸法は (幅, 高さ) で指定してください: {value!r}") from None
    if width < 2 or height < 2:
        raise ValueError(f"動画の最大寸法は2以上で指定してください: {width}x{height}")
    return width, height


def resolve_max_video_dim(settings: UpscaleSettings | None = None) -> tuple[int, int]:
    """動画の最大寸法を settings → 環境変数 → 既定値の順で解決する。"""
    configured = getattr(settings, "max_video_dim", None)
    if configured is None:
        configured = os.environ.get(MAX_VIDEO_DIM_ENV)
    if configured is None:
        return DEFAULT_MAX_VIDEO_DIM
    try:
        return _coerce_video_dim(configured)
    except ValueError:
        # 環境変数の誤記でジョブ自体を落とさず、安全側の既定上限を使う。
        return DEFAULT_MAX_VIDEO_DIM


def _even_up(value: int) -> int:
    return (value + 1) // 2 * 2


def _even_down(value: float) -> int:
    return max(2, int(value) // 2 * 2)


def fit_video_dimensions(
    width: int,
    height: int,
    max_dim: tuple[int, int] = DEFAULT_MAX_VIDEO_DIM,
) -> tuple[int, int]:
    """アスペクト比を維持して最大寸法内へ収めた動画寸法を返す。

    上限超過時は、上限を超えないように制限側を偶数へ切り下げ、もう一方も
    同じ比率から偶数へ切り下げる。上限内なら入力寸法をそのまま返す。
    """
    width, height = int(width), int(height)
    if width <= 0 or height <= 0:
        raise ValueError(f"動画の寸法が不正です: {width}x{height}")
    max_width, max_height = _coerce_video_dim(max_dim)
    max_width = max(2, max_width // 2 * 2)
    max_height = max(2, max_height // 2 * 2)

    # 既存の pad フィルタが偶数へ切り上げるため、実エンコーダ入力寸法で
    # 上限判定する。これにより奇数のカスタム上限も超えない。
    encoded_width = _even_up(width)
    encoded_height = _even_up(height)
    if encoded_width <= max_width and encoded_height <= max_height:
        return width, height

    if max_width / width <= max_height / height:
        target_width = max_width
        target_height = _even_down(target_width * height / width)
    else:
        target_height = max_height
        target_width = _even_down(target_height * width / height)

    # 浮動小数点の境界誤差があっても上限を超えないように最後に再確認する。
    target_width = min(target_width, max_width)
    target_height = min(target_height, max_height)
    return target_width, target_height


def _video_filter_for_dimensions(
    width: int,
    height: int,
    max_dim: tuple[int, int],
) -> tuple[str, tuple[int, int] | None]:
    """入力寸法に必要なフィルタと、適用した出力寸法を返す。"""
    fitted = fit_video_dimensions(width, height, max_dim)
    max_width, max_height = _coerce_video_dim(max_dim)
    needs_fit = _even_up(width) > max(2, max_width // 2 * 2) or _even_up(height) > max(2, max_height // 2 * 2)
    filters: list[str] = []
    if needs_fit:
        filters.append(f"scale={fitted[0]}:{fitted[1]}:flags=lanczos")
        return ",".join((*filters, _VIDEO_COLOR_FILTER)), fitted
    return _VIDEO_COLOR_FILTER, None


def _fit_progress_message(dim: tuple[int, int] | None) -> str | None:
    if dim is None:
        return None
    return f"出力を{dim[0]}x{dim[1]}へ縮小（H.264上限のため）"


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
    AMD専用機では *_amf、`nvidia-smi` がPATH上にある環境では *_nvenc を優先する。

    prefer: "auto"（既定, h264系→hevc系）/ "h264" / "hevc" / "none"。
    """
    if prefer == "none":
        return None

    available = _available_encoders()

    if prefer == "hevc":
        order = _encoder_candidates(_HEVC_CANDIDATES)
    elif prefer == "h264":
        order = _encoder_candidates(_H264_CANDIDATES)
    else:  # auto: Windows標準再生に必要なH.264だけを選ぶ
        order = _encoder_candidates(_H264_CANDIDATES)

    for enc in order:
        if enc not in available:
            continue
        # 列挙されていても GPU/ドライバ非対応なら実行時に失敗するため、
        # 全候補を実機で検証してから採用する（本機では h264_amf が選ばれる）。
        if not _encoder_works(enc):
            continue
        return enc
    return None


def _encoder_candidates(candidates: list[str]) -> list[str]:
    """GPU構成に合わせてハードウェアエンコーダの優先順を返す。"""
    if shutil.which("nvidia-smi") or shutil.which("nvidia-smi.exe"):
        nvenc = [name for name in candidates if name.endswith("_nvenc")]
        other = [name for name in candidates if not name.endswith("_nvenc")]
        return [*nvenc, *other]
    return candidates


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
    info = None
    try:
        info = media.probe(video_path)
        total_frames = info.frame_count
    except Exception:
        total_frames = None
    if info is not None and info.color_transfer in {"smpte2084", "arib-std-b67"}:
        raise ValueError("HDR動画（PQ/HLG）は未対応です。SDRに変換してから処理してください")

    out_pattern = str(out / FRAME_PATTERN)
    cmd = [
        binaries.ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
        "-i", video_path,
        "-fps_mode", "cfr", "-r", repr(float(info.fps if info and info.fps else 30.0)),
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

    # PNG経路では、先頭フレームの実寸がAIアップスケール後の出力寸法。
    # ここで先に判定しておかないと、AMFがscale前の巨大フレームで初期化される。
    frame_width = frame_height = 0
    first_frame = next(iter(sorted(frames.glob(FRAME_GLOB))), None)
    if first_frame is not None:
        try:
            frame_info = media.probe(str(first_frame))
            frame_width, frame_height = frame_info.width, frame_info.height
        except Exception:
            frame_width = frame_height = 0
    max_video_dim = resolve_max_video_dim(settings)
    video_filter = _VIDEO_COLOR_FILTER
    fit_dim: tuple[int, int] | None = None
    if frame_width > 0 and frame_height > 0:
        video_filter, fit_dim = _video_filter_for_dimensions(
            frame_width, frame_height, max_video_dim
        )

    # エンコーダ選択
    encoder = "libx264"
    if settings.hw_encode:
        hw = detect_hw_encoder("h264")
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

    # 映像エンコード設定。元実装と同じBT.709/TVレンジを明示する。
    cmd += ["-vf", video_filter]
    cmd += _encoder_args(encoder, settings.video_quality)
    cmd += [
        "-pix_fmt", "yuv420p", "-r", fps_str,
        "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
    ]

    if want_audio:
        # 映像は入力0、音声は入力1からマップ。
        # 注: -shortest は image2 入力＋音声 copy だと末尾フレームを落とすため使わない
        #     （フレーム列を正としたいので映像を切り詰めない。A/V 長差は通常 1ms 未満）。
        cmd += ["-map", "0:v:0", "-map", "1:a:0", "-c:a", "copy"]

    cmd += [str(out), "-progress", "pipe:1", "-nostats"]

    if progress:
        progress(0.0, _fit_progress_message(fit_dim) or "動画を再結合中…")

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
            cmd2 += ["-vf", video_filter]
            cmd2 += _encoder_args(encoder, settings.video_quality)
            cmd2 += [
                "-pix_fmt", "yuv420p", "-r", fps_str,
                "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
            ]
            cmd2 += ["-map", "0:v:0", "-map", "1:a:0",
                     "-c:a", "aac", "-b:a", "192k"]
            cmd2 += [str(out), "-progress", "pipe:1", "-nostats"]
            _run_with_progress(cmd2, total_frames, progress, cancel, "再結合中…")
        else:
            raise

    if progress:
        progress(1.0, "再結合完了")


SWINIR_PYTHON_ENV = "UEU_SWINIR_PYTHON"
SWINIR_MODEL_ENV = "UEU_SWINIR_MODEL"
SWINIR_CHUNK_FRAMES_ENV = "UEU_SWINIR_CHUNK_FRAMES"
SWINIR_DEFAULT_CHUNK_FRAMES = 150
SWINIR_MODEL_NAME = "003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth"


def _swinir_file_sha256(path: Path, cancel=None) -> str:
    """再開対象の内容を厳密に照合するため、ファイル全体をハッシュする。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _swinir_paths() -> tuple[Path, Path, Path]:
    """SwinIR CUDA worker, script, and weight pathsを解決する。"""
    root = binaries.repo_root()
    python = Path(os.environ.get(
        SWINIR_PYTHON_ENV,
        str(root / "tmp" / "swinir-venv" / "Scripts" / "python.exe"),
    )).expanduser()
    model = Path(os.environ.get(
        SWINIR_MODEL_ENV,
        str(root / "tmp" / "swinir-models" / SWINIR_MODEL_NAME),
    )).expanduser()
    script = root / "tools" / "swinir" / "worker.py"
    if not python.is_file():
        raise RuntimeError(f"SwinIR CUDA用Pythonが見つかりません: {python}")
    if not script.is_file():
        raise RuntimeError(f"SwinIR CUDAワーカーが見つかりません: {script}")
    if not model.is_file():
        raise RuntimeError(f"SwinIR CUDAの重みが見つかりません: {model}")
    return python.resolve(), script.resolve(), model.resolve()


def _swinir_identity(
    in_path: str,
    settings: UpscaleSettings,
    video_encoder: str,
    cancel=None,
) -> dict[str, object]:
    """再開可否を決める入力・設定・モデルの同一性情報を作る。"""
    source = Path(in_path).resolve()
    stat = source.stat()
    _python, script, model = _swinir_paths()

    def file_identity(path: Path) -> dict[str, object]:
        try:
            item = path.stat()
        except OSError:
            return {"path": str(path), "missing": True}
        return {
            "path": str(path), "size": item.st_size,
            "mtime_ns": item.st_mtime_ns,
            "sha256": _swinir_file_sha256(path, cancel),
        }

    def normalize(value):
        # JSON再読込後はtupleがlistになるため、保存前から同じ形へ揃える。
        if hasattr(value, "value"):
            return normalize(value.value)
        if isinstance(value, dict):
            return {str(key): normalize(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [normalize(item) for item in value]
        return value

    normalized = normalize(asdict(settings))
    return {
        "input": {
            "path": str(source), "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": _swinir_file_sha256(source, cancel),
        },
        "settings": normalized,
        "video_encoder": video_encoder,
        "model": file_identity(model),
        "worker": file_identity(script),
    }


def swinir_workdir(in_path: str, out_path: str, settings: UpscaleSettings) -> Path:
    """SwinIRの再開用作業ディレクトリを返す（生成・削除は呼び出し側）。"""
    del settings
    # 内容の厳密なSHA-256検証は処理側で一度だけ行う。ここでは長尺入力を
    # 二重に全読み込みしないよう、安定した入出力パスだけで作業名を決める。
    payload = json.dumps(
        {
            "input": str(Path(in_path).resolve()),
            "output": str(Path(out_path).resolve()),
        },
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    out = Path(out_path).resolve()
    return out.parent / f".{out.name}.swinir-work-{digest}"


def _atomic_json(path: Path, value: object) -> None:
    """manifestを同一ディレクトリ内でfsyncして原子的に更新する。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    except BaseException:
        try:
            os.unlink(name)
        except OSError:
            pass
        raise


def _swinir_chunk_size() -> int:
    value = os.environ.get(SWINIR_CHUNK_FRAMES_ENV)
    if value is None:
        return SWINIR_DEFAULT_CHUNK_FRAMES
    try:
        count = int(value)
    except ValueError:
        return SWINIR_DEFAULT_CHUNK_FRAMES
    return max(100, min(300, count))


def _run_capture_cancelable(cmd: list[str], cancel) -> tuple[int, str, str]:
    """stdout/stderrを詰まらせず、キャンセル可能な形でコマンドを待つ。"""
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=_NO_WINDOW,
    )
    try:
        while True:
            try:
                stdout, stderr = proc.communicate(timeout=0.2)
                return proc.returncode, stdout or "", stderr or ""
            except subprocess.TimeoutExpired:
                if cancel is not None and cancel.is_set():
                    _terminate(proc)
                    raise Cancelled()
    finally:
        if proc.poll() is None:
            _terminate(proc)


def _swinir_count_cfr_frames(in_path: str, fps: float, cancel) -> int:
    """本処理と同じCFR変換を空出力し、実際のrawフレーム数を先に確定する。"""
    cmd = [
        binaries.ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-nostdin",
        "-progress", "pipe:1", "-nostats",
        "-i", in_path, "-map", "0:v:0", "-fps_mode", "cfr",
        "-r", repr(float(fps)), "-f", "null", os.devnull,
    ]
    returncode, stdout, stderr = _run_capture_cancelable(cmd, cancel)
    if returncode != 0:
        raise RuntimeError("SwinIR動画のフレーム数確認に失敗しました:\n" + stderr)
    frames = 0
    for line in stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator and key == "frame":
            try:
                frames = max(frames, int(value))
            except ValueError:
                pass
    if frames <= 0:
        raise RuntimeError("SwinIR動画のCFR変換後フレーム数を取得できません")
    return frames


def _drain_binary_lines(stream, target: list[str]) -> None:
    """ffmpegのstderrを処理中から排出し、OSパイプの上限で止まるのを防ぐ。"""
    if stream is None:
        return
    try:
        for raw in iter(stream.readline, b""):
            target.append(raw.decode("utf-8", errors="replace").rstrip())
    except Exception:
        pass


def _swinir_session(
    settings: UpscaleSettings,
    width: int,
    height: int,
    progress: ProgressCb,
    cancel,
):
    """共通helper解決を使ってPyTorch CUDA workerを常駐起動する。"""
    from . import helper_backend

    return helper_backend.open_session(settings, width, height, progress, cancel)


def _swinir_open_decoder(in_path: str, fps: float):
    """動画を一度だけraw RGB24へ展開するdecoderを起動する。"""
    cmd = [
        binaries.ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-nostdin",
        "-i", in_path, "-map", "0:v:0", "-fps_mode", "cfr",
        "-r", repr(float(fps)),
        "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ]
    return subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL, bufsize=0, creationflags=_NO_WINDOW,
    )


def _swinir_discard_frames(decoder, frame_bytes: int, count: int, cancel) -> None:
    """既に完了済みのチャンクをAI処理せずdecoderから読み捨てる。"""
    assert decoder.stdout is not None
    for _ in range(count):
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        if _read_raw_frame(decoder.stdout, frame_bytes) is None:
            raise RuntimeError("SwinIR動画decoderのフレーム数が不足しています")


def _swinir_encode_chunk(
    path: Path,
    width: int,
    height: int,
    fps: float,
    settings: UpscaleSettings,
    video_encoder: str,
    video_filter: str,
):
    """H.264チャンクのencoderを起動する。チャンクごとに一度だけencodeする。"""
    command = [
        binaries.ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s:v", f"{width}x{height}",
        "-r", repr(float(fps)), "-i", "pipe:0", "-vf", video_filter,
        *_encoder_args(video_encoder, settings.video_quality), "-pix_fmt", "yuv420p",
        "-r", repr(float(fps)), "-an", "-reset_timestamps", "1",
        "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
        str(path),
    ]
    return subprocess.Popen(
        command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE, bufsize=0, creationflags=_NO_WINDOW,
    )


def _swinir_run_chunk(
    in_path: str,
    chunk_path: Path,
    start: int,
    count: int,
    width: int,
    height: int,
    out_width: int,
    out_height: int,
    fps: float,
    settings: UpscaleSettings,
    video_encoder: str,
    video_filter: str,
    session,
    progress: ProgressCb,
    processed_before: int,
    total_frames: int,
    cancel,
    *,
    decoder_proc=None,
) -> int:
    """1チャンクを処理し、成功時だけH.264ファイルを残す。"""
    del in_path, start
    if decoder_proc is None:
        raise ValueError("SwinIRチャンク処理には共有decoderが必要です")
    # 拡張子はffmpegの出力形式判定に使われるため、.tmpを拡張子の前へ置く。
    tmp = chunk_path.with_name(f".{chunk_path.stem}.tmp{chunk_path.suffix}")
    tmp.unlink(missing_ok=True)
    decoder = decoder_proc
    encoder = None
    decoded = encoded = 0
    encoder_stderr: list[str] = []
    encoder_drain = None
    try:
        frame_bytes = width * height * 3
        encoder = _swinir_encode_chunk(
            tmp,
            out_width,
            out_height,
            fps,
            settings,
            video_encoder,
            video_filter,
        )
        assert decoder.stdout is not None and encoder.stdin is not None
        encoder_drain = threading.Thread(
            target=_drain_binary_lines,
            args=(encoder.stderr, encoder_stderr),
            name="ueu-swinir-encode-stderr",
            daemon=True,
        )
        encoder_drain.start()
        while decoded < count:
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            raw = _read_raw_frame(decoder.stdout, frame_bytes)
            if raw is None:
                break
            image = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3).copy()
            output = session.upscale(image, cancel=cancel)
            encoder.stdin.write(np.ascontiguousarray(output).tobytes())
            decoded += 1
            encoded += 1
            progress(
                0.97 * (processed_before + encoded) / total_frames,
                f"SwinIR動画 {processed_before + encoded}/{total_frames}フレーム",
            )
        encoder.stdin.close()
        encoder_ret = encoder.wait()
        encoder_drain.join(timeout=2.0)
        if decoded != count:
            raise RuntimeError(f"SwinIRチャンクのフレーム数が不足しています: {decoded}/{count}")
        if encoder_ret != 0:
            raise RuntimeError(
                "SwinIRチャンクのffmpeg処理に失敗しました:\n"
                + "\n".join(encoder_stderr[-15:])
            )
        if not tmp.is_file() or tmp.stat().st_size == 0:
            raise RuntimeError("SwinIRチャンクが生成されませんでした")
        os.replace(tmp, chunk_path)
        return encoded
    except BaseException:
        if encoder is not None and encoder.poll() is None:
            _terminate(encoder)
        if encoder_drain is not None:
            encoder_drain.join(timeout=2.0)
        tmp.unlink(missing_ok=True)
        raise


def _swinir_concat_mux(
    chunks: list[Path],
    source: str,
    out_path: str,
    settings: UpscaleSettings,
    cancel=None,
) -> None:
    """H.264映像をstream copyし、元音声だけAACへ再エンコードする。"""
    list_path = chunks[0].parent / "concat.txt"
    list_path.write_text(
        "".join(f"file '{path.as_posix().replace(chr(39), chr(39) + chr(92) + chr(39) + chr(39))}'\n" for path in chunks),
        encoding="utf-8",
    )
    cmd = [
        binaries.ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
        "-f", "concat", "-safe", "0", "-i", str(list_path), "-i", source,
        "-map", "0:v:0",
    ]
    if settings.keep_audio:
        cmd += ["-map", "1:a:0?", "-c:a", "aac", "-b:a", "192k"]
    else:
        cmd += ["-an"]
    cmd += ["-c:v", "copy"]
    if Path(out_path).suffix.lower() in {".mp4", ".mov"}:
        cmd += ["-movflags", "+faststart"]
    cmd += [str(out_path)]
    try:
        returncode, _stdout, stderr = _run_capture_cancelable(cmd, cancel)
    finally:
        list_path.unlink(missing_ok=True)
    if returncode != 0:
        raise RuntimeError("SwinIRチャンクの結合に失敗しました:\n" + stderr)
    if not Path(out_path).is_file() or Path(out_path).stat().st_size == 0:
        raise RuntimeError(f"SwinIR動画出力が生成されませんでした: {out_path}")


def upscale_video_swinir_chunked(
    in_path: str,
    out_path: str,
    settings: UpscaleSettings,
    work_dir: str | Path | None = None,
    progress: Optional[ProgressCb] = None,
    cancel=None,
) -> Path:
    """SwinIR CUDA動画をH.264チャンクへ保存し、manifestから再開する。"""
    progress = progress or (lambda _fraction, _message: None)
    if settings.backend != UpscaleBackend.SWINIR_CUDA:
        raise ValueError("SwinIR CUDAチャンク経路にはSWINIR_CUDA backendが必要です")
    if canonical_helper_model(settings.model) != HELPER_MODEL_SWINIR:
        raise ValueError("SwinIR CUDA経路ではSwinIR-Mモデルを選択してください")
    from . import media
    info = media.probe(in_path)
    if info.color_transfer in {"smpte2084", "arib-std-b67"}:
        raise ValueError("HDR動画（PQ/HLG）は未対応です。SDRに変換してから処理してください")
    width, height = info.display_width, info.display_height
    fps = info.fps or 30.0
    if width <= 0 or height <= 0:
        raise RuntimeError("SwinIR動画の解像度を取得できません")
    work = Path(work_dir) if work_dir is not None else swinir_workdir(in_path, out_path, settings)
    work.mkdir(parents=True, exist_ok=True)
    progress(0.0, "SwinIR動画の再開データを確認中…")
    video_encoder = detect_hw_encoder("h264") if settings.hw_encode else None
    video_encoder = video_encoder or "libx264"
    identity = _swinir_identity(in_path, settings, video_encoder, cancel)
    manifest_path = work / "manifest.json"
    manifest: dict[str, object]
    if manifest_path.is_file():
        try:
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            loaded = None
        if isinstance(loaded, dict) and loaded.get("identity") == identity:
            manifest = loaded
        else:
            # 旧設定のチャンクは同じwork_dirへ持ち込まず、新manifestで無効化する。
            manifest = {"version": 2, "identity": identity, "chunks": []}
    else:
        manifest = {"version": 2, "identity": identity, "chunks": []}

    cached_total = manifest.get("cfr_frame_count")
    if isinstance(cached_total, int) and cached_total > 0:
        total = cached_total
    else:
        progress(0.0, "SwinIR動画のフレーム数を確認中…")
        total = _swinir_count_cfr_frames(in_path, fps, cancel)
    manifest["identity"] = identity
    manifest["version"] = 2
    manifest["cfr_frame_count"] = total
    manifest["total_frames"] = total
    manifest["fps"] = fps
    manifest["width"] = width
    manifest["height"] = height
    _atomic_json(manifest_path, manifest)

    chunk_size = _swinir_chunk_size()
    chunk_total = (total + chunk_size - 1) // chunk_size
    records = manifest.get("chunks")
    record_map = {int(item["index"]): item for item in records if isinstance(item, dict) and "index" in item} if isinstance(records, list) else {}
    chunks = [work / f"chunk_{index:06d}.mp4" for index in range(chunk_total)]

    def valid_chunk(index: int) -> bool:
        record = record_map.get(index)
        path = chunks[index]
        if not isinstance(record, dict) or not path.is_file():
            return False
        stat = path.stat()
        return (
            record.get("start") == index * chunk_size
            and record.get("count") == min(chunk_size, total - index * chunk_size)
            and record.get("size") == stat.st_size
            and isinstance(record.get("sha256"), str)
            and record["sha256"] == _swinir_file_sha256(path, cancel)
        )

    valid = {index for index in range(chunk_total) if valid_chunk(index)}
    pending = [index for index in range(chunk_total) if index not in valid]
    session = None
    decoder = None
    decoder_stderr: list[str] = []
    decoder_drain = None
    try:
        if pending:
            session = _swinir_session(settings, width, height, progress, cancel)
            out_width, out_height = width * int(session.scale), height * int(session.scale)
            max_dim = resolve_max_video_dim(settings)
            video_filter, _fit_dim = _video_filter_for_dimensions(out_width, out_height, max_dim)
            decoder = _swinir_open_decoder(in_path, fps)
            if decoder is not None:
                decoder_drain = threading.Thread(
                    target=_drain_binary_lines,
                    args=(decoder.stderr, decoder_stderr),
                    name="ueu-swinir-decode-stderr",
                    daemon=True,
                )
                decoder_drain.start()
            frame_bytes = width * height * 3
            cursor = 0
            for index in range(chunk_total):
                if cancel is not None and cancel.is_set():
                    raise Cancelled()
                start = cursor
                count = min(chunk_size, total - start)
                # decoderは全体で1本だけ維持する。完了済みチャンクは入力だけ
                # 読み捨て、未完了チャンクのみAIとH.264 encodeを実行する。
                if index in valid:
                    _swinir_discard_frames(decoder, frame_bytes, count, cancel)
                    cursor += count
                    continue
                _swinir_run_chunk(
                    in_path, chunks[index], start, count, width, height,
                    out_width, out_height, fps, settings, video_encoder,
                    video_filter, session,
                    progress, start, total, cancel, decoder_proc=decoder,
                )
                cursor += count
                record_map[index] = {
                    "index": index,
                    "start": start,
                    "count": count,
                    "path": chunks[index].name,
                    "size": chunks[index].stat().st_size,
                    "sha256": _swinir_file_sha256(chunks[index], cancel),
                }
                manifest["chunks"] = [record_map[key] for key in sorted(record_map)]
                _atomic_json(manifest_path, manifest)
            if decoder is not None:
                # 事前確認と本番decoderは同じCFR条件なので、余剰は入力変化か
                # ffmpeg条件の不一致を示す。黙って切り捨てずチェックポイントを守る。
                assert decoder.stdout is not None
                extra_frames = 0
                while _read_raw_frame(decoder.stdout, frame_bytes) is not None:
                    if cancel is not None and cancel.is_set():
                        raise Cancelled()
                    extra_frames += 1
                decoder_ret = decoder.wait()
                if decoder_drain is not None:
                    decoder_drain.join(timeout=2.0)
                if decoder_ret != 0:
                    raise RuntimeError(
                        "SwinIR動画のデコードに失敗しました:\n"
                        + "\n".join(decoder_stderr[-15:])
                    )
                if extra_frames:
                    raise RuntimeError(
                        "SwinIR動画のフレーム数が事前確認後に変化しました: "
                        f"{total}+{extra_frames}"
                    )
        progress(0.98, "SwinIR動画のチャンクを結合中…")
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        _swinir_concat_mux(chunks, in_path, out_path, settings, cancel=cancel)
        progress(1.0, "完了")
        return Path(out_path)
    finally:
        if decoder is not None and decoder.poll() is None:
            _terminate(decoder)
        if decoder_drain is not None:
            decoder_drain.join(timeout=2.0)
        if session is not None:
            session.close(force=cancel is not None and cancel.is_set())


def _read_raw_frame(stream, size: int) -> bytes | None:
    """rawvideoから1フレームを厳密に読む。正常EOFはNone。"""
    buf = bytearray()
    while len(buf) < size:
        chunk = stream.read(size - len(buf))
        if not chunk:
            if not buf:
                return None
            raise EOFError(f"rawvideo frame ended early ({len(buf)}/{size} bytes)")
        buf += chunk
    return bytes(buf)


def _ffmpeg_pipeline_error(
    label: str,
    stderr_lines: list[str],
    *,
    cause: BaseException | None = None,
    returncode: int | None = None,
) -> RuntimeError:
    """パイプ経路の失敗をstderr優先で利用者向け例外へ変換する。"""
    tail = "\n".join(line for line in stderr_lines[-15:] if line)
    if tail:
        return RuntimeError(f"{label}に失敗しました:\n{tail}")
    if isinstance(cause, (BrokenPipeError, OSError, EOFError)):
        return RuntimeError(
            f"{label}に失敗しました（ffmpegとのパイプが閉じられました。"
            "ffmpegのstderrは取得できませんでした）"
        )
    if returncode is not None:
        return RuntimeError(
            f"{label}に失敗しました（ffmpeg exit {returncode}、stderrなし）"
        )
    return RuntimeError(f"{label}に失敗しました")


def upscale_video_piped(
    in_path: str,
    out_path: str,
    settings: UpscaleSettings,
    progress: Optional[ProgressCb] = None,
    cancel=None,
) -> None:
    """ffmpeg rawvideo → UEU helper → ffmpegをPNG無しで直結する。"""
    import numpy as np

    from . import helper_backend, media

    progress = progress or (lambda _fraction, _message: None)
    info = media.probe(in_path)
    if info.color_transfer in {"smpte2084", "arib-std-b67"}:
        raise ValueError("HDR動画（PQ/HLG）は未対応です。SDRに変換してから処理してください")
    width, height = info.display_width, info.display_height
    fps = info.fps or 30.0
    total_frames = info.frame_count
    if width <= 0 or height <= 0:
        raise RuntimeError(f"動画の解像度を取得できません: {in_path}")

    progress(0.0, "AI準備中…")
    session = helper_backend.open_session(settings, width, height, progress, cancel)
    out_width = width * session.scale
    out_height = height * session.scale
    max_video_dim = resolve_max_video_dim(settings)
    video_filter, fit_dim = _video_filter_for_dimensions(
        out_width, out_height, max_video_dim
    )
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fit_message = _fit_progress_message(fit_dim)
    if fit_message:
        progress(0.0, fit_message)

    decoder_cmd = [
        binaries.ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-nostdin",
        "-i", in_path, "-map", "0:v:0", "-fps_mode", "cfr", "-r", repr(float(fps)), "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ]

    encoder = "libx264"
    if settings.hw_encode:
        encoder = detect_hw_encoder("h264") or "libx264"
    fps_str = repr(float(fps))
    encoder_cmd = [
        binaries.ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s:v", f"{out_width}x{out_height}", "-r", fps_str, "-i", "pipe:0",
    ]
    want_audio = bool(settings.keep_audio and info.has_audio)
    if want_audio:
        encoder_cmd += ["-i", in_path]
    encoder_cmd += ["-vf", video_filter]
    encoder_cmd += _encoder_args(encoder, settings.video_quality)
    encoder_cmd += [
        "-pix_fmt", "yuv420p", "-r", fps_str,
        "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
    ]
    if want_audio:
        # pipe処理は再試行できないため、コンテナ互換性が安定するAACでmuxする。
        encoder_cmd += ["-map", "0:v:0", "-map", "1:a:0", "-c:a", "aac", "-b:a", "192k"]
    else:
        encoder_cmd += ["-map", "0:v:0"]
    encoder_cmd += [str(out)]

    decoder: subprocess.Popen | None = None
    encoder_proc: subprocess.Popen | None = None
    errors: list[tuple[str, BaseException]] = []
    error_lock = threading.Lock()
    stop = threading.Event()
    decoded: queue.Queue[object] = queue.Queue(maxsize=2)
    upscaled: queue.Queue[object] = queue.Queue(maxsize=2)
    sentinel = object()
    decoder_stderr: list[str] = []
    encoder_stderr: list[str] = []

    def _fail(stage: str, exc: BaseException) -> None:
        with error_lock:
            if not errors:
                errors.append((stage, exc))
        stop.set()

    def _queue_put(target: queue.Queue[object], value: object) -> bool:
        while not stop.is_set():
            try:
                target.put(value, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def _drain(stream, target: list[str]) -> None:
        try:
            for raw in iter(stream.readline, b""):
                target.append(raw.decode("utf-8", errors="replace").rstrip())
        except Exception:
            pass

    try:
        decoder = subprocess.Popen(
            decoder_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            bufsize=0,
            creationflags=_NO_WINDOW,
        )
        encoder_proc = subprocess.Popen(
            encoder_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            bufsize=0,
            creationflags=_NO_WINDOW,
        )
        assert decoder.stdout is not None and decoder.stderr is not None
        assert encoder_proc.stdin is not None and encoder_proc.stderr is not None

        decoder_drain = threading.Thread(
            target=_drain, args=(decoder.stderr, decoder_stderr), name="ueu-decode-stderr", daemon=True
        )
        encoder_drain = threading.Thread(
            target=_drain, args=(encoder_proc.stderr, encoder_stderr), name="ueu-encode-stderr", daemon=True
        )
        decoder_drain.start()
        encoder_drain.start()
        frame_bytes = width * height * 3

        def _decode() -> None:
            try:
                while not stop.is_set():
                    raw = _read_raw_frame(decoder.stdout, frame_bytes)
                    if raw is None:
                        _queue_put(decoded, sentinel)
                        return
                    frame = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3).copy()
                    if not _queue_put(decoded, frame):
                        return
            except BaseException as exc:
                _fail("decoder", exc)

        def _infer() -> None:
            try:
                while not stop.is_set():
                    try:
                        item = decoded.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    if item is sentinel:
                        _queue_put(upscaled, sentinel)
                        return
                    result = (session.upscale(item, cancel=cancel) if cancel is not None else session.upscale(item))  # type: ignore[arg-type]
                    if not _queue_put(upscaled, result):
                        return
            except BaseException as exc:
                _fail("inference", exc)

        processed = 0

        def _encode() -> None:
            nonlocal processed
            try:
                while not stop.is_set():
                    try:
                        item = upscaled.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    if item is sentinel:
                        encoder_proc.stdin.close()
                        return
                    encoder_proc.stdin.write(np.ascontiguousarray(item).tobytes())
                    processed += 1
                    fraction = min(0.999, processed / total_frames) if total_frames else 0.0
                    suffix = f" {processed}/{total_frames}フレーム" if total_frames else f" {processed}フレーム"
                    progress(fraction, "動画をアップスケール中…" + suffix)
            except BaseException as exc:
                _fail("encoder", exc)

        threads = [
            threading.Thread(target=_decode, name="ueu-video-read", daemon=True),
            threading.Thread(target=_infer, name="ueu-video-ai", daemon=True),
            threading.Thread(target=_encode, name="ueu-video-write", daemon=True),
        ]
        for thread in threads:
            thread.start()

        while any(thread.is_alive() for thread in threads):
            if cancel is not None and cancel.is_set():
                _fail("pipeline", Cancelled())
            if stop.is_set():
                _terminate(decoder)
                _terminate(encoder_proc)
                session.close(force=True)
                break
            time.sleep(0.05)
        for thread in threads:
            thread.join(timeout=5)

        # エラー経路でもプロセス終了とstderrドレインを待ってから例外を組み立てる。
        # これにより、Windowsのパイプ書き込み側に出る Errno 22 ではなく、
        # エンコーダ/デコーダ自身の診断を優先して表示できる。
        decoder_ret = decoder.wait(timeout=10)
        encoder_ret = encoder_proc.wait(timeout=60)
        decoder_drain.join(timeout=2)
        encoder_drain.join(timeout=2)

        if errors:
            stage, exc = errors[0]
            if isinstance(exc, Cancelled):
                raise exc
            if stage == "encoder" and isinstance(exc, (BrokenPipeError, OSError, EOFError)):
                raise _ffmpeg_pipeline_error(
                    "動画エンコード", encoder_stderr,
                    cause=exc, returncode=encoder_ret,
                ) from exc
            if stage == "decoder" and isinstance(exc, (BrokenPipeError, OSError, EOFError)):
                raise _ffmpeg_pipeline_error(
                    "動画デコード", decoder_stderr,
                    cause=exc, returncode=decoder_ret,
                ) from exc
            raise exc
        if decoder_ret != 0:
            raise _ffmpeg_pipeline_error(
                "動画デコード", decoder_stderr, returncode=decoder_ret
            )
        if encoder_ret != 0:
            raise _ffmpeg_pipeline_error(
                "動画エンコード", encoder_stderr, returncode=encoder_ret
            )
        if not out.is_file() or out.stat().st_size == 0:
            raise RuntimeError(f"動画出力が生成されませんでした: {out}")
        progress(1.0, "完了")
    finally:
        if decoder is not None and decoder.poll() is None:
            _terminate(decoder)
        if encoder_proc is not None and encoder_proc.poll() is None:
            _terminate(encoder_proc)
        session.close(force=bool(errors) or (cancel is not None and cancel.is_set()))

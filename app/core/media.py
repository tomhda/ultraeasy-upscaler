"""メディア種別判定とメタ情報取得。

`classify` / 拡張子集合 / `MediaInfo` は実装済み。
`probe()` は動画/ffmpeg 担当エージェントが実装する。
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from . import binaries
from .jobs import JobKind

# Windows でコンソールウィンドウを出さない
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".wmv", ".flv",
              ".mpg", ".mpeg", ".ts", ".m2ts"}


def classify(path: str) -> JobKind:
    """パスを画像 / 動画 / フォルダに分類。未対応なら ValueError。"""
    p = Path(path)
    if p.is_dir():
        return JobKind.FOLDER
    ext = p.suffix.lower()
    if ext in IMAGE_EXTS:
        return JobKind.IMAGE
    if ext in VIDEO_EXTS:
        return JobKind.VIDEO
    raise ValueError(f"未対応の形式です: {p.name}")


@dataclass
class MediaInfo:
    kind: str                        # "image" | "video"
    width: int = 0
    height: int = 0
    fps: Optional[float] = None      # 動画のみ
    frame_count: Optional[int] = None
    duration: Optional[float] = None
    has_audio: bool = False
    codec: Optional[str] = None
    size_bytes: int = 0
    rotation: int = 0                  # 表示時の回転角（度）

    @property
    def display_width(self) -> int:
        return self.height if abs(self.rotation) % 180 == 90 else self.width

    @property
    def display_height(self) -> int:
        return self.width if abs(self.rotation) % 180 == 90 else self.height


def _parse_fps(value: Optional[str]) -> Optional[float]:
    """ffprobe の "30000/1001" や "24" 形式の frame_rate を float 化する。"""
    if not value:
        return None
    value = value.strip()
    try:
        if "/" in value:
            num, den = value.split("/", 1)
            num_f, den_f = float(num), float(den)
            if den_f == 0:
                return None
            fps = num_f / den_f
        else:
            fps = float(value)
    except (ValueError, ZeroDivisionError):
        return None
    return fps if fps > 0 else None


def _to_int(value) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_float(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_rotation(stream: dict) -> int:
    """ffprobe の side_data / tags から表示回転角を取得する。"""
    side_data = stream.get("side_data_list") or stream.get("side_data") or []
    for item in side_data:
        value = item.get("rotation") if isinstance(item, dict) else None
        parsed = _to_float(value)
        if parsed is not None:
            return int(round(parsed)) % 360
    tags = stream.get("tags") or {}
    parsed = _to_float(tags.get("rotate"))
    return int(round(parsed)) % 360 if parsed is not None else 0


def _run_ffprobe(path: str) -> dict:
    """ffprobe を JSON 出力で実行して dict を返す。失敗時は空 dict。"""
    cmd = [
        binaries.ffprobe_exe(),
        "-v", "error",
        "-print_format", "json",
        "-show_streams",
        "-show_format",
        path,
    ]
    proc = subprocess.run(
        cmd, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
        creationflags=_NO_WINDOW,
    )
    if proc.returncode != 0 or not proc.stdout:
        return {}
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {}


def probe(path: str) -> MediaInfo:
    """ffprobe（画像は ffprobe か Pillow）でメタ情報を取得する。

    - 画像: width/height/size_bytes を埋め kind="image"。
    - 動画: width/height/fps/frame_count/duration/has_audio/codec/size_bytes を埋め
      kind="video"。fps は avg/r_frame_rate を float 化。frame_count が無ければ
      duration*fps で概算。
    - ffprobe は binaries.ffprobe_exe() を使う。例外時も最低限 kind を返すこと。
    """
    p = Path(path)
    kind = "image" if p.suffix.lower() in IMAGE_EXTS else "video"

    size_bytes = 0
    try:
        size_bytes = p.stat().st_size
    except OSError:
        pass

    info = MediaInfo(kind=kind, size_bytes=size_bytes)

    if kind == "image":
        _probe_image(p, info)
        return info

    _probe_video(p, info)
    return info


def _probe_image(p: Path, info: MediaInfo) -> None:
    """画像の width/height を埋める。Pillow を優先し、失敗時 ffprobe にフォールバック。"""
    try:
        from PIL import Image  # 遅延 import

        with Image.open(p) as im:
            info.width, info.height = im.width, im.height
        if info.width and info.height:
            return
    except Exception:
        pass

    # フォールバック: ffprobe
    data = _run_ffprobe(str(p))
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video":
            info.width = _to_int(stream.get("width")) or 0
            info.height = _to_int(stream.get("height")) or 0
            info.codec = stream.get("codec_name")
            break


def _probe_video(p: Path, info: MediaInfo) -> None:
    """動画のメタ情報を ffprobe から埋める。例外時も kind は維持。"""
    data = _run_ffprobe(str(p))
    if not data:
        return

    fmt = data.get("format", {})
    duration = _to_float(fmt.get("duration"))

    video_stream = None
    for stream in data.get("streams", []):
        ctype = stream.get("codec_type")
        if ctype == "video" and video_stream is None:
            video_stream = stream
        elif ctype == "audio":
            info.has_audio = True

    if video_stream is not None:
        info.width = _to_int(video_stream.get("width")) or 0
        info.height = _to_int(video_stream.get("height")) or 0
        info.rotation = _parse_rotation(video_stream)
        info.codec = video_stream.get("codec_name")

        # fps: avg_frame_rate を優先し、無ければ r_frame_rate
        info.fps = (_parse_fps(video_stream.get("avg_frame_rate"))
                    or _parse_fps(video_stream.get("r_frame_rate")))

        # duration はストリーム側を優先的に補完
        if duration is None:
            duration = _to_float(video_stream.get("duration"))

        # frame_count: nb_frames / nb_read_frames を優先、無ければ概算
        frame_count = (_to_int(video_stream.get("nb_frames"))
                       or _to_int(video_stream.get("nb_read_frames")))
        if frame_count is None and duration and info.fps:
            frame_count = round(duration * info.fps)
        info.frame_count = frame_count

    info.duration = duration

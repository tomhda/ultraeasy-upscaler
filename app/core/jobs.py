"""ジョブ（処理単位）モデルと共通型。"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

# 進捗コールバック: (fraction 0.0-1.0, 表示メッセージ)
ProgressCb = Callable[[float, str], None]


class Cancelled(Exception):
    """ユーザーによるキャンセル。"""


class JobKind(str, Enum):
    IMAGE = "image"
    VIDEO = "video"
    FOLDER = "folder"


class JobStatus(str, Enum):
    QUEUED = "queued"
    PROBING = "probing"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"
    CANCELED = "canceled"


_id_counter = itertools.count(1)


@dataclass
class Job:
    """キュー上の1項目（画像 / 動画 / フォルダ）。"""

    input_path: Path
    kind: JobKind
    id: int = field(default_factory=lambda: next(_id_counter))

    status: JobStatus = JobStatus.QUEUED
    progress: float = 0.0          # 0.0 - 1.0
    message: str = "待機中"

    # メタデータ（probe で埋める。表示用）
    width: int = 0
    height: int = 0
    fps: Optional[float] = None
    frame_count: Optional[int] = None
    has_audio: bool = False
    size_bytes: int = 0

    # 結果
    output_path: Optional[Path] = None
    error: str = ""

    @property
    def name(self) -> str:
        return self.input_path.name

    @classmethod
    def create(cls, path) -> "Job":
        """パスから種別を判定して Job を生成（未対応形式は ValueError）。"""
        from . import media  # 遅延 import（循環回避）

        p = Path(path)
        return cls(input_path=p, kind=media.classify(str(p)))

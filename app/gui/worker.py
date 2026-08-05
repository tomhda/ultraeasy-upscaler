"""キュー処理のバックグラウンドワーカー。

GUI スレッドから切り離した QObject を QThread に move して使う。
ジョブは GPU 競合を避けるため **逐次** に1件ずつ処理する。
ウィジェットには一切触れず、シグナルで GUI スレッドに状態を渡す。
"""
from __future__ import annotations

import threading
from typing import Optional

from PySide6.QtCore import QObject, Signal, Slot

from app.core import engine
from app.core.jobs import Cancelled, Job
from app.core.settings import UpscaleSettings


class QueueWorker(QObject):
    """保留中ジョブを逐次処理するワーカー。

    使い方:
      worker = QueueWorker(jobs, settings, cancel_events, pause_flag)
      worker.moveToThread(thread)
      thread.started.connect(worker.run)
      ...各シグナルを GUI スロットへ接続...
    """

    # job_id, fraction(0..1), message
    progress = Signal(int, float, str)
    # job_id, 出力パス文字列
    job_done = Signal(int, str)
    # job_id, エラーメッセージ
    job_error = Signal(int, str)
    # job_id
    job_canceled = Signal(int)
    # 全保留ジョブを処理し終えた（または一時停止で停止した）
    queue_finished = Signal()

    def __init__(
        self,
        jobs: list[Job],
        settings: UpscaleSettings,
        cancel_events: dict[int, threading.Event],
        pause_flag: threading.Event,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._jobs = jobs
        self._settings = settings
        self._cancel_events = cancel_events
        self._pause = pause_flag

    @Slot()
    def run(self) -> None:
        """保留中ジョブを順番に処理する。一時停止フラグで安全に抜ける。"""
        try:
            for job in self._jobs:
                # 一時停止が要求されたら、現在ジョブ完了済みのこの境界で停止
                if self._pause.is_set():
                    break

                cancel = self._cancel_events.get(job.id)
                # cancel が None = 実行中に行が削除された / すでにキャンセル済み。
                # いずれも処理せずスキップする（削除済みジョブの取りこぼし出力を防ぐ）。
                if cancel is None or cancel.is_set():
                    self.job_canceled.emit(job.id)
                    continue

                def _cb(frac: float, msg: str, _jid: int = job.id) -> None:
                    # ワーカースレッドから GUI へはシグナルのみ
                    self.progress.emit(_jid, float(frac), msg or "")

                try:
                    job_settings = job.settings or self._settings
                    out = engine.process_job(
                        job, job_settings, progress=_cb, cancel=cancel
                    )
                    self.job_done.emit(job.id, str(out))
                except Cancelled:
                    self.job_canceled.emit(job.id)
                except Exception as exc:  # noqa: BLE001 - GUI に渡して表示
                    self.job_error.emit(job.id, str(exc) or exc.__class__.__name__)
        finally:
            self.queue_finished.emit()

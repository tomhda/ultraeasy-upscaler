"""GUI ワーカーのキュー処理ロジックのテスト（GPU 不使用・engine をモック）。"""
from __future__ import annotations

import os
import threading

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from app.core import engine  # noqa: E402
from app.core.jobs import Job, JobKind  # noqa: E402
from app.core.settings import UpscaleSettings  # noqa: E402
from app.gui.worker import QueueWorker  # noqa: E402


def _ensure_app() -> None:
    QApplication.instance() or QApplication([])


def test_worker_skips_removed_job(monkeypatch, tmp_path):
    """実行中に削除された(=cancel_events に無い)ジョブは process_job されない（#4 回帰）。"""
    _ensure_app()
    calls: list[int] = []

    def fake_process(job, settings, progress=None, cancel=None):
        calls.append(job.id)
        return tmp_path / f"{job.id}.out"

    monkeypatch.setattr(engine, "process_job", fake_process)

    f = tmp_path / "a.png"
    f.write_bytes(b"x")
    j1 = Job(input_path=f, kind=JobKind.IMAGE)
    j2 = Job(input_path=f, kind=JobKind.IMAGE)
    # j2 はわざと cancel_events に入れない（実行中に行が削除された状況を再現）
    cancel_events = {j1.id: threading.Event()}
    pause = threading.Event()

    canceled: list[int] = []
    done: list[int] = []
    w = QueueWorker([j1, j2], UpscaleSettings(), cancel_events, pause)
    w.job_canceled.connect(canceled.append)
    w.job_done.connect(lambda jid, _p: done.append(jid))
    w.run()

    assert calls == [j1.id]        # j2 は実処理されない（取りこぼし出力の防止）
    assert j2.id in canceled
    assert done == [j1.id]


def test_worker_pause_stops_before_processing(monkeypatch, tmp_path):
    """pause が立っていればジョブ境界で停止する（処理せず queue_finished）。"""
    _ensure_app()
    calls: list[int] = []
    monkeypatch.setattr(
        engine, "process_job",
        lambda job, settings, progress=None, cancel=None: calls.append(job.id),
    )

    f = tmp_path / "a.png"
    f.write_bytes(b"x")
    j1 = Job(input_path=f, kind=JobKind.IMAGE)
    cancel_events = {j1.id: threading.Event()}
    pause = threading.Event()
    pause.set()  # 開始前から一時停止

    finished: list[bool] = []
    w = QueueWorker([j1], UpscaleSettings(), cancel_events, pause)
    w.queue_finished.connect(lambda: finished.append(True))
    w.run()

    assert calls == []          # 一時停止中なので未処理
    assert finished == [True]   # それでも queue_finished は必ず出る

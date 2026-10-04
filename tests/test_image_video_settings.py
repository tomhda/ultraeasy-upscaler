"""画像/動画の別設定・個別設定・やり直しの GUI テスト（オフスクリーン）。

実際の推論はしない。`engine.process_job` を偽物に差し替える。
"""
from __future__ import annotations

import os
import shutil
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_IMAGE = REPO_ROOT / "vendor" / "realesrgan" / "input.jpg"


@pytest.fixture(scope="module")
def app():
    application = QApplication.instance() or QApplication([])
    yield application
    application.processEvents()


def _wait_until(pred, timeout: float = 20.0) -> bool:
    """条件が満たされるまで GUI イベントを回す。"""
    application = QApplication.instance()
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        if application is not None:
            application.processEvents()
        time.sleep(0.05)
    return pred()


def _install_fake_process(monkeypatch, calls: list, tmp_path, block=None):
    """engine.process_job の偽物（設定を記録して空ファイルを返す）。"""
    from app.core import engine

    def fake(job, settings, progress=None, cancel=None):
        calls.append(job)
        if block is not None:
            block.wait(timeout=20.0)
        if progress is not None:
            progress(1.0, "fake")
        out = tmp_path / f"fake-out-{job.id}.bin"
        out.write_bytes(b"fake")
        return out

    monkeypatch.setattr(engine, "process_job", fake)


def _make_window(app, tmp_path, extra_video: bool = False):
    from app.gui.main_window import MainWindow

    win = MainWindow()
    win.show()
    app.processEvents()
    src = tmp_path / "pic.png"
    shutil.copy(SAMPLE_IMAGE, src)
    ok, _ = win.add_path(str(src))
    assert ok is True
    video_job = None
    if extra_video:
        vsrc = tmp_path / "clip.mp4"
        vsrc.write_bytes(b"not a real video")
        ok, _ = win.add_path(str(vsrc))
        assert ok is True
        video_job = next(j for j in win._jobs.values() if j.input_path == vsrc)
    app.processEvents()
    image_job = next(j for j in win._jobs.values() if j.input_path == src)
    return win, image_job, video_job


def _ensure_rife(win):
    """補間コンボに rife-v4.6 が無ければ足す（環境差の吸収）。"""
    idx = win.interpolation_combo.findData("rife-v4.6")
    if idx < 0:
        win.interpolation_combo.addItem("RIFE v4.6", "rife-v4.6")
        idx = win.interpolation_combo.findData("rife-v4.6")
    return idx


def _set_model(win, value) -> None:
    idx = win.model_combo.findData(value)
    assert idx >= 0, f"モデルが一覧に無い: {value!r}"
    win.model_combo.setCurrentIndex(idx)


def test_mixed_start_applies_each_kind_settings(app, monkeypatch, tmp_path) -> None:
    """画像と動画を混ぜて開始すると、各ジョブに種類の既定が入る。"""
    from app.core.settings import HELPER_MODEL_ANIME, HELPER_MODEL_SPAN

    calls: list = []
    _install_fake_process(monkeypatch, calls, tmp_path)
    win, image_job, video_job = _make_window(app, tmp_path, extra_video=True)
    assert video_job is not None
    try:
        rife = _ensure_rife(win)
        # 画像側の既定: Anime Video v3 / 動画側の既定: SPAN＋補間
        win._set_settings_tab("image")
        _set_model(win, HELPER_MODEL_ANIME)
        win._set_settings_tab("video")
        _set_model(win, HELPER_MODEL_SPAN)
        win.interpolation_combo.setCurrentIndex(rife)
        app.processEvents()

        win._on_start()
        assert _wait_until(lambda: not win._running)
        app.processEvents()

        assert {job.id for job in calls} == {image_job.id, video_job.id}
        assert image_job.settings is not None
        assert video_job.settings is not None
        assert image_job.settings.model == HELPER_MODEL_ANIME
        # 画像に補間設定は無いが、欄の値（動画側）は載ったままになる（互換）
        assert image_job.settings.interpolation_model == "rife-v4.6"
        assert video_job.settings.model == HELPER_MODEL_SPAN
        assert video_job.settings.interpolation_model == "rife-v4.6"
        assert image_job.message == "完了・Anime Video v3"
        assert video_job.message == "完了・4xNomosUni SPAN"
    finally:
        win.close()
        app.processEvents()


def test_per_file_override_wins_and_uncheck_reverts(app, tmp_path) -> None:
    """個別設定が既定より優先され、外すと既定に戻る。"""
    from app.core.jobs import JobStatus
    from app.core.settings import HELPER_MODEL_ANIME, HELPER_MODEL_SPAN

    win, image_job, _ = _make_window(app, tmp_path)
    try:
        assert win._settings_tab == "image"
        assert win._per_file_check.isEnabled() is True
        assert image_job.message == "Anime Video v3・4x"

        win._per_file_check.setChecked(True)
        app.processEvents()
        assert image_job.id in win._overrides
        # 初期値はその時点の既定
        assert win._overrides[image_job.id]["model"] == HELPER_MODEL_ANIME

        _set_model(win, HELPER_MODEL_SPAN)
        app.processEvents()
        assert win.build_settings(image_job).model == HELPER_MODEL_SPAN
        assert win._image_model == HELPER_MODEL_ANIME
        assert image_job.message == "4xNomosUni SPAN・4x（個別）"

        pending = win._pending_jobs()
        win._apply_current_settings(pending)
        assert image_job.settings is not None
        assert image_job.settings.model == HELPER_MODEL_SPAN

        win._per_file_check.setChecked(False)
        app.processEvents()
        assert image_job.id not in win._overrides
        assert image_job.message == "Anime Video v3・4x"
        assert win.build_settings(image_job).model == HELPER_MODEL_ANIME
        assert image_job.status == JobStatus.QUEUED
    finally:
        win.close()
        app.processEvents()


def test_video_model_list_excludes_adcsr(app) -> None:
    """動画側のモデル一覧に AdcSR が無く、画像側にはある。"""
    from app.core.settings import HELPER_MODEL_ADCSR
    from app.gui.main_window import MainWindow

    win = MainWindow()
    win.show()
    app.processEvents()
    try:
        win._set_settings_tab("video")
        app.processEvents()
        video_values = [
            win.model_combo.itemData(i) for i in range(win.model_combo.count())
        ]
        assert HELPER_MODEL_ADCSR not in video_values
        assert win.model_combo.findData(None) == 0

        win._set_settings_tab("image")
        app.processEvents()
        image_values = [
            win.model_combo.itemData(i) for i in range(win.model_combo.count())
        ]
        assert HELPER_MODEL_ADCSR in image_values
    finally:
        win.close()
        app.processEvents()


def test_waiting_row_texts(app, tmp_path) -> None:
    """行の表示が表どおりで、既定を変えると待機中の行が変わる。"""
    from app.core.settings import HELPER_MODEL_SPAN

    win, image_job, video_job = _make_window(app, tmp_path, extra_video=True)
    assert video_job is not None
    try:
        rife = _ensure_rife(win)
        # 既定のまま: 画像は拡大あり、動画は既定（拡大あり・補間なし）
        assert image_job.message == "Anime Video v3・4x"
        assert video_job.message == "Anime Video v3・4x"

        # 動画側に補間を足すと「・補間あり」になる
        win._set_settings_tab("video")
        win.interpolation_combo.setCurrentIndex(rife)
        app.processEvents()
        assert video_job.message == "Anime Video v3・4x・補間あり"
        assert image_job.message == "Anime Video v3・4x"

        # 動画のモデルを「なし」にすると「補間のみ」になる
        none_idx = win.model_combo.findData(None)
        win.model_combo.setCurrentIndex(none_idx)
        app.processEvents()
        assert video_job.message == "補間のみ"

        # 両方なしは「モデル未選択」
        win.interpolation_combo.setCurrentIndex(0)
        app.processEvents()
        assert video_job.message == "モデル未選択"

        # 画像側の既定を変えると、その行だけ変わる
        win._set_settings_tab("image")
        _set_model(win, HELPER_MODEL_SPAN)
        app.processEvents()
        assert image_job.message == "4xNomosUni SPAN・4x"
        assert video_job.message == "モデル未選択"

        # 個別設定の印
        win.queue.select(image_job.id)
        app.processEvents()
        assert win._settings_tab == "image"
        win._per_file_check.setChecked(True)
        app.processEvents()
        assert image_job.message == "4xNomosUni SPAN・4x（個別）"
    finally:
        win.close()
        app.processEvents()


def test_retry_single_and_all(app, monkeypatch, tmp_path) -> None:
    """やり直しで待機中に戻り、次の開始で処理対象になる。"""
    from app.core.jobs import JobStatus

    calls: list = []
    _install_fake_process(monkeypatch, calls, tmp_path)
    win, image_job, video_job = _make_window(app, tmp_path, extra_video=True)
    assert video_job is not None
    try:
        win._on_start()
        assert _wait_until(lambda: not win._running)
        app.processEvents()
        assert image_job.status == JobStatus.DONE
        assert video_job.status == JobStatus.DONE
        assert win.retry_all_btn.isEnabled() is True
        row = win.queue.row(image_job.id)
        assert row is not None
        assert row._retry.isVisible() is True
        assert row._retry.isEnabled() is True

        # 1 行のやり直し
        calls.clear()
        win._on_retry_requested(image_job.id)
        app.processEvents()
        assert image_job.status == JobStatus.QUEUED
        assert image_job.progress == 0.0
        assert image_job.output_path is None
        assert image_job.error == ""
        assert image_job.message == "Anime Video v3・4x"
        assert video_job.status == JobStatus.DONE

        win._on_start()
        assert _wait_until(lambda: not win._running)
        app.processEvents()
        assert [job.id for job in calls] == [image_job.id]
        assert image_job.status == JobStatus.DONE

        # すべてやり直す
        win._on_retry_all()
        app.processEvents()
        assert image_job.status == JobStatus.QUEUED
        assert video_job.status == JobStatus.QUEUED
        assert video_job.message == "Anime Video v3・4x"

        # 待機中の行のやり直しは無視される
        win._on_retry_requested(image_job.id)
        assert image_job.status == JobStatus.QUEUED
    finally:
        win.close()
        app.processEvents()


def test_retry_locked_while_running(app, tmp_path) -> None:
    """処理中はやり直し（1 行・全行）とも押せない。"""
    from app.gui.main_window import MainWindow

    win = MainWindow()
    win.show()
    app.processEvents()
    try:
        src = tmp_path / "pic.png"
        shutil.copy(SAMPLE_IMAGE, src)
        ok, _ = win.add_path(str(src))
        assert ok is True
        job = next(iter(win._jobs.values()))
        from app.core.jobs import JobStatus

        job.status = JobStatus.DONE
        job.settings = win.build_settings(job)
        job.message = win._done_text(job)
        win.queue.refresh(job.id)
        win._update_retry_all()
        row = win.queue.row(job.id)
        assert row is not None
        assert row._retry.isEnabled() is True
        assert win.retry_all_btn.isEnabled() is True

        win._set_running(True)
        app.processEvents()
        assert row._retry.isEnabled() is False
        assert win.retry_all_btn.isEnabled() is False
        # 押しても待機中に戻らない
        win._on_retry_requested(job.id)
        win._on_retry_all()
        assert job.status == JobStatus.DONE

        win._set_running(False)
        app.processEvents()
        assert row._retry.isEnabled() is True
        assert win.retry_all_btn.isEnabled() is True
    finally:
        win.close()
        app.processEvents()


def test_start_validation_messages(app, monkeypatch, tmp_path) -> None:
    """開始時の確認の文言が表どおり。"""
    from app.core import engine
    from app.gui.main_window import MainWindow

    called: list = []
    monkeypatch.setattr(
        engine, "process_job",
        lambda job, settings, progress=None, cancel=None: called.append(job.id),
    )
    win = MainWindow()
    win.show()
    app.processEvents()
    try:
        # 待機中の行が無い
        win._on_start()
        assert win.status_label.text() == "処理するファイルがありません。"
        assert called == []

        src = tmp_path / "pic.png"
        shutil.copy(SAMPLE_IMAGE, src)
        ok, _ = win.add_path(str(src))
        assert ok is True
        image_job = next(iter(win._jobs.values()))
        vsrc = tmp_path / "clip.mp4"
        vsrc.write_bytes(b"not a real video")
        ok, _ = win.add_path(str(vsrc))
        assert ok is True
        video_job = next(j for j in win._jobs.values() if j.input_path == vsrc)

        # 画像（既定）でモデルが「なし」
        win._set_settings_tab("image")
        win.model_combo.setCurrentIndex(win.model_combo.findData(None))
        # 動画側も両方なしにしておく（画像の文言が先に出ることの確認）
        win._set_settings_tab("video")
        win.model_combo.setCurrentIndex(win.model_combo.findData(None))
        win.interpolation_combo.setCurrentIndex(0)
        app.processEvents()
        win._on_start()
        assert win.status_label.text() == "画像のモデルを選んでください。"
        assert called == []
        assert win._running is False

        # 画像を戻すと、次は動画の文言
        win._set_settings_tab("image")
        win.model_combo.setCurrentIndex(1)
        app.processEvents()
        win._on_start()
        assert win.status_label.text() == "動画のモデルかフレーム補間を選んでください。"
        assert called == []

        # 個別設定のファイルが同じ状態ならファイル名で案内
        win.queue.select(video_job.id)
        app.processEvents()
        assert win._settings_tab == "video"
        win._per_file_check.setChecked(True)
        app.processEvents()
        win._on_start()
        assert win.status_label.text() == f"{video_job.name}のモデルを選んでください。"
        assert called == []
        assert win._running is False
        from app.core.jobs import JobStatus

        assert image_job.status == JobStatus.QUEUED
    finally:
        win.close()
        app.processEvents()


def test_trial_uses_selected_file_settings(app, tmp_path) -> None:
    """試すときの設定が選択中のファイル用（個別があればそれ）になる。"""
    from app.core.settings import HELPER_MODEL_ANIME, HELPER_MODEL_SPAN

    win, image_job, video_job = _make_window(app, tmp_path, extra_video=True)
    assert video_job is not None
    try:
        win._set_settings_tab("image")
        _set_model(win, HELPER_MODEL_ANIME)
        win._set_settings_tab("video")
        _set_model(win, HELPER_MODEL_SPAN)
        app.processEvents()

        # 画像を選ぶと画像用の既定（先に動画を選んで選択を動かす）
        win.queue.select(video_job.id)
        app.processEvents()
        assert win._settings_tab == "video"
        win.queue.select(image_job.id)
        app.processEvents()
        assert win._settings_tab == "image"
        settings = win.preview._current_settings_or_none()
        assert settings is not None
        assert settings.model == HELPER_MODEL_ANIME

        # 動画を選ぶと動画用の既定
        win.queue.select(video_job.id)
        app.processEvents()
        assert win._settings_tab == "video"
        settings = win.preview._current_settings_or_none()
        assert settings is not None
        assert settings.model == HELPER_MODEL_SPAN

        # 動画に個別を入れると試しも個別になる
        assert win._per_file_check.isEnabled() is True
        win._per_file_check.setChecked(True)
        _set_model(win, HELPER_MODEL_ANIME)
        app.processEvents()
        settings = win.preview._current_settings_or_none()
        assert settings is not None
        assert settings.model == HELPER_MODEL_ANIME
    finally:
        win.close()
        app.processEvents()

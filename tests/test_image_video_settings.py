"""画像/動画の一括設定・個別設定・やり直しの GUI テスト（オフスクリーン）。

ヘッダー = 一括設定、右列 = 選んだファイルの設定。実際の推論はしない。
`engine.process_job` を偽物に差し替える。
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

# NPU キットの無い PC でも通るよう NPU ありに固定する（なしの表示は別に確かめる）。
pytestmark = pytest.mark.usefixtures("force_npu_available")


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
    """補間コンボ（ヘッダーと右列）に rife-v4.6 が無ければ足す（環境差の吸収）。"""
    for combo in (win.global_interpolation_combo, win.interpolation_combo):
        idx = combo.findData("rife-v4.6")
        if idx < 0:
            combo.addItem("RIFE v4.6", "rife-v4.6")
    return win.global_interpolation_combo.findData("rife-v4.6")


def _set_header_model(win, kind: str, value) -> None:
    """一括設定のモデルを変える（kind は "image" か "video"）。"""
    combo = win.image_model_combo if kind == "image" else win.video_model_combo
    idx = combo.findData(value)
    assert idx >= 0, f"モデルが一覧に無い: {value!r}"
    combo.setCurrentIndex(idx)


def _set_right_model(win, value) -> None:
    idx = win.model_combo.findData(value)
    assert idx >= 0, f"モデルが一覧に無い: {value!r}"
    win.model_combo.setCurrentIndex(idx)


def test_mixed_start_applies_each_kind_settings(app, monkeypatch, tmp_path) -> None:
    """画像と動画を混ぜて開始すると、各ジョブに種類の一括が入る。"""
    from app.core.settings import HELPER_MODEL_ANIME, HELPER_MODEL_SPAN

    calls: list = []
    _install_fake_process(monkeypatch, calls, tmp_path)
    win, image_job, video_job = _make_window(app, tmp_path, extra_video=True)
    assert video_job is not None
    try:
        rife = _ensure_rife(win)
        # 画像側の一括: Anime Video v3 / 動画側の一括: SPAN＋補間
        _set_header_model(win, "image", HELPER_MODEL_ANIME)
        _set_header_model(win, "video", HELPER_MODEL_SPAN)
        win.global_interpolation_combo.setCurrentIndex(rife)
        app.processEvents()

        win._on_start()
        assert _wait_until(lambda: not win._running)
        app.processEvents()

        assert {job.id for job in calls} == {image_job.id, video_job.id}
        assert image_job.settings is not None
        assert video_job.settings is not None
        assert image_job.settings.model == HELPER_MODEL_ANIME
        # 画像に補間設定は無いが、一括の値（動画側）は載ったままになる（互換）
        assert image_job.settings.interpolation_model == "rife-v4.6"
        assert video_job.settings.model == HELPER_MODEL_SPAN
        assert video_job.settings.interpolation_model == "rife-v4.6"
        assert image_job.message == "完了・Anime Video v3"
        assert video_job.message == "完了・4xNomosUni SPAN"
    finally:
        win.close()
        app.processEvents()


def test_per_file_override_wins_and_reverts(app, tmp_path) -> None:
    """個別設定が一括より優先され、一括と同じに戻すと消える。"""
    from app.core.jobs import JobStatus
    from app.core.settings import HELPER_MODEL_ANIME, HELPER_MODEL_SPAN

    win, image_job, _ = _make_window(app, tmp_path)
    try:
        win.queue.select(image_job.id)
        app.processEvents()
        assert image_job.message == "Anime Video v3・4x"
        assert image_job.id not in win._overrides
        assert win._override_box.isVisible() is False

        # 右列で変えるとそのファイルだけ個別になる
        _set_right_model(win, HELPER_MODEL_SPAN)
        app.processEvents()
        assert win._overrides[image_job.id]["model"] == HELPER_MODEL_SPAN
        assert win.build_settings(image_job).model == HELPER_MODEL_SPAN
        assert win._image_model == HELPER_MODEL_ANIME
        assert image_job.message == "4xNomosUni SPAN・4x（個別）"
        assert win._override_box.isVisible() is True

        pending = win._pending_jobs()
        win._apply_current_settings(pending)
        assert image_job.settings is not None
        assert image_job.settings.model == HELPER_MODEL_SPAN

        # 右列で一括と同じ値に戻すと個別が消える
        _set_right_model(win, HELPER_MODEL_ANIME)
        app.processEvents()
        assert image_job.id not in win._overrides
        assert image_job.message == "Anime Video v3・4x"
        assert win._override_box.isVisible() is False
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
        video_values = [
            win.video_model_combo.itemData(i)
            for i in range(win.video_model_combo.count())
        ]
        assert HELPER_MODEL_ADCSR not in video_values
        assert win.video_model_combo.findData(None) == 0

        image_values = [
            win.image_model_combo.itemData(i)
            for i in range(win.image_model_combo.count())
        ]
        assert HELPER_MODEL_ADCSR in image_values
    finally:
        win.close()
        app.processEvents()


def test_waiting_row_texts(app, tmp_path) -> None:
    """行の表示が表どおりで、一括を変えると待機中の行が変わる。"""
    from app.core.settings import HELPER_MODEL_ANIME, HELPER_MODEL_SPAN

    win, image_job, video_job = _make_window(app, tmp_path, extra_video=True)
    assert video_job is not None
    try:
        rife = _ensure_rife(win)
        # 一括のまま: 画像は拡大あり、動画は一括（拡大あり・補間なし）
        assert image_job.message == "Anime Video v3・4x"
        assert video_job.message == "Anime Video v3・4x"

        # 動画の一括に補間を足すと「・補間あり」になる
        win.global_interpolation_combo.setCurrentIndex(rife)
        app.processEvents()
        assert video_job.message == "Anime Video v3・4x・補間あり"
        assert image_job.message == "Anime Video v3・4x"

        # 動画のモデルを「なし」にすると「補間のみ」になる
        win.video_model_combo.setCurrentIndex(win.video_model_combo.findData(None))
        app.processEvents()
        assert video_job.message == "補間のみ"

        # 両方なしは「モデル未選択」
        win.global_interpolation_combo.setCurrentIndex(0)
        app.processEvents()
        assert video_job.message == "モデル未選択"

        # 画像の一括を変えると、その行だけ変わる
        _set_header_model(win, "image", HELPER_MODEL_SPAN)
        app.processEvents()
        assert image_job.message == "4xNomosUni SPAN・4x"
        assert video_job.message == "モデル未選択"

        # 右列で一括と違う値に変えると個別の印
        win.queue.select(image_job.id)
        app.processEvents()
        _set_right_model(win, HELPER_MODEL_ANIME)
        app.processEvents()
        assert image_job.message == "Anime Video v3・4x（個別）"
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

        # 画像・動画の一括をどちらも「なし」に（画像の文言が先に出ることの確認）
        win.image_model_combo.setCurrentIndex(win.image_model_combo.findData(None))
        win.video_model_combo.setCurrentIndex(win.video_model_combo.findData(None))
        win.global_interpolation_combo.setCurrentIndex(0)
        app.processEvents()
        win._on_start()
        assert win.status_label.text() == "画像の拡大モデルを選んでください。"
        assert called == []
        assert win._running is False

        # 画像を戻すと、次は動画の文言
        _set_header_model(win, "image", win.image_model_combo.itemData(1))
        app.processEvents()
        win._on_start()
        assert win.status_label.text() == (
            "動画の拡大モデルかフレーム補間モデルを選んでください。"
        )
        assert called == []

        # 個別設定のファイルが同じ状態ならファイル名で案内
        # （一括に戻してから右列で「なし」にして個別にする）
        _set_header_model(win, "video", win.video_model_combo.itemData(1))
        app.processEvents()
        win.queue.select(video_job.id)
        app.processEvents()
        _set_right_model(win, None)
        app.processEvents()
        assert video_job.id in win._overrides
        win._on_start()
        assert win.status_label.text() == (
            f"{video_job.name}の拡大モデルを選んでください。"
        )
        assert called == []
        assert win._running is False
        from app.core.jobs import JobStatus

        assert image_job.status == JobStatus.QUEUED
    finally:
        win.close()
        app.processEvents()


def test_trial_uses_selected_file_settings(app, tmp_path) -> None:
    """確認するときの設定が選択中のファイル用（個別があればそれ）になる。"""
    from app.core.settings import HELPER_MODEL_ANIME, HELPER_MODEL_SPAN

    win, image_job, video_job = _make_window(app, tmp_path, extra_video=True)
    assert video_job is not None
    try:
        _set_header_model(win, "image", HELPER_MODEL_ANIME)
        _set_header_model(win, "video", HELPER_MODEL_SPAN)
        app.processEvents()

        # 画像を選ぶと画像の一括
        win.queue.select(video_job.id)
        app.processEvents()
        win.queue.select(image_job.id)
        app.processEvents()
        settings = win.preview._current_settings_or_none()
        assert settings is not None
        assert settings.model == HELPER_MODEL_ANIME

        # 動画を選ぶと動画の一括
        win.queue.select(video_job.id)
        app.processEvents()
        settings = win.preview._current_settings_or_none()
        assert settings is not None
        assert settings.model == HELPER_MODEL_SPAN

        # 動画に個別を入れると試しも個別になる
        _set_right_model(win, HELPER_MODEL_ANIME)
        app.processEvents()
        assert video_job.id in win._overrides
        settings = win.preview._current_settings_or_none()
        assert settings is not None
        assert settings.model == HELPER_MODEL_ANIME
    finally:
        win.close()
        app.processEvents()


def test_row_tooltip_uses_display_names_not_internal_ids(app):
    """行の小さな説明には、内部の名前（winml_gpu / animevideov3 など）を出さない。"""
    from app.gui.main_window import MainWindow

    win = MainWindow()
    win.add_paths([str(SAMPLE_IMAGE)])
    app.processEvents()
    job = win._jobs[win._order[0]]
    job.settings = win.build_settings(job)
    win.queue.refresh(job.id)

    text = win.queue.row(job.id)._meta.toolTip()
    assert text == "AI実行先: GPU（DirectML）\nアップスケール: Anime Video v3・4x\nフレーム補間: なし"
    assert "winml_gpu" not in text and "animevideov3" not in text

    win.close()
    app.processEvents()


def test_header_change_follows_plain_rows_and_right_panel(app, tmp_path) -> None:
    """ヘッダーで一括を変えると個別の無い行と右列が追従する。"""
    from app.core.settings import HELPER_MODEL_ANIME, HELPER_MODEL_SPAN

    win, image_job, video_job = _make_window(app, tmp_path, extra_video=True)
    assert video_job is not None
    try:
        win.queue.select(image_job.id)
        app.processEvents()
        assert win.model_combo.currentData() == HELPER_MODEL_ANIME

        _set_header_model(win, "image", HELPER_MODEL_SPAN)
        app.processEvents()
        assert win._image_model == HELPER_MODEL_SPAN
        assert image_job.message == "4xNomosUni SPAN・4x"
        assert video_job.message == "Anime Video v3・4x"
        assert win.model_combo.currentData() == HELPER_MODEL_SPAN
        assert win.image_model_combo.currentData() == HELPER_MODEL_SPAN
    finally:
        win.close()
        app.processEvents()


def test_right_change_affects_only_selected_file(app, tmp_path) -> None:
    """右列で変えると選んだファイルだけ個別になり、ほかは変わらない。"""
    import shutil as _shutil

    from app.core.settings import HELPER_MODEL_SPAN

    win, image_job, _ = _make_window(app, tmp_path)
    try:
        second = tmp_path / "second.png"
        _shutil.copy(SAMPLE_IMAGE, second)
        ok, _ = win.add_path(str(second))
        assert ok is True
        other = next(j for j in win._jobs.values() if j.input_path == second)
        app.processEvents()

        win.queue.select(image_job.id)
        app.processEvents()
        _set_right_model(win, HELPER_MODEL_SPAN)
        app.processEvents()
        assert image_job.id in win._overrides
        assert other.id not in win._overrides
        assert image_job.message.endswith("（個別）")
        assert "（個別）" not in other.message
        assert win.build_settings(other).model != HELPER_MODEL_SPAN
    finally:
        win.close()
        app.processEvents()


def test_header_to_override_value_prunes(app, tmp_path) -> None:
    """ヘッダーを個別と同じ値にするとその個別が消える。"""
    from app.core.settings import HELPER_MODEL_ANIME, HELPER_MODEL_SPAN

    win, image_job, _ = _make_window(app, tmp_path)
    try:
        win.queue.select(image_job.id)
        app.processEvents()
        _set_right_model(win, HELPER_MODEL_SPAN)
        app.processEvents()
        assert image_job.id in win._overrides

        _set_header_model(win, "image", HELPER_MODEL_SPAN)
        app.processEvents()
        assert image_job.id not in win._overrides
        assert image_job.message == "4xNomosUni SPAN・4x"
        assert win._override_box.isVisible() is False

        # 戻すと一括に従う
        _set_header_model(win, "image", HELPER_MODEL_ANIME)
        app.processEvents()
        assert image_job.message == "Anime Video v3・4x"
    finally:
        win.close()
        app.processEvents()


def test_apply_all_video_changes_bulk_and_header(app, tmp_path) -> None:
    """ほかの動画にも使うで一括とヘッダーが変わり、ほかの動画が追従する。"""
    import shutil as _shutil

    from app.core.settings import HELPER_MODEL_ANIME, HELPER_MODEL_SPAN

    win, image_job, video_job = _make_window(app, tmp_path, extra_video=True)
    assert video_job is not None
    try:
        second_video = tmp_path / "second.mp4"
        second_video.write_bytes(b"not a real video")
        ok, _ = win.add_path(str(second_video))
        assert ok is True
        other_video = next(
            j for j in win._jobs.values() if j.input_path == second_video
        )
        app.processEvents()

        win.queue.select(video_job.id)
        app.processEvents()
        _set_right_model(win, HELPER_MODEL_SPAN)
        app.processEvents()
        assert win.apply_all_btn.text() == "ほかの動画にも使う"
        win.apply_all_btn.click()
        app.processEvents()

        assert win._video_model == HELPER_MODEL_SPAN
        assert win.video_model_combo.currentData() == HELPER_MODEL_SPAN
        assert video_job.id not in win._overrides
        assert other_video.message == "4xNomosUni SPAN・4x"
        assert image_job.message == "Anime Video v3・4x"
        assert win._image_model == HELPER_MODEL_ANIME
    finally:
        win.close()
        app.processEvents()


def test_apply_all_image_changes_bulk_and_header(app, tmp_path) -> None:
    """ほかの画像にも使うで画像の一括とヘッダーが変わる（動画は不変）。"""
    from app.core.settings import HELPER_MODEL_SPAN, UpscaleBackend

    win, image_job, video_job = _make_window(app, tmp_path, extra_video=True)
    assert video_job is not None
    try:
        win.queue.select(image_job.id)
        app.processEvents()
        _set_right_model(win, HELPER_MODEL_SPAN)
        app.processEvents()
        assert win.apply_all_btn.text() == "ほかの画像にも使う"
        win.apply_all_btn.click()
        app.processEvents()

        assert win._image_model == HELPER_MODEL_SPAN
        assert win.image_model_combo.currentData() == HELPER_MODEL_SPAN
        assert image_job.id not in win._overrides
        assert image_job.message == "4xNomosUni SPAN・4x"
        assert video_job.message == "Anime Video v3・4x"

        # 倍率の一括は右列→「ほかの画像にも使う」で写す（Vulkanで確認）
        vulkan = win.backend_combo.findData(UpscaleBackend.VULKAN.value)
        win.backend_combo.setCurrentIndex(vulkan)
        app.processEvents()
        win.queue.select(image_job.id)
        app.processEvents()
        assert win._scale_btns[2].isEnabled() is True
        win._set_scale(2)
        app.processEvents()
        assert image_job.id in win._overrides
        win.apply_all_btn.click()
        app.processEvents()
        assert win._image_scale == 2
        assert image_job.id not in win._overrides
    finally:
        win.close()
        app.processEvents()


def test_reset_override_returns_to_bulk(app, tmp_path) -> None:
    """一括設定に戻すで個別が消え、右列が一括の値になる。"""
    from app.core.settings import HELPER_MODEL_ANIME, HELPER_MODEL_SPAN

    win, image_job, _ = _make_window(app, tmp_path)
    try:
        win.queue.select(image_job.id)
        app.processEvents()
        _set_right_model(win, HELPER_MODEL_SPAN)
        app.processEvents()
        assert image_job.id in win._overrides
        assert win.reset_override_btn.text() == "一括設定に戻す"

        win.reset_override_btn.click()
        app.processEvents()
        assert image_job.id not in win._overrides
        assert win.model_combo.currentData() == HELPER_MODEL_ANIME
        assert image_job.message == "Anime Video v3・4x"
        assert win._override_box.isVisible() is False
    finally:
        win.close()
        app.processEvents()


def test_no_file_right_panel(app, tmp_path) -> None:
    """ファイル未選択の右列は題名・案内・出力先だけが見える。"""
    from app.gui.main_window import MainWindow

    win = MainWindow()
    win.show()
    app.processEvents()
    try:
        assert win._settings_title.text() == "設定"
        assert win._no_file_hint.isVisible() is True
        assert win._model_wrap.isVisible() is False
        assert win.model_hint.isVisible() is False
        assert win._scale_wrap.isVisible() is False
        assert win._interp_wrap.isVisible() is False
        assert win._override_box.isVisible() is False
        # 出力先は全体の設定なので、右列ではなく詳細設定の中にある
        assert win.output_combo is win.drawer.output_combo

        src = tmp_path / "pic.png"
        shutil.copy(SAMPLE_IMAGE, src)
        ok, _ = win.add_path(str(src))
        assert ok is True
        app.processEvents()
        job = next(iter(win._jobs.values()))
        assert win._settings_title.toolTip() == job.name

        # 行を消すと未選択に戻る
        win._on_remove_requested(job.id)
        app.processEvents()
        assert win._settings_title.text() == "設定"
        assert win._no_file_hint.isVisible() is True
        assert win._model_wrap.isVisible() is False
    finally:
        win.close()
        app.processEvents()


def test_interp_wrap_only_for_video(app, tmp_path) -> None:
    """画像では補間欄が出ず、動画では出る。"""
    win, image_job, video_job = _make_window(app, tmp_path, extra_video=True)
    assert video_job is not None
    try:
        win.queue.select(image_job.id)
        app.processEvents()
        assert win._interp_wrap.isVisible() is False

        win.queue.select(video_job.id)
        app.processEvents()
        assert win._interp_wrap.isVisible() is True
    finally:
        win.close()
        app.processEvents()


def test_backend_change_replaces_models_everywhere(app, tmp_path, monkeypatch) -> None:
    """AI実行先を変えるとヘッダー・右列・個別の使えないモデルが置き換わる。"""
    from app.core import binaries
    from app.core.settings import HELPER_MODEL_ADCSR, UpscaleBackend

    win, image_job, video_job = _make_window(app, tmp_path, extra_video=True)
    assert video_job is not None
    try:
        # 画像に AdcSR の個別を入れる
        win.queue.select(image_job.id)
        app.processEvents()
        _set_right_model(win, HELPER_MODEL_ADCSR)
        app.processEvents()
        assert image_job.id in win._overrides

        # Vulkan へ（従来モデルなしの想定で AdcSR は使えない）
        monkeypatch.setattr(binaries, "available_models", lambda: [])
        vulkan = win.backend_combo.findData(UpscaleBackend.VULKAN.value)
        win.backend_combo.setCurrentIndex(vulkan)
        app.processEvents()

        assert win._image_model != HELPER_MODEL_ADCSR
        assert win._overrides.get(image_job.id, {}).get("model") != HELPER_MODEL_ADCSR
        image_values = [
            win.image_model_combo.itemData(i)
            for i in range(win.image_model_combo.count())
        ]
        assert HELPER_MODEL_ADCSR not in image_values
        right_values = [
            win.model_combo.itemData(i) for i in range(win.model_combo.count())
        ]
        assert HELPER_MODEL_ADCSR not in right_values
    finally:
        win.close()
        app.processEvents()


def test_running_locks_header_and_override_buttons(app, tmp_path) -> None:
    """実行中はヘッダー欄と2つのボタンがモデル欄と同じ扱いになる。"""
    from app.core.settings import HELPER_MODEL_SPAN

    win, image_job, _ = _make_window(app, tmp_path)
    try:
        win.queue.select(image_job.id)
        app.processEvents()
        _set_right_model(win, HELPER_MODEL_SPAN)
        app.processEvents()
        assert win._override_box.isVisible() is True

        win._set_running(True)
        app.processEvents()
        assert win.image_model_combo.isEnabled() is False
        assert win.video_model_combo.isEnabled() is False
        assert win.global_interpolation_combo.isEnabled() is False
        assert win.model_combo.isEnabled() is False
        assert win.apply_all_btn.isEnabled() is False
        assert win.reset_override_btn.isEnabled() is False

        win._set_running(False)
        app.processEvents()
        assert win.image_model_combo.isEnabled() is True
        assert win.video_model_combo.isEnabled() is True
        assert win.global_interpolation_combo.isEnabled() is True
    finally:
        win.close()
        app.processEvents()


def test_english_header_and_override_buttons(app) -> None:
    """英語のときヘッダー見出しと新しいボタンが英語になる。"""
    from PySide6.QtWidgets import QApplication as _QApp
    from PySide6.QtWidgets import QLabel

    from app.i18n import set_language
    from app.gui.main_window import MainWindow

    set_language("en")
    win = None
    try:
        win = MainWindow()
        win.show()
        _QApp.instance().processEvents()
        headers = [w.text() for w in win.findChildren(QLabel)]
        assert "Image model" in headers
        assert "Video model" in headers
        assert win._override_hint.text() == "This file uses its own settings."
        assert win.apply_all_btn.text() in (
            "Use for other images too", "Use for other videos too",
        )
        assert win.reset_override_btn.text() == "Use the settings at the top"
        assert win._no_file_hint.text() == (
            "Select a file to change the settings for that file only."
        )
    finally:
        if win is not None:
            win.close()
        set_language("ja")
        _QApp.instance().processEvents()

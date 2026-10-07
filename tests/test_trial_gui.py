"""クイック確認の GUI テスト（オフスクリーン）。

実際の推論はしない。`upscaler.upscale_image` を Pillow の最近傍4倍の
偽物に差し替える。ffmpeg のコマ取り出しだけ実実行する。
"""
from __future__ import annotations

import os
import re
import threading
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PIL import Image
from PySide6.QtWidgets import QApplication

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_IMAGE = REPO_ROOT / "vendor" / "realesrgan" / "input.jpg"
DEMO_VIDEO = REPO_ROOT / "vendor" / "realesrgan" / "onepiece_demo.mp4"

# NPU キットの無い PC でも通るよう NPU ありに固定する（なしの表示は別に確かめる）。
pytestmark = pytest.mark.usefixtures("force_npu_available")


@pytest.fixture(scope="module")
def app():
    application = QApplication.instance() or QApplication([])
    yield application
    application.processEvents()


def _wait_until(pred, timeout: float = 15.0) -> bool:
    """条件が満たされるまで GUI イベントを回す。"""
    end = time.time() + timeout
    application = QApplication.instance()
    while time.time() < end:
        if pred():
            return True
        if application is not None:
            application.processEvents()
        time.sleep(0.05)
    return pred()


def _install_fake_upscale(monkeypatch, calls: list, block: threading.Event | None = None,
                           error: str | None = None):
    """upscale_image の偽物（Pillow 最近傍4倍で出力ファイルを書く）。"""
    from app.core import upscaler
    from app.core.jobs import Cancelled

    def fake(in_path, out_path, settings, progress=None, cancel=None) -> None:
        calls.append({"in": in_path, "out": out_path, "settings": settings})
        if error is not None:
            raise ValueError(error)
        if progress is not None:
            progress(0.45, "fake")
        if block is not None:
            while not block.is_set():
                if cancel is not None and cancel.is_set():
                    raise Cancelled()
                time.sleep(0.02)
            if cancel is not None and cancel.is_set():
                raise Cancelled()
        with Image.open(in_path) as im:
            im.resize((im.width * 4, im.height * 4), Image.NEAREST).save(out_path, "PNG")
        if progress is not None:
            progress(1.0, "fake")

    monkeypatch.setattr(upscaler, "upscale_image", fake)


def _make_window(app, source: Path):
    from app.gui.main_window import MainWindow

    win = MainWindow()
    win.show()
    app.processEvents()
    ok, _msg = win.add_path(str(source))
    assert ok is True
    assert _wait_until(lambda: win.preview._source_image is not None), "元画像の読み込み待ちが切れた"
    app.processEvents()
    return win


def test_wording_buttons_and_status(app, monkeypatch) -> None:
    """文言表の各場面（ボタン・札・状況行）。"""
    from app.gui.main_window import MainWindow

    calls: list = []
    _install_fake_upscale(monkeypatch, calls)
    win = MainWindow()
    win.show()
    app.processEvents()
    try:
        # 画像のクイック確認ボタン（画像・動画で同じ文言）。
        ok, _ = win.add_path(str(SAMPLE_IMAGE))
        assert ok
        assert _wait_until(lambda: win.preview._source_image is not None)
        app.processEvents()
        assert win.preview.trial_btn.text() == "クイック確認"
        assert win.preview.range_btn.text() == "範囲を選択してクイック確認"
        assert win.preview.fit_btn.toolTip() == "全体表示"
        assert win.preview.actual_btn.text() == "1:1"
        assert win.preview.actual_btn.toolTip() == "等倍"
        # 札は「元の画像」だけ（右は未確認なので出ない）。
        assert win.preview.view.left_label() == "元の画像"
        assert win.preview.view.left_tag.isVisible() is True
        assert win.preview.view.right_tag.isVisible() is False
        assert win.preview.status_label.text() == ""

        # 範囲選択モード中は画像の上に案内が出る（ボタン文言は変わらない）。
        win.preview.range_btn.click()
        app.processEvents()
        assert win.preview.range_btn.text() == "範囲を選択してクイック確認"
        assert win.preview.view.message_text() == "確認したい部分をドラッグで囲んでください"

        # 範囲ありの解除ボタン（ドラッグの代わりに直接範囲を入れる）。
        win.preview.view.set_selection_mode(False)
        win.preview._rect = (10, 10, 64, 64)
        win.preview._rebuild_tags()
        win.preview.refresh()
        win.preview._update_view()
        app.processEvents()
        assert win.preview.trial_btn.text() == "クイック確認"
        assert win.preview.range_btn.text() == "範囲を解除"
    finally:
        win.close()
        app.processEvents()


def test_wording_video_button_and_time(app, monkeypatch) -> None:
    """動画でも文言は同じで、位置の時刻表示の書式。"""
    from app.gui.main_window import MainWindow

    calls: list = []
    _install_fake_upscale(monkeypatch, calls)
    win = MainWindow()
    win.show()
    app.processEvents()
    try:
        assert DEMO_VIDEO.exists()
        ok, _ = win.add_path(str(DEMO_VIDEO))
        assert ok
        assert _wait_until(lambda: win.preview._source_image is not None, timeout=30.0)
        app.processEvents()
        assert win.preview.trial_btn.text() == "クイック確認"
        assert win.preview.range_btn.text() == "範囲を選択してクイック確認"
        assert win.preview.video_row.isVisible()
        text = win.preview.time_label.text()
        assert re.fullmatch(r"\d+:\d{2} / \d+:\d{2}", text), text
    finally:
        win.close()
        app.processEvents()


def test_trial_success_caches_and_shows_result(app, monkeypatch) -> None:
    """確認が終わったら右の札が結果に切り替わり、同じキーは再実行しない。"""
    from app.gui.main_window import MainWindow

    calls: list = []
    _install_fake_upscale(monkeypatch, calls)
    win = _make_window(app, SAMPLE_IMAGE)
    try:
        win.preview.trial_btn.click()
        assert _wait_until(lambda: not win.preview.is_trial_running(), timeout=30.0)
        app.processEvents()
        assert len(calls) == 1
        assert win.preview.status_label.text() == ""
        # 札: 左は元の画像、右はモデルの表示名。
        assert win.preview.view.left_label() == "元の画像"
        assert win.preview.view.right_label() == "Anime Video v3"
        assert win.preview.view.has_two()
        # 選べるものが 2 つあるので両方の札が押せて、メニューが付く。
        assert win.preview.view.left_tag.isEnabled()
        assert win.preview.view.right_tag.isEnabled()
        left_menu = win.preview.view.left_tag.menu()
        right_menu = win.preview.view.right_tag.menu()
        assert left_menu is not None and len(left_menu.actions()) == 2
        assert right_menu is not None and len(right_menu.actions()) == 2
        assert left_menu.actions()[0].text() == "元の画像"
        assert right_menu.actions()[1].text() == "Anime Video v3"

        # 同じキーをもう一度押したら再実行しない。
        win.preview.trial_btn.click()
        app.processEvents()
        assert len(calls) == 1
        assert win.preview.status_label.text() == ""
    finally:
        win.close()
        app.processEvents()


def test_trial_progress_cancel_and_failure_text(app, monkeypatch) -> None:
    """確認中の進捗・中止・失敗の状況行。"""
    from app.gui.main_window import MainWindow

    calls: list = []
    release = threading.Event()
    _install_fake_upscale(monkeypatch, calls, block=release)
    win = _make_window(app, SAMPLE_IMAGE)
    try:
        win.preview.trial_btn.click()
        assert _wait_until(lambda: win.preview.is_trial_running(), timeout=15.0)
        app.processEvents()
        assert win.preview.trial_btn.text() == "中止"
        assert _wait_until(
            lambda: win.preview.status_label.text() == "確認中… 45%"
        )
        assert win.preview.status_label.text() == "確認中… 45%"
        # 確認中は開始が無効になる。
        assert win.start_btn.isEnabled() is False
        win.preview.trial_btn.click()
        release.set()
        assert _wait_until(lambda: not win.preview.is_trial_running(), timeout=15.0)
        app.processEvents()
        assert win.preview.status_label.text() == "中止しました"
        assert win.start_btn.isEnabled() is True
    finally:
        release.set()
        win.close()
        app.processEvents()

    fail_calls: list = []
    _install_fake_upscale(monkeypatch, fail_calls, error="boom")
    win = _make_window(app, SAMPLE_IMAGE)
    try:
        win.preview.trial_btn.click()
        assert _wait_until(lambda: not win.preview.is_trial_running(), timeout=30.0)
        app.processEvents()
        assert win.preview.status_label.text() == "試せませんでした: boom"
    finally:
        win.close()
        app.processEvents()


def test_main_running_model_none_and_folder_status(app, monkeypatch, tmp_path) -> None:
    """本処理中・モデルなし・フォルダ選択中の状況行と無効化。"""
    from app.gui.main_window import MainWindow

    calls: list = []
    _install_fake_upscale(monkeypatch, calls)
    win = _make_window(app, SAMPLE_IMAGE)
    try:
        win.preview.set_main_running(True)
        app.processEvents()
        assert win.preview.status_label.text() == "処理中はクイック確認できません"
        assert win.preview.trial_btn.isEnabled() is False
        win.preview.set_main_running(False)
        app.processEvents()

        none_index = win.model_combo.findData(None)
        assert none_index >= 0
        win.model_combo.setCurrentIndex(none_index)
        app.processEvents()
        win.preview.refresh()
        assert win.preview.status_label.text() == "拡大モデルを選ぶとクイック確認できます"

        target = tmp_path / "folderjob"
        target.mkdir()
        ok, _ = win.add_path(str(target))
        assert ok
        job = next(job for job in win._jobs.values() if job.input_path == target)
        win.queue.select(job.id)
        app.processEvents()
        assert win.preview.status_label.text() == (
            "フォルダはクイック確認できません。中の画像を 1 枚追加すると確認できます。"
        )
    finally:
        win.close()
        app.processEvents()


def test_remove_file_discards_results(app, monkeypatch) -> None:
    """ファイルを一覧から消したら結果と一時ファイルも消える。"""
    from app.gui.main_window import MainWindow

    calls: list = []
    _install_fake_upscale(monkeypatch, calls)
    win = _make_window(app, SAMPLE_IMAGE)
    try:
        win.preview.trial_btn.click()
        assert _wait_until(lambda: not win.preview.is_trial_running(), timeout=30.0)
        app.processEvents()
        assert len(win.preview._results) == 1
        result_path = next(iter(win.preview._results.values()))["path"]
        assert Path(result_path).exists()
        job = next(iter(win._jobs.values()))
        job_id = job.id
        file_str = str(job.input_path)
        temps = set(win.preview._temp_by_file.get(file_str, set()))
        assert temps, "一時ファイルが記録されていない"
        win._on_remove_requested(job_id)
        app.processEvents()
        assert win.preview._results == {}
        assert win.preview.view.left_tag.isVisible() is False
        assert win.preview.view.right_tag.isVisible() is False
        assert not Path(result_path).exists()
        assert all(not Path(temp).exists() for temp in temps)
    finally:
        win.close()
        app.processEvents()


def test_compare_view_split_fit_and_selection_clamp(app) -> None:
    """比較ビューの境界線・倍率・選択範囲の丸め（オフスクリーン）。"""
    from PySide6.QtGui import QColor, QImage

    from app.gui.compare_view import CompareView

    view = CompareView()
    view.resize(400, 300)
    view.show()
    app.processEvents()
    small = QImage(100, 80, QImage.Format.Format_RGB888)
    small.fill(QColor("#112233"))
    big = QImage(200, 160, QImage.Format.Format_RGB888)
    big.fill(QColor("#445566"))
    view.set_images(small, "元の画像", big, "Anime Video v3")
    app.processEvents()
    assert view.has_two() is True
    assert view.left_label() == "元の画像"
    assert view.right_label() == "Anime Video v3"
    view.set_split(0.25)
    assert view.split() == 0.25
    view.fit_view()
    assert view.scale() > 0
    view.actual_pixels()
    assert view.scale() == 1.0
    # 画外の選択は丸める。
    view.set_selection((-10, -10, 50, 50))
    assert view.selection() == (0, 0, 40, 40)
    view.set_selection(None)
    assert view.selection() is None
    # 同じ画の左右は1枚表示にする。
    view.set_images(small, "元の画像", small, "元の画像")
    assert view.has_two() is False


def test_trial_workdir_removed_on_close(app, monkeypatch) -> None:
    """アプリ終了時に確認の作業フォルダを消す。"""
    from app.gui.main_window import MainWindow

    calls: list = []
    _install_fake_upscale(monkeypatch, calls)
    win = MainWindow()
    win.show()
    app.processEvents()
    workdir = Path(win.preview._workdir)
    assert workdir.is_dir()
    win.close()
    app.processEvents()
    assert not workdir.exists()


def test_position_change_switches_and_restores_results(app, monkeypatch) -> None:
    """位置を変えると結果が切り替わり、戻れば前の結果が選べる。"""
    from app.gui.main_window import MainWindow

    calls: list = []
    _install_fake_upscale(monkeypatch, calls)
    win = _make_window(app, DEMO_VIDEO)
    try:
        assert _wait_until(lambda: win.preview._duration, timeout=15.0)
        win.preview.trial_btn.click()
        assert _wait_until(lambda: not win.preview.is_trial_running(), timeout=30.0)
        app.processEvents()
        assert len(win.preview._matching_keys()) == 1
        first_key = win.preview._right_key
        assert win.preview.view.right_label() == "Anime Video v3"

        duration = win.preview._duration or 1.0
        moved = min(duration, win.preview._seconds + 1.0)
        win.preview._seconds = moved
        win.preview._status_hold = ""
        win.preview._rebuild_tags()
        win.preview.refresh()
        app.processEvents()
        assert win.preview._matching_keys() == []
        assert win.preview.view.right_tag.isVisible() is False

        win.preview._seconds = first_key[1]
        win.preview._status_hold = ""
        win.preview._rebuild_tags()
        win.preview.refresh()
        app.processEvents()
        assert len(win.preview._matching_keys()) == 1
        # 戻れば前の結果が menu から選べる。
        win.preview._on_tag_selected("right", first_key)
        app.processEvents()
        assert win.preview.view.right_label() == "Anime Video v3"
    finally:
        win.close()
        app.processEvents()


def test_compare_view_keeps_zoomed_spot_when_result_arrives(app):
    """拡大して見ている最中に 4 倍後の画が来ても、見ている場所が動かない。"""
    from PySide6.QtGui import QImage

    from app.gui.compare_view import CompareView

    view = CompareView()
    view.resize(800, 600)
    view.show()
    app.processEvents()

    before = QImage(200, 100, QImage.Format.Format_RGB32)
    before.fill(0x202020)
    after = QImage(800, 400, QImage.Format.Format_RGB32)
    after.fill(0x808080)

    view.set_images(before, "元の画像", None, "")
    view.actual_pixels()
    shown = view._dest_rect()

    view.set_images(before, "元の画像", after, "結果")
    assert view.has_two()
    moved = view._dest_rect()
    assert abs(moved.x() - shown.x()) < 0.5 and abs(moved.width() - shown.width()) < 0.5
    assert abs(view.scale() - 0.25) < 1e-6

    view.close()


def test_video_starts_at_two_seconds_or_the_middle_of_a_short_clip(app):
    """動画は 2 秒の位置から始める（冒頭の黒画面を避ける）。短い動画は真ん中。"""
    from pathlib import Path

    from app.core.jobs import Job, JobKind
    from app.gui.compare_view import TrialPanel

    panel = TrialPanel(build_settings=lambda *_a: None, model_label=str)
    panel._load_source_async = lambda: None  # コマの取り出しは回さない

    long_clip = Job(input_path=Path("long.mp4"), kind=JobKind.VIDEO)
    long_clip.fps, long_clip.frame_count = 30.0, 30 * 600
    panel.show_job(long_clip)
    assert panel._seconds == 2.0
    assert panel.time_label.text() == "0:02 / 10:00"

    short_clip = Job(input_path=Path("short.mp4"), kind=JobKind.VIDEO)
    short_clip.fps, short_clip.frame_count = 30.0, 30 * 3
    panel.show_job(short_clip)
    assert panel._seconds == 1.5

    panel.shutdown()


def test_tag_menu_switches_display(app, monkeypatch) -> None:
    """結果が 2 つあるとき札のメニューから切り替えられ、表示が変わる。"""
    from app.core.settings import HELPER_MODEL_ANIME, HELPER_MODEL_SPAN
    from app.gui.main_window import MainWindow

    calls: list = []
    _install_fake_upscale(monkeypatch, calls)
    win = _make_window(app, SAMPLE_IMAGE)
    try:
        win.preview.trial_btn.click()
        assert _wait_until(lambda: not win.preview.is_trial_running(), timeout=30.0)
        app.processEvents()
        assert len(win.preview._matching_keys()) == 1

        # モデルを変えてもう一度確認すると同じ位置に 2 つ目の結果ができる。
        span_index = win.model_combo.findData(HELPER_MODEL_SPAN)
        assert span_index >= 0
        win.model_combo.setCurrentIndex(span_index)
        app.processEvents()
        win.preview.trial_btn.click()
        assert _wait_until(lambda: not win.preview.is_trial_running(), timeout=30.0)
        app.processEvents()
        assert len(calls) == 2
        matches = win.preview._matching_keys()
        assert len(matches) == 2

        left_menu = win.preview.view.left_tag.menu()
        assert left_menu is not None and len(left_menu.actions()) == 3
        assert left_menu.actions()[0].text() == "元の画像"

        # 右の札を 1 つ目の結果に切り替える。
        first_key = next(key for key in matches if key[4] == HELPER_MODEL_ANIME)
        win.preview._on_tag_selected("right", first_key)
        app.processEvents()
        assert win.preview.view.right_label() == "Anime Video v3"
        win.preview._on_tag_selected("right", matches[-1])
        app.processEvents()
        assert win.preview.view.right_label() == "4xNomosUni SPAN"
    finally:
        win.close()
        app.processEvents()


def test_overlay_buttons_call_fit_and_actual(app, monkeypatch) -> None:
    """全体表示・1:1 のボタンが fit_view / actual_pixels を呼ぶ。"""
    from app.gui.main_window import MainWindow

    calls: list = []
    _install_fake_upscale(monkeypatch, calls)
    win = _make_window(app, SAMPLE_IMAGE)
    try:
        view = win.preview.view
        app.processEvents()
        assert view.fit_btn.isVisible()
        assert view.actual_btn.isVisible()
        assert view.actual_btn.text() == "1:1"

        view.actual_btn.click()
        app.processEvents()
        assert view.scale() == 1.0
        view.fit_btn.click()
        app.processEvents()
        assert view.scale() > 0
        assert view.scale() != 1.0
    finally:
        win.close()
        app.processEvents()


def test_quick_check_lives_in_right_column(app, monkeypatch) -> None:
    """クイック確認が右列にあり、ファイル未選択のときは見えない。"""
    from app.gui.main_window import MainWindow

    calls: list = []
    _install_fake_upscale(monkeypatch, calls)
    win = MainWindow()
    win.show()
    app.processEvents()
    try:
        # 中央のレイアウトにはクイック確認の組を置かない。
        assert win.preview.layout().indexOf(win.preview.quick_box) == -1
        # ファイル未選択のときは見えない。
        assert win.preview.quick_box.isVisible() is False

        ok, _ = win.add_path(str(SAMPLE_IMAGE))
        assert ok
        assert _wait_until(lambda: win.preview._source_image is not None)
        app.processEvents()
        assert win.preview.quick_box.isVisible() is True
        assert win.preview.trial_btn.objectName() == "accent"
        assert win.preview.trial_btn.text() == "クイック確認"

        # 動画でも文言は同じ。
        assert DEMO_VIDEO.exists()
        ok, _ = win.add_path(str(DEMO_VIDEO))
        assert ok
        job = next(job for job in win._jobs.values() if job.input_path == DEMO_VIDEO)
        win.queue.select(job.id)
        assert _wait_until(lambda: win.preview._source_image is not None, timeout=30.0)
        app.processEvents()
        assert win.preview.trial_btn.text() == "クイック確認"
        assert win.preview.range_btn.text() == "範囲を選択してクイック確認"
    finally:
        win.close()
        app.processEvents()


def test_range_finish_starts_check_at_once(app, monkeypatch) -> None:
    """範囲を囲み終えると押し直さずに確認が始まり、ボタンが範囲を解除になる。"""
    from app.gui.main_window import MainWindow

    calls: list = []
    _install_fake_upscale(monkeypatch, calls)
    win = _make_window(app, SAMPLE_IMAGE)
    try:
        win.preview.range_btn.click()
        app.processEvents()
        assert win.preview.view.is_selection_mode() is True
        # ドラッグの代わりに範囲を直接渡す（基底画基準）。
        win.preview._on_view_selection((10, 10, 64, 64))
        assert _wait_until(lambda: not win.preview.is_trial_running(), timeout=30.0)
        app.processEvents()
        assert len(calls) == 1
        assert win.preview._rect == (10, 10, 64, 64)
        assert win.preview.range_btn.text() == "範囲を解除"
    finally:
        win.close()
        app.processEvents()


def test_quick_help_text(app) -> None:
    """? に説明文が付いている。"""
    from app.gui.main_window import MainWindow

    win = MainWindow()
    win.show()
    app.processEvents()
    try:
        assert win.preview.quick_help.objectName() == "helpIcon"
        assert win.preview.quick_help.text() == "?"
        assert win.preview.quick_help.help_text == (
            "仕上がり確認のため、1 枚だけ拡大処理を行い、元の画像と比較できます。\n"
            "動画は、画像下のスライダーで拡大するフレームを選択できます。\n"
            "「範囲を選択してクイック確認」を使うと、その 1 枚のさらに一部分だけに処理を限定できます。"
        )
    finally:
        win.close()
        app.processEvents()

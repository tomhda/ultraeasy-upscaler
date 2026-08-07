"""GUI スモークテスト（オフスクリーン）。

表示なしで QApplication + MainWindow を生成し、Job をキューに投入して
行が現れることを確認する。実アップスケールは行わない。
"""
from __future__ import annotations

import os
import shutil

# QApplication 生成前にオフスクリーンを強制
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import pytest

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QCheckBox, QComboBox, QLabel

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_IMAGE = REPO_ROOT / "vendor" / "realesrgan" / "input.jpg"


@pytest.fixture(scope="module")
def app():
    application = QApplication.instance() or QApplication([])
    yield application
    # モジュール終了時に保留イベントを捌く
    application.processEvents()


def test_run_importable():
    """`from app.gui.main_window import run` が import できる。"""
    from app.gui.main_window import run  # noqa: F401

    assert callable(run)


def test_window_creates_and_adds_job(app):
    from app.gui.main_window import MainWindow

    win = MainWindow()
    win.show()
    app.processEvents()

    assert win.windowTitle().startswith("ultraeasy-upscaler")
    assert win.queue.row_count() == 0

    # サンプル画像をキューへ投入（vendor 同梱の input.jpg）
    assert SAMPLE_IMAGE.exists(), f"サンプル画像がありません: {SAMPLE_IMAGE}"
    ok, _msg = win.add_path(str(SAMPLE_IMAGE))
    assert ok is True

    app.processEvents()

    # 行が 1 つ現れる
    assert win.queue.row_count() == 1
    job = next(iter(win._jobs.values()))
    assert win.queue.has_job(job.id)
    row = win.queue.row(job.id)
    assert row is not None
    assert row.job.name == SAMPLE_IMAGE.name

    # 後始末（実行はしない）
    win.close()
    app.processEvents()


def test_drop_zone_click_adds_selected_file(app, monkeypatch):
    from app.gui import main_window

    def fake_get_open_file_names(*_args, **_kwargs):
        return [str(SAMPLE_IMAGE)], ""

    monkeypatch.setattr(
        main_window.QFileDialog, "getOpenFileNames", fake_get_open_file_names
    )

    win = main_window.MainWindow()
    win.show()
    app.processEvents()

    QTest.mouseClick(win.drop_zone, Qt.MouseButton.LeftButton)
    app.processEvents()

    assert win.queue.row_count() == 1
    job = next(iter(win._jobs.values()))
    assert job.input_path == SAMPLE_IMAGE

    win.close()
    app.processEvents()


def test_queue_progress_bars_align_for_different_file_names(app, tmp_path):
    from app.gui.main_window import MainWindow

    sources = [
        tmp_path / "a.jpg",
        tmp_path / "アイコン_緑強め.png",
        tmp_path / "very-very-long-file-name-that-should-not-push-the-bar.jpg",
    ]
    for src in sources:
        shutil.copy(SAMPLE_IMAGE, src)

    win = MainWindow()
    win.resize(1360, 780)
    win.show()
    app.processEvents()

    win.add_paths([str(src) for src in sources])
    app.processEvents()

    rows = [win.queue.row(job_id) for job_id in win._order]
    bar_x = {row._bar.geometry().x() for row in rows if row is not None}
    bar_widths = {row._bar.geometry().width() for row in rows if row is not None}

    assert len(rows) == 3
    assert len(bar_x) == 1
    assert len(bar_widths) == 1

    win.close()
    app.processEvents()


def test_build_settings_maps_widgets(app):
    """ウィジェット値が UpscaleSettings に反映される。"""
    from app.core.settings import OutputLocation, UpscaleBackend
    from app.gui.main_window import MainWindow

    win = MainWindow()
    app.processEvents()

    # 倍率 2x を選択
    win._set_scale(2)
    # 詳細設定の一部を変更
    win.drawer.tta_mode.setChecked(True)
    win.drawer.image_format.setCurrentText("webp")

    s = win.build_settings()
    assert s.scale == 2
    assert s.backend == UpscaleBackend.VULKAN
    assert s.tta_mode is True
    assert s.image_format == "webp"
    assert s.output_location == OutputLocation.SAME  # 既定は「元の場所」

    win.close()
    app.processEvents()


def test_upscale_and_interpolation_models_are_independent(app):
    from app.gui.main_window import MainWindow

    win = MainWindow()
    app.processEvents()

    win.model_combo.setCurrentIndex(0)  # なし（拡大しない）
    rife_index = win.interpolation_combo.findData("rife-v4.6")
    assert rife_index >= 0
    win.interpolation_combo.setCurrentIndex(rife_index)
    app.processEvents()

    settings = win.build_settings()
    assert settings.model is None
    assert settings.interpolation_model == "rife-v4.6"
    assert all(not button.isEnabled() for button in win._scale_btns.values())
    assert win.drawer.target_fps.isEnabled() is True

    win.interpolation_combo.setCurrentIndex(0)
    app.processEvents()
    settings = win.build_settings()
    assert settings.interpolation_model is None
    assert win.drawer.target_fps.isEnabled() is False

    win.close()
    app.processEvents()


def test_job_settings_apply_at_start_not_at_add(app):
    """設定は追加時ではなく「開始」時点のUI値が全保留ジョブへ適用される。"""
    from app.core.settings import DEFAULT_MODEL
    from app.gui.main_window import MainWindow

    win = MainWindow()
    app.processEvents()
    win.model_combo.setCurrentIndex(0)  # なし（拡大しない）
    rife_index = win.interpolation_combo.findData("rife-v4.6")
    assert rife_index >= 0
    win.interpolation_combo.setCurrentIndex(rife_index)

    ok, _ = win.add_path(str(SAMPLE_IMAGE))
    assert ok
    job = next(iter(win._jobs.values()))
    # 追加時点では固定されない
    assert job.settings is None

    # 追加後にUIを変更 → 開始時の適用でその値になる
    model_index = win.model_combo.findData(DEFAULT_MODEL)
    assert model_index >= 0
    win.model_combo.setCurrentIndex(model_index)
    win.interpolation_combo.setCurrentIndex(0)
    app.processEvents()

    win._apply_current_settings(win._pending_jobs())
    assert job.settings is not None
    assert job.settings.model == DEFAULT_MODEL
    assert job.settings.interpolation_model is None
    win.close()
    app.processEvents()


def test_model_combo_reenabled_after_leaving_npu(app):
    """NPUでモデル選択が無効化されても、GPUに戻すと再度選べる。"""
    from app.core.settings import UpscaleBackend
    from app.gui.main_window import MainWindow

    win = MainWindow()
    app.processEvents()

    npu = win.backend_combo.findData(UpscaleBackend.NPU.value)
    assert npu >= 0
    win.backend_combo.setCurrentIndex(npu)
    app.processEvents()
    assert win.model_combo.isEnabled() is False

    vulkan = win.backend_combo.findData(UpscaleBackend.VULKAN.value)
    assert vulkan >= 0
    win.backend_combo.setCurrentIndex(vulkan)
    app.processEvents()
    assert win.model_combo.isEnabled() is True

    win.close()
    app.processEvents()


def test_settings_drawer_explains_specialized_terms(app):
    from app.gui.main_window import MainWindow

    win = MainWindow()
    win.drawer.setVisible(True)
    app.processEvents()

    labels = [w.text() for w in win.drawer.findChildren(QLabel)]
    checks = [w.text() for w in win.drawer.findChildren(QCheckBox)]
    combo_items = [
        child.itemText(i)
        for child in win.drawer.findChildren(QComboBox)
        for i in range(child.count())
    ]
    visible_text = "\n".join(labels + checks + combo_items)
    help_icons = [
        w for w in win.drawer.findChildren(QLabel)
        if w.objectName() == "helpIcon"
    ]

    assert "動画の保存形式" in labels
    assert "動画の画質 (CRF/QP)" in labels
    assert "分割処理 (タイル)" in labels
    assert "出力フォルダ名" in labels
    assert "動画コンテナ" not in visible_text
    assert "サブフォルダ" not in visible_text
    assert win.drawer.tta_mode.text() == "高品質モード (TTA)"
    assert len(help_icons) >= 10
    assert all(icon.text() == "?" for icon in help_icons)
    assert all(getattr(icon, "help_text", "") for icon in help_icons)
    assert any("数字が小さいほど高画質" in icon.help_text for icon in help_icons)
    assert any("かなり遅く" in icon.help_text for icon in help_icons)

    quality_help = next(
        icon for icon in help_icons
        if "数字が小さいほど高画質" in icon.help_text
    )
    quality_help._show_popup()
    app.processEvents()
    assert quality_help._popup is not None
    assert quality_help._popup.text() == quality_help.help_text
    assert quality_help._popup.text() != "?"
    quality_help._hide_popup()
    assert win.drawer.hw_encode.objectName() == "clearCheck"
    assert win.drawer.hw_encode.isChecked() is True
    assert win.drawer.tta_mode.isChecked() is False

    win.drawer.tta_mode.setChecked(True)
    assert win.drawer.tta_mode.isChecked() is True

    idx = win.drawer.video_quality.findData(28)
    assert idx >= 0
    win.drawer.video_quality.setCurrentIndex(idx)
    s = win.build_settings()
    assert s.video_quality == 28

    win.close()
    app.processEvents()


def test_backend_combo_maps_to_npu_and_locks_scale(app):
    """NPU選択は設定へ反映され、倍率は4xに固定される。"""
    from app.core.settings import UpscaleBackend
    from app.gui.main_window import MainWindow

    win = MainWindow()
    app.processEvents()

    idx = win.backend_combo.findData(UpscaleBackend.NPU.value)
    assert idx >= 0
    win.backend_combo.setCurrentIndex(idx)
    app.processEvents()

    s = win.build_settings()
    assert s.backend == UpscaleBackend.NPU
    assert s.scale == 4
    assert win._scale_btns[2].isEnabled() is False
    assert win._scale_btns[4].isEnabled() is True

    win.close()
    app.processEvents()

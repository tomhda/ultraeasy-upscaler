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

# NPU キットの無い PC でも通るよう NPU ありに固定する（なしの表示は別に確かめる）。
pytestmark = pytest.mark.usefixtures("force_npu_available")


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


def test_build_settings_maps_widgets(app, tmp_path):
    """ウィジェット値が UpscaleSettings に反映される。"""
    import shutil as _shutil

    from app.core.settings import OutputLocation, UpscaleBackend
    from app.gui.main_window import MainWindow

    win = MainWindow()
    app.processEvents()
    try:
        src = tmp_path / "pic.png"
        _shutil.copy(SAMPLE_IMAGE, src)
        ok, _ = win.add_path(str(src))
        assert ok is True
        app.processEvents()

        # Vulkanでは従来モデルの倍率を選べる。
        vulkan = win.backend_combo.findData(UpscaleBackend.VULKAN.value)
        assert vulkan >= 0
        win.backend_combo.setCurrentIndex(vulkan)
        app.processEvents()

        # 倍率 2x を選択（右列で選んだファイルの個別になる）
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
    finally:
        win.close()
        app.processEvents()


def test_upscale_and_interpolation_models_are_independent(app, tmp_path):
    """拡大モデル「なし」と補間は独立し、補間の有無でfps欄が切り替わる。"""
    from app.gui.main_window import MainWindow

    win = MainWindow()
    app.processEvents()
    try:
        src = tmp_path / "clip.mp4"
        src.write_bytes(b"not a real video")
        ok, _ = win.add_path(str(src))
        assert ok is True
        app.processEvents()
        job = next(iter(win._jobs.values()))

        from app.core.settings import UpscaleBackend
        vulkan = win.backend_combo.findData(UpscaleBackend.VULKAN.value)
        win.backend_combo.setCurrentIndex(vulkan)
        app.processEvents()
        win.model_combo.setCurrentIndex(0)  # なし（拡大しない）
        rife_index = win.interpolation_combo.findData("rife-v4.6")
        assert rife_index >= 0
        win.interpolation_combo.setCurrentIndex(rife_index)
        app.processEvents()

        settings = win.build_settings(job)
        assert settings.model is None
        assert settings.interpolation_model == "rife-v4.6"
        assert all(not button.isEnabled() for button in win._scale_btns.values())
        assert win.drawer.target_fps.isEnabled() is True

        win.interpolation_combo.setCurrentIndex(0)
        app.processEvents()
        settings = win.build_settings(job)
        assert settings.interpolation_model is None
        assert win.drawer.target_fps.isEnabled() is False
    finally:
        win.close()
        app.processEvents()


def test_helper_model_none_is_available_and_saved_as_upscale_off(app):
    """新AIでも「なし」を選べ、RIFEのみのジョブ設定を作れる。"""
    from app.core.settings import (
        HELPER_MODEL_AMD_RRDB,
        HELPER_MODEL_ANIME,
        HELPER_MODEL_SPAN,
        HELPER_MODEL_SWINIR,
        HELPER_MODEL_ADCSR,
        UpscaleBackend,
    )
    from app.gui.main_window import MainWindow

    win = MainWindow()
    app.processEvents()
    try:
        # ヘッダーの画像一括に新AIモデルが並ぶ
        assert win.image_model_combo.findData(None) == 0
        assert win.image_model_combo.itemText(0) == "なし（拡大しない）"
        assert [win.image_model_combo.itemData(i) for i in range(win.image_model_combo.count())] == [
            None,
            HELPER_MODEL_ANIME,
            HELPER_MODEL_SPAN,
            HELPER_MODEL_AMD_RRDB,
            HELPER_MODEL_SWINIR,
            HELPER_MODEL_ADCSR,
        ]

        win.image_model_combo.setCurrentIndex(0)
        rife_index = win.global_interpolation_combo.findData("rife-v4.6")
        assert rife_index >= 0
        win.global_interpolation_combo.setCurrentIndex(rife_index)
        app.processEvents()

        settings = win.build_settings()
        assert settings.backend == UpscaleBackend.WINML_GPU
        assert settings.model is None
        assert settings.interpolation_model == "rife-v4.6"
        assert settings.upscale_enabled is False
        assert settings.interpolation_enabled is True

        ok, _ = win.add_path(str(SAMPLE_IMAGE))
        assert ok is True
        app.processEvents()
        # 一括「なし」に右列が追従する
        assert win.model_combo.currentData() is None
        assert win.model_combo.isEnabled() is True
        assert all(not button.isEnabled() for button in win._scale_btns.values())
        win._apply_current_settings(win._pending_jobs())
        job = next(iter(win._jobs.values()))
        assert job.settings is not None
        assert job.settings.model is None
        assert job.settings.interpolation_model == "rife-v4.6"

        # 「なし」から新AIモデルへ戻せることも確認する（右列で個別になる）。
        win.model_combo.setCurrentIndex(win.model_combo.findData(HELPER_MODEL_ANIME))
        app.processEvents()
        assert win.build_settings().model == HELPER_MODEL_ANIME
    finally:
        win.close()
        app.processEvents()


def test_job_settings_apply_at_start_not_at_add(app):
    """設定は追加時ではなく「開始」時点のUI値が全保留ジョブへ適用される。"""
    from app.core.settings import DEFAULT_MODEL
    from app.gui.main_window import MainWindow

    win = MainWindow()
    app.processEvents()
    try:
        from app.core.settings import UpscaleBackend
        vulkan = win.backend_combo.findData(UpscaleBackend.VULKAN.value)
        win.backend_combo.setCurrentIndex(vulkan)
        app.processEvents()
        # 一括を「なし」＋補間ありに
        win.image_model_combo.setCurrentIndex(0)  # なし（拡大しない）
        win.video_model_combo.setCurrentIndex(0)
        rife_index = win.global_interpolation_combo.findData("rife-v4.6")
        assert rife_index >= 0
        win.global_interpolation_combo.setCurrentIndex(rife_index)
        app.processEvents()

        ok, _ = win.add_path(str(SAMPLE_IMAGE))
        assert ok
        job = next(iter(win._jobs.values()))
        # 追加時点では固定されない
        assert job.settings is None

        # 追加後に一括を変更 → 開始時の適用でその値になる
        model_index = win.image_model_combo.findData(DEFAULT_MODEL)
        assert model_index >= 0
        win.image_model_combo.setCurrentIndex(model_index)
        win.global_interpolation_combo.setCurrentIndex(0)
        app.processEvents()

        win._apply_current_settings(win._pending_jobs())
        assert job.settings is not None
        assert job.settings.model == DEFAULT_MODEL
        assert job.settings.interpolation_model is None
    finally:
        win.close()
        app.processEvents()


def test_backend_combo_maps_new_helpers_and_limits_scale_to_4x(app, monkeypatch, tmp_path):
    """新AIの具体的3モデルがVulkan資産の有無にかかわらず選択できる。"""
    import shutil as _shutil

    from app.core import binaries
    from app.core.settings import (
        HELPER_MODEL_AMD_RRDB,
        HELPER_MODEL_ANIME,
        HELPER_MODEL_SPAN,
        HELPER_MODEL_SWINIR,
        HELPER_MODEL_ADCSR,
        UpscaleBackend,
    )
    from app.gui.main_window import MainWindow

    available_calls: list[bool] = []

    def no_vulkan_models():
        available_calls.append(True)
        return []

    monkeypatch.setattr(binaries, "available_models", no_vulkan_models)
    win = MainWindow()
    app.processEvents()
    try:
        src = tmp_path / "pic.png"
        _shutil.copy(SAMPLE_IMAGE, src)
        ok, _ = win.add_path(str(src))
        assert ok is True
        app.processEvents()

        assert available_calls == []
        npu = win.backend_combo.findData(UpscaleBackend.NPU_NATIVE.value)
        assert npu >= 0
        win.backend_combo.setCurrentIndex(npu)
        app.processEvents()

        s = win.build_settings()
        assert s.backend == UpscaleBackend.NPU_NATIVE
        assert win._scale_btns[2].isEnabled() is False
        assert win._scale_btns[4].isEnabled() is True
        assert win.image_model_combo.isEnabled() is True
        assert [win.image_model_combo.itemData(i) for i in range(win.image_model_combo.count())] == [
            None, HELPER_MODEL_ANIME, HELPER_MODEL_SPAN, HELPER_MODEL_AMD_RRDB,
            HELPER_MODEL_SWINIR,
            HELPER_MODEL_ADCSR,
        ]
        assert [win.video_model_combo.itemData(i) for i in range(win.video_model_combo.count())] == [
            None, HELPER_MODEL_ANIME, HELPER_MODEL_SPAN, HELPER_MODEL_AMD_RRDB,
            HELPER_MODEL_SWINIR,
        ]
        assert win.image_model_combo.itemText(1).startswith("Anime Video v3")
        assert win.image_model_combo.itemText(2).startswith("4xNomosUni SPAN")
        assert win.image_model_combo.itemText(3).startswith("Real-ESRGAN（AMD縮小版）")
        assert all(
            text not in "\n".join(
                win.image_model_combo.itemText(i) for i in range(win.image_model_combo.count())
            )
            for text in ("実写（質感重視）", "実写（くっきり）")
        )

        photo_idx = win.image_model_combo.findData(HELPER_MODEL_SPAN)
        win.image_model_combo.setCurrentIndex(photo_idx)
        assert win.build_settings().model == HELPER_MODEL_SPAN

        auto_idx = win.backend_combo.findData("auto")
        win.backend_combo.setCurrentIndex(auto_idx)
        app.processEvents()
        assert win.image_model_combo.isEnabled() is True
        assert win.image_model_combo.count() == 6
        assert win.build_settings().backend == UpscaleBackend.WINML_GPU

        vulkan = win.backend_combo.findData(UpscaleBackend.VULKAN.value)
        win.backend_combo.setCurrentIndex(vulkan)
        app.processEvents()
        assert available_calls == [True]
        assert win.image_model_combo.count() == 2
        assert win.image_model_combo.itemData(0) is None
    finally:
        win.close()
        app.processEvents()


def test_swinir_cuda_backend_only_offers_swinir_and_warns_about_speed(app, monkeypatch, tmp_path):
    import shutil as _shutil

    from app.core import helper_backend
    monkeypatch.setattr(helper_backend, "swinir_cuda_available", lambda: True)
    from app.core.settings import HELPER_MODEL_SWINIR, UpscaleBackend
    from app.gui.main_window import MainWindow

    win = MainWindow()
    app.processEvents()
    try:
        src = tmp_path / "pic.png"
        _shutil.copy(SAMPLE_IMAGE, src)
        ok, _ = win.add_path(str(src))
        assert ok is True
        app.processEvents()

        index = win.backend_combo.findData(UpscaleBackend.SWINIR_CUDA.value)
        assert index >= 0
        win.backend_combo.setCurrentIndex(index)
        app.processEvents()

        assert [win.image_model_combo.itemData(i) for i in range(win.image_model_combo.count())] == [
            None,
            HELPER_MODEL_SWINIR,
        ]
        settings = win.build_settings()
        assert settings.backend == UpscaleBackend.SWINIR_CUDA
        assert settings.model == HELPER_MODEL_SWINIR
        assert settings.scale == 4
        assert win._scale_btns[2].isEnabled() is False
        assert win._scale_btns[4].isEnabled() is True
        assert win.model_hint.text() == "実写の静止画向け・高精細・遅い"
        assert win.drawer.backend.findData(UpscaleBackend.SWINIR_CUDA.value) >= 0
    finally:
        win.close()
        app.processEvents()


def test_swinir_cuda_backend_is_hidden_without_runtime(app, monkeypatch):
    from app.core import helper_backend
    from app.core.settings import UpscaleBackend
    from app.gui.main_window import MainWindow

    monkeypatch.setattr(helper_backend, "swinir_cuda_available", lambda: False)
    win = MainWindow()
    try:
        assert win.backend_combo.findData(UpscaleBackend.SWINIR_CUDA.value) == -1
        assert win.drawer.backend.findData(UpscaleBackend.SWINIR_CUDA.value) == -1
    finally:
        win.close()
        app.processEvents()


def test_model_combo_filters_legacy_npu_api_is_removed_from_gui(app):
    """旧npu_worker用のNPU値は新しいGUIの選択肢に出さない。"""
    from app.core.settings import UpscaleBackend
    from app.gui.main_window import MainWindow

    win = MainWindow()
    app.processEvents()
    assert win.backend_combo.findData(UpscaleBackend.NPU.value) == -1
    win.close()
    app.processEvents()


def test_backend_combo_maps_to_npu_native_and_locks_scale(app, tmp_path):
    """NPU_NATIVE選択は設定へ反映され、倍率は4xに固定される。"""
    import shutil as _shutil

    from app.core.settings import UpscaleBackend
    from app.gui.main_window import MainWindow

    win = MainWindow()
    app.processEvents()
    try:
        src = tmp_path / "pic.png"
        _shutil.copy(SAMPLE_IMAGE, src)
        ok, _ = win.add_path(str(src))
        assert ok is True
        app.processEvents()

        idx = win.backend_combo.findData(UpscaleBackend.NPU_NATIVE.value)
        assert idx >= 0
        win.backend_combo.setCurrentIndex(idx)
        app.processEvents()

        assert win.image_model_combo.isEnabled() is True
        s = win.build_settings()
        assert s.backend == UpscaleBackend.NPU_NATIVE
        assert s.scale == 4
        assert win._scale_btns[2].isEnabled() is False
        assert win._scale_btns[4].isEnabled() is True
    finally:
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


def test_model_picker_shows_speed_quality_info(app, tmp_path):
    """モデルコンボにバッジ、説明行に 1 行の説明が出る。"""
    import shutil as _shutil

    from app.core.settings import (
        HELPER_MODEL_AMD_RRDB,
        HELPER_MODEL_ANIME,
        HELPER_MODEL_SPAN,
        HELPER_MODEL_SWINIR,
        HELPER_MODEL_ADCSR,
        UpscaleBackend,
    )
    from app.gui.main_window import MainWindow

    win = MainWindow()
    app.processEvents()
    try:
        src = tmp_path / "pic.png"
        _shutil.copy(SAMPLE_IMAGE, src)
        ok, _ = win.add_path(str(src))
        assert ok is True
        app.processEvents()

        # 自動（GPU優先）では実モデル名の新AIモデルを表示する。
        # ヘッダー（画像一括）と右列（選んだファイル）で同じ並びになる。
        assert [win.image_model_combo.itemData(i) for i in range(win.image_model_combo.count())] == [
            None, HELPER_MODEL_ANIME, HELPER_MODEL_SPAN, HELPER_MODEL_AMD_RRDB,
            HELPER_MODEL_SWINIR,
            HELPER_MODEL_ADCSR,
        ]
        assert [win.model_combo.itemData(i) for i in range(win.model_combo.count())] == [
            None, HELPER_MODEL_ANIME, HELPER_MODEL_SPAN, HELPER_MODEL_AMD_RRDB,
            HELPER_MODEL_SWINIR,
            HELPER_MODEL_ADCSR,
        ]
        assert win.image_model_combo.itemText(1).startswith("Anime Video v3")
        assert "速度◎" in win.image_model_combo.itemText(1)
        # 開いた一覧の印は今のまま残る。
        assert win.model_hint.text() == "アニメ向け・速い"

        # SPAN/AMD縮小版も同じ実モデル名の仕組みで選べる。
        # 右列で変えると個別になるが、説明行の内容は同じ規則で出る。
        win.model_combo.setCurrentIndex(win.model_combo.findData(HELPER_MODEL_SPAN))
        app.processEvents()
        assert win.model_combo.currentText().startswith("4xNomosUni SPAN")
        assert win.model_hint.text() == "実写向け・速い"
        win.model_combo.setCurrentIndex(win.model_combo.findData(HELPER_MODEL_AMD_RRDB))
        app.processEvents()
        assert win.model_combo.currentText().startswith("Real-ESRGAN（AMD縮小版）")
        assert win.model_hint.text() == "実写向け・くっきり・やや遅い"

        # Vulkanでは旧モデルのバッジは残るが、説明は 1 行になる。
        vulkan = win.backend_combo.findData(UpscaleBackend.VULKAN.value)
        win.backend_combo.setCurrentIndex(vulkan)
        app.processEvents()
        idx = win.image_model_combo.findData("realesrgan-x4plus")
        assert idx >= 0
        assert "速" in win.image_model_combo.itemText(idx)
        assert win.model_hint.text() == "実写向け・高画質・遅い"

        # NPU_NATIVEに切替 → 説明は実行先によらず同じ 1 行。
        npu = win.backend_combo.findData(UpscaleBackend.NPU_NATIVE.value)
        win.backend_combo.setCurrentIndex(npu)
        app.processEvents()
        assert win.model_combo.currentText().startswith("Anime Video v3")
        assert win.model_hint.text() == "アニメ向け・速い"

        # モデル「なし」では補間のみの案内
        win.backend_combo.setCurrentIndex(
            win.backend_combo.findData(UpscaleBackend.VULKAN.value))
        app.processEvents()
        win.image_model_combo.setCurrentIndex(0)
        win.video_model_combo.setCurrentIndex(0)
        app.processEvents()
        assert "フレーム補間だけ" in win.model_hint.text()
    finally:
        win.close()
        app.processEvents()


def test_workspace_switches_between_drop_zone_media_and_details(app):
    from app.gui.main_window import MainWindow

    win = MainWindow()
    win.show()
    app.processEvents()

    # 空のときは左＋中央の全面がドロップ枠。開始ボタンは常に見えている。
    assert win.stack.currentIndex() == 0
    assert win.start_btn.isVisible()

    win.add_paths([str(SAMPLE_IMAGE)])
    app.processEvents()
    assert win.stack.currentIndex() == 1
    first = win._order[0]
    assert win.queue.selected_id() == first

    # 歯車で詳細設定に切り替わり、もう一度押すと一覧＋プレビューへ戻る。
    win._toggle_drawer()
    assert win.stack.currentIndex() == 2
    assert win.start_btn.isVisible()
    win._toggle_drawer()
    assert win.stack.currentIndex() == 1

    # 最後の 1 件を消すとドロップ枠へ戻り、選択も外れる。
    win._on_remove_requested(first)
    app.processEvents()
    assert win.stack.currentIndex() == 0
    assert win.queue.selected_id() is None

    win.close()
    app.processEvents()


def test_removing_selected_row_moves_selection_to_remaining_row(app, tmp_path):
    from app.gui.main_window import MainWindow

    sources = [tmp_path / "a.jpg", tmp_path / "b.jpg"]
    for src in sources:
        shutil.copy(SAMPLE_IMAGE, src)

    win = MainWindow()
    win.show()
    win.add_paths([str(src) for src in sources])
    app.processEvents()
    first, second = win._order

    win.queue.select(second)
    assert win.queue.selected_id() == second
    win._on_remove_requested(second)
    app.processEvents()
    assert win.queue.selected_id() == first
    assert win.queue.row_count() == 1

    win.close()
    app.processEvents()


def test_dropping_anywhere_on_the_window_adds_files(app):
    """左下の小さい枠だけでなく、ウィンドウ全体がドロップ先になっている。"""
    from PySide6.QtCore import QMimeData, QPoint, QUrl
    from PySide6.QtGui import QDragEnterEvent, QDropEvent

    from app.gui.main_window import MainWindow

    win = MainWindow()
    win.show()
    win.add_paths([str(SAMPLE_IMAGE)])
    app.processEvents()
    assert win.acceptDrops()
    # 一覧・プレビュー・設定の各列は自分ではドロップを受けず、ウィンドウに任せる
    for area in (win.media_card, win.queue, win.preview, win.preview.view):
        assert not area.acceptDrops()

    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(SAMPLE_IMAGE))])
    pos = QPoint(win.width() // 2, win.height() // 2)  # 中央のプレビューの上
    enter = QDragEnterEvent(pos, Qt.DropAction.CopyAction, mime,
                            Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    win.dragEnterEvent(enter)
    assert enter.isAccepted()
    assert win.media_card.property("dragActive") is True

    drop = QDropEvent(pos, Qt.DropAction.CopyAction, mime,
                      Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    win.dropEvent(drop)
    app.processEvents()
    assert win.queue.row_count() == 2
    assert win.media_card.property("dragActive") is False

    win.close()
    app.processEvents()


def test_media_header_buttons_are_icon_only(app):
    """見出しの 2 ボタンは文字なし・ツールチップ付き。削除だけ別の見た目。"""
    from app.gui.main_window import MainWindow

    win = MainWindow()
    win.show()
    app.processEvents()
    try:
        assert win.retry_all_btn.text() == ""
        assert win.clear_btn.text() == ""
        assert win.retry_all_btn.toolTip() == "すべてやり直す"
        assert win.clear_btn.toolTip() == "すべて削除"
        assert win.clear_btn.objectName() != win.retry_all_btn.objectName()
        assert win.clear_btn.objectName() == "dangerIcon"
    finally:
        win.close()
        app.processEvents()


def test_header_has_no_output_button(app):
    """ヘッダーに「出力先を開く」が無い。"""
    from PySide6.QtWidgets import QPushButton

    from app.gui.main_window import MainWindow

    win = MainWindow()
    win.show()
    app.processEvents()
    try:
        assert not hasattr(win, "output_open_btn")
        texts = [w.text() for w in win.findChildren(QPushButton)]
        assert "出力先を開く" not in texts
    finally:
        win.close()
        app.processEvents()


def test_completed_row_has_output_button(app, tmp_path):
    """完了した行にだけ保存先ボタンが出て、押すとフォルダを開こうとする。"""
    import shutil as _shutil

    from app.core.jobs import JobStatus
    from app.gui.main_window import MainWindow

    win = MainWindow()
    win.show()
    app.processEvents()
    try:
        src = tmp_path / "pic.png"
        _shutil.copy(SAMPLE_IMAGE, src)
        out = tmp_path / "pic_upscaled.png"
        out.write_bytes(b"fake")
        ok, _ = win.add_path(str(src))
        assert ok is True
        app.processEvents()
        job = next(iter(win._jobs.values()))
        row = win.queue.row(job.id)
        assert row is not None
        # 未完了の行には出ない。
        assert row._folder.isVisible() is False

        job.status = JobStatus.DONE
        job.output_path = out
        job.message = win._done_text(job)
        win.queue.refresh(job.id)
        app.processEvents()
        assert row._folder.isVisible() is True
        assert row._folder.toolTip() == "保存先を開く"

        launched: list = []
        win._launch_explorer = launched.append  # type: ignore[method-assign]
        row._folder.click()
        app.processEvents()
        # 出力ファイルが残っていればそれを選んだ状態で開く。
        assert launched == [["explorer", "/select,", str(out)]]

        # 出力ファイルが消えていればフォルダだけ開く。
        out.unlink()
        launched.clear()
        row._folder.click()
        app.processEvents()
        assert launched == [["explorer", str(tmp_path)]]

        # 実行中でも押せる。
        win._set_running(True)
        app.processEvents()
        assert row._folder.isEnabled() is True
        win._set_running(False)
        app.processEvents()
    finally:
        win.close()
        app.processEvents()


def test_model_hints_are_single_line_ja_and_en(app, tmp_path):
    """各モデルの説明が表の 1 行になる（日本語・英語）。"""
    import shutil as _shutil

    from app import i18n
    from app.core.settings import UpscaleBackend
    from app.gui.main_window import MainWindow

    win = MainWindow()
    win.show()
    app.processEvents()
    try:
        src = tmp_path / "pic.png"
        _shutil.copy(SAMPLE_IMAGE, src)
        ok, _ = win.add_path(str(src))
        assert ok is True
        app.processEvents()

        vulkan = win.backend_combo.findData(UpscaleBackend.VULKAN.value)
        win.backend_combo.setCurrentIndex(vulkan)
        app.processEvents()

        table = {
            "realesr-animevideov3": ("アニメ向け・速い", "For anime · Fast"),
            "realesr-general-x4v3": (
                "実写・アニメ兼用・ノイズ除去強め",
                "For live action and anime · Strong denoising",
            ),
            "realesr-general-wdn-x4v3": (
                "実写向け・ノイズ除去弱め",
                "For live action · Light denoising",
            ),
            "realesrgan-x4plus": (
                "実写向け・高画質・遅い",
                "For live action · High quality · Slow",
            ),
            "realesrgan-x4plus-anime": (
                "アニメ向け・高画質・遅い",
                "For anime · High quality · Slow",
            ),
        }
        for key, (ja, en) in table.items():
            idx = win.model_combo.findData(key)
            if idx < 0:
                continue
            win.model_combo.setCurrentIndex(idx)
            app.processEvents()
            assert win.model_hint.text() == ja, key

        i18n.set_language("en")
        try:
            win.preview._rebuild_tags()
            win.preview.refresh()
            app.processEvents()
            for key, (ja, en) in table.items():
                idx = win.model_combo.findData(key)
                if idx < 0:
                    continue
                win.model_combo.setCurrentIndex(idx)
                app.processEvents()
                win._update_all_model_info()
                assert win.model_hint.text() == en, key
            # クイック確認の文言も英語になる。
            assert win.preview.trial_btn.text() == "Quick check"
            assert win.preview.view.left_label() == "Original"
        finally:
            i18n.set_language("ja")
            win._update_all_model_info()
            win.preview.refresh()
            app.processEvents()
    finally:
        win.close()
        app.processEvents()

"""GUI スモークテスト（オフスクリーン）。

表示なしで QApplication + MainWindow を生成し、Job をキューに投入して
行が現れることを確認する。実アップスケールは行わない。
"""
from __future__ import annotations

import os

# QApplication 生成前にオフスクリーンを強制
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


def test_build_settings_maps_widgets(app):
    """ウィジェット値が UpscaleSettings に反映される。"""
    from app.core.settings import OutputLocation
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
    assert s.tta_mode is True
    assert s.image_format == "webp"
    assert s.output_location == OutputLocation.SAME  # 既定は「元の場所」

    win.close()
    app.processEvents()

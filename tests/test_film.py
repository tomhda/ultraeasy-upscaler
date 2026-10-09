"""FILM (Style) 補間の載せ込み。実行はせず、選択・計算・文言を確かめる。"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from app.core import binaries, engine, film, interpolator, media, video
from app.core.jobs import Job, JobKind
from app.core.settings import UpscaleSettings


@pytest.fixture(scope="module")
def app():
    application = QApplication.instance() or QApplication([])
    yield application
    application.processEvents()


def _patch_interpolation(monkeypatch, models: list[str]) -> None:
    monkeypatch.setattr(
        binaries, "available_interpolation_models", lambda: list(models)
    )


# --- 倍率の決め方 ---

def test_film_factor_defaults_to_double() -> None:
    assert film.resolve_factor(UpscaleSettings(target_fps=None), 23.976) == 2
    assert (
        film.resolve_factor(
            UpscaleSettings(target_fps=None, interpolation_factor=4), 23.976
        )
        == 4
    )
    assert (
        film.resolve_factor(
            UpscaleSettings(target_fps=None, interpolation_factor=8), 30.0
        )
        == 8
    )


def test_film_factor_matches_named_fps() -> None:
    assert film.resolve_factor(UpscaleSettings(target_fps=120), 30) == 4
    assert film.resolve_factor(UpscaleSettings(target_fps=60), 30) == 2
    assert film.resolve_factor(UpscaleSettings(target_fps=240), 30) == 8


def test_film_factor_rejects_fractional_fps() -> None:
    with pytest.raises(ValueError, match="2 倍・4 倍・8 倍"):
        film.resolve_factor(UpscaleSettings(target_fps=60), 24)
    with pytest.raises(ValueError, match="元: 24.000fps / 指定: 60.000fps"):
        film.resolve_factor(UpscaleSettings(target_fps=60), 24)
    with pytest.raises(ValueError, match="元動画のfpsを取得できません"):
        film.resolve_factor(UpscaleSettings(target_fps=None), 0.0)


def test_rife_uses_interpolation_factor_without_target() -> None:
    assert (
        interpolator.requested_fps(
            UpscaleSettings(target_fps=None, interpolation_factor=4), 30.0
        )
        == 120.0
    )
    assert interpolator.requested_fps(
        UpscaleSettings(target_fps=None), 23.976
    ) == pytest.approx(47.952)
    assert (
        interpolator.requested_fps(
            UpscaleSettings(target_fps=60.0, interpolation_factor=8), 30.0
        )
        == 60.0
    )


def test_output_suffix_follows_factor() -> None:
    assert (
        UpscaleSettings(
            model="realesr-animevideov3",
            scale=4,
            interpolation_model="film-style",
        ).output_suffix()
        == "_x4_FILM-Style_2xfps"
    )
    assert (
        UpscaleSettings(
            model="realesr-animevideov3",
            scale=4,
            interpolation_model="film-style",
            interpolation_factor=4,
        ).output_suffix()
        == "_x4_FILM-Style_4xfps"
    )
    assert (
        UpscaleSettings(
            model=None, interpolation_model="rife-v4.6", target_fps=60.0
        ).output_suffix()
        == "_RIFE-v4.6_60fps"
    )


# --- 資材の有無と一覧 ---

def test_film_is_listed_only_with_both_model_and_helper(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(binaries, "repo_root", lambda: tmp_path)
    monkeypatch.delenv("UEU_FILM_HELPER", raising=False)
    monkeypatch.delenv("UEU_FILM_MODEL", raising=False)
    helper = tmp_path / "vendor" / "winml-film" / "winml-film.exe"
    model = tmp_path / "models" / "film" / "film_style_fp32.onnx"
    helper.parent.mkdir(parents=True)
    model.parent.mkdir(parents=True)
    binaries.available_interpolation_models.cache_clear()
    try:
        assert binaries.FILM_MODEL not in binaries.available_interpolation_models()
        helper.touch()
        binaries.available_interpolation_models.cache_clear()
        assert binaries.FILM_MODEL not in binaries.available_interpolation_models()
        model.touch()
        binaries.available_interpolation_models.cache_clear()
        assert binaries.FILM_MODEL in binaries.available_interpolation_models()
    finally:
        binaries.available_interpolation_models.cache_clear()


def test_interpolator_routes_film_without_rife(monkeypatch) -> None:
    called = []

    def fake_film(input_dir, output_dir, settings, source_fps, progress=None, cancel=None):
        called.append((input_dir, output_dir, source_fps))
        return 8, 60.0

    monkeypatch.setattr(film, "interpolate_folder", fake_film)
    result = interpolator.interpolate_folder(
        "source",
        "output",
        UpscaleSettings(interpolation_model=binaries.FILM_MODEL),
        30.0,
    )
    assert result == (8, 60.0)
    assert called == [("source", "output", 30.0)]


def test_unsupported_film_fps_fails_before_extracting_frames(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        media,
        "probe",
        lambda _path: media.MediaInfo(kind="video", width=128, height=72, fps=24.0),
    )
    monkeypatch.setattr(
        video,
        "extract_frames",
        lambda *_args, **_kwargs: pytest.fail("unsupported fps must fail before extraction"),
    )
    settings = UpscaleSettings(
        model=None, interpolation_model=binaries.FILM_MODEL, target_fps=60.0
    )
    job = Job(input_path=tmp_path / "24fps.mp4", kind=JobKind.VIDEO)
    with pytest.raises(ValueError, match="2 倍・4 倍・8 倍"):
        engine.process_job(job, settings)


# --- 画面: 補間モデルの 3 択 ---

def _make_video_window(app, tmp_path: Path):
    from app.gui.main_window import MainWindow

    win = MainWindow()
    win.show()
    app.processEvents()
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"not a real video")
    ok, _ = win.add_path(str(src))
    assert ok is True
    app.processEvents()
    job = next(iter(win._jobs.values()))
    assert job.kind == JobKind.VIDEO
    return win, job


def test_interpolation_combos_list_rife_and_film(app, monkeypatch, tmp_path) -> None:
    from app.gui.main_window import MainWindow

    _patch_interpolation(monkeypatch, ["rife-v4.6", binaries.FILM_MODEL])
    win = MainWindow()
    try:
        for combo in (win.global_interpolation_combo, win.interpolation_combo):
            assert [combo.itemData(i) for i in range(combo.count())] == [
                None,
                "rife-v4.6",
                binaries.FILM_MODEL,
            ]
            assert combo.itemText(0) == "なし（補間しない）"
            assert combo.itemText(1) == "RIFE v4.6"
            assert combo.itemText(2) == "FILM (Style)"
        film_index = win.global_interpolation_combo.findData(binaries.FILM_MODEL)
        win.global_interpolation_combo.setCurrentIndex(film_index)
        app.processEvents()
        assert win.build_settings().interpolation_model == binaries.FILM_MODEL
        assert win.drawer.target_fps.isEnabled() is True
    finally:
        win.close()
        app.processEvents()


def test_interpolation_combos_show_uninstalled_film_disabled(
    app, monkeypatch, tmp_path
) -> None:
    from app.gui.main_window import MainWindow

    _patch_interpolation(monkeypatch, [])
    win = MainWindow()
    try:
        for combo in (win.global_interpolation_combo, win.interpolation_combo):
            index = combo.findData(binaries.FILM_MODEL)
            assert index >= 0
            assert combo.itemText(index) == "FILM (Style)（未導入）"
            item = combo.model().item(index)
            assert item is not None
            assert item.isEnabled() is False
            assert (
                item.toolTip()
                == "詳細設定の「追加キット」からダウンロードできます"
            )
    finally:
        win.close()
        app.processEvents()


def test_interpolation_hint_shows_next_to_right_combo(
    app, monkeypatch, tmp_path
) -> None:
    _patch_interpolation(monkeypatch, ["rife-v4.6", binaries.FILM_MODEL])
    win, job = _make_video_window(app, tmp_path)
    try:
        # なしのときは行を出さない（場所も取らない）。
        assert win.interpolation_hint.isVisible() is False
        assert win.interpolation_hint.text() == ""

        # 一括で選んだ値は個別の無いファイルの右列にも出る。
        rife = win.global_interpolation_combo.findData("rife-v4.6")
        win.global_interpolation_combo.setCurrentIndex(rife)
        app.processEvents()
        assert win.interpolation_combo.currentData() == "rife-v4.6"
        assert win.interpolation_hint.isVisible() is True
        assert win.interpolation_hint.text() == "高速"
        assert win.global_interpolation_combo.toolTip() == "高速"

        # 右列で変えると個別になり、行の説明だけが変わる。
        film_index = win.interpolation_combo.findData(binaries.FILM_MODEL)
        win.interpolation_combo.setCurrentIndex(film_index)
        app.processEvents()
        assert win.interpolation_combo.currentData() == binaries.FILM_MODEL
        assert win.interpolation_hint.text() == "低速・高品質"
        assert win.global_interpolation_combo.toolTip() == "高速"
    finally:
        win.close()
        app.processEvents()


def test_interpolation_hint_in_english(app, monkeypatch, tmp_path) -> None:
    from app.i18n import set_language

    _patch_interpolation(monkeypatch, ["rife-v4.6", binaries.FILM_MODEL])
    win, _job = _make_video_window(app, tmp_path)
    set_language("en")
    try:
        rife = win.global_interpolation_combo.findData("rife-v4.6")
        win.global_interpolation_combo.setCurrentIndex(rife)
        app.processEvents()
        assert win.interpolation_hint.text() == "Fast"
        assert win.global_interpolation_combo.toolTip() == "Fast"

        film_index = win.interpolation_combo.findData(binaries.FILM_MODEL)
        win.interpolation_combo.setCurrentIndex(film_index)
        app.processEvents()
        assert win.interpolation_hint.text() == "Slow · High quality"
    finally:
        set_language("ja")
        win.close()
        app.processEvents()


def test_film_override_follows_bulk_rules(app, monkeypatch, tmp_path) -> None:
    """補間 3 択でも個別・一括・ほかの動画にも使う・戻すは同じ規則で働く。"""
    _patch_interpolation(monkeypatch, ["rife-v4.6", binaries.FILM_MODEL])
    win, job = _make_video_window(app, tmp_path)
    try:
        # 右列で FILM を選ぶと個別になる。
        film_right = win.interpolation_combo.findData(binaries.FILM_MODEL)
        win.interpolation_combo.setCurrentIndex(film_right)
        app.processEvents()
        assert win._overrides[job.id]["interpolation"] == binaries.FILM_MODEL
        assert win.build_settings(job).interpolation_model == binaries.FILM_MODEL

        # ほかの動画にも使うと一括へ写り、個別は消える。
        win._on_apply_all()
        app.processEvents()
        assert job.id not in win._overrides
        assert win._video_interpolation == binaries.FILM_MODEL
        assert win.global_interpolation_combo.currentData() == binaries.FILM_MODEL

        # 右列で一括と同じに戻すと個別は作られない。
        win.interpolation_combo.setCurrentIndex(film_right)
        app.processEvents()
        assert job.id not in win._overrides

        # 一括を RIFE に変えると個別の無い行は追従する。
        rife = win.global_interpolation_combo.findData("rife-v4.6")
        win.global_interpolation_combo.setCurrentIndex(rife)
        app.processEvents()
        assert win.build_settings(job).interpolation_model == "rife-v4.6"
    finally:
        win.close()
        app.processEvents()


def test_describe_settings_shows_film_name(app, monkeypatch, tmp_path) -> None:
    _patch_interpolation(monkeypatch, ["rife-v4.6", binaries.FILM_MODEL])
    win, job = _make_video_window(app, tmp_path)
    try:
        film_index = win.global_interpolation_combo.findData(binaries.FILM_MODEL)
        win.global_interpolation_combo.setCurrentIndex(film_index)
        app.processEvents()
        settings = win.build_settings(job)
        assert "FILM (Style)" in win._describe_settings(settings)
        assert "film-style" not in win._describe_settings(settings)
    finally:
        win.close()
        app.processEvents()


# --- 画面: 補間の倍率 ---

def test_fps_row_hidden_by_default(app, monkeypatch, tmp_path) -> None:
    _patch_interpolation(monkeypatch, ["rife-v4.6", binaries.FILM_MODEL])
    win, job = _make_video_window(app, tmp_path)
    try:
        drawer = win.drawer
        assert drawer.detail_fps_check.isChecked() is False
        assert drawer.target_fps_label.isHidden() is True
        assert drawer.target_fps.isHidden() is True
        settings = win.build_settings(job)
        assert settings.target_fps is None
        assert settings.interpolation_factor == 2
    finally:
        win.close()
        app.processEvents()


def test_fps_detail_offers_five_choices(app, monkeypatch, tmp_path) -> None:
    _patch_interpolation(monkeypatch, ["rife-v4.6", binaries.FILM_MODEL])
    win, job = _make_video_window(app, tmp_path)
    try:
        drawer = win.drawer
        drawer.detail_fps_check.setChecked(True)
        app.processEvents()
        assert drawer.target_fps_label.isHidden() is False
        assert drawer.target_fps.isHidden() is False
        assert [drawer.target_fps.itemText(i) for i in range(drawer.target_fps.count())] == [
            "元動画の2倍",
            "元動画の4倍",
            "元動画の8倍",
            "60 fps",
            "120 fps",
        ]

        drawer.target_fps.setCurrentIndex(drawer.target_fps.findData("x4"))
        settings = win.build_settings(job)
        assert settings.target_fps is None
        assert settings.interpolation_factor == 4

        drawer.target_fps.setCurrentIndex(drawer.target_fps.findData("x8"))
        settings = win.build_settings(job)
        assert settings.interpolation_factor == 8

        drawer.target_fps.setCurrentIndex(drawer.target_fps.findData("fps60"))
        settings = win.build_settings(job)
        assert settings.target_fps == 60.0
        assert settings.interpolation_factor == 2

        # 切ると元動画の 2 倍に戻る。
        drawer.detail_fps_check.setChecked(False)
        app.processEvents()
        assert drawer.target_fps.currentData() == "x2"
        settings = win.build_settings(job)
        assert settings.target_fps is None
        assert settings.interpolation_factor == 2
    finally:
        win.close()
        app.processEvents()


def _fps_item(drawer, key):
    index = drawer.target_fps.findData(key)
    assert index >= 0
    item = drawer.target_fps.model().item(index)
    assert item is not None
    return item


def test_film_disables_fixed_fps_choices(app, monkeypatch, tmp_path) -> None:
    _patch_interpolation(monkeypatch, ["rife-v4.6", binaries.FILM_MODEL])
    win, _job = _make_video_window(app, tmp_path)
    try:
        drawer = win.drawer
        drawer.detail_fps_check.setChecked(True)
        drawer.target_fps.setCurrentIndex(drawer.target_fps.findData("fps60"))
        app.processEvents()

        # 一括で FILM を選ぶと 60/120 は選べず、2 倍に戻る。
        film_bulk = win.global_interpolation_combo.findData(binaries.FILM_MODEL)
        win.global_interpolation_combo.setCurrentIndex(film_bulk)
        app.processEvents()
        for key in ("fps60", "fps120"):
            item = _fps_item(drawer, key)
            assert item.isEnabled() is False
            assert item.toolTip() == "FILM (Style) は元動画の倍数だけに対応します"
        assert drawer.target_fps.currentData() == "x2"

        # RIFE に戻すと選べるようになる。
        rife_bulk = win.global_interpolation_combo.findData("rife-v4.6")
        win.global_interpolation_combo.setCurrentIndex(rife_bulk)
        app.processEvents()
        for key in ("fps60", "fps120"):
            assert _fps_item(drawer, key).isEnabled() is True

        # 個別で FILM を選んだ間も同じ。
        film_right = win.interpolation_combo.findData(binaries.FILM_MODEL)
        win.interpolation_combo.setCurrentIndex(film_right)
        app.processEvents()
        assert _fps_item(drawer, "fps60").isEnabled() is False
    finally:
        win.close()
        app.processEvents()


# --- 画面: 追加キット ---

def _canned_kits(installed: dict[str, bool]):
    from app.core.addon_kits import AddonKit

    return [
        AddonKit("film", "FILM (Style)", "178 MB",
                 "ultraeasy-upscaler-film-kit.zip", installed["film"]),
        AddonKit("npu", "NPU キット", "56 MB",
                 "ultraeasy-upscaler-npu-kit.zip", installed["npu"]),
        AddonKit("adcsr_gpu", "AdcSR（GPU 用）", "1.7 GB",
                 "ultraeasy-upscaler-adcsr-kit.zip", installed["adcsr_gpu"]),
        AddonKit("adcsr_npu", "AdcSR（NPU 用）", "1.7 GB",
                 "ultraeasy-upscaler-npu-kit-adcsr.zip", installed["adcsr_npu"]),
    ]


def test_kit_rows_follow_asset_presence(app, monkeypatch) -> None:
    from app.core import addon_kits
    from app.gui.settings_drawer import SettingsDrawer

    monkeypatch.setattr(
        addon_kits, "addon_kits",
        lambda: _canned_kits(
            {"film": True, "npu": False, "adcsr_gpu": True, "adcsr_npu": False}
        ),
    )
    drawer = SettingsDrawer()
    app.processEvents()
    try:
        assert set(drawer.kit_rows) == {"film", "npu", "adcsr_gpu", "adcsr_npu"}
        assert drawer.kit_rows["film"]["status"].text() == "導入済み"
        assert drawer.kit_rows["npu"]["status"].text() == "未導入"
        assert drawer.kit_rows["film"]["name"].text() == "FILM (Style)"
        assert drawer.kit_rows["adcsr_gpu"]["name"].text() == "AdcSR（GPU 用）"
        assert drawer.kit_rows["film"]["button"].text() == "ダウンロード"

        monkeypatch.setattr(
            addon_kits, "addon_kits",
            lambda: _canned_kits(
                {"film": False, "npu": True, "adcsr_gpu": False, "adcsr_npu": True}
            ),
        )
        drawer.refresh_kit_rows()
        app.processEvents()
        assert drawer.kit_rows["film"]["status"].text() == "未導入"
        assert drawer.kit_rows["npu"]["status"].text() == "導入済み"
    finally:
        drawer.close()
        app.processEvents()


def test_kit_button_opens_release_url(app, monkeypatch) -> None:
    from PySide6.QtGui import QDesktopServices

    from app.core import addon_kits
    from app.gui.settings_drawer import SettingsDrawer

    opened: list[str] = []
    monkeypatch.setattr(
        QDesktopServices, "openUrl", lambda url: opened.append(url.toString())
    )
    monkeypatch.setattr(
        addon_kits, "addon_kits",
        lambda: _canned_kits(
            {"film": False, "npu": False, "adcsr_gpu": False, "adcsr_npu": False}
        ),
    )
    drawer = SettingsDrawer()
    app.processEvents()
    try:
        drawer.kit_rows["film"]["button"].click()
        drawer.kit_rows["adcsr_npu"]["button"].click()
        app.processEvents()
        assert opened == [
            "https://github.com/tomhda/ultraeasy-upscaler/releases/latest/download/"
            "ultraeasy-upscaler-film-kit.zip",
            "https://github.com/tomhda/ultraeasy-upscaler/releases/latest/download/"
            "ultraeasy-upscaler-npu-kit-adcsr.zip",
        ]
    finally:
        drawer.close()
        app.processEvents()


def test_addon_kit_detection_reads_files_only(monkeypatch, tmp_path: Path) -> None:
    from app.core import addon_kits, binaries, helper_backend
    from app.core.settings import (
        ADCSR_NPU_MANIFEST,
        HELPER_MODEL_ADCSR,
        HELPER_MODEL_FILES,
        HELPER_MODEL_NPU_BACK,
        UpscaleBackend,
    )

    monkeypatch.setattr(binaries, "repo_root", lambda: tmp_path)
    models = tmp_path / "models-ai"
    models.mkdir()
    monkeypatch.setenv(helper_backend.MODELS_DIR_ENV, str(models))
    monkeypatch.delenv("UEU_FILM_HELPER", raising=False)
    monkeypatch.delenv("UEU_FILM_MODEL", raising=False)

    states = {kit.key: kit.installed for kit in addon_kits.addon_kits()}
    assert states == {
        "film": False,
        "npu": False,
        "adcsr_gpu": False,
        "adcsr_npu": False,
    }

    (tmp_path / "vendor" / "winml-film").mkdir(parents=True)
    (tmp_path / "vendor" / "winml-film" / "winml-film.exe").touch()
    (tmp_path / "models" / "film").mkdir(parents=True)
    (tmp_path / "models" / "film" / "film_style_fp32.onnx").touch()
    (tmp_path / "tools" / "npu-serve").mkdir(parents=True)
    (tmp_path / "tools" / "npu-serve" / "npu_serve.py").touch()
    gpu_name = HELPER_MODEL_FILES[UpscaleBackend.WINML_GPU][HELPER_MODEL_ADCSR][128]
    (models / gpu_name).touch()
    front_name = HELPER_MODEL_FILES[UpscaleBackend.NPU_NATIVE][HELPER_MODEL_ADCSR][128]
    (models / front_name).touch()
    (models / HELPER_MODEL_NPU_BACK[HELPER_MODEL_ADCSR]).touch()
    (models / ADCSR_NPU_MANIFEST).touch()

    states = {kit.key: kit.installed for kit in addon_kits.addon_kits()}
    assert states == {
        "film": True,
        "npu": True,
        "adcsr_gpu": True,
        "adcsr_npu": True,
    }


def test_self_test_returns_6_when_film_kit_partial(monkeypatch, tmp_path: Path) -> None:
    from app import main as app_main
    from app.core import binaries

    for name in (
        "ffmpeg_exe",
        "ffprobe_exe",
        "realesrgan_exe",
        "rife_exe",
    ):
        monkeypatch.setattr(binaries, name, lambda: "dummy")
    monkeypatch.setattr(binaries, "available_models", lambda: ["realesrgan-x4plus"])
    monkeypatch.setattr(
        binaries, "available_interpolation_models", lambda: ["rife-v4.6"]
    )
    monkeypatch.setattr(binaries, "repo_root", lambda: tmp_path)
    monkeypatch.delenv("UEU_FILM_HELPER", raising=False)
    monkeypatch.delenv("UEU_FILM_MODEL", raising=False)

    assert app_main.portable_self_test() == 0

    (tmp_path / "vendor" / "winml-film").mkdir(parents=True)
    (tmp_path / "vendor" / "winml-film" / "winml-film.exe").touch()
    assert app_main.portable_self_test() == 6

    (tmp_path / "models" / "film").mkdir(parents=True)
    (tmp_path / "models" / "film" / "film_style_fp32.onnx").touch()
    assert app_main.portable_self_test() == 0

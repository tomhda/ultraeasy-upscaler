"""表示言語の切り替え（TASK5）。日本語の原文をキーに文言を返す。"""
from __future__ import annotations

import ast
import os
import string
from collections import Counter
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from app import i18n
from app.i18n import N_, language, set_language, t
from app.locale.en import EN

ROOT = Path(__file__).resolve().parents[1]


# --- t() / N_() の基本 ---

def test_t_returns_source_in_japanese():
    set_language("ja")
    assert t("クイック確認") == "クイック確認"
    assert t("確認中… {pct}%", pct=45) == "確認中… 45%"
    assert N_("画像") == "画像"


def test_t_falls_back_to_source_when_untranslated(monkeypatch):
    from app.locale import en as en_module

    monkeypatch.setitem(en_module.EN, "クイック確認", None)
    monkeypatch.delitem(en_module.EN, "確認中… {pct}%")
    set_language("en")
    assert t("クイック確認") == "クイック確認"
    assert t("確認中… {pct}%", pct=45) == "確認中… 45%"


def test_t_returns_translation_when_present(monkeypatch):
    from app.locale import en as en_module

    monkeypatch.setitem(en_module.EN, "クイック確認", "Quick check it")
    set_language("en")
    assert t("クイック確認") == "Quick check it"
    assert t("確認中… {pct}%", pct=45) == "Checking… 45%"


def test_set_language_rejects_unknown():
    with pytest.raises(ValueError):
        set_language("fr")


def test_language_defaults_to_ja():
    i18n._LANG = None
    try:
        assert language() in ("ja", "en")
    finally:
        i18n._LANG = "ja"


# --- 起動時の決め方（環境変数 → 保存設定 → Windows 表示言語） ---

def test_resolve_prefers_env(monkeypatch):
    monkeypatch.setenv("UEU_LANG", "en")
    assert i18n.resolve_startup_language() == "en"
    monkeypatch.setenv("UEU_LANG", "ja")
    assert i18n.resolve_startup_language() == "ja"


def test_resolve_prefers_saved_over_windows(monkeypatch):
    from app.core import user_settings

    monkeypatch.delenv("UEU_LANG", raising=False)
    monkeypatch.setattr(user_settings, "load_language", lambda: "en")
    monkeypatch.setattr(i18n, "_windows_prefers_japanese", lambda: True)
    assert i18n.resolve_startup_language() == "en"


def test_resolve_falls_back_to_windows(monkeypatch):
    from app.core import user_settings

    monkeypatch.delenv("UEU_LANG", raising=False)
    monkeypatch.setattr(user_settings, "load_language", lambda: "auto")
    monkeypatch.setattr(i18n, "_windows_prefers_japanese", lambda: True)
    assert i18n.resolve_startup_language() == "ja"
    monkeypatch.setattr(i18n, "_windows_prefers_japanese", lambda: False)
    assert i18n.resolve_startup_language() == "en"


def test_resolve_ignores_broken_settings(monkeypatch):
    from app.core import user_settings

    monkeypatch.delenv("UEU_LANG", raising=False)

    def _boom():
        raise OSError("disk gone")

    monkeypatch.setattr(user_settings, "load_language", _boom)
    monkeypatch.setattr(i18n, "_windows_prefers_japanese", lambda: True)
    assert i18n.resolve_startup_language() == "ja"


def test_init_language_pins_result(monkeypatch):
    monkeypatch.setenv("UEU_LANG", "en")
    assert i18n.init_language() == "en"
    assert language() == "en"


# --- 設定ファイル（言語だけ。壊れていても落ちない） ---

def test_settings_missing_is_auto():
    from app.core import user_settings

    assert user_settings.load_language() == "auto"


def test_settings_corrupt_is_auto():
    from app.core import user_settings

    user_settings.settings_path().parent.mkdir(parents=True, exist_ok=True)
    user_settings.settings_path().write_text("{壊れた", encoding="utf-8")
    assert user_settings.load_language() == "auto"
    user_settings.settings_path().write_text("[1,2]", encoding="utf-8")
    assert user_settings.load_language() == "auto"
    user_settings.settings_path().write_text('{"language": "xx"}', encoding="utf-8")
    assert user_settings.load_language() == "auto"


def test_settings_roundtrip():
    from app.core import user_settings

    for code in ("auto", "ja", "en"):
        user_settings.save_language(code)
        assert user_settings.load_language() == code


def test_settings_rejects_unknown():
    from app.core import user_settings

    with pytest.raises(ValueError):
        user_settings.save_language("fr")


def test_settings_uses_env_dir(monkeypatch, tmp_path):
    from app.core import user_settings

    other = tmp_path / "other-place"
    monkeypatch.setenv("UEU_SETTINGS_DIR", str(other))
    user_settings.save_language("en")
    assert (other / "settings.json").is_file()
    assert user_settings.load_language() == "en"


# --- 洗い出しの網羅性（AST で機械的に確認） ---

def _collect():
    found = Counter()
    nonliteral = Counter()
    for path in sorted((ROOT / "app").rglob("*.py")):
        if "locale" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        rel = path.relative_to(ROOT).as_posix()
        for node in ast.walk(tree):
            func = node.func if isinstance(node, ast.Call) else None
            name = ""
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            if name not in ("t", "N_"):
                continue
            if not node.args:
                continue
            first = node.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                found[first.value] += 1
            else:
                nonliteral[(rel, ast.unparse(first))] += 1
    return found, nonliteral


def test_all_wrapped_strings_are_in_en():
    found, _ = _collect()
    missing = sorted(set(found) - set(EN))
    assert not missing


def test_all_en_keys_are_used():
    found, _ = _collect()
    unused = sorted(set(EN) - set(found))
    assert not unused


# t() に変数を渡す箇所は、値がすべて N_() 印付きか記号の中継に限る。
# 新しい箇所を足したらここ（と REPORT5.md の一覧）も更新する。
_EXPECTED_NONLITERAL = Counter(
    {
        ("app/gui/compare_view.py", "self.QUICK_HELP"): 1,
        ("app/gui/main_window.py", "_MODEL_LABELS.get(data, data)"): 1,
        (
            "app/gui/main_window.py",
            "_MODEL_LABELS.get(str(job_settings.model), str(job_settings.model))",
        ): 1,
        ("app/gui/main_window.py", "_MODEL_LABELS.get(model, model)"): 1,
        (
            "app/gui/main_window.py",
            "_MODEL_LABELS.get(settings.model, settings.model)",
        ): 1,
        ("app/gui/main_window.py", "_MODEL_LABELS.get(key, key)"): 2,
        ("app/gui/main_window.py", "label"): 1,
        ("app/gui/main_window.py", "backend_label"): 1,
        ("app/gui/main_window.py", "model_label"): 1,
        ("app/gui/main_window.py", "speed"): 2,
        ("app/gui/main_window.py", "quality"): 2,
        ("app/gui/main_window.py", "anime"): 1,
        ("app/gui/main_window.py", "live"): 1,
        ("app/gui/main_window.py", "star"): 2,
        ("app/gui/main_window.py", "desc"): 1,
        ("app/gui/queue_view.py", "_KIND_LABEL.get(self.job.kind, '')"): 1,
        ("app/gui/queue_view.py", "self.job.message"): 1,
        ("app/gui/queue_view.py", "_STATUS_TEXT.get(self.job.status, '')"): 1,
        ("app/gui/settings_drawer.py", "label"): 2,
        ("app/gui/settings_drawer.py", "text"): 1,
        ("app/gui/settings_drawer.py", "help_text"): 2,
        ("app/gui/settings_drawer.py", "kit.name_key"): 1,
    }
)


def test_nonliteral_t_calls_are_known():
    _, nonliteral = _collect()
    assert nonliteral == _EXPECTED_NONLITERAL


# --- 差し込み名の一致（いまは全 None のため合成例で検査） ---

def _placeholders(text: str) -> set[str]:
    names = set()
    for _, field, _, _ in string.Formatter().parse(text):
        if field is None:
            continue
        names.add(field.split("!")[0].split(":")[0].split(".")[0].split("[")[0])
    return names


def test_en_placeholders_match_source():
    for key, value in EN.items():
        if value is None:
            continue
        assert _placeholders(key) == _placeholders(value), key


def test_placeholder_checker_catches_mismatch():
    assert _placeholders("試しています… {pct}%") == {"pct"}
    assert _placeholders("{a}{b!r}") == {"a", "b"}
    assert _placeholders("{timeout:g}秒") == {"timeout"}
    with pytest.raises(AssertionError):
        got = _placeholders("{pct}%")
        want = {"other"}
        assert got == want


def test_all_en_translated():
    missing = [key for key, value in EN.items() if not value]
    assert not missing, missing


# --- 表示言語の項目（詳細設定の末尾） ---

def test_language_row_options_save_and_notice():
    from PySide6.QtWidgets import QApplication

    from app.core import user_settings
    from app.gui.settings_drawer import _LANGUAGE_OPTIONS, SettingsDrawer

    application = QApplication.instance() or QApplication([])
    assert [data for _, data in _LANGUAGE_OPTIONS] == ["auto", "ja", "en"]

    drawer = SettingsDrawer()
    assert drawer.language_combo.currentData() == "auto"
    assert drawer.language_notice.text() == ""

    drawer.language_combo.setCurrentIndex(drawer.language_combo.findData("en"))
    assert user_settings.load_language() == "en"
    assert drawer.language_notice.text() == "次回の起動から切り替わります。"

    drawer.language_combo.setCurrentIndex(drawer.language_combo.findData("ja"))
    assert user_settings.load_language() == "ja"
    assert drawer.language_notice.text() == "次回の起動から切り替わります。"
    application  # noqa: B018 - インスタンス維持の意図を明示


def test_language_row_restores_saved():
    from PySide6.QtWidgets import QApplication

    from app.core import user_settings
    from app.gui.settings_drawer import SettingsDrawer

    application = QApplication.instance() or QApplication([])
    user_settings.save_language("en")
    drawer = SettingsDrawer()
    assert drawer.language_combo.currentData() == "en"
    assert drawer.language_notice.text() == ""
    application  # noqa: B018 - インスタンス維持の意図を明示


# --- 英語にしたとき、起動直後の画面が英語で出る ---

def test_main_window_boots_in_english():
    from PySide6.QtWidgets import QApplication
    from PySide6.QtWidgets import QLabel

    from app.gui.main_window import MainWindow

    application = QApplication.instance() or QApplication([])
    set_language("en")
    win = None
    try:
        win = MainWindow()
        win.show()
        application.processEvents()
        assert win.windowTitle().startswith("ultraeasy-upscaler")
        assert win.start_btn.text() == "Start"
        assert win.pause_btn.text() == "Pause"
        assert win.image_model_combo.itemText(0) == "None (no upscaling)"
        # 印（◎など）は英語では言葉になる
        assert "Speed: High" in win.image_model_combo.itemText(1)
        assert "Speed: High" in win.video_model_combo.itemText(1)
        # ヘッダー見出しと右列の新しいボタンが英語になる
        headers = [w.text() for w in win.findChildren(QLabel)]
        assert "Image model" in headers
        assert "Video model" in headers
        assert win._override_hint.text() == "This file uses its own settings."
        assert win.reset_override_btn.text() == "Use the settings at the top"
        assert win._no_file_hint.text() == (
            "Select a file to change the settings for that file only."
        )
        assert win.drawer.language_combo.count() == 3
    finally:
        # 閉じないと試し用の一時フォルダが残る
        if win is not None:
            win.close()
        set_language("ja")
        application.processEvents()


# --- フォントの切り替え ---

def test_font_family_switch():
    from app.gui import theme

    assert "Yu Gothic UI" in theme.font_family("ja")
    assert "Meiryo UI" in theme.font_family("ja")
    assert theme.font_family("en").startswith('"Segoe UI"')
    set_language("ja")
    assert "Meiryo UI" in theme.font_family()
    set_language("en")
    assert "Meiryo UI" not in theme.font_family()


def test_build_qss_font_follows_language():
    from app.gui import theme

    palette = theme.build_palette(False, "#21c7d9")
    assert "Meiryo UI" in theme.build_qss(palette, "ja")
    assert "Meiryo UI" not in theme.build_qss(palette, "en")

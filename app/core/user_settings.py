"""表示言語とアクセントカラーの保存と読み込み（Qt に依存しない）。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from app.i18n import N_

SETTINGS_DIR_ENV = "TOGU_SETTINGS_DIR"
SETTINGS_FILENAME = "settings.json"

AUTO = "auto"
JA = "ja"
EN = "en"
LANGUAGE_CODES = (AUTO, JA, EN)

# アクセントカラーの表（値、表示名の原文）。1 か所に置く（Qt 非依存）。
# 英語の訳は app/locale/en.py にある。
ACCENT_AUTO = "auto"
ACCENT_OPTIONS: tuple[tuple[str, str], ...] = (
    (ACCENT_AUTO, N_("Windows に合わせる")),
    ("#3b82f6", N_("青")),
    ("#21c7d9", N_("水色")),
    ("#3ecf5a", N_("緑")),
    ("#e6b422", N_("黄")),
    ("#f08a24", N_("オレンジ")),
    ("#e5484d", N_("赤")),
    ("#e0569b", N_("ピンク")),
    ("#8b5cf6", N_("紫")),
)
ACCENT_VALUES = tuple(value for value, _label in ACCENT_OPTIONS)

# 改名前の設定フォルダ（読み継ぎ元。消さない・書き換えない）。
_LEGACY_DIR_NAME = "ultraeasy-upscaler"


def _new_settings_dir() -> Path:
    """新しい設定フォルダ。"""
    from app import APP_SLUG

    local = os.environ.get("LOCALAPPDATA")
    if local:
        return Path(local) / APP_SLUG
    return Path.home() / APP_SLUG


def _old_settings_dir() -> Path:
    """改名前の設定フォルダ（読み継ぎ元）。"""
    local = os.environ.get("LOCALAPPDATA")
    if local:
        return Path(local) / _LEGACY_DIR_NAME
    return Path.home() / _LEGACY_DIR_NAME


def settings_dir() -> Path:
    """設定フォルダ。TOGU_SETTINGS_DIR があればそちらを使う（テスト用）。"""
    override = os.environ.get(SETTINGS_DIR_ENV)
    if override:
        return Path(override).expanduser()
    return _new_settings_dir()


def settings_path() -> Path:
    """settings.json の場所（保存は常にこちら）。"""
    return settings_dir() / SETTINGS_FILENAME


def _effective_path() -> Path:
    """読み込む settings.json の場所。

    TOGU_SETTINGS_DIR があるときはそのフォルダだけを使う（引き継ぎはしない）。
    そうでなく、新しいフォルダに settings.json が無く古いフォルダにあるときは
    古いほうを読む。保存は常に settings_path() へ。
    """
    if os.environ.get(SETTINGS_DIR_ENV):
        return settings_path()
    new_path = settings_path()
    if not new_path.is_file():
        old_path = _old_settings_dir() / SETTINGS_FILENAME
        if old_path.is_file():
            return old_path
    return new_path


def _read_data() -> dict:
    """有効な設定ファイルの中身（辞書でなければ空）。"""
    try:
        data = json.loads(_effective_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_data(data: dict) -> None:
    """settings_path() へ書き込む（古いフォルダは触らない）。"""
    path = settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(SETTINGS_FILENAME + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def load_language() -> str:
    """保存された言語（"auto"|"ja"|"en"）。無い・壊れているときは "auto"。"""
    code = _read_data().get("language", AUTO)
    return code if code in LANGUAGE_CODES else AUTO


def save_language(code: str) -> None:
    """言語を保存する。不正な値は ValueError。もう片方の設定は残す。"""
    if code not in LANGUAGE_CODES:
        raise ValueError(f"unknown language: {code!r}")
    data = _read_data()
    data["language"] = code
    _write_data(data)


def load_accent() -> str:
    """保存されたアクセントカラー（"auto" か #rrggbb）。無い・壊れている・
    表に無い値のときは "auto"。"""
    value = _read_data().get("accent", ACCENT_AUTO)
    return value if value in ACCENT_VALUES else ACCENT_AUTO


def save_accent(value: str) -> None:
    """アクセントカラーを保存する。不正な値は ValueError。もう片方の設定は残す。"""
    if value not in ACCENT_VALUES:
        raise ValueError(f"unknown accent: {value!r}")
    data = _read_data()
    data["accent"] = value
    _write_data(data)

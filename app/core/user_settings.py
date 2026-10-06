"""表示言語の保存と読み込み（Qt に依存しない）。

いま保存する設定は言語だけ。他の設定は足さない。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

SETTINGS_DIR_ENV = "UEU_SETTINGS_DIR"
SETTINGS_FILENAME = "settings.json"

AUTO = "auto"
JA = "ja"
EN = "en"
LANGUAGE_CODES = (AUTO, JA, EN)


def settings_dir() -> Path:
    """設定フォルダ。UEU_SETTINGS_DIR があればそちらを使う（テスト用）。"""
    override = os.environ.get(SETTINGS_DIR_ENV)
    if override:
        return Path(override).expanduser()
    local = os.environ.get("LOCALAPPDATA")
    if local:
        return Path(local) / "ultraeasy-upscaler"
    return Path.home() / "ultraeasy-upscaler"


def settings_path() -> Path:
    """settings.json の場所。"""
    return settings_dir() / SETTINGS_FILENAME


def load_language() -> str:
    """保存された言語（"auto"|"ja"|"en"）。無い・壊れているときは "auto"。"""
    try:
        data = json.loads(settings_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return AUTO
    if not isinstance(data, dict):
        return AUTO
    code = data.get("language", AUTO)
    return code if code in LANGUAGE_CODES else AUTO


def save_language(code: str) -> None:
    """言語を保存する。不正な値は ValueError。"""
    if code not in LANGUAGE_CODES:
        raise ValueError(f"unknown language: {code!r}")
    path = settings_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            data = {}
    except (OSError, ValueError):
        data = {}
    data["language"] = code
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(SETTINGS_FILENAME + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)

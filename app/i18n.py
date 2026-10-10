"""画面の表示言語（日本語/英語）の切り替え。Qt に依存しない。

日本語の原文をキーにして現在の言語の文言を返す。言語の切り替えは
次回の起動から有効（実行中の画面は貼り替えない）。
"""

from __future__ import annotations

import os

from app.locale.en import EN

LANG_ENV = "TOGU_LANG"

JA = "ja"
EN_CODE = "en"

_LANG: str | None = None


def N_(text: str) -> str:
    """辞書定義側の印。何もせず原文のまま返す（洗い出し用）。"""
    return text


def set_language(code: str) -> None:
    """現在の言語を切り替える（次回起動までは呼ばない。テスト用）。"""
    if code not in (JA, EN_CODE):
        raise ValueError(f"unknown language: {code!r}")
    global _LANG
    _LANG = code


def language() -> str:
    """現在の言語（"ja" か "en"）。未設定なら起動時の決め方で求める。"""
    if _LANG is not None:
        return _LANG
    return resolve_startup_language()


def t(source: str, **kwargs: object) -> str:
    """原文をキーに現在の言語の文言を返す。f 文字列は使わない。"""
    text = source
    if language() == EN_CODE:
        translated = EN.get(text)
        if translated:
            text = translated
    if kwargs:
        return text.format(**kwargs)
    return text


def _windows_prefers_japanese() -> bool:
    """Windows の表示言語が日本語か。非 Windows・失敗時は True。"""
    try:
        import ctypes

        lang_id = int(ctypes.windll.kernel32.GetUserDefaultUILanguage())  # type: ignore[attr-defined]
    except Exception:
        return True
    return (lang_id & 0x3FF) == 0x11


def resolve_startup_language() -> str:
    """起動時の言語を優先順（環境変数 → 保存設定 → Windows 表示言語）で決める。"""
    env = os.environ.get(LANG_ENV)
    if env in (JA, EN_CODE):
        return env
    try:
        from app.core.user_settings import load_language
    except Exception:
        saved = "auto"
    else:
        try:
            saved = load_language()
        except Exception:
            saved = "auto"
    if saved in (JA, EN_CODE):
        return saved
    return JA if _windows_prefers_japanese() else EN_CODE


def init_language() -> str:
    """起動時に一度だけ呼び、決まった言語を固定する。"""
    code = resolve_startup_language()
    set_language(code)
    return code

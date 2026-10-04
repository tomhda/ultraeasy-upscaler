"""pytest 共通設定: プロジェクトルートを import path に追加。"""
from __future__ import annotations

import sys
from pathlib import Path

# tests/ の親（リポジトリルート）を sys.path 先頭に追加して `import app` を可能にする。
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pytest


@pytest.fixture
def force_npu_available(monkeypatch, tmp_path):
    """GUI テストは NPU キットの有無に依らず NPU ありで実行する。

    NPU が無い PC でも既存の GUI テストが通るようにする。
    実キャッシュ・実モデルを読まないよう両方を空の一時フォルダへ向ける
    （NPU なしの表示と変換済み判定は tests/test_npu_prepare.py で別に確かめる）。
    """
    from app.core import helper_backend, npu_prepare

    monkeypatch.setattr(npu_prepare, "npu_available", lambda: True)
    empty_models = tmp_path / "empty-models"
    empty_models.mkdir()
    monkeypatch.setenv(helper_backend.MODELS_DIR_ENV, str(empty_models))
    empty_cache = tmp_path / "empty-npu-cache"
    empty_cache.mkdir()
    monkeypatch.setenv(helper_backend.NPU_CACHE_ENV, str(empty_cache))

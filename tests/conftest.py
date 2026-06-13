"""pytest 共通設定: プロジェクトルートを import path に追加。"""
from __future__ import annotations

import sys
from pathlib import Path

# tests/ の親（リポジトリルート）を sys.path 先頭に追加して `import app` を可能にする。
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

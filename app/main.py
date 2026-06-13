"""エントリポイント: GUI を起動する。"""
from __future__ import annotations

import sys


def main() -> int:
    from app.gui.main_window import run  # GUI エージェントが提供する run(argv)->int
    return run(sys.argv)


if __name__ == "__main__":
    raise SystemExit(main())

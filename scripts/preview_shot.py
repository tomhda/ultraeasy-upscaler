"""GUI のプレビュー用スクリーンショットを生成する（offscreen レンダリング）。

実画面にウィンドウを出さずにレイアウトを PNG 化して確認できる。
    .venv\\Scripts\\python.exe scripts\\preview_shot.py [out.png]
既定の出力: C:\\tmp\\ueu_gui_shot.png
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from app.gui.main_window import MainWindow  # noqa: E402
from app.gui.theme import apply_theme  # noqa: E402


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"C:\tmp\ueu_gui_shot.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    app = QApplication([])
    apply_theme(app)
    win = MainWindow()
    win.resize(980, 660)

    # 見栄え確認用にサンプルを数件投入（probe でソース解像度も表示される）
    vendor = ROOT / "vendor" / "realesrgan"
    samples = [vendor / "input.jpg", vendor / "input2.jpg", vendor / "bbb_demo.mp4"]
    win.add_paths([str(p) for p in samples if p.exists()])
    win.show()

    def _grab_quit() -> None:
        win.grab().save(str(out))
        print("SHOT_SAVED", out)
        app.quit()

    QTimer.singleShot(600, _grab_quit)
    app.exec()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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
from PySide6.QtGui import QFontDatabase  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from app.core.settings import DEFAULT_HELPER_MODEL  # noqa: E402
from app.gui.main_window import MainWindow  # noqa: E402
from app.gui.theme import apply_theme  # noqa: E402


def _load_windows_preview_fonts() -> None:
    """offscreen Qtでも通常のWindowsフォントで文字を描画できるようにする。"""
    for path in (
        Path(r"C:\Windows\Fonts\YuGothR.ttc"),
        Path(r"C:\Windows\Fonts\SegoeIcons.ttf"),
    ):
        if path.is_file():
            QFontDatabase.addApplicationFont(str(path))


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"C:\tmp\ueu_gui_shot.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    app = QApplication([])
    _load_windows_preview_fonts()
    apply_theme(app)
    win = MainWindow()
    win.resize(980, 660)

    # 見栄え確認用にサンプルを数件投入（probe でソース解像度も表示される）
    vendor = ROOT / "vendor" / "realesrgan"
    samples = [vendor / "input.jpg", vendor / "input2.jpg", vendor / "bbb_demo.mp4"]
    win.add_paths([str(p) for p in samples if p.exists()])
    # 回帰確認用に、新AIのモデル欄で「なし」が先頭に残っていることを確認する。
    none_index = win.model_combo.findData(None)
    if none_index != 0:
        raise RuntimeError("モデル欄の「なし」が先頭にありません")
    win.model_combo.setCurrentIndex(none_index)
    # 実モデル名と、選択中の処理×モデルの説明行が読める状態を撮る。
    model_index = win.model_combo.findData(DEFAULT_HELPER_MODEL)
    if model_index >= 0:
        win.model_combo.setCurrentIndex(model_index)
    win.show()

    def _grab_quit() -> None:
        win.grab().save(str(out))
        print("SHOT_SAVED", out)
        print("MODEL_PREVIEW", win.model_combo.currentText())
        print("HINT_PREVIEW", win.model_hint.text())
        app.quit()

    QTimer.singleShot(600, _grab_quit)
    app.exec()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

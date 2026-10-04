"""選択中メディアのプレビュー（中央ペイン）。

画像はそのまま、動画は 1 コマを ffmpeg で取り出して表示する。
読み込みは別スレッドで行い、GUI スレッドはシグナルで結果だけ受け取る。
"""
from __future__ import annotations

import subprocess
import threading

from PySide6.QtCore import QRect, QSize, Qt, Signal
from PySide6.QtGui import QImage, QImageReader, QPainter, QPixmap
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout

from app.core import binaries
from app.core.jobs import Job, JobKind

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
# 表示用に読み込む最大の長辺。これより大きい画像は読み込み時に縮小する。
_MAX_SIDE = 2560


def _load_image(path: str) -> QImage:
    reader = QImageReader(path)
    reader.setAutoTransform(True)
    size = reader.size()
    if size.isValid() and max(size.width(), size.height()) > _MAX_SIDE:
        reader.setScaledSize(
            size.scaled(QSize(_MAX_SIDE, _MAX_SIDE), Qt.AspectRatioMode.KeepAspectRatio)
        )
    return reader.read()


def _load_video_frame(path: str, seconds: float) -> QImage:
    """動画の seconds 秒付近の 1 コマを取り出す。失敗時は null の QImage。"""
    cmd = [
        binaries.ffmpeg_exe(), "-hide_banner", "-loglevel", "error",
        "-ss", f"{max(0.0, seconds):.3f}", "-i", path,
        "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-",
    ]
    try:
        done = subprocess.run(
            cmd, capture_output=True, timeout=20, creationflags=_NO_WINDOW
        )
    except (OSError, subprocess.SubprocessError, binaries.BinaryError):
        return QImage()
    image = QImage()
    image.loadFromData(done.stdout, "PNG")
    return image


class PreviewPane(QFrame):
    """選択中のメディアを枠内に収めて表示する。"""

    _loaded = Signal(int, QImage)  # (要求番号, 画像)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("previewPane")
        self.setMinimumSize(320, 200)
        self._pixmap: QPixmap | None = None
        self._token = 0  # 古い読み込み結果を捨てるための要求番号

        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 12, 12, 12)
        self._caption = QLabel("処理前")
        self._caption.setObjectName("previewCaption")
        lay.addWidget(self._caption, 0, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self._message = QLabel("")
        self._message.setObjectName("previewMessage")
        self._message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self._message, 1)

        self._loaded.connect(self._on_loaded)
        self.show_job(None)

    def show_job(self, job: Job | None) -> None:
        """job の処理前の画を表示する（None なら空にする）。"""
        self._token += 1
        self._pixmap = None
        self._caption.setVisible(False)
        if job is None:
            self._message.setText("左の一覧からファイルを選ぶと、ここに表示します。")
            self.update()
            return
        if job.kind == JobKind.FOLDER:
            self._message.setText("フォルダの中身はここには表示されません。")
            self.update()
            return
        self._message.setText("読み込み中…")
        self.update()

        token = self._token
        path = str(job.input_path)
        if job.kind == JobKind.VIDEO:
            # 冒頭は黒画面のことが多いので、長さが分かるなら 1 割進んだ位置を使う
            seconds = 0.0
            if job.frame_count and job.fps:
                seconds = job.frame_count / job.fps * 0.1
            target = lambda: _load_video_frame(path, seconds)  # noqa: E731
        else:
            target = lambda: _load_image(path)  # noqa: E731

        def work() -> None:
            image = target()
            try:
                self._loaded.emit(token, image)
            except RuntimeError:
                pass  # 読み込み中にウィンドウが閉じられた

        threading.Thread(target=work, daemon=True).start()

    def _on_loaded(self, token: int, image: QImage) -> None:
        if token != self._token:
            return
        if image.isNull():
            self._message.setText("このファイルは表示できません。処理はできます。")
            return
        self._pixmap = QPixmap.fromImage(image)
        self._message.setText("")
        self._caption.setVisible(True)
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        super().paintEvent(event)
        if self._pixmap is None:
            return
        area = self.contentsRect().adjusted(12, 12, -12, -12)
        size = self._pixmap.size().scaled(area.size(), Qt.AspectRatioMode.KeepAspectRatio)
        target = QRect(0, 0, size.width(), size.height())
        target.moveCenter(area.center())
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.drawPixmap(target, self._pixmap)
        painter.end()

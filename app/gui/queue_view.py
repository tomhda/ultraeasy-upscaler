"""処理キューの表示（各ジョブ = 1 行）。

行内容: サムネ/グリフ + 名前 + 種別/寸法 + 個別進捗バー + 状態 + パーセント + ✕。
ロジック（開始/キャンセル）は MainWindow 側。ここは表示と remove 要求の発火のみ。
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.core.jobs import Job, JobKind, JobStatus

from .icons import Icon, apply_icon_font

_KIND_GLYPH = {
    JobKind.IMAGE: Icon.IMAGE,
    JobKind.VIDEO: Icon.VIDEO,
    JobKind.FOLDER: Icon.FOLDER,
}
_KIND_LABEL = {
    JobKind.IMAGE: "画像",
    JobKind.VIDEO: "動画",
    JobKind.FOLDER: "フォルダ",
}
_INFO_COL_WIDTH = 170
_META_COL_WIDTH = 136

# 状態ごとの表示文言（メッセージが無い場合のフォールバック）
_STATUS_TEXT = {
    JobStatus.QUEUED: "待機中",
    JobStatus.PROBING: "解析中…",
    JobStatus.RUNNING: "処理中…",
    JobStatus.DONE: "完了",
    JobStatus.ERROR: "エラー",
    JobStatus.CANCELED: "キャンセル",
}


class QueueRow(QFrame):
    """キュー内の 1 ジョブを表す行ウィジェット。"""

    removeRequested = Signal(int)  # job_id

    def __init__(self, job: Job, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("queueRow")
        self.job = job
        self.setMinimumHeight(76)
        self._build()
        self.refresh()

    def _build(self) -> None:
        row = QHBoxLayout(self)
        row.setContentsMargins(14, 8, 12, 8)
        row.setSpacing(16)

        # サムネ / グリフ
        self._thumb = QLabel()
        self._thumb.setObjectName("thumb")
        self._thumb.setFixedSize(82, 48)
        self._thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._set_thumb()
        row.addWidget(self._thumb)

        text_widget = QWidget()
        text_widget.setObjectName("rowText")
        text_widget.setFixedWidth(_INFO_COL_WIDTH)
        text_col = QVBoxLayout(text_widget)
        text_col.setContentsMargins(0, 0, 0, 0)
        text_col.setSpacing(5)

        self._name = QLabel(self.job.name)
        self._name.setObjectName("rowName")
        self._name.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._name.setFixedWidth(_INFO_COL_WIDTH)
        self._name.setToolTip(self.job.name)
        text_col.addWidget(self._name)

        self._status = QLabel()
        self._status.setObjectName("rowStatus")
        self._status.setFixedWidth(_INFO_COL_WIDTH)
        text_col.addWidget(self._status)
        row.addWidget(text_widget)

        self._meta = QLabel()
        self._meta.setObjectName("rowMeta")
        self._meta.setFixedWidth(_META_COL_WIDTH)
        self._meta.setAlignment(Qt.AlignmentFlag.AlignCenter)
        row.addWidget(self._meta)

        self._bar = QProgressBar()
        self._bar.setRange(0, 100)
        self._bar.setValue(0)
        self._bar.setTextVisible(False)
        self._bar.setMinimumWidth(360)
        self._bar.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        row.addWidget(self._bar, 1)

        self._percent = QLabel("0%")
        self._percent.setObjectName("rowPercent")
        self._percent.setFixedWidth(44)
        self._percent.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(self._percent)

        # ✕ 削除/キャンセル
        self._close = QPushButton("×")
        self._close.setObjectName("rowClose")
        self._close.setCursor(Qt.CursorShape.PointingHandCursor)
        self._close.setFixedSize(28, 28)
        self._close.setToolTip("キューから削除 / 処理中ならキャンセル")
        self._close.clicked.connect(lambda: self.removeRequested.emit(self.job.id))
        row.addWidget(self._close, 0, Qt.AlignmentFlag.AlignTop)

    def _set_thumb(self) -> None:
        """画像なら縮小サムネ、それ以外は種別グリフ。"""
        if self.job.kind == JobKind.IMAGE:
            pm = QPixmap(str(self.job.input_path))
            if not pm.isNull():
                pm = pm.scaled(
                    82, 48,
                    Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                    Qt.TransformationMode.SmoothTransformation,
                )
                self._thumb.setPixmap(pm)
                return
        apply_icon_font(self._thumb, 24)
        self._thumb.setText(_KIND_GLYPH.get(self.job.kind, "?"))

    def _meta_text(self) -> str:
        parts = [_KIND_LABEL.get(self.job.kind, "")]
        if self.job.width and self.job.height:
            parts.append(f"{self.job.width}x{self.job.height}")
        return "  ".join(p for p in parts if p)

    def _settings_text(self) -> str:
        settings = self.job.settings
        if settings is None:
            return ""
        upscale = settings.model or "なし"
        interpolation = settings.interpolation_model or "なし"
        return f"アップスケール: {upscale}\nフレーム補間: {interpolation}"

    def _name_text(self) -> str:
        return self._name.fontMetrics().elidedText(
            self.job.name,
            Qt.TextElideMode.ElideRight,
            _INFO_COL_WIDTH,
        )

    def refresh(self) -> None:
        """job の現在状態を行に反映する。"""
        self._name.setText(self._name_text())
        self._meta.setText(self._meta_text())
        self._meta.setToolTip(self._settings_text())
        pct = int(round(self.job.progress * 100))
        self._bar.setValue(max(0, min(100, pct)))
        self._percent.setText(f"{pct}%")

        msg = self.job.message or _STATUS_TEXT.get(self.job.status, "")
        self._status.setText(msg)

        # 状態に応じたツールチップ（出力先・エラー詳細）
        if self.job.status == JobStatus.DONE and self.job.output_path:
            self._status.setToolTip(f"出力先: {self.job.output_path}")
        elif self.job.status == JobStatus.ERROR and self.job.error:
            self._status.setToolTip(self.job.error)
        else:
            self._status.setToolTip("")

        # 完了/キャンセル後は ✕ をグレーアウトせず（再削除可）残す
        self._close.setEnabled(True)

    def set_busy_icon(self, busy: bool) -> None:
        """処理中はツールチップを「キャンセル」寄りにする。"""
        self._close.setToolTip("処理を中止" if busy else "キューから削除")


class QueueView(QWidget):
    """ジョブ行を縦に積むコンテナ。"""

    removeRequested = Signal(int)  # job_id

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._rows: dict[int, QueueRow] = {}
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(8)
        self._empty = QLabel("キューは空です。ファイルをドロップするか「追加」してください。")
        self._empty.setObjectName("hint")
        self._empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._layout.addWidget(self._empty)
        self._layout.addStretch(1)

    def row_count(self) -> int:
        return len(self._rows)

    def has_job(self, job_id: int) -> bool:
        return job_id in self._rows

    def add_job(self, job: Job) -> QueueRow:
        """ジョブ行を追加（既存 id は再利用）。"""
        if job.id in self._rows:
            return self._rows[job.id]
        self._empty.setVisible(False)
        row = QueueRow(job)
        row.removeRequested.connect(self.removeRequested.emit)
        # stretch の手前に挿入
        self._layout.insertWidget(self._layout.count() - 1, row)
        self._rows[job.id] = row
        return row

    def remove_job(self, job_id: int) -> None:
        row = self._rows.pop(job_id, None)
        if row is not None:
            self._layout.removeWidget(row)
            row.deleteLater()
        if not self._rows:
            self._empty.setVisible(True)

    def row(self, job_id: int) -> QueueRow | None:
        return self._rows.get(job_id)

    def refresh(self, job_id: int) -> None:
        row = self._rows.get(job_id)
        if row is not None:
            row.refresh()

    def jobs(self) -> list[Job]:
        return [r.job for r in self._rows.values()]

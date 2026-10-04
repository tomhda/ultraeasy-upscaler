"""処理キューの表示（各ジョブ = 1 行）。

行内容: サムネ/グリフ + 名前 + 種別/寸法・状態 + ✕、その下に個別進捗バー + パーセント。
左の細い列に置く前提の縦積みレイアウト。行のクリックで選択を通知する。
ロジック（開始/キャンセル）は MainWindow 側。ここは表示と選択/remove 要求の発火のみ。
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
    clicked = Signal(int)  # job_id

    def __init__(self, job: Job, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("queueRow")
        self.job = job
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._build()
        self.refresh()

    def _build(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 8, 8, 8)
        outer.setSpacing(6)

        top = QHBoxLayout()
        top.setSpacing(10)
        outer.addLayout(top)

        # サムネ / グリフ
        self._thumb = QLabel()
        self._thumb.setObjectName("thumb")
        self._thumb.setFixedSize(64, 40)
        self._thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._set_thumb()
        top.addWidget(self._thumb)

        text_col = QVBoxLayout()
        text_col.setContentsMargins(0, 0, 0, 0)
        text_col.setSpacing(2)

        self._name = QLabel(self.job.name)
        self._name.setObjectName("rowName")
        self._name.setToolTip(self.job.name)
        # 長い名前で列幅が押し広げられないようにする（表示は refresh で省略）
        self._name.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        text_col.addWidget(self._name)

        self._meta = QLabel()
        self._meta.setObjectName("rowMeta")
        self._meta.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        text_col.addWidget(self._meta)
        top.addLayout(text_col, 1)

        # ✕ 削除/キャンセル
        self._close = QPushButton("×")
        self._close.setObjectName("rowClose")
        self._close.setCursor(Qt.CursorShape.PointingHandCursor)
        self._close.setFixedSize(28, 28)
        self._close.setToolTip("一覧から削除 / 処理中ならキャンセル")
        self._close.clicked.connect(lambda: self.removeRequested.emit(self.job.id))
        top.addWidget(self._close, 0, Qt.AlignmentFlag.AlignTop)

        self._status = QLabel()
        self._status.setObjectName("rowStatus")
        self._status.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        outer.addWidget(self._status)

        bar_row = QHBoxLayout()
        bar_row.setSpacing(8)
        self._bar = QProgressBar()
        self._bar.setRange(0, 100)
        self._bar.setValue(0)
        self._bar.setTextVisible(False)
        self._bar.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        bar_row.addWidget(self._bar, 1)

        self._percent = QLabel("0%")
        self._percent.setObjectName("rowPercent")
        self._percent.setFixedWidth(40)
        self._percent.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        bar_row.addWidget(self._percent)
        outer.addLayout(bar_row)

    def set_selected(self, selected: bool) -> None:
        self.setProperty("selected", selected)
        self.style().unpolish(self)
        self.style().polish(self)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.job.id)
        super().mousePressEvent(event)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._elide()

    def _set_thumb(self) -> None:
        """画像なら縮小サムネ、それ以外は種別グリフ。"""
        if self.job.kind == JobKind.IMAGE:
            pm = QPixmap(str(self.job.input_path))
            if not pm.isNull():
                pm = pm.scaled(
                    64, 40,
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
        return "・".join(p for p in parts if p)

    def _settings_text(self) -> str:
        settings = self.job.settings
        if settings is None:
            return ""
        backend = getattr(settings.backend, "value", str(settings.backend))
        if backend in {"winml_gpu", "npu_native"}:
            upscale = "なし" if settings.model is None else f"{settings.model}（4x）"
        else:
            upscale = settings.model or "なし"
        interpolation = settings.interpolation_model or "なし"
        return (
            f"AI実行先: {backend}\n"
            f"アップスケール: {upscale}\n"
            f"フレーム補間: {interpolation}"
        )

    def _elide(self) -> None:
        """名前と状態を、いまの行幅に収まるよう末尾省略で表示する。"""
        self._name.setText(self._name.fontMetrics().elidedText(
            self.job.name, Qt.TextElideMode.ElideMiddle, max(40, self._name.width())
        ))
        self._status.setText(self._status.fontMetrics().elidedText(
            self._status_full, Qt.TextElideMode.ElideRight, max(40, self._status.width())
        ))

    def refresh(self) -> None:
        """job の現在状態を行に反映する。"""
        self._meta.setText(self._meta_text())
        self._meta.setToolTip(self._settings_text())
        pct = int(round(self.job.progress * 100))
        self._bar.setValue(max(0, min(100, pct)))
        self._percent.setText(f"{pct}%")

        # 進捗バーは処理中だけ出す（待機中・完了後は行を低く保つ）
        busy = self.job.status in (JobStatus.PROBING, JobStatus.RUNNING)
        self._bar.setVisible(busy)
        self._percent.setVisible(busy)

        self._status_full = self.job.message or _STATUS_TEXT.get(self.job.status, "")
        self._elide()

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
        self._close.setToolTip("処理を中止" if busy else "一覧から削除")


class QueueView(QWidget):
    """ジョブ行を縦に積むコンテナ。"""

    removeRequested = Signal(int)  # job_id
    selectionChanged = Signal(object)  # 選択中の job_id（無ければ None）

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._rows: dict[int, QueueRow] = {}
        self._selected: int | None = None
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(8)
        self._layout.addStretch(1)

    def selected_id(self) -> int | None:
        return self._selected

    def select(self, job_id: int | None) -> None:
        """job_id の行を選択状態にする（None で選択なし）。"""
        if job_id is not None and job_id not in self._rows:
            job_id = None
        if job_id == self._selected:
            return
        for jid, row in self._rows.items():
            row.set_selected(jid == job_id)
        self._selected = job_id
        self.selectionChanged.emit(job_id)

    def row_count(self) -> int:
        return len(self._rows)

    def has_job(self, job_id: int) -> bool:
        return job_id in self._rows

    def add_job(self, job: Job) -> QueueRow:
        """ジョブ行を追加（既存 id は再利用）。"""
        if job.id in self._rows:
            return self._rows[job.id]
        row = QueueRow(job)
        row.removeRequested.connect(self.removeRequested.emit)
        row.clicked.connect(self.select)
        # stretch の手前に挿入
        self._layout.insertWidget(self._layout.count() - 1, row)
        self._rows[job.id] = row
        if self._selected is None:
            self.select(job.id)
        return row

    def remove_job(self, job_id: int) -> None:
        row = self._rows.pop(job_id, None)
        if row is not None:
            self._layout.removeWidget(row)
            row.deleteLater()
        if job_id == self._selected:
            # 選択中の行が消えたら、残っている先頭の行へ選択を移す
            self._selected = None
            self.select(next(iter(self._rows), None))
            if not self._rows:
                self.selectionChanged.emit(None)

    def row(self, job_id: int) -> QueueRow | None:
        return self._rows.get(job_id)

    def refresh(self, job_id: int) -> None:
        row = self._rows.get(job_id)
        if row is not None:
            row.refresh()

    def jobs(self) -> list[Job]:
        return [r.job for r in self._rows.values()]

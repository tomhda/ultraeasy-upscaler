"""中央の比較ビューと試しの操作部。

`PreviewPane` の置き換え。左の画と右の画を同じ表示領域に重ね、
縦の境界線で左右に分けて描く。AとBの解像度が違っても、同じ被写体位置が
重なるよう同じ矩形へ拡大縮小して描く（呼び出し側が範囲選択時は切り出し済みの
画を渡すため、ここでは2枚を同じ dest 矩形へ描くだけでよい）。
"""
from __future__ import annotations

import tempfile
import threading
from pathlib import Path

from PySide6.QtCore import QObject, QPointF, QRectF, QSize, Qt, QThread, Signal, Slot
from PySide6.QtGui import (
    QColor, QImage, QImageReader, QPainter, QPen, QPixmap, QPolygonF,
)
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from app.core import helper_backend, npu_prepare
from app.core import trial as trial_core
from app.core.jobs import Job, JobKind
from app.core.settings import UpscaleBackend, UpscaleSettings
from app.i18n import N_, t

from .icons import Icon, make_icon

# 境界線の掴み判定の片側幅（ピクセル）。細すぎると掴めないため余裕を持つ。
_DIVIDER_GRAB = 7
_HANDLE_RADIUS = 15  # 分割線の中央に出す、つまめる丸の半径
# 境界線が端に張り付かないよう可動域を制限する。
_SPLIT_MIN = 0.04
_SPLIT_MAX = 0.96
# ホイール1ノッチの倍率。
_ZOOM_IN = 1.25
_ZOOM_OUT = 1.0 / 1.25
# 拡大・縮小の限界（等倍基準の倍率）。
_SCALE_MIN = 0.02
_SCALE_MAX = 32.0
# 選択範囲の最小（元画像基準のピクセル）。小さすぎるドラッグは取り消す。
_MIN_SELECT = 32

# 動画を選んだときに最初に表示する位置（秒）
_DEFAULT_START_SECONDS = 2.0

# プレビュー枠の中だけの例外色（既存の #0b0f13 系と同じ扱い）。
# 枠内は常に暗いため、白系で描く。QSS の対象外。
_DIVIDER_COLOR = "#ffffff"
_SELECT_COLOR = "#ffffff"


def _clamp_rect_to_base(
    rect: tuple[int, int, int, int], width: int, height: int
) -> tuple[int, int, int, int] | None:
    """範囲を画内に丸める。重ならなければ None。"""
    x, y, w, h = (int(v) for v in rect)
    x0 = min(max(0, x), width)
    y0 = min(max(0, y), height)
    x1 = min(max(0, x + w), width)
    y1 = min(max(0, y + h), height)
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1 - x0, y1 - y0)


def _crop_qimage(image: QImage, rect: tuple[int, int, int, int]) -> QImage | None:
    """QImage を元画像基準の rect で切り出す。範囲外は丸める。"""
    clamped = _clamp_rect_to_base(rect, image.width(), image.height())
    if clamped is None:
        return None
    x, y, w, h = clamped
    return image.copy(x, y, w, h)


def _load_source_qimage(path: str) -> QImage:
    """表示・試し用の元画像をフル解像度で読み込む（回転メタデータを反映）。"""
    reader = QImageReader(path)
    reader.setAutoTransform(True)
    return reader.read()


def sanitize_error_message(text: str, redactions: list[str]) -> str:
    """状態行に出すエラー文から内部識別子を取り除く。

    複数行の実行ログは1行目だけ使う（コマンド行や一時パスが混ざるため）。
    既知の識別子（AI実行先・モデルキー・一時パス・ヘルパー名）は取り除く。
    """
    first = str(text).strip().splitlines()
    msg = first[0].strip() if first else ""
    for token in redactions:
        if token and len(token) >= 2:
            msg = msg.replace(token, "")
    # 置換で空いた穴を詰める（なぜ置換だけか: 文言の形は変えない）。
    msg = " ".join(msg.split())
    msg = msg.replace("（）", "").replace("()", "").replace("''", "").strip()
    if len(msg) > 200:
        msg = msg[:200]
    return msg or t("不明なエラー")


class CompareView(QFrame):
    """左右比較の描画域。左画と右画を同じ矩形へ重ねて境界線で分ける。

    選択矩形はここでは表示先頭の画（基底画）のピクセル座標で保持する。
    試しの範囲（元画像基準）との相互変換は呼び出し側が行う。
    """

    selectionChanged = Signal(object)  # tuple[int,int,int,int] | None
    splitChanged = Signal(float)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("previewPane")
        self.setMinimumSize(320, 200)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self._left: QImage | None = None
        self._right: QImage | None = None
        self._left_pm: QPixmap | None = None
        self._right_pm: QPixmap | None = None
        self._left_label = ""
        self._right_label = ""
        self._split = 0.5
        self._scale = 1.0
        self._offset = QPointF(0, 0)
        self._fit_on_resize = True
        self._selection_mode = False
        self._selection: tuple[int, int, int, int] | None = None
        self._drag_mode: str | None = None
        self._drag_pos: QPointF | None = None
        self._drag_offset = QPointF(0, 0)
        self._drag_split = 0.5
        self._sel_anchor: QPointF | None = None
        self._last_base = QSize(0, 0)

        self._message = QLabel("", self)
        self._message.setObjectName("previewMessage")
        self._message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._message.setWordWrap(True)
        # 画像の上の札（押して切り替えるボタン）。選べるものが 2 つ以上ある
        # ときだけ押せて、文末に下向きの小さな印を付ける。
        self._left_tag = QPushButton("", self)
        self._left_tag.setObjectName("previewCaption")
        self._left_tag.setCursor(Qt.CursorShape.PointingHandCursor)
        self._left_tag.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._right_tag = QPushButton("", self)
        self._right_tag.setObjectName("previewCaption")
        self._right_tag.setCursor(Qt.CursorShape.PointingHandCursor)
        self._right_tag.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._left_menu: QMenu | None = None
        self._right_menu: QMenu | None = None
        self._left_tag.setVisible(False)
        self._right_tag.setVisible(False)
        self._message.setVisible(False)
        # 画像の右下に重ねる小さなボタン 2 つ（不透明で下の絵と混ざらない）。
        self.fit_btn = QPushButton("", self)
        self.fit_btn.setObjectName("previewTool")
        self.fit_btn.setIcon(make_icon(Icon.FIT, 16, "#f3f5f7"))
        self.fit_btn.setToolTip(t("全体表示"))
        self.fit_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.fit_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.fit_btn.clicked.connect(self.fit_view)
        self.actual_btn = QPushButton("1:1", self)
        self.actual_btn.setObjectName("previewTool")
        self.actual_btn.setToolTip(t("等倍"))
        self.actual_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.actual_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.actual_btn.clicked.connect(self.actual_pixels)

    # ---------------------------------------------------------- 設定と状態
    def set_images(
        self,
        left: QImage | None,
        left_label: str,
        right: QImage | None,
        right_label: str,
    ) -> None:
        """左右の画とラベルを設定する。None はその側に画が無いことを表す。"""
        self._left = None if left is None or left.isNull() else left
        self._right = None if right is None or right.isNull() else right
        self._left_pm = QPixmap.fromImage(self._left) if self._left else None
        self._right_pm = QPixmap.fromImage(self._right) if self._right else None
        if self._left_pm is not None and self._left_pm.isNull():
            self._left_pm = None
        if self._right_pm is not None and self._right_pm.isNull():
            self._right_pm = None
        self._left_label = left_label
        self._right_label = right_label
        # 同じ画が両側に来たら1枚表示にする（比較相手が元の画像だけのとき）。
        if self._has_two() and self._left is not None and self._right is not None:
            if self._left.cacheKey() == self._right.cacheKey():
                self._right = None
                self._right_pm = None
        base = self._base_size()
        if base != self._last_base:
            old = self._last_base
            self._last_base = base
            self._selection = None
            same_shape = (
                not old.isEmpty() and not base.isEmpty()
                and abs(old.width() * base.height() - old.height() * base.width())
                <= max(old.width(), base.width())
            )
            if same_shape and not self._fit_on_resize:
                # 同じ画の解像度違い（元の画像 → 4 倍後など）に切り替わっただけなら、
                # 拡大して見ていた場所をそのまま保つ。
                self._scale *= old.width() / base.width()
            elif not base.isEmpty():
                self.fit_view()
        self._refresh_tags()
        self.update()

    def clear(self) -> None:
        """画を消す（文言は呼び出し側が set_message で出す）。"""
        self._left = None
        self._right = None
        self._left_pm = None
        self._right_pm = None
        self._left_label = ""
        self._right_label = ""
        self._selection = None
        self._last_base = QSize(0, 0)
        self._refresh_tags()
        self.update()

    def set_message(self, text: str) -> None:
        """中央の文言を出す。空文字で消す。"""
        self._message.setText(text)
        self._message.setVisible(bool(text))
        self._layout_children()
        self.update()

    def left_label(self) -> str:
        """左タグの文言（テスト用。印を除いた表示名）。"""
        return self._left_label

    def right_label(self) -> str:
        """右タグの文言（テスト用。印を除いた表示名）。"""
        return self._right_label

    @property
    def left_tag(self) -> QPushButton:
        """左上の札ボタン（メニュー操作用）。"""
        return self._left_tag

    @property
    def right_tag(self) -> QPushButton:
        """右上の札ボタン（メニュー操作用）。"""
        return self._right_tag

    def set_left_menu(self, menu: QMenu | None) -> None:
        """左札のメニューを差し替える（選べるものが 1 つなら None）。"""
        self._left_menu = menu
        if menu is not None:
            self._left_tag.setMenu(menu)
        else:
            self._left_tag.setMenu(None)
        self._refresh_tags()

    def set_right_menu(self, menu: QMenu | None) -> None:
        """右札のメニューを差し替える（選べるものが 1 つなら None）。"""
        self._right_menu = menu
        if menu is not None:
            self._right_tag.setMenu(menu)
        else:
            self._right_tag.setMenu(None)
        self._refresh_tags()

    def message_text(self) -> str:
        """中央文言（テスト用）。"""
        return self._message.text()

    def set_split(self, value: float) -> None:
        """境界線の位置（0..1）。"""
        self._split = min(_SPLIT_MAX, max(_SPLIT_MIN, float(value)))
        self.update()

    def split(self) -> float:
        """境界線の位置（テスト用）。"""
        return self._split

    def scale(self) -> float:
        """現在の表示倍率（基底画の1ピクセルあたりの画面ピクセル数）。"""
        return self._scale

    def has_two(self) -> bool:
        """左右2枚を分けているか（テスト用）。"""
        return self._has_two()

    def set_selection(self, rect: tuple[int, int, int, int] | None) -> None:
        """選択矩形（基底画基準）を外から与える。画外は丸める。"""
        if rect is None:
            self._selection = None
        else:
            base = self._base_size()
            clamped = _clamp_rect_to_base(rect, base.width(), base.height())
            self._selection = clamped
        self.update()

    def selection(self) -> tuple[int, int, int, int] | None:
        """選択矩形（基底画基準。テスト用）。"""
        return self._selection

    def set_selection_mode(self, enabled: bool) -> None:
        """範囲選択モード（ドラッグで矩形を選ぶ）。"""
        self._selection_mode = bool(enabled)
        if not enabled:
            self._drag_mode = None
        self.setCursor(
            Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.ArrowCursor
        )

    def is_selection_mode(self) -> bool:
        """範囲選択モードか（テスト用）。"""
        return self._selection_mode

    def fit_view(self) -> None:
        """枠に収める。"""
        base = self._base_size()
        viewport = self.contentsRect()
        if base.isEmpty() or viewport.isEmpty():
            return
        scale = min(viewport.width() / base.width(), viewport.height() / base.height())
        if scale <= 0:
            return
        self._scale = scale
        self._offset = QPointF(
            viewport.center().x() - (base.width() * scale) / 2,
            viewport.center().y() - (base.height() * scale) / 2,
        )
        self._fit_on_resize = True
        self.update()

    def actual_pixels(self) -> None:
        """等倍（基底画の1ピクセル=画面の1ピクセル）。"""
        base = self._base_size()
        viewport = self.contentsRect()
        if base.isEmpty() or viewport.isEmpty():
            return
        self._scale = 1.0
        self._offset = QPointF(
            viewport.center().x() - base.width() / 2,
            viewport.center().y() - base.height() / 2,
        )
        self._fit_on_resize = False
        self.update()

    # ---------------------------------------------------------- 内部計算
    def _has_two(self) -> bool:
        return (
            self._left is not None
            and not self._left.isNull()
            and self._right is not None
            and not self._right.isNull()
        )

    def _base_size(self) -> QSize:
        base = self._right if self._right is not None else self._left
        if base is None or base.isNull():
            return QSize(0, 0)
        return QSize(base.width(), base.height())

    def _dest_rect(self) -> QRectF:
        base = self._base_size()
        return QRectF(self._offset, QSize(int(base.width() * self._scale),
                                         int(base.height() * self._scale)))

    def _widget_to_base(self, pos: QPointF) -> QPointF:
        if self._scale <= 0:
            return QPointF(0, 0)
        return QPointF(
            (pos.x() - self._offset.x()) / self._scale,
            (pos.y() - self._offset.y()) / self._scale,
        )

    def _divider_x(self) -> float:
        viewport = self.contentsRect()
        return viewport.left() + self._split * viewport.width()

    def _handle_center(self) -> QPointF:
        viewport = self.contentsRect()
        return QPointF(self._divider_x(), viewport.center().y())

    def _on_divider(self, pos: QPointF) -> bool:
        """分割線か、その中央の丸の上にいるか。"""
        if not self._has_two():
            return False
        if abs(pos.x() - self._divider_x()) <= _DIVIDER_GRAB:
            return True
        c = self._handle_center()
        return (pos.x() - c.x()) ** 2 + (pos.y() - c.y()) ** 2 <= _HANDLE_RADIUS ** 2

    @staticmethod
    def _paint_handle(painter: QPainter, c: QPointF) -> None:
        """線だけでは動かせると気づきにくいので、左右の矢印が入った丸を付ける。"""
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(QPen(QColor(0, 0, 0, 90), 1))
        painter.setBrush(QColor(_DIVIDER_COLOR))
        painter.drawEllipse(c, _HANDLE_RADIUS, _HANDLE_RADIUS)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#1b1f27"))
        for sign in (-1, 1):
            tip = c.x() + sign * 10
            base = c.x() + sign * 4
            painter.drawPolygon(QPolygonF([
                QPointF(tip, c.y()),
                QPointF(base, c.y() - 5.5),
                QPointF(base, c.y() + 5.5),
            ]))

    def _refresh_tags(self) -> None:
        has_any = self._left is not None or self._right is not None
        two = self._has_two()
        left_menu = self._left_menu if self._left_menu is not None and len(self._left_menu.actions()) >= 2 else None
        right_menu = self._right_menu if self._right_menu is not None and len(self._right_menu.actions()) >= 2 else None
        # 選べるものが 2 つ以上ある側だけ印を付けて押せるようにする。
        self._left_tag.setText(self._left_label)
        self._right_tag.setText(self._right_label)
        self._left_tag.setEnabled(left_menu is not None)
        self._right_tag.setEnabled(right_menu is not None)
        self._left_tag.setCursor(
            Qt.CursorShape.PointingHandCursor if left_menu is not None
            else Qt.CursorShape.ArrowCursor
        )
        self._right_tag.setCursor(
            Qt.CursorShape.PointingHandCursor if right_menu is not None
            else Qt.CursorShape.ArrowCursor
        )
        self._left_tag.setVisible(has_any and bool(self._left_label))
        # 1枚表示のときは右タグを出さない（境界線も出さない）。
        self._right_tag.setVisible(two and bool(self._right_label))
        has_image = not self._base_size().isEmpty()
        self.fit_btn.setVisible(has_image)
        self.actual_btn.setVisible(has_image)
        self._layout_children()

    def _layout_children(self) -> None:
        rect = self.contentsRect()
        self._message.setGeometry(rect.adjusted(24, 24, -24, -24))
        self._left_tag.adjustSize()
        self._right_tag.adjustSize()
        self._left_tag.move(rect.left() + 12, rect.top() + 8)
        self._right_tag.move(
            rect.right() - 12 - self._right_tag.width(), rect.top() + 8
        )
        # 右下の隅に重ねる小さなボタン 2 つ（等倍の左に全体表示）。
        self.actual_btn.adjustSize()
        self.fit_btn.adjustSize()
        margin = 12
        gap = 8
        actual_w = self.actual_btn.width()
        actual_h = self.actual_btn.height()
        fit_w = self.fit_btn.width()
        fit_h = self.fit_btn.height()
        row_h = max(actual_h, fit_h)
        base_y = rect.bottom() - margin - row_h
        self.actual_btn.move(rect.right() - margin - actual_w, base_y + (row_h - actual_h) // 2)
        self.fit_btn.move(
            rect.right() - margin - actual_w - gap - fit_w,
            base_y + (row_h - fit_h) // 2,
        )

    # ---------------------------------------------------------- 描画
    def paintEvent(self, event) -> None:  # noqa: N802
        super().paintEvent(event)
        base = self._base_size()
        if base.isEmpty():
            return
        dest = self._dest_rect()
        painter = QPainter(self)
        # 等倍を超えて拡大したときだけ補間なしで描く（画素を見るため）。
        # 等倍までは滑らかに描く。解像度の低い元の画像の画が、等倍表示で
        # 実際より粗く見えてしまうのを避ける。
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform,
                              self._scale <= 1.0 + 1e-6)
        viewport = self.contentsRect()
        if not self._has_two():
            pm = self._left_pm if self._left_pm is not None else self._right_pm
            if pm is not None and not pm.isNull():
                painter.drawPixmap(dest, pm, QRectF(pm.rect()))
        else:
            divider_x = self._divider_x()
            left_clip = QRectF(viewport.left(), viewport.top(),
                               divider_x - viewport.left(), viewport.height())
            right_clip = QRectF(divider_x, viewport.top(),
                                viewport.right() - divider_x, viewport.height())
            if self._left_pm is not None:
                painter.save()
                painter.setClipRect(left_clip)
                painter.drawPixmap(dest, self._left_pm, QRectF(self._left_pm.rect()))
                painter.restore()
            if self._right_pm is not None:
                painter.save()
                painter.setClipRect(right_clip)
                painter.drawPixmap(dest, self._right_pm, QRectF(self._right_pm.rect()))
                painter.restore()
            painter.save()
            painter.setPen(QPen(QColor(_DIVIDER_COLOR), 2))
            painter.drawLine(
                QPointF(divider_x, viewport.top()),
                QPointF(divider_x, viewport.bottom()),
            )
            self._paint_handle(painter, self._handle_center())
            painter.restore()
        if self._selection is not None:
            x, y, w, h = self._selection
            rect = QRectF(
                self._offset.x() + x * self._scale,
                self._offset.y() + y * self._scale,
                w * self._scale,
                h * self._scale,
            )
            painter.save()
            pen = QPen(QColor(_SELECT_COLOR), 1.5)
            pen.setStyle(Qt.PenStyle.DotLine)
            painter.setPen(pen)
            painter.drawRect(rect)
            painter.restore()
        painter.end()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._layout_children()
        if self._fit_on_resize and not self._base_size().isEmpty():
            self.fit_view()

    # ---------------------------------------------------------- 操作
    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if self._base_size().isEmpty():
            return
        pos = event.position()
        if self._selection_mode:
            anchor = self._widget_to_base(pos)
            base = self._base_size()
            anchor.setX(min(max(0.0, anchor.x()), float(base.width())))
            anchor.setY(min(max(0.0, anchor.y()), float(base.height())))
            self._drag_mode = "select"
            self._sel_anchor = anchor
            self._selection = (int(anchor.x()), int(anchor.y()), 0, 0)
            self.update()
            return
        if self._on_divider(pos):
            self._drag_mode = "split"
            self._drag_pos = pos
            self._drag_split = self._split
            return
        self._drag_mode = "pan"
        self._drag_pos = pos
        self._drag_offset = QPointF(self._offset)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        pos = event.position()
        if self._drag_mode is None:
            # 押す前から、つまめる場所ではカーソルを左右の矢印にする
            if not self._selection_mode and self._on_divider(pos):
                self.setCursor(Qt.CursorShape.SplitHCursor)
            elif not self._selection_mode:
                self.unsetCursor()
            return
        if self._base_size().isEmpty():
            return
        if self._drag_mode == "select" and self._sel_anchor is not None:
            base = self._base_size()
            cur = self._widget_to_base(pos)
            cur.setX(min(max(0.0, cur.x()), float(base.width())))
            cur.setY(min(max(0.0, cur.y()), float(base.height())))
            x0 = int(min(self._sel_anchor.x(), cur.x()))
            y0 = int(min(self._sel_anchor.y(), cur.y()))
            w = int(abs(cur.x() - self._sel_anchor.x()))
            h = int(abs(cur.y() - self._sel_anchor.y()))
            clamped = _clamp_rect_to_base((x0, y0, w, h),
                                          base.width(), base.height())
            self._selection = clamped if clamped is not None else (x0, y0, 0, 0)
            self.update()
        elif self._drag_mode == "split":
            viewport = self.contentsRect()
            if viewport.width() > 0:
                ratio = (pos.x() - viewport.left()) / viewport.width()
                self._split = min(_SPLIT_MAX, max(_SPLIT_MIN, ratio))
                self.update()
        elif self._drag_mode == "pan" and self._drag_pos is not None:
            self._offset = QPointF(
                self._drag_offset.x() + (pos.x() - self._drag_pos.x()),
                self._drag_offset.y() + (pos.y() - self._drag_pos.y()),
            )
            self._fit_on_resize = False
            self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if self._drag_mode == "select":
            rect = self._selection
            if rect is not None and (rect[2] <= 0 or rect[3] <= 0):
                rect = None
                self._selection = None
                self.update()
            try:
                self.selectionChanged.emit(rect)
            except RuntimeError:
                pass
        elif self._drag_mode == "split":
            try:
                self.splitChanged.emit(self._split)
            except RuntimeError:
                pass
        self._drag_mode = None
        self._drag_pos = None
        self._sel_anchor = None

    def wheelEvent(self, event) -> None:  # noqa: N802
        if self._base_size().isEmpty():
            return
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = _ZOOM_IN if delta > 0 else _ZOOM_OUT
        anchor = self._widget_to_base(event.position())
        new_scale = min(_SCALE_MAX, max(_SCALE_MIN, self._scale * factor))
        pos = event.position()
        self._offset = QPointF(
            pos.x() - anchor.x() * new_scale,
            pos.y() - anchor.y() * new_scale,
        )
        self._scale = new_scale
        self._fit_on_resize = False
        self.update()
        event.accept()


class _TrialWorker(QObject):
    """試し拡大を別スレッドで走らせる QObject（QThread へ move する）。"""

    succeeded = Signal(str)  # 出力 PNG パス
    failed = Signal(str)  # エラー文
    canceled = Signal()
    progressed = Signal(float, str)

    def __init__(
        self,
        source_png: str,
        settings: UpscaleSettings,
        out_png: str,
        cancel: threading.Event,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._source_png = source_png
        self._settings = settings
        self._out_png = out_png
        self._cancel = cancel

    @Slot()
    def run(self) -> None:
        """作業スレッド上で試しを実行する。結果はシグナルでのみ返す。"""
        from app.core.jobs import Cancelled

        def _cb(frac: float, msg: str) -> None:
            try:
                self.progressed.emit(float(frac), msg or "")
            except RuntimeError:
                pass

        try:
            trial_core.run_trial(
                self._source_png, self._settings, self._out_png,
                progress=_cb, cancel=self._cancel,
            )
        except Cancelled:
            try:
                self.canceled.emit()
            except RuntimeError:
                pass
        except Exception as exc:  # noqa: BLE001 - 文言にして状態行へ
            try:
                self.failed.emit(str(exc) or exc.__class__.__name__)
            except RuntimeError:
                pass
        else:
            if self._cancel.is_set():
                try:
                    self.canceled.emit()
                except RuntimeError:
                    pass
            else:
                try:
                    self.succeeded.emit(self._out_png)
                except RuntimeError:
                    pass


# 結果キャッシュのキー（ファイル, 位置の秒, 範囲, AI実行先, モデル, 倍率）。
_TrialKey = tuple[str, float, tuple[int, int, int, int] | None, str, str, int]


class StatusLabel(QLabel):
    """文言が空の間は場所を取らない状況行（右列の高さを節約する）。"""

    def __init__(self, text: str = "", parent=None) -> None:
        super().__init__(text, parent)
        self.setVisible(bool(text))

    def setText(self, text: str) -> None:  # noqa: N802
        super().setText(text)
        self.setVisible(bool(text))


class TrialPanel(QWidget):
    """中央列: 比較ビューと動画の位置行。クイック確認の組も持つ。

    中央のレイアウトには画と動画のスライダーだけ置く。クイック確認の
    ボタンと状況行は `quick_box` にまとめ、MainWindow が右列に差し込む。
    MainWindow は `show_job`・`set_main_running`・`discard_file` だけ呼ぶ。
    確認の設定は押した時点の、そのファイル用の `build_settings` 値を使う。
    """

    trial_running_changed = Signal(bool)
    _frame_ready = Signal(int, QImage, str)

    # クイック確認の「?」の説明文（改行も仕様どおり）。
    QUICK_HELP = N_(
        "仕上がり確認のため、1 枚だけ拡大処理を行い、元の画像と比較できます。\n"
        "動画は、画像下のスライダーで拡大するフレームを選択できます。\n"
        "「範囲を選択してクイック確認」を使うと、その 1 枚のさらに一部分だけに処理を限定できます。"
    )

    def __init__(self, build_settings, model_label, parent=None) -> None:
        super().__init__(parent)
        self._build_settings = build_settings
        self._model_label = model_label
        self._workdir = trial_core.create_work_dir()
        self._job: Job | None = None
        self._seconds = 0.0
        self._duration: float | None = None
        self._rect: tuple[int, int, int, int] | None = None
        self._source_image: QImage | None = None
        self._source_png: str | None = None
        self._results: dict[_TrialKey, dict] = {}
        self._order: list[_TrialKey] = []
        self._file_state: dict[str, dict] = {}
        self._temp_by_file: dict[str, set[str]] = {}
        self._frame_token = 0
        self._main_running = False
        self._npu_converting = False
        self._trial_thread: QThread | None = None
        self._trial_worker: _TrialWorker | None = None
        self._trial_cancel: threading.Event | None = None
        self._trial_key: _TrialKey | None = None
        self._status_hold = ""
        self._left_key: _TrialKey | None = None
        self._right_key: _TrialKey | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)

        self.view = CompareView()
        root.addWidget(self.view, 1)

        # 動画のときだけ出す行: 位置スライダーと時刻表示。
        # 画像のとき中央の下段には何も出さない。
        self.video_row = QWidget()
        video_lay = QHBoxLayout(self.video_row)
        video_lay.setContentsMargins(0, 0, 0, 0)
        video_lay.setSpacing(8)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)
        self.slider.setSingleStep(1)
        video_lay.addWidget(self.slider, 1)
        self.time_label = QLabel("0:00 / 0:00")
        self.time_label.setObjectName("hint")
        video_lay.addWidget(self.time_label)
        root.addWidget(self.video_row)

        # 全体表示・等倍は画像の右下に重ねる（CompareView が持つ）。
        # テストから使う属性名はここに残す。
        self.fit_btn = self.view.fit_btn
        self.actual_btn = self.view.actual_btn

        # クイック確認の組（右列に差し込む。ファイル選択時のみ表示）。
        self.trial_btn = QPushButton(t("クイック確認"))
        self.trial_btn.setObjectName("accent")
        self.trial_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.trial_btn.clicked.connect(self._on_trial_button)
        from .settings_drawer import HelpIcon

        self.quick_help = HelpIcon(t(self.QUICK_HELP))
        self.range_btn = QPushButton(t("範囲を選択してクイック確認"))
        self.range_btn.setObjectName("quickSecondary")
        self.range_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.range_btn.clicked.connect(self._on_range_button)
        self.status_label = StatusLabel("")
        self.status_label.setObjectName("hint")
        self.status_label.setWordWrap(True)
        self.quick_box = QWidget()
        self.quick_box.setObjectName("scaleWrap")  # 背景を持たない包み
        quick = QVBoxLayout(self.quick_box)
        quick.setContentsMargins(0, 0, 0, 0)
        quick.setSpacing(8)
        # 2 つのボタンを同じ幅・同じ高さにそろえ、? はその右に置く
        quick_grid = QGridLayout()
        quick_grid.setContentsMargins(0, 0, 0, 0)
        quick_grid.setHorizontalSpacing(6)
        quick_grid.setVerticalSpacing(8)
        quick_grid.addWidget(self.trial_btn, 0, 0)
        quick_grid.addWidget(self.quick_help, 0, 1)
        quick_grid.addWidget(self.range_btn, 1, 0)
        quick_grid.setColumnStretch(0, 1)
        quick.addLayout(quick_grid)
        quick.addWidget(self.status_label)

        self.slider.valueChanged.connect(self._on_slider_moved)
        self.slider.sliderReleased.connect(self._on_slider_released)
        self.view.selectionChanged.connect(self._on_view_selection)
        self._frame_ready.connect(self._on_frame_ready)
        self._rebuild_tags()
        self.refresh()

    # ---------------------------------------------------------- ファイル切替
    def show_job(self, job: Job | None) -> None:
        """選択中ファイルの試し状態を表示する（None なら空にする）。"""
        self._save_file_state()
        self._job = job
        self._source_image = None
        self._source_png = None
        self._status_hold = ""
        if job is None:
            self._seconds = 0.0
            self._duration = None
            self._rect = None
            self.view.clear()
            self.view.set_message(t("左の一覧からファイルを選ぶと、ここに表示します。"))
            self._rebuild_tags()
            self.refresh()
            return
        saved = self._file_state.get(str(job.input_path))
        if job.kind == JobKind.VIDEO:
            duration = self._video_duration(job)
            self._duration = duration
            if saved is not None and "seconds" in saved:
                self._seconds = min(max(0.0, float(saved["seconds"])),
                                    duration or 0.0)
            else:
                # 冒頭は黒画面のことがあるため、2 秒の位置から始める。
                # 4 秒より短い動画では真ん中を使う。
                self._seconds = (
                    min(_DEFAULT_START_SECONDS, duration / 2) if duration else 0.0
                )
            self._rect = saved.get("rect") if saved else None
        else:
            self._duration = None
            self._seconds = 0.0
            self._rect = saved.get("rect") if saved else None
        self.view.set_selection_mode(False)
        self.view.set_message("")
        self._rebuild_tags(restore=saved)
        self.refresh()
        self._load_source_async()

    def _save_file_state(self) -> None:
        """いまのファイルの位置・範囲・左右の選択を覚える。"""
        if self._job is None:
            return
        self._file_state[str(self._job.input_path)] = {
            "seconds": self._seconds,
            "rect": self._rect,
            "left": self._left_key,
            "right": self._right_key,
        }

    def _video_duration(self, job: Job) -> float | None:
        """動画長（秒）。frame_count/fps があればそれを使い、無ければ probe する。"""
        if job.frame_count and job.fps:
            try:
                duration = job.frame_count / job.fps
                if duration > 0:
                    return float(duration)
            except (TypeError, ZeroDivisionError):
                pass
        try:
            from app.core import media

            info = media.probe(str(job.input_path))
            if info.duration and info.duration > 0:
                return float(info.duration)
            if info.frame_count and info.fps:
                return float(info.frame_count / info.fps)
        except Exception:
            pass
        return None

    # ---------------------------------------------------------- コマ読み込み
    def _load_source_async(self) -> None:
        """現在の元画像を別スレッドで読み込む。連続要求は最後だけ反映する。"""
        job = self._job
        if job is None or job.kind == JobKind.FOLDER:
            return
        self._frame_token += 1
        token = self._frame_token
        path = str(job.input_path)
        kind = job.kind
        seconds = self._seconds
        workdir = str(self._workdir)

        def work() -> None:
            image = QImage()
            png_path: str | None = None
            if kind == JobKind.IMAGE:
                image = _load_source_qimage(path)
                png_path = path
            else:
                try:
                    fd, name = tempfile.mkstemp(prefix="src-", suffix=".png",
                                                dir=workdir)
                    import os as _os

                    _os.close(fd)
                    trial_core.extract_frame(path, seconds, name)
                    png_path = name
                    image = _load_source_qimage(name)
                except Exception:
                    image = QImage()
                    png_path = None
            if png_path is not None and kind == JobKind.VIDEO:
                self._temp_by_file.setdefault(path, set()).add(png_path)
            try:
                self._frame_ready.emit(token, image, png_path or "")
            except RuntimeError:
                pass

        threading.Thread(target=work, daemon=True).start()

    @Slot(int, QImage, str)
    def _on_frame_ready(self, token: int, image: QImage, png_path: str) -> None:
        if token != self._frame_token:
            return
        job = self._job
        if job is None:
            return
        if image.isNull():
            self._source_image = None
            self._source_png = None
            self.view.clear()
            self.view.set_message(t("このファイルは表示できません。処理はできます。"))
            self.refresh()
            return
        self._source_image = image
        if job.kind == JobKind.IMAGE:
            self._source_png = str(job.input_path)
        else:
            self._source_png = png_path or None
        self.view.set_message("")
        self._update_view()
        self.refresh()

    # ---------------------------------------------------------- 位置と範囲
    def _on_slider_moved(self, _value: int) -> None:
        if self._job is None or self._job.kind != JobKind.VIDEO:
            return
        duration = self._duration or 0.0
        seconds = self.slider.value() / 10.0
        self.time_label.setText(
            trial_core.format_time_position(seconds, duration)
        )

    def _on_slider_released(self) -> None:
        if self.is_trial_running():
            return
        if self._job is None or self._job.kind != JobKind.VIDEO:
            return
        self._seconds = self.slider.value() / 10.0
        self._status_hold = ""
        self._rebuild_tags()
        self.refresh()
        self._load_source_async()

    def _on_view_selection(self, rect: object) -> None:
        """比較ビューでのドラッグ結果を試しの範囲にする。囲み終えたらすぐ始める。"""
        if not self.view.is_selection_mode():
            return
        base = self.view._base_size()
        if rect is None or base.isEmpty():
            self.view.set_selection(None)
            return
        x, y, w, h = (int(v) for v in rect)  # type: ignore[union-attr]
        # 選択モード中は全画を表示しているため、基底画=元画像として扱う。
        src_w = self._source_image.width() if self._source_image else base.width()
        src_h = self._source_image.height() if self._source_image else base.height()
        sx = src_w / base.width() if base.width() else 1.0
        sy = src_h / base.height() if base.height() else 1.0
        conv = _clamp_rect_to_base(
            (int(x * sx), int(y * sy), int(w * sx), int(h * sy)), src_w, src_h
        )
        if conv is None or conv[2] < _MIN_SELECT or conv[3] < _MIN_SELECT:
            # 小さすぎるドラッグは取り消して選び直せるように残す。
            self.view.set_selection(None)
            return
        self._rect = conv
        self.view.set_selection_mode(False)
        self.view.set_selection(None)
        self._status_hold = ""
        self._rebuild_tags()
        self.refresh()
        self._update_view()
        # もう一度押させず、そのまま確認を始める。
        self._on_trial_button()

    def _on_range_button(self) -> None:
        if self.is_trial_running() or self._main_running:
            return
        if self._rect is not None:
            self._rect = None
            self.view.set_selection_mode(False)
            self.view.set_selection(None)
            self._status_hold = ""
            self._rebuild_tags()
            self.refresh()
            self._update_view()
            return
        if self.view.is_selection_mode():
            self.view.set_selection_mode(False)
            self.view.set_selection(None)
            self.refresh()
            self._update_view()
            return
        if self._job is None or self._job.kind == JobKind.FOLDER:
            return
        if self._source_image is None:
            return
        self.view.set_selection_mode(True)
        self.view.set_selection(None)
        self.refresh()
        self._update_view()

    # ---------------------------------------------------------- 比べる相手
    def _matching_keys(self) -> list[_TrialKey]:
        if self._job is None:
            return []
        file_str = str(self._job.input_path)
        seconds = round(self._seconds, 3) if self._job.kind == JobKind.VIDEO else 0.0
        return [
            key for key in self._order
            if key[0] == file_str and key[1] == seconds and key[2] == self._rect
            and key in self._results
        ]

    def _rebuild_tags(self, restore: dict | None = None) -> None:
        """札の選択を今の結果に合わせる（左右コンボと同じ規則）。"""
        left_prev = self._left_key
        right_prev = self._right_key
        if restore is not None:
            left_prev = restore.get("left")
            right_prev = restore.get("right")
        matches = self._matching_keys()
        if self._left_key_valid(left_prev, matches):
            self._left_key = left_prev  # type: ignore[assignment]
        else:
            self._left_key = None
        if self._right_key_valid(right_prev, matches):
            self._right_key = right_prev  # type: ignore[assignment]
        elif matches:
            self._right_key = matches[-1]
        else:
            self._right_key = None
        self._update_view()

    @staticmethod
    def _left_key_valid(value: object, matches: list) -> bool:
        if value is None:
            return True
        return value in matches

    @staticmethod
    def _right_key_valid(value: object, matches: list) -> bool:
        if value is None:
            return True
        return value in matches

    def _menu_for_side(self, side: str) -> QMenu | None:
        """その側の札メニュー（選べるものが 2 つ以上あるときだけ作る）。"""
        matches = self._matching_keys()
        if not matches:
            return None
        current = self._left_key if side == "left" else self._right_key
        menu = QMenu(self.view)
        original = menu.addAction(t("元の画像"))
        original.setCheckable(True)
        original.setChecked(current is None)
        original.triggered.connect(lambda _=False: self._on_tag_selected(side, None))
        for key in matches:
            label = self._results[key]["label"]
            action = menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(key == current)
            action.triggered.connect(
                lambda _=False, k=key: self._on_tag_selected(side, k)
            )
        return menu

    def _on_tag_selected(self, side: str, key: _TrialKey | None) -> None:
        """札メニューでの切り替え。"""
        if side == "left":
            self._left_key = key
        else:
            self._right_key = key
        self._save_file_state()
        self._update_view()
        self.refresh()

    def _resolve_side(self, key: object) -> tuple[QImage | None, str]:
        if key is None:
            return self._display_source(), t("元の画像")
        entry = self._results.get(key)  # type: ignore[arg-type]
        if entry is None:
            return self._display_source(), t("元の画像")
        image = entry.get("qimage")
        if image is None or image.isNull():
            try:
                loaded = QImage(str(entry["path"]))
            except Exception:
                loaded = QImage()
            if loaded.isNull():
                return self._display_source(), t("元の画像")
            entry["qimage"] = loaded
            image = loaded
        return image, str(entry["label"])

    def _display_source(self) -> QImage | None:
        """表示用の元画像。範囲ありならその切り出し（比較対象をそろえるため）。"""
        if self._source_image is None:
            return None
        if self._rect is None or self.view.is_selection_mode():
            return self._source_image
        return _crop_qimage(self._source_image, self._rect)

    def _update_view(self) -> None:
        job = self._job
        if job is None:
            return
        if job.kind == JobKind.FOLDER:
            self.view.clear()
            self.view.set_message(t("フォルダの中身はここには表示されません。"))
            return
        if self._source_image is None:
            self.view.clear()
            self.view.set_message(t("読み込み中…"))
            return
        self.view.set_message("")
        left_image, left_label = self._resolve_side(self._left_key)
        right_image, right_label = self._resolve_side(self._right_key)
        self.view.set_images(left_image, left_label, right_image, right_label)
        self.view.set_left_menu(self._menu_for_side("left"))
        self.view.set_right_menu(self._menu_for_side("right"))
        self.view.set_selection(None)
        if self.view.is_selection_mode():
            # 範囲選択中は画像の上に案内を出す。
            self.view.set_message(t("確認したい部分をドラッグで囲んでください"))

    # ---------------------------------------------------------- 文言と有効化
    def refresh(self) -> None:
        """ボタン文言・表示/無効・状況行を現在の状態に合わせる。"""
        job = self._job
        running = self.is_trial_running()
        # 画像と動画で文言は同じにする。確認中だけ中止になる。
        if running:
            self.trial_btn.setText(t("中止"))
        else:
            self.trial_btn.setText(t("クイック確認"))
        if self._rect is not None:
            self.range_btn.setText(t("範囲を解除"))
        else:
            self.range_btn.setText(t("範囲を選択してクイック確認"))

        is_video = job is not None and job.kind == JobKind.VIDEO
        self.video_row.setVisible(is_video)
        if is_video:
            duration = self._duration or 0.0
            maximum = max(0, int(round(duration * 10)))
            block = self.slider.blockSignals(True)
            try:
                self.slider.setMaximum(maximum)
                self.slider.setValue(int(round(self._seconds * 10)))
            finally:
                self.slider.blockSignals(block)
            self.time_label.setText(
                trial_core.format_time_position(
                    self.slider.value() / 10.0, duration
                )
            )

        settings = self._current_settings_or_none()
        model_missing = settings is not None and settings.model is None
        can_pick = (
            job is not None
            and job.kind in (JobKind.IMAGE, JobKind.VIDEO)
            and self._source_image is not None
        )
        self.trial_btn.setEnabled(
            running or (can_pick and not self._main_running
                        and not model_missing and not self._npu_converting)
        )
        editable = not running and not self._main_running
        self.range_btn.setEnabled(editable and can_pick)
        self.slider.setEnabled(editable)
        has_image = self._source_image is not None
        self.fit_btn.setEnabled(has_image)
        self.actual_btn.setEnabled(has_image)

        if running:
            return
        if self._status_hold:
            self.status_label.setText(self._status_hold)
        elif self._npu_converting:
            self.status_label.setText(t("NPU の変換中はクイック確認できません"))
        elif self._main_running:
            self.status_label.setText(t("処理中はクイック確認できません"))
        elif job is not None and job.kind == JobKind.FOLDER:
            self.status_label.setText(
                t("フォルダはクイック確認できません。中の画像を 1 枚追加すると確認できます。")
            )
        elif model_missing:
            self.status_label.setText(t("拡大モデルを選ぶとクイック確認できます"))
        else:
            self.status_label.setText("")

    def _current_settings_or_none(self) -> UpscaleSettings | None:
        """選択中のファイル用の設定（個別があればそれ、無ければ種類の既定）。"""
        try:
            try:
                return self._build_settings(self._job)
            except TypeError:
                return self._build_settings()
        except Exception:
            return None

    def set_main_running(self, running: bool) -> None:
        """本処理の実行状態を反映する（確認ボタンの無効化と状況行のため）。"""
        self._main_running = bool(running)
        self.refresh()

    def set_npu_converting(self, converting: bool) -> None:
        """NPU 変換の実行状態を反映する（確認ボタンの無効化と状況行のため）。"""
        self._npu_converting = bool(converting)
        self.refresh()

    def is_trial_running(self) -> bool:
        """試しが実行中か（テスト用）。"""
        return self._trial_thread is not None

    # ---------------------------------------------------------- 試しの実行
    def _on_trial_button(self) -> None:
        if self.is_trial_running():
            if self._trial_cancel is not None:
                self._trial_cancel.set()
            return
        job = self._job
        if job is None or job.kind == JobKind.FOLDER or self._main_running:
            return
        if self._npu_converting:
            return
        settings = self._current_settings_or_none()
        if settings is None or settings.model is None:
            self.refresh()
            return
        if self._source_image is None or self._source_png is None:
            return
        if (settings.backend == UpscaleBackend.NPU_NATIVE
                and not npu_prepare.is_converted(str(settings.model))):
            # 短辺不足で GPU に自動切替する入力は止めない。
            width, height = self._source_image.width(), self._source_image.height()
            if (
                width <= 0 or height <= 0
                or helper_backend.effective_backend(
                    settings.backend, width, height
                ) == UpscaleBackend.NPU_NATIVE
            ):
                try:
                    label = self._model_label(str(settings.model))
                except Exception:
                    label = str(settings.model)
                self._status_hold = npu_prepare.not_converted_message(label)
                self.refresh()
                return
        key = self._make_key(settings)
        hit = self._results.get(key)
        if hit is not None and Path(str(hit["path"])).exists():
            self._status_hold = ""
            self._right_key = key
            self._save_file_state()
            self._update_view()
            self.refresh()
            return
        self._status_hold = ""
        self._start_trial(key, settings)

    def _make_key(self, settings: UpscaleSettings) -> _TrialKey:
        assert self._job is not None
        file_str = str(self._job.input_path)
        seconds = (
            round(self._seconds, 3) if self._job.kind == JobKind.VIDEO else 0.0
        )
        backend = settings.backend.value
        return (file_str, seconds, self._rect, backend, str(settings.model), int(settings.scale))

    def _redactions(self, settings: UpscaleSettings) -> list[str]:
        job_path = str(self._job.input_path) if self._job else ""
        return [
            settings.backend.value,
            str(settings.model or ""),
            job_path,
            Path(job_path).name if job_path else "",
            str(self._workdir),
            str(self._source_png or ""),
            "realesrgan-ncnn-vulkan",
            "rife-ncnn-vulkan",
            "winml_gpu",
            "npu_native",
            "swinir_cuda",
            "vulkan",
        ]

    def _start_trial(self, key: _TrialKey, settings: UpscaleSettings) -> None:
        assert self._job is not None and self._source_png is not None
        file_str = str(self._job.input_path)
        try:
            if self._rect is not None:
                fd, crop_name = tempfile.mkstemp(prefix="range-", suffix=".png",
                                                 dir=str(self._workdir))
                import os as _os

                _os.close(fd)
                trial_core.crop_image(self._source_png, self._rect, crop_name)
                source_png = crop_name
                self._temp_by_file.setdefault(file_str, set()).add(crop_name)
            else:
                source_png = self._source_png
            fd, out_name = tempfile.mkstemp(prefix="trial-", suffix=".png",
                                            dir=str(self._workdir))
            import os as _os2

            _os2.close(fd)
            self._temp_by_file.setdefault(file_str, set()).add(out_name)
        except Exception as exc:
            self._status_hold = t(
                "試せませんでした: {message}",
                message=sanitize_error_message(str(exc), self._redactions(settings)),
            )
            self.refresh()
            return
        cancel = threading.Event()
        worker = _TrialWorker(source_png, settings, out_name, cancel)
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progressed.connect(self._on_trial_progress)
        worker.succeeded.connect(self._on_trial_succeeded)
        worker.failed.connect(self._on_trial_failed)
        worker.canceled.connect(self._on_trial_canceled)
        self._trial_worker = worker
        self._trial_cancel = cancel
        self._trial_key = key
        self._trial_thread = thread
        self.status_label.setText(t("確認中…"))
        self.refresh()
        try:
            self.trial_running_changed.emit(True)
        except RuntimeError:
            pass
        thread.start()

    @Slot(float, str)
    def _on_trial_progress(self, frac: float, _msg: str) -> None:
        if not self.is_trial_running():
            return
        try:
            pct = int(round(max(0.0, min(1.0, float(frac))) * 100))
        except (TypeError, ValueError):
            self.status_label.setText(t("確認中…"))
            return
        self.status_label.setText(t("確認中… {pct}%", pct=pct))

    @Slot(str)
    def _on_trial_succeeded(self, out_path: str) -> None:
        key = self._trial_key
        job = self._job
        if key is None:
            self._stop_trial_thread()
            return
        try:
            label = self._model_label(str(key[4]))
        except Exception:
            label = str(key[4])
        image = QImage(out_path)
        self._results[key] = {"path": out_path, "label": label, "qimage": image}
        if key not in self._order:
            self._order.append(key)
        current = self._current_base_key()
        self._stop_trial_thread()
        if current is not None and current[:3] == key[:3]:
            self._status_hold = ""
            self._right_key = key
            self._save_file_state()
            self._rebuild_tags()
        self.refresh()

    @Slot(str)
    def _on_trial_failed(self, message: str) -> None:
        settings = self._current_settings_or_none()
        redactions = self._redactions(settings) if settings is not None else []
        self._status_hold = t(
            "試せませんでした: {message}",
            message=sanitize_error_message(message, redactions),
        )
        self._remove_running_output()
        self._stop_trial_thread()
        self.refresh()

    @Slot()
    def _on_trial_canceled(self) -> None:
        self._remove_running_output()
        self._status_hold = t("中止しました")
        self._stop_trial_thread()
        self.refresh()

    def _current_base_key(self) -> tuple[str, float, object] | None:
        if self._job is None:
            return None
        seconds = round(self._seconds, 3) if self._job.kind == JobKind.VIDEO else 0.0
        return (str(self._job.input_path), seconds, self._rect)

    def _remove_running_output(self) -> None:
        if self._trial_key is None:
            return
        # 失敗・中止時はその回の出力だけ消す（覚えた結果は残す）。
        thread_out = None
        if self._trial_worker is not None:
            thread_out = getattr(self._trial_worker, "_out_png", None)
        if thread_out:
            try:
                Path(str(thread_out)).unlink(missing_ok=True)
            except OSError:
                pass
            paths = self._temp_by_file.get(self._trial_key[0])
            if paths is not None:
                paths.discard(str(thread_out))

    def _stop_trial_thread(self) -> None:
        thread, worker = self._trial_thread, self._trial_worker
        self._trial_thread = None
        self._trial_worker = None
        self._trial_cancel = None
        self._trial_key = None
        if thread is not None:
            try:
                thread.quit()
                thread.wait(5000)
            except RuntimeError:
                pass
            try:
                thread.deleteLater()
            except RuntimeError:
                pass
        if worker is not None:
            try:
                worker.deleteLater()
            except RuntimeError:
                pass
        try:
            self.trial_running_changed.emit(False)
        except RuntimeError:
            pass

    # ---------------------------------------------------------- 後始末
    def discard_file(self, file_str: str) -> None:
        """一覧から消えたファイルの結果と一時ファイルを消す。"""
        if self._trial_key is not None and self._trial_key[0] == file_str:
            if self._trial_cancel is not None:
                self._trial_cancel.set()
        for key in [key for key in self._order if key[0] == file_str]:
            entry = self._results.pop(key, None)
            if entry is not None:
                try:
                    Path(str(entry["path"])).unlink(missing_ok=True)
                except OSError:
                    pass
        self._order = [key for key in self._order if key[0] != file_str]
        for temp in self._temp_by_file.pop(file_str, set()):
            try:
                Path(temp).unlink(missing_ok=True)
            except OSError:
                pass
        self._file_state.pop(file_str, None)
        if self._job is not None and str(self._job.input_path) == file_str:
            self._status_hold = ""
        self._rebuild_tags()
        self.refresh()

    def discard_all(self) -> None:
        """全ファイルの結果と一時ファイルを消す。"""
        if self._trial_cancel is not None:
            self._trial_cancel.set()
        for key, entry in list(self._results.items()):
            try:
                Path(str(entry["path"])).unlink(missing_ok=True)
            except OSError:
                pass
        self._results.clear()
        self._order.clear()
        for _file_str, temps in list(self._temp_by_file.items()):
            for temp in temps:
                try:
                    Path(temp).unlink(missing_ok=True)
                except OSError:
                    pass
        self._temp_by_file.clear()
        self._file_state.clear()
        self._status_hold = ""
        self._rebuild_tags()
        self.refresh()

    def cancel_trial(self) -> None:
        """実行中の試しに中止を要求する（子プロセスを残さない）。"""
        if self._trial_cancel is not None:
            self._trial_cancel.set()

    def shutdown(self) -> None:
        """試し作業フォルダを消す。アプリ終了時に呼ぶ。"""
        self._save_file_state()
        if self._trial_cancel is not None:
            self._trial_cancel.set()
        thread = self._trial_thread
        if thread is not None:
            try:
                thread.quit()
                thread.wait(5000)
            except RuntimeError:
                pass
        self._trial_thread = None
        self._trial_worker = None
        self._trial_cancel = None
        self._trial_key = None
        trial_core.cleanup_work_dir(str(self._workdir))

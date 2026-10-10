"""詳細設定ドロワー（折りたたみ）。"""
from __future__ import annotations

import threading
import time

from PySide6.QtCore import QObject, QPointF, QRectF, QSize, Qt, QThread, QTimer, Signal, QUrl
from PySide6.QtGui import QColor, QDesktopServices, QIcon, QPaintEvent, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.core import addon_kits, catalog, helper_backend, npu_prepare
from app.core import user_settings
from app.core.jobs import Cancelled
from app.core.settings import (
    ProcessingOrder,
    UpscaleBackend,
    UpscaleSettings,
)
from app.i18n import N_, t

from . import theme
from .compare_view import sanitize_error_message

_IMAGE_FORMATS = ["png", "jpg", "webp"]
# mp4/mkv/mov は H.264(+AAC) を収容できる。webm は VP9/Opus が必要で
# 現状のエンコーダ選択(H.264系)とは噛み合わないため除外（フェーズ2で対応検討）。
_VIDEO_FORMATS = ["mp4", "mkv", "mov"]
_VIDEO_QUALITY_OPTIONS = [
    (N_("高画質（容量大）"), 18),
    (N_("標準"), 23),
    (N_("軽量（容量小）"), 28),
]
# 「補間後のfps」の選択肢。値はキーで持ち、実値は _FPS_CHOICES で引く
# （タプルをそのまま値にすると findData が等しい値を探せないため）。
_FPS_CHOICES: dict[str, tuple[float | None, int]] = {
    "x2": (None, 2),
    "x4": (None, 4),
    "x8": (None, 8),
    "fps60": (60.0, 2),
    "fps120": (120.0, 2),
}
# FILM 選択中に選べなくする固定 fps のキー。
_FPS_FIXED_KEYS = ("fps60", "fps120")
_TARGET_FPS_OPTIONS = [
    (N_("元動画の2倍"), "x2"),
    (N_("元動画の4倍"), "x4"),
    (N_("元動画の8倍"), "x8"),
    ("60 fps", "fps60"),
    ("120 fps", "fps120"),
]


def _fps_key_for(target: float | None, factor: int) -> str:
    """設定値に対応する選択肢のキー（無ければ元動画の 2 倍）。"""
    for key, (choice_target, choice_factor) in _FPS_CHOICES.items():
        if choice_target is None and target is None:
            if int(choice_factor) == int(factor):
                return key
        elif choice_target is not None and target is not None:
            if float(choice_target) == float(target):
                return key
    return "x2"
_PROCESSING_ORDER_OPTIONS = [
    (N_("アプコン → 補間（速い）"), ProcessingOrder.UPSCALE_FIRST.value),
    (N_("補間 → アプコン（省メモリ）"), ProcessingOrder.INTERPOLATE_FIRST.value),
]
_TILE_OPTIONS = [
    (N_("自動"), 0),
    (N_("メモリ節約"), 128),
    (N_("強めに節約"), 64),
]
_GPU_OPTIONS = [
    (N_("自動"), -1),
    ("GPU 0", 0),
    ("GPU 1", 1),
    ("GPU 2", 2),
    ("GPU 3", 3),
]
# AI 実行先の選択肢は app/core/catalog.py が唯一の出どころ（CLI と共有）。
_BACKEND_OPTIONS = catalog.BACKEND_OPTIONS


def available_backend_options() -> list[tuple[str, str]]:
    """表示する AI 実行先の一覧。

    NPU はキットがある PC だけ、SwinIR CUDA は実行環境が揃う PC だけ出す。
    """
    return [
        (label, value) for label, value in _BACKEND_OPTIONS
        if (value != UpscaleBackend.NPU_NATIVE.value or npu_prepare.npu_available())
        and (value != UpscaleBackend.SWINIR_CUDA.value
             or helper_backend.swinir_cuda_available())
    ]
_LANGUAGE_OPTIONS = [
    (N_("Windows の設定に合わせる"), "auto"),
    (N_("日本語"), "ja"),
    (N_("English"), "en"),
]
# アクセントカラーの選択肢（値・表示名の表は user_settings が唯一の出どころ）。
_ACCENT_OPTIONS = [
    (label, value) for value, label in user_settings.ACCENT_OPTIONS
]
_HELP = {
    "backend": N_("AIの実行先です。自動はDirectML GPUを優先し、起動できない場合はVulkanへ切り替えます。NPUはGPU負荷を抑えます。SwinIR CUDAは実写向けですが動画処理は非常に低速です。"),
    "image_format": N_("画像を書き出す形式です。pngは劣化なし、jpgは容量小、webpは容量を抑えやすい形式です。"),
    "video_format": N_("動画ファイルの保存形式です。mp4は再生互換性が高く、mkv/movは用途に合わせて選びます。"),
    "video_quality": N_("CRF/QPは動画の圧縮品質です。数字が小さいほど高画質で容量は大きくなります。"),
    "tile_size": N_("タイルは画像を分割して処理する単位です。通常は自動でOK。メモリ不足で失敗するときだけ節約側にします。"),
    "gpu_id": N_("通常は自動でOK。GPUが複数あるPCで、使うGPUを固定したい時だけ番号を選びます。"),
    "subfolder": N_("出力をまとめるフォルダ名です。上段の出力先が「元の場所」なら、元画像の横にこの名前のフォルダを作ります。"),
    "hw_encode": N_("動画の書き出しにGPUを使います。対応していれば速くなります。失敗時は通常エンコードに戻します。"),
    "keep_audio": N_("元動画の音声を、新しく作る動画にも入れます。"),
    "tta": N_("TTAは同じ画像を反転などで複数回処理して仕上げる高品質モードです。少し良くなる場合がありますが、かなり遅くなります。"),
    "output": N_("処理したファイルの保存先です。「元の場所」は元のファイルと同じ場所、「フォルダ選択…」は指定したフォルダに保存します。"),
    "create_folder": N_("チェックすると、出力を指定名のフォルダにまとめます。外すと入力ファイルと同じ場所へ直接出力します。"),
    "accent": N_("画面の強調に使う色です。「Windows に合わせる」では、Windows の個人用設定で選んだ色を使います。"),
    "target_fps": N_("フレーム補間後の滑らかさです。通常は元動画の2倍を選びます。指定fpsが元動画以下なら処理できません。"),
    "processing_order": N_("アップスケールとフレーム補間を両方行うときの順番です。通常は「アプコン→補間」が速くおすすめ。高解像度でメモリ不足になるときだけ「補間→アプコン」にします。FILM (Style) は、メモリを抑えるため常に「補間 → アプコン」の順で処理します。"),
}


def format_convert_elapsed(seconds: float) -> str:
    """変換中の状態表示（1 時間以上は H:MM:SS、それ未満は M:SS）。"""
    total = max(0, int(seconds))
    if total >= 3600:
        body = f"{total // 3600}:{(total % 3600) // 60:02d}:{total % 60:02d}"
    else:
        body = f"{total // 60}:{total % 60:02d}"
    return t("変換中（経過 {body}）", body=body)


def format_convert_estimate(minutes: int) -> str:
    """未変換の行に出す目安（60 分以上は「約○時間」「約○時間○分」の形）。"""
    if minutes >= 60:
        hours, rest = divmod(int(minutes), 60)
        if rest:
            return t(
                "初回変換の目安: 約{hours}時間{rest}分",
                hours=hours,
                rest=rest,
            )
        return t("初回変換の目安: 約{hours}時間", hours=hours)
    return t("初回変換の目安: 約{minutes}分", minutes=int(minutes))


class _NpuConvertWorker(QObject):
    """NPU 変換を別スレッドで 1 本だけ行う（進捗文言は出さず経過時間で示す）。"""

    finished = Signal()
    failed = Signal(str)
    canceled = Signal()

    def __init__(self, model_key: str, cancel: threading.Event) -> None:
        super().__init__()
        self._model_key = model_key
        self._cancel = cancel

    def run(self) -> None:
        try:
            npu_prepare.convert(self._model_key, cancel=self._cancel)
        except Cancelled:
            self.canceled.emit()
        except Exception as exc:  # 失敗は文言にして状況行へ出す
            self.failed.emit(str(exc))
        else:
            self.finished.emit()


class ClearCheckBox(QCheckBox):
    """詳細設定用の見やすいチェックボックス。"""

    def __init__(self, label: str, parent=None) -> None:
        super().__init__(parent)
        self.setText(t(label))
        self.setObjectName("clearCheck")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(34)

    def sizeHint(self) -> QSize:  # noqa: N802
        hint = super().sizeHint()
        # 文字は自前で x=34 から描くので、幅も同じ前提で測る（長い英語の名前が切れないように）
        text_width = 34 + self.fontMetrics().horizontalAdvance(self.text()) + 8
        return QSize(max(hint.width(), text_width, 220), max(hint.height(), 34))

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        checked = self.isChecked()
        enabled = self.isEnabled()
        hovered = self.underMouse() and enabled

        rect = self.rect()
        box_size = 22
        box = QRectF(2, (rect.height() - box_size) / 2, box_size, box_size)

        p = theme.current()
        border = QColor(p.accent if checked else p.scrollbar)
        fill = QColor(p.accent if checked else (p.button_hover if hovered else p.input))
        if not enabled:
            border = QColor(p.border)
            fill = QColor(p.button_disabled)

        painter.setPen(QPen(border, 2))
        painter.setBrush(fill)
        painter.drawRoundedRect(box, 5, 5)

        if checked:
            pen = QPen(QColor(p.on_accent), 3)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)
            painter.drawLine(
                QPointF(box.left() + 5.5, box.top() + 11.5),
                QPointF(box.left() + 9.5, box.top() + 15.5),
            )
            painter.drawLine(
                QPointF(box.left() + 9.5, box.top() + 15.5),
                QPointF(box.left() + 16.5, box.top() + 7.0),
            )

        painter.setPen(QColor(p.text if enabled else p.text_mute))
        text_rect = QRectF(34, 0, max(0, rect.width() - 34), rect.height())
        painter.drawText(
            text_rect,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            self.text(),
        )


class HelpIcon(QLabel):
    """マウスオーバーで説明を出す小さなヘルプアイコン。"""

    def __init__(self, text: str, parent=None) -> None:
        super().__init__("?", parent)
        self.help_text = text
        self._popup: QLabel | None = None
        self.setObjectName("helpIcon")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setFixedSize(18, 18)
        self.setCursor(Qt.CursorShape.WhatsThisCursor)

    def enterEvent(self, event) -> None:  # noqa: N802
        self._show_popup()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802
        self._hide_popup()
        super().leaveEvent(event)

    def _show_popup(self) -> None:
        if self._popup is None:
            self._popup = QLabel(self.help_text)
            self._popup.setObjectName("helpPopup")
            self._popup.setWindowFlags(
                Qt.WindowType.ToolTip | Qt.WindowType.FramelessWindowHint
            )
            self._popup.setWordWrap(True)
            self._popup.setFixedWidth(340)
        self._popup.setText(self.help_text)
        self._popup.adjustSize()

        pos = self.mapToGlobal(self.rect().bottomLeft())
        pos.setX(pos.x() + 2)
        pos.setY(pos.y() + 8)

        screen = QApplication.screenAt(pos) or QApplication.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            if pos.x() + self._popup.width() > available.right():
                pos.setX(max(available.left(), available.right() - self._popup.width()))
            if pos.y() + self._popup.height() > available.bottom():
                above_icon = (
                    self.mapToGlobal(self.rect().topLeft()).y()
                    - self._popup.height()
                    - 8
                )
                pos.setY(max(available.top(), above_icon))

        self._popup.move(pos)
        self._popup.show()

    def _hide_popup(self) -> None:
        if self._popup is not None:
            self._popup.hide()


class SettingsDrawer(QFrame):
    """折りたたみ可能な詳細設定パネル。"""

    # 変換の開始・終了を MainWindow へ知らせる（開始/確認の無効化と連動用）。
    npu_converting_changed = Signal(bool)

    def __init__(self, parent=None, model_label=None) -> None:
        super().__init__(parent)
        self.setObjectName("card")
        # モデルの表示名は右列の表とそろえる（既定はキーのまま）。
        self._model_label = model_label or (lambda key: str(key))
        self.npu_rows: dict[str, dict[str, QWidget]] = {}
        self._npu_thread: QThread | None = None
        self._npu_worker: _NpuConvertWorker | None = None
        self._npu_cancel: threading.Event | None = None
        self._npu_convert_key: str | None = None
        self._npu_convert_start = 0.0
        self._npu_minutes: dict[str, int] = {}
        self._external_busy = False  # 本処理・試しの実行中
        self._npu_timer = QTimer(self)
        self._npu_timer.setInterval(1000)
        self._npu_timer.timeout.connect(self._on_npu_tick)
        self._build()

    def _label(self, text: str, help_text: str | None = None) -> QWidget:
        wrap = QWidget()
        wrap.setObjectName("fieldLabelWrap")
        row = QHBoxLayout(wrap)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        lab = QLabel(t(text))
        lab.setObjectName("fieldLabel")
        row.addWidget(lab)
        if help_text:
            row.addWidget(HelpIcon(t(help_text)))
        row.addStretch(1)
        return wrap

    def _check_row(self, checkbox: ClearCheckBox, help_text: str) -> QWidget:
        wrap = QWidget()
        wrap.setObjectName("checkRow")
        row = QHBoxLayout(wrap)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        row.addWidget(checkbox)
        row.addWidget(HelpIcon(t(help_text)))
        row.addStretch(1)
        return wrap

    def _combo_with_data(self, options: list[tuple[str, object]]) -> QComboBox:
        combo = QComboBox()
        for label, value in options:
            combo.addItem(t(label), value)
        return combo

    def _set_combo_value(self, combo: QComboBox, value: object) -> None:
        idx = combo.findData(value)
        combo.setCurrentIndex(idx if idx >= 0 else 0)

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 14, 16, 14)
        root.setSpacing(12)

        title = QLabel(t("詳細設定"))
        title.setObjectName("sectionTitle")
        root.addWidget(title)

        grid = QGridLayout()
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(10)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)

        # --- AI実行先 / 画像の保存形式 ---
        self.backend = self._combo_with_data(available_backend_options())
        grid.addWidget(self._label(N_("AI実行先"), _HELP["backend"]), 0, 0)
        grid.addWidget(self.backend, 0, 1)

        # --- 画像の保存形式 ---
        self.image_format = QComboBox()
        self.image_format.addItems(_IMAGE_FORMATS)
        grid.addWidget(self._label(N_("画像の保存形式"), _HELP["image_format"]), 0, 2)
        grid.addWidget(self.image_format, 0, 3)

        # --- 動画の保存形式 ---
        self.video_format = QComboBox()
        self.video_format.addItems(_VIDEO_FORMATS)
        grid.addWidget(self._label(N_("動画の保存形式"), _HELP["video_format"]), 1, 0)
        grid.addWidget(self.video_format, 1, 1)

        # --- 動画の画質 ---
        self.video_quality = self._combo_with_data(_VIDEO_QUALITY_OPTIONS)
        grid.addWidget(self._label(N_("動画の画質 (CRF/QP)"), _HELP["video_quality"]), 1, 2)
        grid.addWidget(self.video_quality, 1, 3)

        # --- 分割処理 ---
        self.tile_size = self._combo_with_data(_TILE_OPTIONS)
        grid.addWidget(self._label(N_("分割処理 (タイル)"), _HELP["tile_size"]), 2, 0)
        grid.addWidget(self.tile_size, 2, 1)

        # --- 使うGPU ---
        self.gpu_id = self._combo_with_data(_GPU_OPTIONS)
        grid.addWidget(self._label(N_("使うGPU"), _HELP["gpu_id"]), 2, 2)
        grid.addWidget(self.gpu_id, 2, 3)

        # --- 出力フォルダ名 ---
        self.subfolder_name = QLineEdit()
        self.subfolder_name.setPlaceholderText("upscaled")
        grid.addWidget(self._label(N_("出力フォルダ名"), _HELP["subfolder"]), 3, 0)
        grid.addWidget(self.subfolder_name, 3, 1)

        # --- フレーム補間後のfps（細かく選ぶときだけ出す） ---
        self.target_fps = self._combo_with_data(_TARGET_FPS_OPTIONS)
        self.target_fps_label = self._label(N_("補間後のfps"), _HELP["target_fps"])
        grid.addWidget(self.target_fps_label, 3, 2)
        grid.addWidget(self.target_fps, 3, 3)

        # --- 処理の順番（アプコン×補間 併用時） ---
        self.processing_order = self._combo_with_data(_PROCESSING_ORDER_OPTIONS)
        grid.addWidget(self._label(N_("処理の順番"), _HELP["processing_order"]), 4, 0)
        grid.addWidget(self.processing_order, 4, 1)

        # --- 出力先（全体の設定。選んだあとの処理は MainWindow が持つ） ---
        self.output_combo = QComboBox()
        self.output_combo.addItems([t("元の場所"), t("フォルダ選択…")])
        grid.addWidget(self._label(N_("出力先"), _HELP["output"]), 4, 2)
        grid.addWidget(self.output_combo, 4, 3)

        # --- 表示言語（既存の項目の並びの最後。切り替えは次回起動から） ---
        self.language_combo = self._combo_with_data(_LANGUAGE_OPTIONS)
        grid.addWidget(self._label(N_("表示言語"), None), 5, 0)
        grid.addWidget(self.language_combo, 5, 1)
        self.language_notice = QLabel("")
        self.language_notice.setObjectName("hint")
        grid.addWidget(self.language_notice, 6, 0, 1, 4)
        self.language_combo.currentIndexChanged.connect(
            self._on_language_changed
        )
        self._load_language_combo()

        # --- アクセントカラー（表示言語の近く。選ぶとすぐ画面全体に反映） ---
        self.accent_combo = self._combo_with_data(_ACCENT_OPTIONS)
        for i in range(self.accent_combo.count()):
            value = self.accent_combo.itemData(i)
            if value != user_settings.ACCENT_AUTO:
                pix = QPixmap(14, 14)
                pix.fill(QColor(str(value)))
                self.accent_combo.setItemIcon(i, QIcon(pix))
        grid.addWidget(self._label(N_("アクセントカラー"), _HELP["accent"]), 5, 2)
        grid.addWidget(self.accent_combo, 5, 3)
        self.accent_combo.currentIndexChanged.connect(
            self._on_accent_changed
        )
        self._load_accent_combo()

        root.addLayout(grid)

        # --- トグル群 ---
        toggles = QGridLayout()
        toggles.setHorizontalSpacing(12)
        toggles.setVerticalSpacing(8)
        self.hw_encode = ClearCheckBox(N_("動画の保存を速くする"))
        self.keep_audio = ClearCheckBox(N_("動画の音声を残す"))
        self.tta_mode = ClearCheckBox(N_("高品質モード (TTA)"))
        self.create_subfolder = ClearCheckBox(N_("出力フォルダを作る"))
        toggles.addWidget(self._check_row(self.hw_encode, _HELP["hw_encode"]), 0, 0)
        toggles.addWidget(self._check_row(self.keep_audio, _HELP["keep_audio"]), 0, 1)
        toggles.addWidget(self._check_row(self.tta_mode, _HELP["tta"]), 1, 0)
        toggles.addWidget(self._check_row(self.create_subfolder, _HELP["create_folder"]), 1, 1)
        # 補間の倍率を細かく選ぶ（切っている間は元動画の 2 倍。保存しない）。
        self.detail_fps_check = ClearCheckBox(N_("補間の倍率を細かく選ぶ"))
        toggles.addWidget(
            self._check_row(
                self.detail_fps_check,
                N_("入れると「補間後のfps」を選べるようになります。"
                   "切っている間は、元動画の 2 倍になります。"),
            ),
            2, 0, 1, 2,
        )
        self.detail_fps_check.toggled.connect(self._on_detail_fps_toggled)
        toggle_wrap = QWidget()
        toggle_wrap.setObjectName("toggleWrap")
        toggle_wrap.setLayout(toggles)
        root.addWidget(toggle_wrap)

        # --- 追加キット（NPU の準備の上に置く） ---
        self.kit_section = QWidget()
        self.kit_section.setObjectName("toggleWrap")
        kit_section = QVBoxLayout(self.kit_section)
        kit_section.setContentsMargins(0, 8, 0, 0)
        kit_section.setSpacing(8)
        kit_line = QFrame()
        kit_line.setObjectName("separator")
        kit_line.setFixedHeight(1)
        kit_section.addWidget(kit_line)
        kit_heading = QLabel(t("追加キット"))
        kit_heading.setObjectName("sectionTitle")
        kit_section.addWidget(kit_heading)
        kit_desc = QLabel(
            t(
                "ダウンロードした zip を、togu-scaler.exe のあるフォルダに"
                "展開してください。次回の起動から使えます。"
            )
        )
        kit_desc.setObjectName("hint")
        kit_desc.setWordWrap(True)
        kit_section.addWidget(kit_desc)
        self.kit_grid = QGridLayout()
        self.kit_grid.setHorizontalSpacing(12)
        self.kit_grid.setVerticalSpacing(6)
        self.kit_grid.setColumnStretch(4, 1)  # ボタンは伸ばさず、余りは右へ
        kit_section.addLayout(self.kit_grid)
        self.kit_rows: dict[str, dict[str, QWidget]] = {}
        root.addWidget(self.kit_section)
        self._rebuild_kit_rows()

        # --- NPU の準備（キットが無い PC では欄ごと出さない） ---
        self.npu_section = QWidget()
        self.npu_section.setObjectName("toggleWrap")
        section = QVBoxLayout(self.npu_section)
        section.setContentsMargins(0, 8, 0, 0)
        section.setSpacing(8)
        line = QFrame()
        line.setObjectName("separator")
        line.setFixedHeight(1)
        section.addWidget(line)
        heading = QLabel(t("NPU の準備"))
        heading.setObjectName("sectionTitle")
        section.addWidget(heading)
        desc = QLabel(
            t(
                "NPU で使うモデルは、最初に一度だけ変換が必要です。"
                "使うモデルだけ変換してください。"
                "変換中も PC は使えますが、SwinIR-M と AdcSR はメモリを多く使います。"
            )
        )
        desc.setObjectName("hint")
        desc.setWordWrap(True)
        section.addWidget(desc)
        self.npu_grid = QGridLayout()
        self.npu_grid.setHorizontalSpacing(12)
        self.npu_grid.setVerticalSpacing(6)
        self.npu_grid.setColumnStretch(2, 1)
        section.addLayout(self.npu_grid)
        self.npu_status = QLabel("")
        self.npu_status.setObjectName("hint")
        self.npu_status.setWordWrap(True)
        section.addWidget(self.npu_status)
        root.addWidget(self.npu_section)
        self._rebuild_npu_rows()
        self.npu_section.setVisible(npu_prepare.npu_available())

        root.addStretch(1)  # 広い領域に置かれても項目を上に詰める

        # 出力フォルダ名はチェック時のみ有効
        self.create_subfolder.toggled.connect(self.subfolder_name.setEnabled)

        # 選択欄は中身の長さに引っ張られず縮められるようにする
        # （英語の表示では項目が長く、欄が領域からはみ出すため）。
        for combo in self.findChildren(QComboBox):
            combo.setMinimumContentsLength(8)
            combo.setSizeAdjustPolicy(
                QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
            )

        self.load_defaults()
        # 補間の倍率は既定で切（保存しない）。「補間後のfps」の欄は出さない。
        self.detail_fps_check.setChecked(False)
        self._sync_detail_fps_row()

    # --- 表示言語（切り替えは次回の起動から有効） ---
    def _load_language_combo(self) -> None:
        """保存された言語を欄に反映する（案内の1行は出さない）。"""
        saved = user_settings.load_language()
        index = self.language_combo.findData(saved)
        previous = self.language_combo.blockSignals(True)
        try:
            self.language_combo.setCurrentIndex(index if index >= 0 else 0)
        finally:
            self.language_combo.blockSignals(previous)
        self.language_notice.setText("")

    def _on_language_changed(self) -> None:
        """選択を保存し、次回起動から有効になる案内を出す。"""
        user_settings.save_language(str(self.language_combo.currentData()))
        self.language_notice.setText(t("次回の起動から切り替わります。"))

    def _load_accent_combo(self) -> None:
        """保存されたアクセントカラーを欄に反映する。"""
        saved = user_settings.load_accent()
        index = self.accent_combo.findData(saved)
        previous = self.accent_combo.blockSignals(True)
        try:
            self.accent_combo.setCurrentIndex(index if index >= 0 else 0)
        finally:
            self.accent_combo.blockSignals(previous)

    def _on_accent_changed(self) -> None:
        """選択を保存し、すぐ画面全体に反映する（再起動は求めない）。"""
        user_settings.save_accent(str(self.accent_combo.currentData()))
        app = QApplication.instance()
        if app is not None:
            theme.refresh(app)

    # --- 既定値の読込／設定への反映 ---
    def load_defaults(self, settings: UpscaleSettings | None = None) -> None:
        s = settings or UpscaleSettings()
        self._set_combo_value(
            self.backend,
            s.backend.value if settings is not None else "auto",
        )
        self.image_format.setCurrentText(s.image_format)
        self.video_format.setCurrentText(s.video_format)
        self._set_combo_value(self.video_quality, s.video_quality)
        self._set_combo_value(self.tile_size, s.tile_size)
        self._set_combo_value(self.gpu_id, s.gpu_id)
        self._set_combo_value(
            self.target_fps, _fps_key_for(s.target_fps, s.interpolation_factor)
        )
        self._set_combo_value(self.processing_order, s.processing_order.value)
        self.subfolder_name.setText(s.subfolder_name)
        self.hw_encode.setChecked(s.hw_encode)
        self.keep_audio.setChecked(s.keep_audio)
        self.tta_mode.setChecked(s.tta_mode)
        self.create_subfolder.setChecked(s.create_subfolder)
        self.subfolder_name.setEnabled(s.create_subfolder)

    def apply_to(self, s: UpscaleSettings) -> None:
        """ドロワーの値を UpscaleSettings に書き込む。"""
        backend_value = self.backend.currentData()
        s.backend = (
            UpscaleBackend.WINML_GPU
            if backend_value == "auto"
            else UpscaleBackend(backend_value)
        )
        s.image_format = self.image_format.currentText()
        s.video_format = self.video_format.currentText()
        s.video_quality = int(self.video_quality.currentData())
        s.tile_size = int(self.tile_size.currentData())
        s.gpu_id = int(self.gpu_id.currentData())
        if self.detail_fps_check.isChecked():
            target, factor = _FPS_CHOICES.get(self.target_fps.currentData(), (None, 2))
            s.target_fps = float(target) if target is not None else None
            s.interpolation_factor = int(factor)
        else:
            # 切っている間は常に元動画の 2 倍。
            s.target_fps = None
            s.interpolation_factor = 2
        s.processing_order = ProcessingOrder(self.processing_order.currentData())
        name = self.subfolder_name.text().strip() or "upscaled"
        s.subfolder_name = name
        s.hw_encode = self.hw_encode.isChecked()
        s.keep_audio = self.keep_audio.isChecked()
        s.tta_mode = self.tta_mode.isChecked()
        s.create_subfolder = self.create_subfolder.isChecked()

    def set_interpolation_enabled(self, enabled: bool) -> None:
        self.target_fps.setEnabled(enabled)
        # 順番は補間とアプコンの併用時のみ意味を持つ
        self.processing_order.setEnabled(enabled)

    def _on_detail_fps_toggled(self, checked: bool) -> None:
        """細かく選ぶを切ったら元動画の 2 倍に戻す。"""
        if not checked:
            self._set_combo_value(self.target_fps, "x2")
        self._sync_detail_fps_row()

    def _sync_detail_fps_row(self) -> None:
        """「補間後のfps」の欄は細かく選ぶときだけ出す。"""
        show = self.detail_fps_check.isChecked()
        self.target_fps_label.setVisible(show)
        self.target_fps.setVisible(show)

    def set_film_selected(self, film: bool) -> None:
        """FILM 選択中は 60/120fps を選べなくする（倍数だけに対応するため）。"""
        for i in range(self.target_fps.count()):
            if self.target_fps.itemData(i) not in _FPS_FIXED_KEYS:
                continue
            item = self.target_fps.model().item(i)
            if item is None:
                continue
            item.setEnabled(not film)
            item.setToolTip(
                t("FILM (Style) は元動画の倍数だけに対応します") if film else ""
            )
        if film and self.target_fps.currentData() in _FPS_FIXED_KEYS:
            self._set_combo_value(self.target_fps, "x2")

    # ------------------------------------------------------- 追加キット
    def _rebuild_kit_rows(self) -> None:
        """追加キットの 4 行を作り直す（判定はファイルの有無だけ）。"""
        while self.kit_grid.count():
            item = self.kit_grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.kit_rows.clear()
        for row, kit in enumerate(addon_kits.addon_kits()):
            name_label = QLabel(t(kit.name_key))
            name_label.setObjectName("fieldLabel")
            size_label = QLabel(kit.size)
            size_label.setObjectName("hint")
            status_label = QLabel(
                t("導入済み") if kit.installed else t("未導入")
            )
            status_label.setObjectName("hint")
            button = QPushButton(t("ダウンロード"))
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.clicked.connect(
                lambda _=False, filename=kit.filename: self._on_kit_button(filename)
            )
            self.kit_grid.addWidget(name_label, row, 0)
            self.kit_grid.addWidget(size_label, row, 1)
            self.kit_grid.addWidget(status_label, row, 2)
            self.kit_grid.addWidget(button, row, 3)
            self.kit_rows[kit.key] = {
                "name": name_label,
                "size": size_label,
                "status": status_label,
                "button": button,
            }

    def refresh_kit_rows(self) -> None:
        """詳細設定を開いたときに導入状態を判定し直す。"""
        self._rebuild_kit_rows()

    @staticmethod
    def _on_kit_button(filename: str) -> None:
        """既定のブラウザで配布ページを開く（入手と展開は利用者が行う）。"""
        QDesktopServices.openUrl(QUrl(addon_kits.kit_download_url(filename)))

    # ------------------------------------------------------- NPU の準備
    def _rebuild_npu_rows(self) -> None:
        """モデルごとの行を作り直す（ファイルが揃わないものは出さない）。"""
        while self.npu_grid.count():
            item = self.npu_grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.npu_rows.clear()
        self._npu_minutes.clear()
        for row, model in enumerate(npu_prepare.npu_models()):
            try:
                name = self._model_label(model.label_key)
            except Exception:
                name = model.label_key
            name_label = QLabel(str(name))
            name_label.setObjectName("fieldLabel")
            status_label = QLabel(
                t("変換済み") if model.converted else t("未変換")
            )
            status_label.setObjectName("hint")
            estimate_label = QLabel(
                "" if model.converted else format_convert_estimate(model.minutes)
            )
            estimate_label.setObjectName("hint")
            button = QPushButton(t("NPU 用に変換"))
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.clicked.connect(
                lambda _=False, key=model.key: self._on_npu_button(key)
            )
            if model.converted:
                button.setVisible(False)
            self.npu_grid.addWidget(name_label, row, 0)
            self.npu_grid.addWidget(status_label, row, 1)
            self.npu_grid.addWidget(estimate_label, row, 2)
            self.npu_grid.addWidget(button, row, 3)
            self.npu_rows[model.key] = {
                "status": status_label,
                "estimate": estimate_label,
                "button": button,
            }
            self._npu_minutes[model.key] = model.minutes

    def refresh_npu_rows(self) -> None:
        """変換画面を開いたとき等に行を載せ替える（変換中は触らない）。"""
        if self.is_converting():
            return
        self._rebuild_npu_rows()

    def set_external_busy(self, busy: bool) -> None:
        """本処理・試しの実行状態を覚える（押されたら案内を出すため）。"""
        self._external_busy = bool(busy)

    def is_converting(self) -> bool:
        """変換が実行中か（テスト・終了処理用）。"""
        return self._npu_thread is not None

    def cancel_conversion(self) -> None:
        """実行中の変換に中止を要求する（子プロセスを残さない）。"""
        if self._npu_cancel is not None:
            self._npu_cancel.set()

    def _on_npu_button(self, model_key: str) -> None:
        if self.is_converting():
            if model_key == self._npu_convert_key:
                self.cancel_conversion()
            return
        if self._external_busy:
            self.npu_status.setText(t("処理中は変換できません"))
            return
        if model_key not in self.npu_rows:
            return
        self.npu_status.setText("")
        self._start_conversion(model_key)

    def _start_conversion(self, model_key: str) -> None:
        cancel = threading.Event()
        worker = _NpuConvertWorker(model_key, cancel)
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_convert_finished)
        worker.failed.connect(self._on_convert_failed)
        worker.canceled.connect(self._on_convert_canceled)
        self._npu_thread = thread
        self._npu_worker = worker
        self._npu_cancel = cancel
        self._npu_convert_key = model_key
        self._npu_convert_start = time.monotonic()
        for key, widgets in self.npu_rows.items():
            button = widgets["button"]
            if key == model_key:
                widgets["status"].setText(
                    format_convert_elapsed(0.0))
                widgets["estimate"].setText("")
                button.setText(t("中止"))
                button.setVisible(True)
                button.setEnabled(True)
            else:
                button.setEnabled(False)
        self._npu_timer.start()
        try:
            self.npu_converting_changed.emit(True)
        except RuntimeError:
            pass
        thread.start()

    def _on_npu_tick(self) -> None:
        """経過時間の表示を 1 秒ごとに更新する。"""
        if not self.is_converting() or self._npu_convert_key is None:
            return
        widgets = self.npu_rows.get(self._npu_convert_key)
        if widgets is None:
            return
        widgets["status"].setText(format_convert_elapsed(
            time.monotonic() - self._npu_convert_start))

    def _stop_convert_thread(self) -> None:
        self._npu_timer.stop()
        thread, worker = self._npu_thread, self._npu_worker
        self._npu_thread = None
        self._npu_worker = None
        self._npu_cancel = None
        self._npu_convert_key = None
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

    def _finish_conversion(self) -> None:
        """共通後始末（行の再有効化と終了通知）。"""
        self._stop_convert_thread()
        for widgets in self.npu_rows.values():
            widgets["button"].setEnabled(True)
        try:
            self.npu_converting_changed.emit(False)
        except RuntimeError:
            pass

    def _on_convert_finished(self) -> None:
        key = self._npu_convert_key
        self._finish_conversion()
        if key is not None and key in self.npu_rows:
            widgets = self.npu_rows[key]
            widgets["status"].setText(t("変換済み"))
            widgets["estimate"].setText("")
            widgets["button"].setVisible(False)
        self.npu_status.setText(t("変換が終わりました。"))

    def _on_convert_failed(self, message: str) -> None:
        key = self._npu_convert_key
        minutes = self._npu_minutes.get(key or "", 0) if key else 0
        self._finish_conversion()
        if key is not None and key in self.npu_rows:
            widgets = self.npu_rows[key]
            widgets["status"].setText(t("失敗"))
            widgets["estimate"].setText(
                format_convert_estimate(minutes) if minutes else "")
            button = widgets["button"]
            button.setText(t("NPU 用に変換"))
            button.setVisible(True)
        self.npu_status.setText(
            t(
                "変換できませんでした: {message}",
                message=sanitize_error_message(message, self._npu_redactions(key)),
            )
        )

    def _on_convert_canceled(self) -> None:
        key = self._npu_convert_key
        minutes = self._npu_minutes.get(key or "", 0) if key else 0
        self._finish_conversion()
        if key is not None and key in self.npu_rows:
            widgets = self.npu_rows[key]
            widgets["status"].setText(t("未変換"))
            widgets["estimate"].setText(
                format_convert_estimate(minutes) if minutes else "")
            button = widgets["button"]
            button.setText(t("NPU 用に変換"))
            button.setVisible(True)
        self.npu_status.setText(t("中止しました"))

    @staticmethod
    def _npu_redactions(model_key: str | None) -> list[str]:
        """失敗文から取り除く内部識別子（試しの整形と同じ流儀）。"""
        return [
            str(model_key or ""),
            UpscaleBackend.NPU_NATIVE.value,
            UpscaleBackend.WINML_GPU.value,
            UpscaleBackend.SWINIR_CUDA.value,
            UpscaleBackend.VULKAN.value,
            "VAIML",
            "vaiml",
            str(helper_backend.npu_cache_dir()),
            str(helper_backend.models_dir()),
        ]

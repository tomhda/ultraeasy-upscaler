"""メインウィンドウと GUI エントリポイント。

レイアウト（ダーク・単一画面・3 列）:
  ヘッダ
  → 左: メディア一覧（ドロップ先を兼ねる） / 中央: プレビュー / 右: 設定と開始。
  メディアが 1 件も無いときは、左と中央を合わせた全面をドロップ先にする。
  歯車を押すと、左と中央の領域が詳細設定に切り替わる。

スレッド方針: GPU 競合回避のため、保留ジョブは QThread 上の QueueWorker が
**逐次** 処理する。ウィジェット更新は GUI スレッドのスロットのみで行う。
"""
from __future__ import annotations

import subprocess
import threading
from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import QSize, Qt, QThread
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import QStyle, QStyleOptionComboBox, QStylePainter
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from app.core import binaries, helper_backend, npu_prepare
from app.core.jobs import Job, JobKind, JobStatus
from app.core.settings import (
    DEFAULT_MODEL,
    DEFAULT_HELPER_MODEL,
    HELPER_MODEL_AMD_RRDB,
    HELPER_MODEL_ANIME,
    HELPER_MODEL_SPAN,
    HELPER_MODEL_SWINIR,
    HELPER_MODEL_ADCSR,
    OutputLocation,
    UpscaleBackend,
    UpscaleSettings,
    helper_model_family,
)

from app.i18n import N_, t

from .compare_view import StatusLabel, TrialPanel
from .drop_zone import DropZone
from .icons import Icon, apply_icon_font, make_icon
from .queue_view import QueueView
from .settings_drawer import _BACKEND_OPTIONS, SettingsDrawer
from . import theme
from .theme import apply_theme
from .worker import QueueWorker

# 画像用・動画用の区別（フォルダは画像用）。右列の切り替えは無いが、
# 内部で種類ごとの一括設定・個別設定を分けるために使う。
_IMAGE_TAB = "image"
_VIDEO_TAB = "video"
# ヘッダー3欄の幅（同じ幅。狭いときは縮む）
_HEADER_FIELD_WIDTH = 280
_HEADER_FIELD_MIN_WIDTH = 120

# 倍率トグルに出す候補（モデルがサポートする倍率のみ有効化）
_SCALE_CHOICES = (2, 4)
# 左右の列の幅（中央のプレビューが残りを使う）
_MEDIA_COL_WIDTH = 330
_SETTINGS_COL_WIDTH = 330
_HELPER_BACKENDS = {
    UpscaleBackend.WINML_GPU,
    UpscaleBackend.NPU_NATIVE,
    UpscaleBackend.SWINIR_CUDA,
}
_MODEL_LABELS = {
    "realesrgan-x4plus": N_("Real-ESRGAN"),
    "realesrgan-x4plus-anime": N_("Real-ESRGAN Anime"),
    "realesr-animevideov3": N_("Anime Video v3"),
    HELPER_MODEL_ANIME: N_("Anime Video v3"),
    "realesr-general-x4v3": N_("General Video v3（ノイズ除去強）"),
    "realesr-general-wdn-x4v3": N_("General Video v3（ノイズ除去弱）"),
    HELPER_MODEL_SPAN: N_("4xNomosUni SPAN"),
    HELPER_MODEL_AMD_RRDB: N_("Real-ESRGAN（AMD縮小版）"),
    HELPER_MODEL_SWINIR: N_("SwinIR-M"),
    HELPER_MODEL_ADCSR: N_("AdcSR"),
}
_HELPER_MODEL_OPTIONS = [
    (N_("なし（拡大しない）"), None),
    (_MODEL_LABELS[HELPER_MODEL_ANIME], HELPER_MODEL_ANIME),
    (_MODEL_LABELS[HELPER_MODEL_SPAN], HELPER_MODEL_SPAN),
    (_MODEL_LABELS[HELPER_MODEL_AMD_RRDB], HELPER_MODEL_AMD_RRDB),
    (_MODEL_LABELS[HELPER_MODEL_SWINIR], HELPER_MODEL_SWINIR),
    (_MODEL_LABELS[HELPER_MODEL_ADCSR], HELPER_MODEL_ADCSR),
]
_HELPER_MODEL_VALUES = {
    value for _label, value in _HELPER_MODEL_OPTIONS if value is not None
}
_SWINIR_CUDA_MODEL_OPTIONS = [
    (N_("なし（拡大しない）"), None),
    (N_("SwinIR-M（real-world x4）"), HELPER_MODEL_SWINIR),
]

# モデルの説明行（1 行）。速さ・実行先の注意は付けない。
_MODEL_HINT = {
    HELPER_MODEL_ANIME: N_("アニメ向け・速い"),
    "realesr-animevideov3": N_("アニメ向け・速い"),
    HELPER_MODEL_SPAN: N_("実写向け・速い"),
    HELPER_MODEL_AMD_RRDB: N_("実写向け・くっきり・やや遅い"),
    HELPER_MODEL_SWINIR: N_("実写の静止画向け・高精細・遅い"),
    HELPER_MODEL_ADCSR: N_("実写の静止画向け・最高画質・とても遅い"),
    "realesrgan-x4plus": N_("実写向け・高画質・遅い"),
    "realesrgan-x4plus-anime": N_("アニメ向け・高画質・遅い"),
    "realesr-general-x4v3": N_("実写・アニメ兼用・ノイズ除去強め"),
    "realesr-general-wdn-x4v3": N_("実写向け・ノイズ除去弱め"),
}

# (backend, model) → (速度, 画質, アニメ適性, 実写適性, 推奨タグ or None)
# 速度・画質・適性の印。英語の表示では言葉に置き換える（◎○△✕ は日本語圏の記号のため）。
_RATING_MARKS = (N_("◎◎"), N_("◎"), N_("○"), N_("△"), N_("✕"))

_MODEL_INFO: dict[tuple[UpscaleBackend, str],
                  tuple[str, str, str, str, str | None]] = {
    (UpscaleBackend.VULKAN, "realesr-animevideov3"):
        ("◎", "◎", "◎", "△", N_("アニメ")),
    (UpscaleBackend.VULKAN, "realesr-general-x4v3"):
        ("◎", "○", "○", "○", None),
    (UpscaleBackend.VULKAN, "realesr-general-wdn-x4v3"):
        ("◎", "○", "○", "◎", N_("実写")),
    (UpscaleBackend.VULKAN, "realesrgan-x4plus"):
        ("✕", "◎", "○", "◎", None),
    (UpscaleBackend.VULKAN, "realesrgan-x4plus-anime"):
        ("✕", "◎", "◎", "○", None),
    (UpscaleBackend.NPU, "realesrgan-x4plus"):
        ("◎", "◎", "○", "◎", N_("実写")),
    (UpscaleBackend.NPU, "realesrgan-x4plus-anime"):
        ("○", "◎", "◎", "○", None),
    (UpscaleBackend.NPU, "realesr-animevideov3"):
        ("◎", "◎", "◎", "△", N_("アニメ")),
    (UpscaleBackend.WINML_GPU, HELPER_MODEL_ANIME):
        ("◎", "◎", "◎", "△", N_("アニメ")),
    (UpscaleBackend.WINML_GPU, HELPER_MODEL_SPAN):
        ("◎", "◎", "○", "◎", N_("実写")),
    (UpscaleBackend.WINML_GPU, HELPER_MODEL_AMD_RRDB):
        ("△", "◎", "○", "◎", None),
    (UpscaleBackend.WINML_GPU, HELPER_MODEL_SWINIR):
        ("✕", "◎", "○", "◎", N_("静止画")),
    (UpscaleBackend.WINML_GPU, HELPER_MODEL_ADCSR):
        ("✕", "◎◎", "○", "◎", N_("実写")),
    (UpscaleBackend.NPU_NATIVE, HELPER_MODEL_ADCSR):
        ("✕", "◎◎", "○", "◎", N_("実写")),
    (UpscaleBackend.NPU_NATIVE, HELPER_MODEL_ANIME):
        ("○", "◎", "◎", "△", N_("アニメ")),
    (UpscaleBackend.NPU_NATIVE, HELPER_MODEL_SPAN):
        ("◎", "◎", "○", "◎", N_("実写")),
    (UpscaleBackend.NPU_NATIVE, HELPER_MODEL_AMD_RRDB):
        ("△", "◎", "○", "◎", None),
    (UpscaleBackend.NPU_NATIVE, HELPER_MODEL_SWINIR):
        ("✕", "◎", "○", "◎", N_("静止画")),
    (UpscaleBackend.SWINIR_CUDA, HELPER_MODEL_SWINIR):
        (N_("極遅"), "◎◎", "△", "◎", N_("実写・再開可")),
}


def _combo_closed_text(text: str) -> str:
    """閉じた表示はモデル名だけ（一覧の印｜…・（未変換）は出さない）。"""
    return str(text).split("｜")[0].removesuffix(t("（未変換）"))


class ModelCombo(QComboBox):
    """閉じた状態ではモデル名だけを見せるコンボ（特性の印は開いた一覧に出す）。"""

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QStylePainter(self)
        opt = QStyleOptionComboBox()
        self.initStyleOption(opt)
        # 収まらない名前は途中で切らず、末尾を「…」にする
        field = self.style().subControlRect(
            QStyle.ComplexControl.CC_ComboBox, opt,
            QStyle.SubControl.SC_ComboBoxEditField, self,
        )
        opt.currentText = self.fontMetrics().elidedText(
            _combo_closed_text(opt.currentText),
            Qt.TextElideMode.ElideRight, max(0, field.width()),
        )
        painter.drawComplexControl(QStyle.ComplexControl.CC_ComboBox, opt)
        painter.drawControl(QStyle.ControlElement.CE_ComboBoxLabel, opt)


class MainWindow(QWidget):
    """ultraeasy-upscaler のメイン画面。"""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ultraeasy-upscaler")
        # 小さい画面（1366x768 や 1080p@125% 等）でもはみ出さないよう、
        # 起動サイズは利用可能領域にクランプする
        screen = QApplication.primaryScreen()
        if screen is not None:
            avail = screen.availableGeometry()
            self.resize(
                min(1360, avail.width() - 80),
                min(820, avail.height() - 80),
            )
        else:
            self.resize(1360, 780)

        # ジョブ管理
        self._jobs: dict[int, Job] = {}
        self._order: list[int] = []                       # 追加順
        self._cancel_events: dict[int, threading.Event] = {}
        self._pause = threading.Event()
        self._thread: QThread | None = None
        self._worker: QueueWorker | None = None
        self._running = False
        self._closing = False
        self._current_job_id: int | None = None
        self._npu_converting = False

        # 種類ごとの一括設定（モデル・倍率。動画は補間モデルも持つ）。
        # 個別設定は _overrides[job_id] にだけ置き、ここには一括だけを持つ。
        self._image_model: str | None = DEFAULT_HELPER_MODEL
        self._image_scale = 4
        self._video_model: str | None = DEFAULT_HELPER_MODEL
        self._video_scale = 4
        self._video_interpolation: str | None = None
        self._overrides: dict[int, dict[str, object]] = {}
        self._syncing_settings = False  # 載せ替え中は保存しない
        self._build()
        self._sync_settings_widgets()
        self._refresh_interpolation_enabled()

    # ------------------------------------------------------------------ UI
    def _build(self) -> None:
        self.setAcceptDrops(True)  # ウィンドウのどこに落としても追加できる
        outer = QVBoxLayout(self)
        outer.setContentsMargins(20, 16, 20, 16)
        outer.setSpacing(12)

        # AI実行先は詳細設定ドロワーにあるコンボをそのまま使う
        # モデルの表示名は右列の表とそろえる（NPU 変換画面へ渡す）。
        self.drawer = SettingsDrawer(
            model_label=lambda key: t(_MODEL_LABELS.get(key, key))
        )
        self.drawer.npu_converting_changed.connect(
            self._on_npu_converting_changed)
        self._drawer_open = False
        self.backend_combo = self.drawer.backend

        outer.addWidget(self._build_header())

        body = QHBoxLayout()
        body.setSpacing(12)
        outer.addLayout(body, 1)

        # 左＋中央。メディアが無い間は全面ドロップ枠、入ったら一覧＋プレビュー。
        self.drop_zone = DropZone()
        self.drop_zone.pathsDropped.connect(self._on_paths_dropped)
        self.drop_zone.browseRequested.connect(self._pick_files)

        workspace = QWidget()
        work_row = QHBoxLayout(workspace)
        work_row.setContentsMargins(0, 0, 0, 0)
        work_row.setSpacing(12)
        work_row.addWidget(self._build_media_panel())
        self.preview = TrialPanel(
            build_settings=self.build_settings,
            model_label=lambda key: t(_MODEL_LABELS.get(key, key)),
        )
        self.preview.trial_running_changed.connect(self._on_trial_running_changed)
        work_row.addWidget(self.preview, 1)

        # 詳細設定は歯車で左＋中央の領域に開く（右列の設定と開始は残す）
        drawer_scroll = QScrollArea()
        drawer_scroll.setWidgetResizable(True)
        drawer_scroll.setFrameShape(QFrame.Shape.NoFrame)
        drawer_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        drawer_scroll.setWidget(self.drawer)

        self.stack = QStackedWidget()
        self.stack.addWidget(self.drop_zone)
        self.stack.addWidget(workspace)
        self.stack.addWidget(drawer_scroll)
        body.addWidget(self.stack, 1)

        body.addWidget(self._build_settings_panel())

        self.queue.selectionChanged.connect(self._on_selection_changed)
        self.queue.retryRequested.connect(self._on_retry_requested)
        self.backend_combo.currentIndexChanged.connect(self._on_backend_changed)
        self.image_model_combo.currentIndexChanged.connect(
            self._on_header_image_model_changed
        )
        self.video_model_combo.currentIndexChanged.connect(
            self._on_header_video_model_changed
        )
        self.global_interpolation_combo.currentIndexChanged.connect(
            self._on_header_interpolation_changed
        )
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        self.interpolation_combo.currentIndexChanged.connect(
            self._on_right_interpolation_changed
        )
        self._refresh_model_options()

        self._apply_icons()
        theme.notifier.changed.connect(self._apply_icons)

    def _apply_icons(self) -> None:
        """アイコンを現在の配色で描き直す（QSS では色を変えられないため）。"""
        p = theme.current()
        self.settings_btn.setIcon(make_icon(Icon.SETTINGS, 24, p.text_soft))
        self.pause_btn.setIcon(make_icon(Icon.PAUSE, 20, p.text_soft))
        self.start_btn.setIcon(make_icon(Icon.PLAY, 26, p.on_accent))
        self.retry_all_btn.setIcon(make_icon(Icon.RETRY, 18, p.text_dim))
        # 削除だけ危険色ではっきり見せる（枠は QSS の dangerIcon）。
        self.clear_btn.setIcon(make_icon(Icon.DELETE, 18, p.danger))

    def _build_header(self) -> QFrame:
        header = QFrame()
        header.setObjectName("header")
        row = QHBoxLayout(header)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(12)

        app_icon = QLabel(Icon.UPLOAD)
        app_icon.setObjectName("appIcon")
        apply_icon_font(app_icon, 22)
        app_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        row.addWidget(app_icon)

        title = QLabel("ultraeasy-upscaler")
        title.setObjectName("appTitle")
        row.addWidget(title)

        # 一括設定の3欄（画像・動画の拡大モデル、フレーム補間）。
        # 同じ幅で、狭いときは最小まで縮む。
        self.image_model_combo = ModelCombo()
        self._fix_header_combo_width(self.image_model_combo)
        row.addWidget(self._header_field(t("画像の拡大モデル"), self.image_model_combo), 10)
        self.video_model_combo = ModelCombo()
        self._fix_header_combo_width(self.video_model_combo)
        row.addWidget(self._header_field(t("動画の拡大モデル"), self.video_model_combo), 10)
        self.global_interpolation_combo = ModelCombo()
        self._fix_header_combo_width(self.global_interpolation_combo)
        self._populate_interpolation_combo(self.global_interpolation_combo)
        row.addWidget(
            self._header_field(t("フレーム補間モデル"), self.global_interpolation_combo),
            10,
        )
        row.addStretch(1)

        self.settings_btn = QPushButton("")
        self.settings_btn.setObjectName("iconButton")
        self.settings_btn.setIconSize(QSize(24, 24))
        self.settings_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.settings_btn.setToolTip(t("詳細設定"))
        self.settings_btn.clicked.connect(self._toggle_drawer)
        row.addWidget(self.settings_btn)

        return header

    @staticmethod
    def _fix_header_combo_width(combo: QComboBox) -> None:
        """ヘッダー3欄を同じ幅にする（狭いときは最小まで縮む）。"""
        combo.setMinimumWidth(_HEADER_FIELD_MIN_WIDTH)
        combo.setMaximumWidth(_HEADER_FIELD_WIDTH)
        combo.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )

    def _header_field(self, label: str, widget: QWidget) -> QWidget:
        """ヘッダー用の2行の欄（見出し＋プルダウン）。幅は3欄でそろえる。"""
        wrap = QWidget()
        wrap.setMinimumWidth(_HEADER_FIELD_MIN_WIDTH)
        wrap.setMaximumWidth(_HEADER_FIELD_WIDTH)
        wrap.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        # ヘッダーを厚くしないよう、見出しは小さく、プルダウンは低くする
        box = QVBoxLayout(wrap)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(2)
        lab = QLabel(label)
        lab.setObjectName("headerLabel")
        box.addWidget(lab)
        widget.setObjectName("headerCombo")
        box.addWidget(widget)
        return wrap

    @staticmethod
    def _populate_interpolation_combo(combo: QComboBox) -> None:
        """フレーム補間コンボの選択肢を入れる（ヘッダーと右列で同じ）。

        FILM が使えないときは「FILM (Style)（未導入）」を選べない項目で出し、
        詳細設定の追加キットへ案内する。
        """
        combo.addItem(t("なし（補間しない）"), None)
        models = binaries.available_interpolation_models()
        if "rife-v4.6" in models:
            combo.addItem(t("RIFE v4.6"), "rife-v4.6")
        if binaries.FILM_MODEL in models:
            combo.addItem(t("FILM (Style)"), binaries.FILM_MODEL)
        else:
            combo.addItem(
                t("{name}（未導入）", name=t("FILM (Style)")),
                binaries.FILM_MODEL,
            )
            item = combo.model().item(combo.count() - 1)
            if item is not None:
                item.setEnabled(False)
                item.setToolTip(t("詳細設定の「追加キット」からダウンロードできます"))

    def _field(self, label: str, widget: QWidget) -> QVBoxLayout:
        box = QVBoxLayout()
        box.setSpacing(4)
        lab = QLabel(label)
        lab.setObjectName("fieldLabel")
        box.addWidget(lab)
        box.addWidget(widget)
        return box

    @staticmethod
    def _compact(combo: QComboBox) -> QComboBox:
        """最長項目でなく一定幅を最小とし、細い列でも収まるようにする。"""
        combo.setMinimumContentsLength(8)
        combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        return combo

    def _build_settings_panel(self) -> QFrame:
        """右列: 選んだファイルの設定と、開始/一時停止（一括はヘッダー）。"""
        panel = QFrame()
        panel.setObjectName("controlPanel")
        panel.setFixedWidth(_SETTINGS_COL_WIDTH)
        self._settings_panel = panel
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(16, 14, 16, 16)
        outer.setSpacing(12)

        # 題名はファイル選択時はその名前、無ければ「設定」。
        self._settings_title = QLabel(t("設定"))
        self._settings_title.setObjectName("sectionTitle")
        self._settings_title.setWordWrap(False)
        outer.addWidget(self._settings_title)

        # 設定欄は画面が低いときだけスクロールし、開始ボタンは常に見える位置に残す
        fields = QWidget()
        fields.setObjectName("scaleWrap")
        col = QVBoxLayout(fields)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(12)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(fields)
        outer.addWidget(scroll, 1)

        # ファイル未選択のときだけ出す案内
        self._no_file_hint = QLabel(
            t("ファイルを選ぶと、そのファイルだけ設定を変えられます")
        )
        self._no_file_hint.setObjectName("hint")
        self._no_file_hint.setWordWrap(True)
        col.addWidget(self._no_file_hint)

        self.model_combo = ModelCombo()
        self._compact(self.model_combo)
        self._model_wrap = QWidget()
        self._model_wrap.setObjectName("scaleWrap")
        model_layout = self._field(t("拡大モデル"), self.model_combo)
        model_layout.setContentsMargins(0, 0, 0, 0)
        self._model_wrap.setLayout(model_layout)
        col.addWidget(self._model_wrap)

        # 選択中の 処理×モデル の説明
        self.model_hint = QLabel("")
        self.model_hint.setObjectName("hint")
        self.model_hint.setWordWrap(True)
        col.addWidget(self.model_hint)

        # 倍率トグル（2x / 4x）
        scale_box = QHBoxLayout()
        scale_box.setContentsMargins(0, 0, 0, 0)
        scale_box.setSpacing(6)
        self._scale_btns: dict[int, QPushButton] = {}
        for s in _SCALE_CHOICES:
            btn = QPushButton(f"{s}x")
            btn.setObjectName("scaleBtn")
            btn.setCheckable(True)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setChecked(s == 4)
            btn.clicked.connect(lambda _=False, v=s: self._set_scale(v))
            self._scale_btns[s] = btn
            scale_box.addWidget(btn, 1)
        self._scale_wrap = QWidget()
        self._scale_wrap.setObjectName("scaleWrap")
        scale_wrap_inner = QWidget()
        scale_wrap_inner.setObjectName("scaleWrap")
        scale_wrap_inner.setLayout(scale_box)
        scale_layout = QVBoxLayout()
        scale_layout.setSpacing(4)
        scale_lab = QLabel(t("倍率"))
        scale_lab.setObjectName("fieldLabel")
        scale_layout.addWidget(scale_lab)
        scale_layout.addWidget(scale_wrap_inner)
        scale_layout.setContentsMargins(0, 0, 0, 0)
        self._scale_wrap.setLayout(scale_layout)
        col.addWidget(self._scale_wrap)

        # フレーム補間モデル（動画のファイルを選んだときだけ出す）
        self.interpolation_combo = QComboBox()
        self._populate_interpolation_combo(self.interpolation_combo)
        self._interp_wrap = QWidget()
        self._interp_wrap.setObjectName("scaleWrap")
        interp_layout = self._field(
            t("フレーム補間モデル"), self._compact(self.interpolation_combo)
        )
        interp_layout.setContentsMargins(0, 0, 0, 0)
        self._interp_wrap.setLayout(interp_layout)
        col.addWidget(self._interp_wrap)

        # フレーム補間モデルの説明（補間を選んでいるときだけ出す）
        self.interpolation_hint = QLabel("")
        self.interpolation_hint.setObjectName("hint")
        self.interpolation_hint.setWordWrap(True)
        col.addWidget(self.interpolation_hint)

        # クイック確認の組（TrialPanel が持ち、右列に差し込む）
        col.addWidget(self.preview.quick_box)

        # 個別設定のときだけ出す箱（案内＋2ボタン）
        self._override_box = QWidget()
        self._override_box.setObjectName("scaleWrap")
        override_col = QVBoxLayout(self._override_box)
        override_col.setContentsMargins(0, 0, 0, 0)
        override_col.setSpacing(8)
        self._override_hint = QLabel(t("このファイルだけ別の設定です"))
        self._override_hint.setObjectName("hint")
        self._override_hint.setWordWrap(True)
        override_col.addWidget(self._override_hint)
        self.apply_all_btn = QPushButton(t("ほかの画像にも使う"))
        self.apply_all_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.apply_all_btn.clicked.connect(self._on_apply_all)
        override_col.addWidget(self.apply_all_btn)
        self.reset_override_btn = QPushButton(t("一括設定に戻す"))
        self.reset_override_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.reset_override_btn.clicked.connect(self._on_reset_override)
        override_col.addWidget(self.reset_override_btn)
        col.addWidget(self._override_box)

        # 出力先（全体の設定なので常に出す）
        # 出力先は全体の設定なので、欄は詳細設定の中にある
        self.output_combo = self.drawer.output_combo
        self.output_combo.activated.connect(self._on_output_changed)
        self._output_dir: str | None = None
        col.addStretch(1)

        self.status_label = StatusLabel("")
        self.status_label.setObjectName("hint")
        self.status_label.setWordWrap(True)
        outer.addWidget(self.status_label)

        self.pause_btn = QPushButton(t("一時停止"))
        self.pause_btn.setIconSize(QSize(20, 20))
        self.pause_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.pause_btn.setEnabled(False)
        self.pause_btn.setToolTip(t("現在のジョブ完了後に停止します"))
        self.pause_btn.clicked.connect(self._on_pause)
        outer.addWidget(self.pause_btn)

        self.start_btn = QPushButton(t("開始"))
        self.start_btn.setObjectName("primary")
        self.start_btn.setIconSize(QSize(26, 26))
        self.start_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.start_btn.clicked.connect(self._on_start)
        outer.addWidget(self.start_btn)

        return panel

    def _build_media_panel(self) -> QFrame:
        """左列: メディア一覧。下端の枠とウィンドウ全体がドロップ先になる。"""
        card = QFrame()
        card.setObjectName("card")
        card.setFixedWidth(_MEDIA_COL_WIDTH)
        self.media_card = card
        lay = QVBoxLayout(card)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(8)

        head = QHBoxLayout()
        title = QLabel(t("メディア"))
        title.setObjectName("sectionTitle")
        head.addWidget(title)
        # やり直すは見出しのすぐ右に置くアイコンだけのボタン
        self.retry_all_btn = QPushButton("")
        self.retry_all_btn.setObjectName("iconButton")
        self.retry_all_btn.setIconSize(QSize(18, 18))
        self.retry_all_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.retry_all_btn.setToolTip(t("すべてやり直す"))
        self.retry_all_btn.setEnabled(False)
        self.retry_all_btn.clicked.connect(self._on_retry_all)
        head.addWidget(self.retry_all_btn)
        head.addStretch(1)
        # 削除は右端に離し、危険色のアイコンと枠で示す
        self.clear_btn = QPushButton("")
        self.clear_btn.setObjectName("dangerIcon")
        self.clear_btn.setIconSize(QSize(18, 18))
        self.clear_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.clear_btn.setToolTip(t("すべて削除"))
        self.clear_btn.clicked.connect(self._clear_queue)
        head.addWidget(self.clear_btn)
        lay.addLayout(head)

        self.queue = QueueView(describe_settings=self._describe_settings)
        self.queue.removeRequested.connect(self._on_remove_requested)
        self.queue.outputRequested.connect(self._on_row_output_requested)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.queue)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        lay.addWidget(scroll, 1)

        self.add_zone = DropZone(compact=True)
        self.add_zone.pathsDropped.connect(self._on_paths_dropped)
        self.add_zone.browseRequested.connect(self._pick_files)
        lay.addWidget(self.add_zone)

        return card

    # ------------------------------------------------------ 一覧と表示の同期
    def _sync_workspace(self) -> None:
        """詳細設定の開閉とメディアの有無に合わせて、左＋中央の表示を切り替える。"""
        if self._drawer_open:
            self.stack.setCurrentIndex(2)
        else:
            self.stack.setCurrentIndex(1 if self._order else 0)

    def _on_selection_changed(self, job_id: object) -> None:
        job = self._jobs.get(job_id) if job_id is not None else None
        self.preview.show_job(job)
        self._sync_settings_widgets()

    def _set_drop_highlight(self, active: bool) -> None:
        """ドラッグ中、ウィンドウのどこでも落とせることを一覧全体の色で示す。"""
        for widget in (self.media_card, self.drop_zone):
            widget.setProperty("dragActive", active)
            widget.style().unpolish(widget)
            widget.style().polish(widget)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802
        if event.mimeData().hasUrls() and not self._running:
            event.acceptProposedAction()
            self._set_drop_highlight(True)
        else:
            event.ignore()

    def dragLeaveEvent(self, event) -> None:  # noqa: N802
        self._set_drop_highlight(False)
        super().dragLeaveEvent(event)

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        self._set_drop_highlight(False)
        paths = [u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()]
        paths = [p for p in paths if p]
        if paths:
            event.acceptProposedAction()
            self.add_paths(paths)
        else:
            event.ignore()

    # --------------------------------------------------------- ジョブ追加
    def add_path(self, path: str) -> tuple[bool, str]:
        """1 パスを Job 化してキューへ。(成功, メッセージ) を返す。"""
        try:
            job = Job.create(path)
        except ValueError as exc:
            return False, str(exc)
        # 表示用にソース解像度などのメタ情報を取得（失敗しても追加は続行）
        if job.kind in (JobKind.IMAGE, JobKind.VIDEO):
            try:
                from app.core import media
                info = media.probe(str(job.input_path))
                job.width, job.height = info.width, info.height
                job.fps = info.fps
                job.frame_count = info.frame_count
                job.has_audio = info.has_audio
            except Exception:
                pass
        # 設定はここでは固定しない。「開始」時点でファイルごとの設定
        # （個別があればそれ、無ければ種類の既定）が入る。
        self._jobs[job.id] = job
        self._order.append(job.id)
        self._cancel_events[job.id] = threading.Event()
        job.message = self._waiting_text(job)
        self.queue.add_job(job)
        self.queue.refresh(job.id)
        self._update_retry_all()
        self._sync_workspace()
        return True, job.name

    def add_paths(self, paths: list[str]) -> None:
        """複数パスを追加し、未対応はスキップしてフッタヒントに件数を出す。"""
        added = 0
        skipped: list[str] = []
        for p in paths:
            ok, msg = self.add_path(p)
            if ok:
                added += 1
            else:
                skipped.append(Path(p).name)
        if skipped:
            self._flash_hint(
                t(
                    "{added} 件追加 / {skipped} 件は未対応のためスキップ: {names}{more}",
                    added=added,
                    skipped=len(skipped),
                    names=", ".join(skipped[:3]),
                    more="…" if len(skipped) > 3 else "",
                )
            )

    def _on_paths_dropped(self, paths: list[str]) -> None:
        self.add_paths(paths)

    def _pick_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self, t("ファイルを選択（複数可）"), "",
            t(
                "対応ファイル (*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff "
                "*.mp4 *.mkv *.mov *.avi *.webm *.m4v *.wmv *.flv *.mpg *.mpeg *.ts *.m2ts);;"
                "すべて (*.*)"
            ),
        )
        if files:
            self.add_paths(files)

    def _pick_folder(self) -> None:
        d = QFileDialog.getExistingDirectory(self, t("フォルダを選択"))
        if d:
            self.add_paths([d])

    # --------------------------------------------- 一括設定と個別設定
    @staticmethod
    def _job_kind_tab(job: Job) -> str:
        """ファイルの種類に対応する設定の側（フォルダは画像用）。"""
        return _VIDEO_TAB if job.kind == JobKind.VIDEO else _IMAGE_TAB

    def _selected_job(self) -> Job | None:
        jid = self.queue.selected_id()
        return self._jobs.get(jid) if jid is not None else None

    def _ensure_override(self, job: Job) -> dict[str, object]:
        """右列の変更用に個別設定を用意する（初期値はその時点の一括設定）。"""
        override = self._overrides.get(job.id)
        if override is None:
            if job.kind == JobKind.VIDEO:
                override = {
                    "model": self._video_model,
                    "scale": self._video_scale,
                    "interpolation": self._video_interpolation,
                }
            else:
                override = {
                    "model": self._image_model,
                    "scale": self._image_scale,
                }
            self._overrides[job.id] = override
        return override

    @staticmethod
    def _norm_choice(value: object) -> str | None:
        """モデル・補間の比較用（「なし」と未検出はどちらも無し扱い）。"""
        return None if value in (None, "__missing__") else str(value)

    def _is_override_redundant(self, job: Job) -> bool:
        """個別設定が一括設定と開始時の実効値で同じか（同じなら消す）。"""
        override = self._overrides.get(job.id)
        if override is None:
            return False
        is_video = job.kind == JobKind.VIDEO
        eff_model, eff_scale, eff_interp, _ = self._resolve_effective(job)
        if is_video:
            bulk_model = self._norm_choice(self._video_model)
            bulk_scale = int(self._video_scale)
            bulk_interp = self._norm_choice(self._video_interpolation)
        else:
            bulk_model = self._norm_choice(self._image_model)
            bulk_scale = int(self._image_scale)
            bulk_interp = None
        if self._selected_backend() in _HELPER_BACKENDS and bulk_model is not None:
            bulk_scale = 4
        if eff_model != bulk_model or eff_scale != bulk_scale:
            return False
        return eff_interp == bulk_interp if is_video else True

    def _prune_redundant_overrides(self, job_ids=None) -> None:
        """一括と同じになった個別設定を消す（判定は1か所にまとめる）。"""
        targets = list(self._overrides.keys()) if job_ids is None else list(job_ids)
        for jid in targets:
            job = self._jobs.get(jid)
            if job is None:
                self._overrides.pop(jid, None)
                continue
            if self._is_override_redundant(job):
                self._overrides.pop(jid, None)

    def _update_settings_title(self, name: str) -> None:
        """右列の題名をファイル名にする（長いときは中ほどを省略）。"""
        width = self._settings_title.width()
        if width <= 0 and hasattr(self, "_settings_panel"):
            width = self._settings_panel.width() - 32
        if width <= 0:
            width = _SETTINGS_COL_WIDTH - 32
        elided = self._settings_title.fontMetrics().elidedText(
            name, Qt.TextElideMode.ElideMiddle, max(40, width)
        )
        self._settings_title.setText(elided)

    def _rebuild_right_combo(
        self, job: Job | None, models: list[str] | None = None
    ) -> None:
        """右列のモデル欄をそのファイルの種類に合わせて作り直す。

        選択が変わると画像／動画で選択肢が変わる（AdcSR の有無）ため、
        載せ替えのたびに作り直す。ヘッダー側は種類が固定なので作り直さない。
        models を渡せば従来モデル一覧の再取得を省く。
        """
        backend = self._selected_backend()
        is_video = job is not None and job.kind == JobKind.VIDEO
        if job is not None:
            effective = self._overrides.get(job.id, {}).get(
                "model",
                self._video_model if is_video else self._image_model,
            )
        else:
            effective = self._image_model
        self._replace_model_items(
            self.model_combo,
            self._base_options(backend, is_video, models),
            effective,
        )
        self._disable_missing_item(self.model_combo)

    def _sync_settings_widgets(self, models: list[str] | None = None) -> None:
        """選択に合わせてヘッダーと右列の欄を載せ替える。"""
        self._syncing_settings = True
        try:
            job = self._selected_job()
            # ヘッダーは常に一括設定の値
            self._set_combo_data(self.image_model_combo, self._image_model)
            self._set_combo_data(self.video_model_combo, self._video_model)
            self._set_combo_data(
                self.global_interpolation_combo, self._video_interpolation
            )
            # 右列の選択肢は種類で変わるため先に作り直す
            self._rebuild_right_combo(job, models)
            if job is None:
                self._settings_title.setText(t("設定"))
                self._settings_title.setToolTip("")
                self._no_file_hint.setVisible(True)
                self._model_wrap.setVisible(False)
                self.model_hint.setVisible(False)
                self._scale_wrap.setVisible(False)
                self._interp_wrap.setVisible(False)
                self.interpolation_hint.setVisible(False)
                self.preview.quick_box.setVisible(False)
                self._override_box.setVisible(False)
            else:
                self._settings_title.setToolTip(job.name)
                self._update_settings_title(job.name)
                is_video = job.kind == JobKind.VIDEO
                self._no_file_hint.setVisible(False)
                self._model_wrap.setVisible(True)
                self.model_hint.setVisible(True)
                self._scale_wrap.setVisible(True)
                self._interp_wrap.setVisible(is_video)
                self.preview.quick_box.setVisible(True)
                model, scale, interp, _individual = self._resolve_effective(job)
                # 補間の説明は動画で補間を選んでいるときだけ出す。
                self.interpolation_hint.setVisible(bool(is_video and interp))
                self._set_combo_data(self.model_combo, model)
                for s, btn in self._scale_btns.items():
                    btn.setChecked(s == scale)
                if is_video:
                    self._set_combo_data(self.interpolation_combo, interp)
                has_override = job.id in self._overrides
                self._override_box.setVisible(has_override)
                if has_override:
                    if is_video:
                        self.apply_all_btn.setText(t("ほかの動画にも使う"))
                    else:
                        self.apply_all_btn.setText(t("ほかの画像にも使う"))
                    self.apply_all_btn.setEnabled(not self._running)
                    self.reset_override_btn.setEnabled(not self._running)
        finally:
            self._syncing_settings = False
        self._refresh_scale_enabled()
        self._refresh_interpolation_enabled()
        if hasattr(self, "preview"):
            self.preview.refresh()

    def _on_apply_all(self) -> None:
        """個別設定を同じ種類の一括設定へ写す（写した分は一括と同じで消える）。"""
        if self._running or self._syncing_settings:
            return
        job = self._selected_job()
        if job is None:
            return
        override = self._overrides.get(job.id)
        if override is None:
            return
        if job.kind == JobKind.VIDEO:
            self._video_model = override.get("model")  # type: ignore[assignment]
            self._video_scale = int(override.get("scale", 4))
            self._video_interpolation = override.get("interpolation")
        else:
            self._image_model = override.get("model")  # type: ignore[assignment]
            self._image_scale = int(override.get("scale", 4))
        self._prune_redundant_overrides()
        self._sync_settings_widgets()
        self._refresh_all_waiting()
        if hasattr(self, "preview"):
            self.preview.refresh()

    def _on_reset_override(self) -> None:
        """そのファイルの個別設定を消して一括設定に戻す。"""
        if self._running or self._syncing_settings:
            return
        job = self._selected_job()
        if job is None:
            return
        self._overrides.pop(job.id, None)
        self._sync_settings_widgets()
        self._refresh_all_waiting()
        if hasattr(self, "preview"):
            self.preview.refresh()

    def _resolve_effective(
        self, job: Job
    ) -> tuple[str | None, int, str | None, bool]:
        """そのファイルが開始時に使われる (モデル, 倍率, 補間, 個別か)。"""
        is_video = job.kind == JobKind.VIDEO
        override = self._overrides.get(job.id)
        if override is not None:
            model = override.get("model")
            scale = int(override.get("scale", 4))
            interp = override.get("interpolation") if is_video else None
            individual = True
        elif is_video:
            model = self._video_model
            scale = self._video_scale
            interp = self._video_interpolation
            individual = False
        else:
            model = self._image_model
            scale = self._image_scale
            interp = None
            individual = False
        # 新AIヘルパーは 4x 固定（古い個別値が残っていても開始時は 4x）
        if self._selected_backend() in _HELPER_BACKENDS and model is not None:
            scale = 4
        return (
            None if model in (None, "__missing__") else str(model),
            scale,
            None if interp in (None, "__missing__") else str(interp),
            individual,
        )

    def _set_scale(self, value: int) -> None:
        """右列の倍率変更。そのファイルの個別設定を作って値を入れる。"""
        if self._syncing_settings:
            return
        job = self._selected_job()
        if job is None:
            return
        override = self._ensure_override(job)
        override["scale"] = value
        self._prune_redundant_overrides([job.id])
        self._sync_settings_widgets()
        self._refresh_all_waiting()
        if hasattr(self, "preview"):
            self.preview.refresh()

    @staticmethod
    def _first_supported_scale(model: object) -> int | None:
        """Vulkan用にそのモデルが対応する最初の倍率（無ければ None）。"""
        for s in _SCALE_CHOICES:
            try:
                supported = binaries.model_supports_scale(model, s)
            except Exception:
                return s
            if supported:
                return s
        return None

    def _fix_stored_scale(self, job: Job) -> None:
        """Vulkanで右列の倍率が対応外なら、保存側を対応する値に直す。"""
        _model, scale, _interp, _ind = self._resolve_effective(job)
        model = self.model_combo.currentData()
        if model in (None, "__missing__"):
            return
        try:
            supported = binaries.model_supports_scale(model, scale)
        except Exception:
            return
        if supported:
            return
        first = self._first_supported_scale(model)
        if first is None:
            return
        if job.id in self._overrides:
            self._overrides[job.id]["scale"] = first
        elif job.kind == JobKind.VIDEO:
            self._video_scale = first
        else:
            self._image_scale = first
        self._prune_redundant_overrides([job.id])

    def _refresh_scale_enabled(self) -> None:
        """バックエンドごとに倍率ボタンの有効範囲を更新する。"""
        backend = self._selected_backend()
        job = self._selected_job()
        model = None
        if job is not None:
            model, _s, _i, _ind = self._resolve_effective(job)
            # currentData は実効値と同じはずだが、載せ替え直後は
            # こちらを正とする（なぜ上書きしないか: 実効値が正）。
            try:
                model = self.model_combo.currentData()
            except RuntimeError:
                pass
        self.backend_combo.setEnabled(not self._running)
        for combo in (
            self.image_model_combo,
            self.video_model_combo,
            self.global_interpolation_combo,
            self.model_combo,
            self.interpolation_combo,
        ):
            try:
                combo.setEnabled(not self._running)
            except RuntimeError:
                pass

        if job is None:
            for button in self._scale_btns.values():
                button.setEnabled(False)
            self._update_all_model_info()
            return
        if backend in _HELPER_BACKENDS:
            # 「なし」は無効にするだけで、コンボ自体は戻せるよう有効のままにする。
            for scale, button in self._scale_btns.items():
                button.setEnabled(
                    model in _HELPER_MODEL_VALUES
                    and scale == 4
                    and not self._running
                )
            self._update_all_model_info()
            return

        upscale_enabled = model not in (None, "__missing__")
        if not upscale_enabled:
            for button in self._scale_btns.values():
                button.setEnabled(False)
            self._update_all_model_info()
            return

        self._fix_stored_scale(job)
        for scale, button in self._scale_btns.items():
            try:
                supported = binaries.model_supports_scale(model, scale)
            except Exception:
                supported = True
            button.setEnabled(supported and not self._running)
        self._update_all_model_info()

    def _film_selected(self) -> bool:
        """一括か個別のどれかで FILM (Style) が選ばれているか。"""
        if self._norm_choice(self._video_interpolation) == binaries.FILM_MODEL:
            return True
        return any(
            self._norm_choice(override.get("interpolation")) == binaries.FILM_MODEL
            for override in self._overrides.values()
        )

    def _refresh_interpolation_enabled(self) -> None:
        """詳細設定の補間後fpsは、一括か個別のどちらかで補間を使うとき有効。"""
        enabled = self._video_interpolation not in (None, "__missing__")
        if not enabled:
            for override in self._overrides.values():
                if override.get("interpolation") not in (None, "__missing__"):
                    enabled = True
                    break
        try:
            self.drawer.set_interpolation_enabled(bool(enabled))
            self.drawer.set_film_selected(self._film_selected())
        except RuntimeError:
            pass

    @staticmethod
    def _without_adcsr(
        options: list[tuple[str, object]],
    ) -> list[tuple[str, object]]:
        """動画側のモデル一覧（AdcSR は動画に使えないため出さない）。"""
        return [
            (label, value)
            for label, value in options
            if value != HELPER_MODEL_ADCSR
        ]

    def _base_options(
        self, backend: UpscaleBackend, is_video: bool,
        models: list[str] | None = None,
    ) -> list[tuple[str, object]]:
        """その種類で使えるモデル一覧（ヘッダーと右列で同じ規則）。"""
        if backend in _HELPER_BACKENDS:
            if backend == UpscaleBackend.SWINIR_CUDA:
                return list(_SWINIR_CUDA_MODEL_OPTIONS)
            base = list(_HELPER_MODEL_OPTIONS)
            return self._without_adcsr(base) if is_video else base
        if models is None:
            try:
                models = binaries.available_models()
            except Exception:
                models = []
        options: list[tuple[str, object]] = [(N_("なし（拡大しない）"), None)]
        options.extend((_MODEL_LABELS.get(model, model), model) for model in models)
        if not models:
            options.append((N_("モデル未検出"), "__missing__"))
        return options

    def _coerce_one_model(
        self, backend: UpscaleBackend, value: object, is_video: bool,
        models: list[str] | None, combo: QComboBox,
    ) -> object:
        """一括1欄を使えない値から置き換える（初回は具体的な既定を使う）。"""
        if backend in _HELPER_BACKENDS:
            default_model = (
                HELPER_MODEL_SWINIR
                if backend == UpscaleBackend.SWINIR_CUDA
                else DEFAULT_HELPER_MODEL
            )
            allowed = {v for _label, v in self._base_options(backend, is_video)}
            if combo.count() == 0:
                return default_model
            return value if value in allowed else default_model
        if models is None:
            try:
                models = binaries.available_models()
            except Exception:
                models = []
        if not models:
            return value if value in (None, "__missing__") else None
        if value is not None and value not in models:
            return DEFAULT_MODEL if DEFAULT_MODEL in models else None
        if combo.count() == 0 and value is None and DEFAULT_MODEL in models:
            return DEFAULT_MODEL
        return value

    def _refresh_model_options(self, models: list[str] | None = None) -> None:
        """バックエンドに応じてヘッダー3欄と右列のモデル欄を再構成する。

        新AIバックエンドでは旧Vulkan資産を列挙せず、具体的な実モデルを表示する。
        Vulkanを選んだときだけ vendor/realesrgan の従来モデルを表示する。
        動画側の一覧には AdcSR を出さない。
        models を渡せば従来モデル一覧の再取得を省く（呼出回数の互換のため）。
        """
        backend = self._selected_backend()
        if backend not in _HELPER_BACKENDS and models is None:
            try:
                models = binaries.available_models()
            except Exception:
                models = []
        self._image_model = self._coerce_one_model(  # type: ignore[assignment]
            backend, self._image_model, False, models, self.image_model_combo
        )
        self._video_model = self._coerce_one_model(  # type: ignore[assignment]
            backend, self._video_model, True, models, self.video_model_combo
        )
        self._replace_model_items(
            self.image_model_combo,
            self._base_options(backend, False, models),
            self._image_model,
        )
        self._disable_missing_item(self.image_model_combo)
        self._replace_model_items(
            self.video_model_combo,
            self._base_options(backend, True, models),
            self._video_model,
        )
        self._disable_missing_item(self.video_model_combo)
        job = self._selected_job()
        if job is not None:
            is_video = job.kind == JobKind.VIDEO
            effective = self._overrides.get(job.id, {}).get(
                "model",
                self._video_model if is_video else self._image_model,
            )
            # 個別が使えない値なら一括の既定に寄せる（_coerce と同じ規則）
            allowed = {
                v for _label, v in self._base_options(backend, is_video, models)
            }
            if effective not in allowed:
                default = self._default_model_for(backend, models)
                if job.id in self._overrides:
                    self._overrides[job.id]["model"] = default
                elif is_video:
                    self._video_model = default  # type: ignore[assignment]
                else:
                    self._image_model = default  # type: ignore[assignment]
        self._sync_settings_widgets(models)
        self._refresh_all_waiting()
        if hasattr(self, "preview"):
            self.preview.refresh()

    @staticmethod
    def _disable_missing_item(combo: QComboBox) -> None:
        """「モデル未検出」の項目は選べないようにする。"""
        idx = combo.findData("__missing__")
        if idx >= 0:
            item = combo.model().item(idx)
            if item is not None:
                item.setEnabled(False)

    def _replace_model_items(
        self, combo: QComboBox,
        options: list[tuple[str, object]], selected: object | None,
    ) -> None:
        """モデルコンボの項目を差し替え、可能なら選択値を維持する。"""
        previous = combo.blockSignals(True)
        try:
            combo.clear()
            for label, value in options:
                combo.addItem(t(label), value)
            if selected is not None:
                index = combo.findData(selected)
                if index >= 0:
                    combo.setCurrentIndex(index)
            if combo.currentIndex() < 0 and combo.count():
                combo.setCurrentIndex(0)
        finally:
            combo.blockSignals(previous)

    @staticmethod
    def _set_combo_data(combo: QComboBox, value: object) -> None:
        """コンボの値を変更する（変更通知は発火させない）。"""
        index = combo.findData(value)
        if index < 0:
            return
        previous = combo.blockSignals(True)
        try:
            combo.setCurrentIndex(index)
        finally:
            combo.blockSignals(previous)

    @staticmethod
    def _compose_model_hint(backend: UpscaleBackend, data: str) -> str:
        info = _MODEL_INFO.get((backend, data))
        if info is None:
            return ""
        speed, quality, anime, live, star = info
        parts = [
            t("速度{speed}", speed=t(speed)),
            t("画質{quality}", quality=t(quality)),
            t("アニメ{anime}・実写{live}", anime=t(anime), live=t(live)),
        ]
        if star:
            parts.append(t("★{star}に推奨", star=t(star)))
        return "／".join(parts)

    @staticmethod
    def _compose_hint_line(backend: UpscaleBackend, data: str) -> str:
        """モデルの説明を 1 行で返す（速さ・実行先の注意は付けない）。"""
        del backend  # 説明は実行先によらない
        desc = _MODEL_HINT.get(data, "")
        return t(desc) if desc else ""

    @staticmethod
    def _compose_interp_hint(data: object) -> str:
        """フレーム補間モデルの説明を 1 行で返す。"""
        if data == "rife-v4.6":
            return t("高速")
        if data == binaries.FILM_MODEL:
            return t("低速・高品質")
        return ""

    def _update_combo_badges(self, combo: QComboBox) -> None:
        """1つのモデルコンボにバッジと未変換印・ツールチップを付ける。"""
        backend = self._selected_backend()
        # NPU のとき未変換のモデルは開いた一覧に印を付ける（閉じた表示は名だけ）。
        show_unconverted = backend == UpscaleBackend.NPU_NATIVE
        item_model = combo.model()
        for i in range(combo.count()):
            data = combo.itemData(i)
            if data in (None, "__missing__"):
                continue
            base = t(_MODEL_LABELS.get(data, data))
            suffix = ""
            if show_unconverted and not npu_prepare.is_converted(str(data)):
                suffix = t("（未変換）")
            info = _MODEL_INFO.get((backend, data))
            item = item_model.item(i)
            if info is None:
                combo.setItemText(i, f"{base}{suffix}")
                if item is not None:
                    item.setToolTip("")
                continue
            speed, quality, _anime, _live, star = info
            badge = t(
                "速度{speed} 画質{quality}",
                speed=t(speed),
                quality=t(quality),
            ) + (t(" ★{star}", star=t(star)) if star else "")
            combo.setItemText(i, f"{base}{suffix}｜{badge}")
            if item is not None:
                item.setToolTip(self._compose_model_hint(backend, data))
        # 閉じた状態はコンパクト幅のままでよいが、開いたリストは全文が
        # 収まる幅へ広げる（切れて読めない問題の対策）
        try:
            view = combo.view()
            fm = view.fontMetrics()
            widest = max((fm.horizontalAdvance(combo.itemText(i))
                          for i in range(combo.count())), default=0)
            view.setMinimumWidth(widest + 48)
        except RuntimeError:
            pass

    def _update_all_model_info(self) -> None:
        """ヘッダー3欄と右列のバッジ・説明・ツールチップを更新する。"""
        if not hasattr(self, "model_hint"):
            return
        backend = self._selected_backend()
        for combo in (
            self.image_model_combo,
            self.video_model_combo,
            self.model_combo,
        ):
            try:
                self._update_combo_badges(combo)
            except RuntimeError:
                pass
        # ヘッダーのツールチップは選択中のモデルの説明（右列の説明と同じ文）
        try:
            cur_image = self.image_model_combo.currentData()
            self.image_model_combo.setToolTip(
                "" if cur_image in (None, "__missing__")
                else self._compose_hint_line(backend, cur_image)
            )
            cur_video = self.video_model_combo.currentData()
            self.video_model_combo.setToolTip(
                "" if cur_video in (None, "__missing__")
                else self._compose_hint_line(backend, cur_video)
            )
        except RuntimeError:
            pass

        # ヘッダーの補間プルダウンには説明をツールチップで出す。
        try:
            self.global_interpolation_combo.setToolTip(
                self._compose_interp_hint(
                    self.global_interpolation_combo.currentData()
                )
            )
        except RuntimeError:
            pass

        job = self._selected_job()
        if job is None:
            self.model_hint.setText("")
            if hasattr(self, "interpolation_hint"):
                self.interpolation_hint.setText("")
            return
        try:
            cur = self.model_combo.currentData()
        except RuntimeError:
            return
        if cur in (None, "__missing__"):
            self.model_hint.setText(t("拡大はしません。フレーム補間だけ実行できます。"))
        else:
            self.model_hint.setText(self._compose_hint_line(backend, cur))
        try:
            self.interpolation_hint.setText(
                self._compose_interp_hint(self.interpolation_combo.currentData())
            )
        except RuntimeError:
            pass

    def _update_model_info(self) -> None:
        """互換のための別名（実体は _update_all_model_info）。"""
        self._update_all_model_info()

    def _selected_backend(self) -> UpscaleBackend:
        value = self.backend_combo.currentData() if hasattr(self, "backend_combo") else None
        if value == "auto":
            return UpscaleBackend.WINML_GPU
        try:
            return UpscaleBackend(value or UpscaleBackend.WINML_GPU.value)
        except ValueError:
            return UpscaleBackend.WINML_GPU

    def _allowed_models(
        self, backend: UpscaleBackend, tab: str, models: list[str] | None = None
    ) -> set:
        """その側で選べるモデル値（「なし」含む）。"""
        if backend in _HELPER_BACKENDS:
            if backend == UpscaleBackend.SWINIR_CUDA:
                return {value for _label, value in _SWINIR_CUDA_MODEL_OPTIONS}
            allowed = {value for _label, value in _HELPER_MODEL_OPTIONS}
            if tab == _VIDEO_TAB:
                allowed.discard(HELPER_MODEL_ADCSR)
            return allowed
        if models is None:
            try:
                models = binaries.available_models()
            except Exception:
                models = []
        return set(models) | {None}

    def _default_model_for(
        self, backend: UpscaleBackend, models: list[str] | None = None
    ) -> str | None:
        if backend == UpscaleBackend.SWINIR_CUDA:
            return HELPER_MODEL_SWINIR
        if backend in _HELPER_BACKENDS:
            return DEFAULT_HELPER_MODEL
        if models is None:
            try:
                models = binaries.available_models()
            except Exception:
                models = []
        if DEFAULT_MODEL in models:
            return DEFAULT_MODEL
        return None

    def _coerce_stored_models(
        self, backend: UpscaleBackend, models: list[str] | None = None
    ) -> None:
        """一括・個別のすべてを、新しい実行先で選べる値に直す。"""
        for tab in (_IMAGE_TAB, _VIDEO_TAB):
            allowed = self._allowed_models(backend, tab, models)
            default = self._default_model_for(backend, models)
            if tab == _IMAGE_TAB:
                if self._image_model not in allowed:
                    self._image_model = default
                if backend in _HELPER_BACKENDS:
                    self._image_scale = 4
            else:
                if self._video_model not in allowed:
                    self._video_model = default
                if backend in _HELPER_BACKENDS:
                    self._video_scale = 4
        for jid, override in self._overrides.items():
            job = self._jobs.get(jid)
            tab = self._job_kind_tab(job) if job is not None else _IMAGE_TAB
            allowed = self._allowed_models(backend, tab, models)
            if override.get("model") not in allowed:
                override["model"] = self._default_model_for(backend, models)
            if backend in _HELPER_BACKENDS:
                override["scale"] = 4

    def _on_backend_changed(self, *_args) -> None:
        backend = self._selected_backend()
        # 従来モデル一覧の取得はここで1回だけ行い、下へ渡す
        models = None
        if backend not in _HELPER_BACKENDS:
            try:
                models = binaries.available_models()
            except Exception:
                models = []
        self._coerce_stored_models(backend, models)
        self._prune_redundant_overrides()
        self._refresh_model_options(models)
        if hasattr(self, "preview"):
            self.preview.refresh()

    def _fix_bulk_scale_if_unsupported(self, is_video: bool) -> None:
        """Vulkanで一括の倍率が対応外なら、対応する値に直す。"""
        if self._selected_backend() in _HELPER_BACKENDS:
            return
        model = self._video_model if is_video else self._image_model
        scale = int(self._video_scale if is_video else self._image_scale)
        if model in (None, "__missing__"):
            return
        try:
            supported = binaries.model_supports_scale(model, scale)
        except Exception:
            return
        if supported:
            return
        first = self._first_supported_scale(model)
        if first is None:
            return
        if is_video:
            self._video_scale = first
        else:
            self._image_scale = first

    def _on_header_image_model_changed(self, *_args) -> None:
        """ヘッダーの画像一括モデル変更。個別の無い画像はそれに従う。"""
        if self._syncing_settings:
            return
        self._image_model = self.image_model_combo.currentData()
        self._fix_bulk_scale_if_unsupported(False)
        self._prune_redundant_overrides()
        self._sync_settings_widgets()
        self._refresh_all_waiting()
        if hasattr(self, "preview"):
            self.preview.refresh()

    def _on_header_video_model_changed(self, *_args) -> None:
        """ヘッダーの動画一括モデル変更。個別の無い動画はそれに従う。"""
        if self._syncing_settings:
            return
        self._video_model = self.video_model_combo.currentData()
        self._fix_bulk_scale_if_unsupported(True)
        self._prune_redundant_overrides()
        self._sync_settings_widgets()
        self._refresh_all_waiting()
        if hasattr(self, "preview"):
            self.preview.refresh()

    def _on_header_interpolation_changed(self, *_args) -> None:
        """ヘッダーの一括補間変更。個別の無い動画はそれに従う。"""
        if self._syncing_settings:
            return
        self._video_interpolation = self.global_interpolation_combo.currentData()
        self._prune_redundant_overrides()
        self._sync_settings_widgets()
        self._refresh_all_waiting()
        if hasattr(self, "preview"):
            self.preview.refresh()

    def _on_model_changed(self, *_args) -> None:
        """右列の拡大モデル変更。そのファイルの個別設定を作って値を入れる。"""
        if self._syncing_settings:
            return
        job = self._selected_job()
        if job is None:
            return
        override = self._ensure_override(job)
        override["model"] = self.model_combo.currentData()
        self._prune_redundant_overrides([job.id])
        self._sync_settings_widgets()
        self._refresh_all_waiting()
        if hasattr(self, "preview"):
            self.preview.refresh()

    def _on_right_interpolation_changed(self, *_args) -> None:
        """右列の補間変更（動画のみ）。そのファイルの個別設定を作って入れる。"""
        if self._syncing_settings:
            return
        job = self._selected_job()
        if job is None or job.kind != JobKind.VIDEO:
            return
        override = self._ensure_override(job)
        override["interpolation"] = self.interpolation_combo.currentData()
        self._prune_redundant_overrides([job.id])
        self._sync_settings_widgets()
        self._refresh_all_waiting()
        if hasattr(self, "preview"):
            self.preview.refresh()

    def _on_output_changed(self, index: int) -> None:
        # index 1 = 「フォルダ選択…」
        if index == 1:
            d = QFileDialog.getExistingDirectory(self, t("出力先フォルダを選択"))
            if d:
                self._output_dir = d
                # 選択フォルダ名を項目テキストに反映
                self.output_combo.setItemText(1, t("📁 {name}", name=Path(d).name))
            else:
                # キャンセル時は「元の場所」に戻す
                self.output_combo.setCurrentIndex(0)
                self.output_combo.setItemText(1, t("フォルダ選択…"))
        # index 0 = 「元の場所」: 何もしない

    def _toggle_drawer(self) -> None:
        show = not self._drawer_open
        self._drawer_open = show
        if show:
            self.drawer.refresh_npu_rows()
            self.drawer.refresh_kit_rows()
        self._sync_workspace()
        self.settings_btn.setProperty("active", show)
        self.settings_btn.style().unpolish(self.settings_btn)
        self.settings_btn.style().polish(self.settings_btn)

    # ----------------------------------------------------- 設定の組み立て
    def build_settings(self, job: Job | None = None) -> UpscaleSettings:
        """UpscaleSettings を構築する。

        ジョブを渡せばそのファイル用の設定（個別があればそれ、無ければ
        種類の一括）を返す。引数なしは互換のため、選択中があればその
        ファイル用、無ければ画像の一括を返す。
        """
        s = UpscaleSettings()
        s.backend = self._selected_backend()
        if job is None:
            selected = self._selected_job()
            if selected is not None:
                job = selected
        if job is None:
            model = self._image_model
            scale = self._image_scale
            interpolation = self._video_interpolation
        else:
            model, scale, _interp, _individual = self._resolve_effective(job)
            if job.kind == JobKind.VIDEO:
                interpolation = _interp
            else:
                # 画像・フォルダに補間設定は無いが、欄の値（動画側の値）は
                # 設定に載せたままにする。従来の単一設定との互換のため。
                interpolation = self._video_interpolation
        if s.backend in _HELPER_BACKENDS and model is not None:
            scale = 4
        s.scale = scale
        # helperも表示ラベルではなく、選択肢の具体的なモデルキーを保存する。
        # 「なし」はアップスケール無効として扱い、前回のモデルキーを残さない。
        s.model = None if model in (None, "__missing__") else str(model)
        if s.backend in _HELPER_BACKENDS and s.model is not None:
            # 旧API利用者向けに系統値も併記するが、解決の主キーは s.model。
            s.model_family = helper_model_family(s.model)
        s.interpolation_model = (
            None if interpolation in (None, "__missing__") else str(interpolation)
        )
        # 出力先
        if self.output_combo.currentIndex() == 1 and self._output_dir:
            s.output_location = OutputLocation.CUSTOM
            s.output_dir = self._output_dir
        else:
            s.output_location = OutputLocation.SAME
            s.output_dir = None
        # 詳細設定ドロワーの値を反映
        self.drawer.apply_to(s)
        return s

    # ----------------------------------------------------------- 実行制御
    def _pending_jobs(self) -> list[Job]:
        """未処理（QUEUED / 過去エラー再実行除く）かつ未キャンセルのジョブ。"""
        out: list[Job] = []
        for jid in self._order:
            job = self._jobs.get(jid)
            if job is None:
                continue
            if job.status in (JobStatus.QUEUED, JobStatus.PROBING):
                ev = self._cancel_events.get(jid)
                if ev is None or not ev.is_set():
                    out.append(job)
        return out

    def _apply_current_settings(self, pending: list[Job]) -> UpscaleSettings:
        """開始時点の設定を保留ジョブごとに適用して返す。

        各ジョブには「個別設定があればそれ、無ければ種類の既定」が入る。
        戻り値はワーカーの予備設定（全ジョブに設定が入るため実質使わない）。
        """
        fallback: UpscaleSettings | None = None
        for job in pending:
            settings = self.build_settings(job)
            job.settings = replace(settings)
            if fallback is None:
                fallback = settings
            self.queue.refresh(job.id)
        if fallback is None:
            fallback = self.build_settings()
        return fallback

    def _on_start(self) -> None:
        if self._running:
            return
        if self._npu_converting:
            self._flash_hint(t("NPU の変換中は処理を始められません"))
            return
        pending = self._pending_jobs()
        if not pending:
            self._flash_hint(t("処理するファイルがありません。"))
            return

        # 開始前にファイルごとの設定を確認する（個別はそのファイル名で案内）
        for job in pending:
            model, _scale, interpolation, individual = self._resolve_effective(job)
            is_video = job.kind == JobKind.VIDEO
            if is_video:
                missing = model is None and interpolation is None
            else:
                missing = model is None
            if not missing:
                continue
            if individual:
                self._flash_hint(
                    t("{name}の拡大モデルを選んでください。", name=job.name)
                )
            elif is_video:
                self._flash_hint(t("動画の拡大モデルかフレーム補間モデルを選んでください。"))
            else:
                self._flash_hint(t("画像の拡大モデルを選んでください。"))
            return

        # NPU 未変換の確認（ファイルごとの設定で判定。
        # 短辺不足で GPU に自動切替する入力は止めない）。
        for job in pending:
            job_settings = self.build_settings(job)
            if job_settings.backend != UpscaleBackend.NPU_NATIVE:
                continue
            if job_settings.model is None:
                continue
            if (
                job.width and job.height
                and helper_backend.effective_backend(
                    job_settings.backend, job.width, job.height
                ) != UpscaleBackend.NPU_NATIVE
            ):
                continue
            if npu_prepare.is_converted(str(job_settings.model)):
                continue
            label = t(_MODEL_LABELS.get(
                str(job_settings.model), str(job_settings.model)))
            self._flash_hint(npu_prepare.not_converted_message(label))
            return

        settings = self._apply_current_settings(pending)
        self._pause.clear()

        # ワーカーを別スレッドへ
        self._thread = QThread(self)
        self._worker = QueueWorker(
            pending, settings, self._cancel_events, self._pause
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.job_done.connect(self._on_job_done)
        self._worker.job_error.connect(self._on_job_error)
        self._worker.job_canceled.connect(self._on_job_canceled)
        self._worker.queue_finished.connect(self._on_queue_finished)
        self._thread.start()

        self._set_running(True)
        if hasattr(self, "preview"):
            self.preview.set_main_running(True)

    def _on_trial_running_changed(self, running: bool) -> None:
        """試し中は開始を無効にする。終了待ちなら試し完了後に閉じる。"""
        if not self._running:
            self.start_btn.setEnabled(
                not running and not self._npu_converting)
        self._update_npu_busy()
        if (not running and self._closing and not self._running
                and not self._npu_converting):
            self.preview.shutdown()
            self.close()

    def _update_npu_busy(self) -> None:
        """変換ボタンの案内用に本処理・試しの実行状態をドロワーへ伝える。"""
        if hasattr(self, "drawer") and hasattr(self, "preview"):
            self.drawer.set_external_busy(
                self._running or self.preview.is_trial_running())

    def _on_npu_converting_changed(self, converting: bool) -> None:
        """変換中は開始・試しを止め、終わったら未変換表示を更新する。"""
        self._npu_converting = bool(converting)
        if hasattr(self, "preview"):
            self.preview.set_npu_converting(self._npu_converting)
        if not converting:
            self._update_model_info()
        if not self._running:
            self.start_btn.setEnabled(
                not self._npu_converting
                and not self.preview.is_trial_running())
        if (not converting and self._closing and not self._running
                and not self.preview.is_trial_running()):
            self.preview.shutdown()
            self.close()

    def _on_pause(self) -> None:
        """一時停止: 現在ジョブ完了後にワーカーを抜けさせる。"""
        if self._running:
            self._pause.set()
            self.pause_btn.setEnabled(False)
            self.pause_btn.setText(t("停止中…"))
            # 動画1本の途中では長時間効かないため、即時中止の手段を案内する
            self._flash_hint(
                t("現在のジョブ完了後に停止します。今すぐ中止するには行の × を押してください。")
            )

    def _set_running(self, running: bool) -> None:
        self._running = running
        self.start_btn.setEnabled(
            not running and not getattr(self, "_npu_converting", False))
        self.start_btn.setText(t("処理中…") if running else t("開始"))
        self.pause_btn.setEnabled(running)
        self.pause_btn.setText(t("一時停止"))
        # 実行中は入力系をロック（ヘッダー・右列のモデル/倍率/出力先/追加）
        for w in (self.backend_combo, self.image_model_combo,
                  self.video_model_combo, self.global_interpolation_combo,
                  self.model_combo, self.interpolation_combo, self.output_combo,
                  self.clear_btn, self.settings_btn,
                  self.drop_zone, self.add_zone, self.retry_all_btn,
                  self.apply_all_btn, self.reset_override_btn):
            try:
                w.setEnabled(not running)
            except RuntimeError:
                pass
        self.drawer.setEnabled(not running)
        for btn in self._scale_btns.values():
            btn.setEnabled(not running)
        self.queue.set_retry_locked(running)
        if hasattr(self, "preview"):
            self.preview.set_main_running(running)
        self._update_npu_busy()
        # ボタン有効範囲はモデル依存のため載せ替えで戻す
        self._sync_settings_widgets()
        self._update_retry_all()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        # 右列題名の省略幅を枠に合わせる
        try:
            job = self._selected_job()
            if job is not None and hasattr(self, "_settings_title"):
                self._update_settings_title(job.name)
        except RuntimeError:
            pass

    # ------------------------------------------------- 行の表示とやり直し
    def _describe_settings(self, settings: UpscaleSettings) -> str:
        """行の小さな説明に出す設定の内訳。内部の名前ではなく、画面の表示名で作る。"""
        backend_value = getattr(settings.backend, "value", str(settings.backend))
        backend_names = {value: label for label, value in _BACKEND_OPTIONS}
        backend_label = backend_names.get(backend_value)
        backend = t(backend_label) if backend_label else backend_value
        if settings.model is None:
            upscale = t("なし")
        else:
            model_label = _MODEL_LABELS.get(settings.model)
            upscale = "{}・{}x".format(
                t(model_label) if model_label else settings.model, settings.scale
            )
        interpolation = settings.interpolation_model
        if interpolation is None:
            interpolation = t("なし")
        elif interpolation == "rife-v4.6":
            interpolation = t("RIFE v4.6")
        elif interpolation == binaries.FILM_MODEL:
            interpolation = t("FILM (Style)")
        return t(
            "AI実行先: {backend}\nアップスケール: {upscale}\nフレーム補間: {interpolation}",
            backend=backend,
            upscale=upscale,
            interpolation=interpolation,
        )

    def _waiting_text(self, job: Job) -> str:
        """待機中の行に出す「何で処理されるか」の表示。"""
        model, scale, interpolation, individual = self._resolve_effective(job)
        suffix = t("（個別）") if individual else ""
        if model is not None:
            label = t(_MODEL_LABELS.get(model, model))
            if interpolation is not None:
                return t(
                    "{model}・{scale}x・補間あり{suffix}",
                    model=label,
                    scale=scale,
                    suffix=suffix,
                )
            return t(
                "{model}・{scale}x{suffix}",
                model=label,
                scale=scale,
                suffix=suffix,
            )
        if interpolation is not None:
            return t("補間のみ{suffix}", suffix=suffix)
        return t("モデル未選択{suffix}", suffix=suffix)

    def _done_text(self, job: Job) -> str:
        """完了した行に出す表示（使った設定で決まる）。"""
        settings = job.settings
        if settings is not None and settings.model is not None:
            return t(
                "完了・{model}",
                model=t(_MODEL_LABELS.get(settings.model, settings.model)),
            )
        if settings is not None and settings.interpolation_model is not None:
            return t("完了・補間のみ")
        return t("完了")

    def _refresh_all_waiting(self) -> None:
        """待機中の行の表示を、いまの既定・個別・実行先に合わせる。"""
        if not hasattr(self, "queue"):
            return
        for jid in self._order:
            job = self._jobs.get(jid)
            if job is None or job.status != JobStatus.QUEUED:
                continue
            job.message = self._waiting_text(job)
            self.queue.refresh(jid)

    def _reset_job_to_queued(self, job: Job) -> None:
        """完了・エラー・キャンセルの行を待機中に戻す。"""
        job.status = JobStatus.QUEUED
        job.progress = 0.0
        job.output_path = None
        job.error = ""
        job.settings = None
        job.message = self._waiting_text(job)
        self._cancel_events[job.id] = threading.Event()
        self.queue.refresh(job.id)

    def _on_retry_requested(self, job_id: int) -> None:
        if self._running:
            return
        job = self._jobs.get(job_id)
        if job is None:
            return
        if job.status not in (JobStatus.DONE, JobStatus.ERROR, JobStatus.CANCELED):
            return
        self._reset_job_to_queued(job)
        self._update_retry_all()

    def _on_retry_all(self) -> None:
        """完了・エラー・キャンセルの全行を待機中に戻す。"""
        if self._running:
            return
        for jid in list(self._order):
            job = self._jobs.get(jid)
            if job is None:
                continue
            if job.status not in (JobStatus.DONE, JobStatus.ERROR, JobStatus.CANCELED):
                continue
            self._reset_job_to_queued(job)
        self._update_retry_all()

    def _update_retry_all(self) -> None:
        """「すべてやり直す」は対象行があるときだけ有効（処理中は押せない）。"""
        if not hasattr(self, "retry_all_btn"):
            return
        has_terminal = any(
            job.status in (JobStatus.DONE, JobStatus.ERROR, JobStatus.CANCELED)
            for job in self._jobs.values()
        )
        self.retry_all_btn.setEnabled(bool(has_terminal) and not self._running)

    # ----------------------------------------------------- ワーカースロット
    def _on_progress(self, job_id: int, frac: float, msg: str) -> None:
        job = self._jobs.get(job_id)
        if job is None:
            return
        self._current_job_id = job_id
        job.status = JobStatus.RUNNING
        job.progress = max(0.0, min(1.0, frac))
        if msg:
            job.message = msg
        row = self.queue.row(job_id)
        if row is not None:
            row.set_busy_icon(True)
        self.queue.refresh(job_id)

    def _on_job_done(self, job_id: int, out_path: str) -> None:
        job = self._jobs.get(job_id)
        if job is None:
            return
        job.status = JobStatus.DONE
        job.progress = 1.0
        job.output_path = Path(out_path)
        job.message = self._done_text(job)
        self.queue.refresh(job_id)
        self._update_retry_all()

    def _on_job_error(self, job_id: int, message: str) -> None:
        job = self._jobs.get(job_id)
        if job is None:
            return
        job.status = JobStatus.ERROR
        job.error = message
        job.message = t("エラー: {message}", message=message)
        self.queue.refresh(job_id)
        self._update_retry_all()

    def _on_job_canceled(self, job_id: int) -> None:
        job = self._jobs.get(job_id)
        if job is None:
            return
        job.status = JobStatus.CANCELED
        job.message = t("キャンセルされました")
        self.queue.refresh(job_id)
        self._update_retry_all()

    def _on_queue_finished(self) -> None:
        # スレッド後始末
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait()
            self._thread.deleteLater()
        self._worker = None
        self._thread = None
        self._current_job_id = None
        self._set_running(False)
        if (self._closing and not self.preview.is_trial_running()
                and not self._npu_converting):
            self.preview.shutdown()
            self.close()
            return
        if self._pause.is_set():
            self._flash_hint(t("一時停止しました。「開始」で再開できます。"))
            self._pause.clear()
        if self._closing:
            # 終了待ちだった → ワーカー停止が完了したのでウィンドウを閉じる
            self.close()

    # ------------------------------------------------------------ 行削除
    def _on_remove_requested(self, job_id: int) -> None:
        job = self._jobs.get(job_id)
        if job is None:
            return
        if self._running and job.status == JobStatus.RUNNING:
            # 処理中ジョブ → cancel イベントをセット（engine が Cancelled を投げる）
            ev = self._cancel_events.get(job_id)
            if ev is not None:
                ev.set()
            job.message = t("キャンセル中…")
            self.queue.refresh(job_id)
            return
        # 未処理/完了済み → 行ごと削除
        ev = self._cancel_events.get(job_id)
        if ev is not None:
            ev.set()  # 念のため（開始前キャンセル扱い）
        # 選択の移動先を一覧が正しく引けるよう、台帳から先に外す
        self._jobs.pop(job_id, None)
        self._cancel_events.pop(job_id, None)
        self._overrides.pop(job_id, None)
        if job_id in self._order:
            self._order.remove(job_id)
        self.queue.remove_job(job_id)
        self.preview.discard_file(str(job.input_path))
        self._update_retry_all()
        self._sync_workspace()

    def _clear_queue(self) -> None:
        if self._running:
            return
        ids = list(self._order)
        self._jobs.clear()
        self._order.clear()
        self._cancel_events.clear()
        self._overrides.clear()
        for jid in ids:
            self.queue.remove_job(jid)
        self.preview.discard_all()
        self._update_retry_all()
        self._sync_workspace()

    # -------------------------------------------------------------- 補助
    def _flash_hint(self, text: str) -> None:
        """開始ボタンの上に状況メッセージを出す（ウィンドウタイトルにも併記）。"""
        self.setWindowTitle(t("ultraeasy-upscaler — {text}", text=text))
        self.status_label.setText(text)

    def _launch_explorer(self, args: list[str]) -> None:
        """エクスプローラーを起動する（テストで差し替え可能）。"""
        subprocess.Popen(args)

    def _on_row_output_requested(self, job_id: int) -> None:
        """完了した行の保存先ボタンを押したときの処理（実行中でも押せる）。"""
        job = self._jobs.get(job_id)
        if job is None:
            return
        output = Path(job.output_path) if job.output_path else None
        if output is not None and output.is_file():
            self._launch_explorer(["explorer", "/select,", str(output)])
            return
        folder = output.parent if output is not None else None
        if folder is not None and folder.is_dir():
            self._launch_explorer(["explorer", str(folder)])
            return
        # フォルダも無ければ状況行に案内を出す（既存の出力先の文言を流用）。
        missing = str(folder) if folder is not None else str(job.input_path)
        self._flash_hint(t("出力先: {path}", path=missing))

    def closeEvent(self, event) -> None:  # noqa: N802
        # 実行中は QThread 走行中の破棄（クラッシュ要因）を避けるため、即閉じない。
        # キャンセル要求 + 一時停止だけ行い、ワーカー完了(queue_finished)後に閉じる。
        # 変換中も同じ扱い（中止してから閉じる）。
        trial_running = (
            self.preview.is_trial_running() if hasattr(self, "preview") else False
        )
        converting = (
            self.drawer.is_converting() if hasattr(self, "drawer") else False
        )
        if self._running or trial_running or converting:
            self._closing = True
            self._pause.set()
            for ev in self._cancel_events.values():
                ev.set()
            if hasattr(self, "preview") and self.preview.is_trial_running():
                self.preview.cancel_trial()
            if converting and hasattr(self, "drawer"):
                self.drawer.cancel_conversion()
            self._flash_hint(t("終了処理中… 現在の処理を停止しています"))
            event.ignore()
            return
        if hasattr(self, "preview"):
            self.preview.shutdown()
        super().closeEvent(event)


def run(argv: list[str]) -> int:
    """GUI エントリポイント。app/main.py から呼ばれる。"""
    app = QApplication.instance() or QApplication(argv)
    apply_theme(app)
    win = MainWindow()
    win.show()
    return app.exec()

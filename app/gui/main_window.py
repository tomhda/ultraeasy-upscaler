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

import threading
from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import QSize, Qt, QThread, QUrl
from PySide6.QtGui import QDesktopServices, QDragEnterEvent, QDropEvent
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

from app.core import binaries
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

from .compare_view import TrialPanel
from .drop_zone import DropZone
from .icons import Icon, apply_icon_font, make_icon
from .queue_view import QueueView
from .settings_drawer import SettingsDrawer
from . import theme
from .theme import apply_theme
from .worker import QueueWorker

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
    "realesrgan-x4plus": "Real-ESRGAN",
    "realesrgan-x4plus-anime": "Real-ESRGAN Anime",
    "realesr-animevideov3": "Anime Video v3",
    HELPER_MODEL_ANIME: "Anime Video v3",
    "realesr-general-x4v3": "General Video v3（ノイズ除去強）",
    "realesr-general-wdn-x4v3": "General Video v3（ノイズ除去弱）",
    HELPER_MODEL_SPAN: "4xNomosUni SPAN",
    HELPER_MODEL_AMD_RRDB: "Real-ESRGAN（AMD縮小版）",
    HELPER_MODEL_SWINIR: "SwinIR-M",
    HELPER_MODEL_ADCSR: "AdcSR",
}
_HELPER_MODEL_OPTIONS = [
    ("なし（拡大しない）", None),
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
    ("なし（拡大しない）", None),
    ("SwinIR-M（real-world x4）", HELPER_MODEL_SWINIR),
]

# 説明行に出す文。モデルの向き不向き → 速さ → 実行先ごとの注意、の順に並べる。
_MODEL_DESC = {
    HELPER_MODEL_ANIME: "アニメ向け。線をくっきり仕上げます。実写には向きません。",
    "realesr-animevideov3": "アニメ向け。線をくっきり仕上げます。実写には向きません。",
    HELPER_MODEL_SPAN: "実写向け。元の質感を残して自然に仕上げます。",
    HELPER_MODEL_AMD_RRDB: "実写向け。輪郭をくっきり仕上げます。",
    HELPER_MODEL_SWINIR: "実写向け。細部まで丁寧に仕上げます。",
    HELPER_MODEL_ADCSR: "実写の静止画向け。いちばん高画質です。動画には使えません。",
    "realesrgan-x4plus": "実写向け。高画質です。",
    "realesrgan-x4plus-anime": "アニメ向け。高画質です。",
    "realesr-general-x4v3": "実写・アニメ兼用。ノイズを強めに消します。",
    "realesr-general-wdn-x4v3": "実写向け。ノイズ消しは控えめです。",
}
_SPEED_TEXT = {
    "◎": "処理は速いです。",
    "○": "速さはふつうです。",
    "△": "少し時間がかかります。",
    "✕": "かなり時間がかかります。",
    "極遅": "動画は1秒ぶんに数十秒かかる超低速です。",
}
_BACKEND_NOTE = {
    UpscaleBackend.NPU_NATIVE: "NPUで処理するので、GPUを空けたまま他の作業ができます。",
    UpscaleBackend.NPU: "NPUで処理するので、GPUを空けたまま他の作業ができます。",
    UpscaleBackend.SWINIR_CUDA: "NVIDIAのGPUが必要です。",
    UpscaleBackend.VULKAN: "処理中はPCが熱くなり、他の作業が重くなります。",
}

# (backend, model) → (速度, 画質, アニメ適性, 実写適性, 推奨タグ or None)
_MODEL_INFO: dict[tuple[UpscaleBackend, str],
                  tuple[str, str, str, str, str | None]] = {
    (UpscaleBackend.VULKAN, "realesr-animevideov3"):
        ("◎", "◎", "◎", "△", "アニメ"),
    (UpscaleBackend.VULKAN, "realesr-general-x4v3"):
        ("◎", "○", "○", "○", None),
    (UpscaleBackend.VULKAN, "realesr-general-wdn-x4v3"):
        ("◎", "○", "○", "◎", "実写"),
    (UpscaleBackend.VULKAN, "realesrgan-x4plus"):
        ("✕", "◎", "○", "◎", None),
    (UpscaleBackend.VULKAN, "realesrgan-x4plus-anime"):
        ("✕", "◎", "◎", "○", None),
    (UpscaleBackend.NPU, "realesrgan-x4plus"):
        ("◎", "◎", "○", "◎", "実写"),
    (UpscaleBackend.NPU, "realesrgan-x4plus-anime"):
        ("○", "◎", "◎", "○", None),
    (UpscaleBackend.NPU, "realesr-animevideov3"):
        ("◎", "◎", "◎", "△", "アニメ"),
    (UpscaleBackend.WINML_GPU, HELPER_MODEL_ANIME):
        ("◎", "◎", "◎", "△", "アニメ"),
    (UpscaleBackend.WINML_GPU, HELPER_MODEL_SPAN):
        ("◎", "◎", "○", "◎", "実写"),
    (UpscaleBackend.WINML_GPU, HELPER_MODEL_AMD_RRDB):
        ("△", "◎", "○", "◎", None),
    (UpscaleBackend.WINML_GPU, HELPER_MODEL_SWINIR):
        ("✕", "◎", "○", "◎", "静止画"),
    (UpscaleBackend.WINML_GPU, HELPER_MODEL_ADCSR):
        ("✕", "◎◎", "○", "◎", "実写"),
    (UpscaleBackend.NPU_NATIVE, HELPER_MODEL_ADCSR):
        ("✕", "◎◎", "○", "◎", "実写"),
    (UpscaleBackend.NPU_NATIVE, HELPER_MODEL_ANIME):
        ("○", "◎", "◎", "△", "アニメ"),
    (UpscaleBackend.NPU_NATIVE, HELPER_MODEL_SPAN):
        ("◎", "◎", "○", "◎", "実写"),
    (UpscaleBackend.NPU_NATIVE, HELPER_MODEL_AMD_RRDB):
        ("△", "◎", "○", "◎", None),
    (UpscaleBackend.NPU_NATIVE, HELPER_MODEL_SWINIR):
        ("✕", "◎", "○", "◎", "静止画"),
    (UpscaleBackend.SWINIR_CUDA, HELPER_MODEL_SWINIR):
        ("極遅", "◎◎", "△", "◎", "実写・再開可"),
}


class ModelCombo(QComboBox):
    """閉じた状態ではモデル名だけを見せるコンボ（特性の印は開いた一覧に出す）。"""

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QStylePainter(self)
        opt = QStyleOptionComboBox()
        self.initStyleOption(opt)
        opt.currentText = opt.currentText.split("｜")[0]
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

        self._scale = 4  # 現在の倍率（既定 4x）
        self._build()
        self._refresh_scale_enabled()
        self._on_interpolation_changed()

    # ------------------------------------------------------------------ UI
    def _build(self) -> None:
        self.setAcceptDrops(True)  # ウィンドウのどこに落としても追加できる
        outer = QVBoxLayout(self)
        outer.setContentsMargins(20, 16, 20, 16)
        outer.setSpacing(12)

        # AI実行先は詳細設定ドロワーにあるコンボをそのまま使う
        self.drawer = SettingsDrawer()
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
            model_label=lambda key: _MODEL_LABELS.get(key, key),
        )
        self.preview.trial_running_changed.connect(self._on_trial_running_changed)
        work_row.addWidget(self.preview, 1)

        # 詳細設定は歯車で左＋中央の領域に開く（右列の設定と開始は残す）
        drawer_scroll = QScrollArea()
        drawer_scroll.setWidgetResizable(True)
        drawer_scroll.setFrameShape(QFrame.Shape.NoFrame)
        drawer_scroll.setWidget(self.drawer)

        self.stack = QStackedWidget()
        self.stack.addWidget(self.drop_zone)
        self.stack.addWidget(workspace)
        self.stack.addWidget(drawer_scroll)
        body.addWidget(self.stack, 1)

        body.addWidget(self._build_settings_panel())

        self.queue.selectionChanged.connect(self._on_selection_changed)
        self.backend_combo.currentIndexChanged.connect(self._on_backend_changed)
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        self._refresh_model_options()

        self._apply_icons()
        theme.notifier.changed.connect(self._apply_icons)

    def _apply_icons(self) -> None:
        """アイコンを現在の配色で描き直す（QSS では色を変えられないため）。"""
        p = theme.current()
        self.output_open_btn.setIcon(make_icon(Icon.FOLDER, 22, p.text_soft))
        self.settings_btn.setIcon(make_icon(Icon.SETTINGS, 24, p.text_soft))
        self.pause_btn.setIcon(make_icon(Icon.PAUSE, 20, p.text_soft))
        self.start_btn.setIcon(make_icon(Icon.PLAY, 26, p.on_accent))
        self.clear_btn.setIcon(make_icon(Icon.DELETE, 18, p.text_dim))

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
        row.addStretch(1)

        self.output_open_btn = QPushButton("出力先を開く")
        self.output_open_btn.setObjectName("toolbarButton")
        self.output_open_btn.setIconSize(QSize(22, 22))
        self.output_open_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.output_open_btn.clicked.connect(self._open_output_folder)
        row.addWidget(self.output_open_btn)

        self.settings_btn = QPushButton("")
        self.settings_btn.setObjectName("iconButton")
        self.settings_btn.setIconSize(QSize(24, 24))
        self.settings_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.settings_btn.setToolTip("詳細設定")
        self.settings_btn.clicked.connect(self._toggle_drawer)
        row.addWidget(self.settings_btn)

        return header

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
        """右列: モデルなどの設定と、開始/一時停止。"""
        panel = QFrame()
        panel.setObjectName("controlPanel")
        panel.setFixedWidth(_SETTINGS_COL_WIDTH)
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(16, 14, 16, 16)
        outer.setSpacing(12)

        title = QLabel("設定")
        title.setObjectName("sectionTitle")
        outer.addWidget(title)

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

        self.model_combo = ModelCombo()
        self._compact(self.model_combo)
        col.addLayout(self._field("モデル", self.model_combo))

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
            btn.setChecked(s == self._scale)
            btn.clicked.connect(lambda _=False, v=s: self._set_scale(v))
            self._scale_btns[s] = btn
            scale_box.addWidget(btn, 1)
        scale_wrap = QWidget()
        scale_wrap.setObjectName("scaleWrap")
        scale_wrap.setLayout(scale_box)
        col.addLayout(self._field("倍率", scale_wrap))

        # フレーム補間モデル（アップスケールとは独立）
        self.interpolation_combo = QComboBox()
        self.interpolation_combo.addItem("なし（補間しない）", None)
        for model in binaries.available_interpolation_models():
            label = "RIFE v4.6" if model == "rife-v4.6" else model
            self.interpolation_combo.addItem(label, model)
        if self.interpolation_combo.count() == 1:
            self.interpolation_combo.addItem("モデル未検出", "__missing__")
            self.interpolation_combo.model().item(1).setEnabled(False)
        self.interpolation_combo.currentIndexChanged.connect(
            lambda _i: self._on_interpolation_changed()
        )
        col.addLayout(
            self._field("フレーム補間モデル", self._compact(self.interpolation_combo))
        )

        # 出力先
        self.output_combo = QComboBox()
        self.output_combo.addItems(["元の場所", "フォルダ選択…"])
        self.output_combo.activated.connect(self._on_output_changed)
        self._output_dir: str | None = None
        col.addLayout(self._field("出力先", self._compact(self.output_combo)))
        col.addStretch(1)

        self.status_label = QLabel("")
        self.status_label.setObjectName("hint")
        self.status_label.setWordWrap(True)
        outer.addWidget(self.status_label)

        self.pause_btn = QPushButton("一時停止")
        self.pause_btn.setIconSize(QSize(20, 20))
        self.pause_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.pause_btn.setEnabled(False)
        self.pause_btn.setToolTip("現在のジョブ完了後に停止します")
        self.pause_btn.clicked.connect(self._on_pause)
        outer.addWidget(self.pause_btn)

        self.start_btn = QPushButton("開始")
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
        lay = QVBoxLayout(card)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(8)

        head = QHBoxLayout()
        title = QLabel("メディア")
        title.setObjectName("sectionTitle")
        head.addWidget(title)
        head.addStretch(1)
        self.clear_btn = QPushButton("すべて削除")
        self.clear_btn.setObjectName("link")
        self.clear_btn.setIconSize(QSize(18, 18))
        self.clear_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.clear_btn.clicked.connect(self._clear_queue)
        head.addWidget(self.clear_btn)
        lay.addLayout(head)

        self.queue = QueueView()
        self.queue.removeRequested.connect(self._on_remove_requested)
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
        self.preview.show_job(self._jobs.get(job_id) if job_id is not None else None)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802
        if event.mimeData().hasUrls() and not self._running:
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
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
        # 設定はここでは固定しない。「開始」時点のUI設定が全保留ジョブに
        # 一括適用される（追加後にモデルを変えても反映されるように）。
        self._jobs[job.id] = job
        self._order.append(job.id)
        self._cancel_events[job.id] = threading.Event()
        self.queue.add_job(job)
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
                f"{added} 件追加 / {len(skipped)} 件は未対応のためスキップ: "
                + ", ".join(skipped[:3]) + ("…" if len(skipped) > 3 else "")
            )

    def _on_paths_dropped(self, paths: list[str]) -> None:
        self.add_paths(paths)

    def _pick_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self, "ファイルを選択（複数可）", "",
            "対応ファイル (*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff "
            "*.mp4 *.mkv *.mov *.avi *.webm *.m4v *.wmv *.flv *.mpg *.mpeg *.ts *.m2ts);;"
            "すべて (*.*)",
        )
        if files:
            self.add_paths(files)

    def _pick_folder(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "フォルダを選択")
        if d:
            self.add_paths([d])

    # ------------------------------------------------------- 倍率/モデル
    def _set_scale(self, value: int) -> None:
        self._scale = value
        for s, btn in self._scale_btns.items():
            btn.setChecked(s == value)

    def _refresh_scale_enabled(self) -> None:
        """バックエンドごとにモデルと倍率の選択可能範囲を更新する。"""
        backend = self._selected_backend()
        model = self.model_combo.currentData()
        self.backend_combo.setEnabled(not self._running)
        self.model_combo.setEnabled(not self._running)

        if backend in _HELPER_BACKENDS:
            # 「なし」はアップスケールを無効にするだけで、コンボ自体は
            # 有効のままにして別のモデルへ戻せるようにする。
            self.model_combo.setEnabled(not self._running)
            self._set_scale(4)
            for scale, button in self._scale_btns.items():
                button.setEnabled(
                    model in _HELPER_MODEL_VALUES
                    and scale == 4
                    and not self._running
                )
            self._update_model_info()
            return

        upscale_enabled = model not in (None, "__missing__")
        if not upscale_enabled:
            for button in self._scale_btns.values():
                button.setEnabled(False)
            self._update_model_info()
            return

        first_enabled: int | None = None
        for scale, button in self._scale_btns.items():
            try:
                supported = binaries.model_supports_scale(model, scale)
            except Exception:
                supported = True
            button.setEnabled(supported and not self._running)
            if supported and first_enabled is None:
                first_enabled = scale
        current = self._scale_btns.get(self._scale)
        if current is not None and not current.isEnabled() and first_enabled is not None:
            self._set_scale(first_enabled)
        self._update_model_info()

    def _refresh_model_options(self) -> None:
        """バックエンドに応じてモデル欄を再構成する。

        新AIバックエンドでは旧Vulkan資産を列挙せず、具体的な3モデルを表示する。
        Vulkanを選んだときだけ vendor/realesrgan の従来モデルを表示する。
        """
        backend = self._selected_backend()
        if backend in _HELPER_BACKENDS:
            selected = self.model_combo.currentData()
            options = (
                _SWINIR_CUDA_MODEL_OPTIONS
                if backend == UpscaleBackend.SWINIR_CUDA
                else _HELPER_MODEL_OPTIONS
            )
            allowed = {value for _label, value in options}
            # 初回（項目未構築）だけ具体的な既定モデルを使う。
            # 既に「なし」が選択されている場合は、バックエンド切替時にも
            # その明示的な選択を維持する。
            default_model = (
                HELPER_MODEL_SWINIR
                if backend == UpscaleBackend.SWINIR_CUDA
                else DEFAULT_HELPER_MODEL
            )
            if self.model_combo.count() == 0:
                selected = default_model
            elif selected is not None and selected not in allowed:
                selected = default_model
            self._replace_model_items(options, selected)
            self._refresh_scale_enabled()
            return

        selected = self.model_combo.currentData()
        models = binaries.available_models()
        options: list[tuple[str, object]] = [("なし（拡大しない）", None)]
        options.extend((_MODEL_LABELS.get(model, model), model) for model in models)
        if not models:
            options.append(("モデル未検出", "__missing__"))
        elif selected is not None and selected not in models:
            selected = DEFAULT_MODEL if DEFAULT_MODEL in models else None
        self._replace_model_items(options, selected)
        if not models:
            item = self.model_combo.model().item(self.model_combo.count() - 1)
            if item is not None:
                item.setEnabled(False)
        self._refresh_scale_enabled()

    def _replace_model_items(
        self, options: list[tuple[str, object]], selected: object | None
    ) -> None:
        """モデルコンボの項目を差し替え、可能なら選択値を維持する。"""
        previous = self.model_combo.blockSignals(True)
        try:
            self.model_combo.clear()
            for label, value in options:
                self.model_combo.addItem(label, value)
            if selected is not None:
                index = self.model_combo.findData(selected)
                if index >= 0:
                    self.model_combo.setCurrentIndex(index)
            if self.model_combo.currentIndex() < 0 and self.model_combo.count():
                self.model_combo.setCurrentIndex(0)
        finally:
            self.model_combo.blockSignals(previous)

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
        parts = [f"速度{speed}", f"画質{quality}", f"アニメ{anime}・実写{live}"]
        if star:
            parts.append(f"★{star}に推奨")
        return "／".join(parts)

    @staticmethod
    def _compose_hint_line(backend: UpscaleBackend, data: str) -> str:
        info = _MODEL_INFO.get((backend, data))
        speed = _SPEED_TEXT.get(info[0], "") if info else ""
        return _MODEL_DESC.get(data, "") + speed + _BACKEND_NOTE.get(backend, "")

    def _update_model_info(self) -> None:
        """モデルコンボのバッジ（速度/画質/★推奨）と、選択中構成の説明行を更新する。"""
        if not hasattr(self, "model_hint"):
            return
        backend = self._selected_backend()
        item_model = self.model_combo.model()
        for i in range(self.model_combo.count()):
            data = self.model_combo.itemData(i)
            if data in (None, "__missing__"):
                continue
            base = _MODEL_LABELS.get(data, data)
            info = _MODEL_INFO.get((backend, data))
            item = item_model.item(i)
            if info is None:
                self.model_combo.setItemText(i, base)
                if item is not None:
                    item.setToolTip("")
                continue
            speed, quality, _anime, _live, star = info
            badge = f"速度{speed} 画質{quality}" + (f" ★{star}" if star else "")
            self.model_combo.setItemText(i, f"{base}｜{badge}")
            if item is not None:
                item.setToolTip(self._compose_model_hint(backend, data))
        # 閉じた状態はコンパクト幅のままでよいが、開いたリストは全文が
        # 収まる幅へ広げる（切れて読めない問題の対策）
        view = self.model_combo.view()
        fm = view.fontMetrics()
        widest = max((fm.horizontalAdvance(self.model_combo.itemText(i))
                      for i in range(self.model_combo.count())), default=0)
        view.setMinimumWidth(widest + 48)

        cur = self.model_combo.currentData()
        if cur in (None, "__missing__"):
            self.model_hint.setText("拡大はしません。フレーム補間だけ実行できます。")
            return
        self.model_hint.setText(self._compose_hint_line(backend, cur))

    def _selected_backend(self) -> UpscaleBackend:
        value = self.backend_combo.currentData() if hasattr(self, "backend_combo") else None
        if value == "auto":
            return UpscaleBackend.WINML_GPU
        try:
            return UpscaleBackend(value or UpscaleBackend.WINML_GPU.value)
        except ValueError:
            return UpscaleBackend.WINML_GPU

    def _on_backend_changed(self, *_args) -> None:
        self._refresh_model_options()
        self._refresh_scale_enabled()
        if hasattr(self, "preview"):
            self.preview.refresh()

    def _on_model_changed(self, *_args) -> None:
        self._refresh_scale_enabled()
        if hasattr(self, "preview"):
            self.preview.refresh()

    def _on_interpolation_changed(self) -> None:
        model = self.interpolation_combo.currentData()
        enabled = model not in (None, "__missing__")
        self.drawer.set_interpolation_enabled(enabled)

    def _on_output_changed(self, index: int) -> None:
        # index 1 = 「フォルダ選択…」
        if index == 1:
            d = QFileDialog.getExistingDirectory(self, "出力先フォルダを選択")
            if d:
                self._output_dir = d
                # 選択フォルダ名を項目テキストに反映
                self.output_combo.setItemText(1, f"📁 {Path(d).name}")
            else:
                # キャンセル時は「元の場所」に戻す
                self.output_combo.setCurrentIndex(0)
                self.output_combo.setItemText(1, "フォルダ選択…")
        # index 0 = 「元の場所」: 何もしない

    def _toggle_drawer(self) -> None:
        show = not self._drawer_open
        self._drawer_open = show
        self._sync_workspace()
        self.settings_btn.setProperty("active", show)
        self.settings_btn.style().unpolish(self.settings_btn)
        self.settings_btn.style().polish(self.settings_btn)

    # ----------------------------------------------------- 設定の組み立て
    def build_settings(self) -> UpscaleSettings:
        """現在のウィジェット値から UpscaleSettings を構築する。"""
        s = UpscaleSettings()
        s.backend = self._selected_backend()
        s.scale = self._scale
        model = self.model_combo.currentData()
        if s.backend in _HELPER_BACKENDS:
            # helperも表示ラベルではなく、選択肢の具体的なモデルキーを保存する。
            # 「なし」はアップスケール無効として扱い、前回のモデルキーを残さない。
            s.model = None if model in (None, "__missing__") else str(model)
            if s.model is not None:
                # 旧API利用者向けに系統値も併記するが、解決の主キーは s.model。
                s.model_family = helper_model_family(s.model)
        else:
            s.model = None if model in (None, "__missing__") else str(model)
        interpolation = self.interpolation_combo.currentData()
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
        """開始時点のUI設定を保留ジョブすべてに適用して返す。"""
        settings = self.build_settings()
        for job in pending:
            job.settings = replace(settings)
            self.queue.refresh(job.id)
        return settings

    def _on_start(self) -> None:
        if self._running:
            return
        pending = self._pending_jobs()
        if not pending:
            self._flash_hint("処理するジョブがありません。")
            return

        settings = self._apply_current_settings(pending)
        kinds = {job.kind for job in pending}
        if kinds & {JobKind.IMAGE, JobKind.FOLDER} and not settings.upscale_enabled:
            self._flash_hint("画像にはアップスケーラーモデルを選択してください。")
            return
        if (JobKind.VIDEO in kinds and not settings.upscale_enabled
                and not settings.interpolation_enabled):
            self._flash_hint(
                "動画にはアップスケーラーかフレーム補間モデルを選択してください。"
            )
            return
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
            self.start_btn.setEnabled(not running)
        if not running and self._closing and not self._running:
            self.preview.shutdown()
            self.close()

    def _on_pause(self) -> None:
        """一時停止: 現在ジョブ完了後にワーカーを抜けさせる。"""
        if self._running:
            self._pause.set()
            self.pause_btn.setEnabled(False)
            self.pause_btn.setText("停止中…")
            # 動画1本の途中では長時間効かないため、即時中止の手段を案内する
            self._flash_hint(
                "現在のジョブ完了後に停止します。今すぐ中止するには行の × を押してください。"
            )

    def _set_running(self, running: bool) -> None:
        self._running = running
        self.start_btn.setEnabled(not running)
        self.start_btn.setText("処理中…" if running else "開始")
        self.pause_btn.setEnabled(running)
        self.pause_btn.setText("一時停止")
        # 実行中は入力系をロック（モデル/倍率/出力先/追加）
        for w in (self.backend_combo, self.model_combo, self.interpolation_combo, self.output_combo,
                  self.clear_btn, self.output_open_btn, self.settings_btn,
                  self.drop_zone, self.add_zone):
            w.setEnabled(not running)
        self.drawer.setEnabled(not running)
        for btn in self._scale_btns.values():
            btn.setEnabled(not running)
        if hasattr(self, "preview"):
            self.preview.set_main_running(running)
        if not running:
            self._refresh_scale_enabled()

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
        job.message = "完了"
        self.queue.refresh(job_id)

    def _on_job_error(self, job_id: int, message: str) -> None:
        job = self._jobs.get(job_id)
        if job is None:
            return
        job.status = JobStatus.ERROR
        job.error = message
        job.message = f"エラー: {message}"
        self.queue.refresh(job_id)

    def _on_job_canceled(self, job_id: int) -> None:
        job = self._jobs.get(job_id)
        if job is None:
            return
        job.status = JobStatus.CANCELED
        job.message = "キャンセルされました"
        self.queue.refresh(job_id)

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
        if self._closing and not self.preview.is_trial_running():
            self.preview.shutdown()
            self.close()
            return
        if self._pause.is_set():
            self._flash_hint("一時停止しました。「開始」で再開できます。")
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
            job.message = "キャンセル中…"
            self.queue.refresh(job_id)
            return
        # 未処理/完了済み → 行ごと削除
        ev = self._cancel_events.get(job_id)
        if ev is not None:
            ev.set()  # 念のため（開始前キャンセル扱い）
        # 選択の移動先を一覧が正しく引けるよう、台帳から先に外す
        self._jobs.pop(job_id, None)
        self._cancel_events.pop(job_id, None)
        if job_id in self._order:
            self._order.remove(job_id)
        self.queue.remove_job(job_id)
        self.preview.discard_file(str(job.input_path))
        self._sync_workspace()

    def _clear_queue(self) -> None:
        if self._running:
            return
        ids = list(self._order)
        self._jobs.clear()
        self._order.clear()
        self._cancel_events.clear()
        for jid in ids:
            self.queue.remove_job(jid)
        self.preview.discard_all()
        self._sync_workspace()

    # -------------------------------------------------------------- 補助
    def _flash_hint(self, text: str) -> None:
        """開始ボタンの上に状況メッセージを出す（ウィンドウタイトルにも併記）。"""
        self.setWindowTitle(f"ultraeasy-upscaler — {text}")
        self.status_label.setText(text)

    def _open_output_folder(self) -> None:
        target = self._output_dir or str(Path.home())
        QDesktopServices.openUrl(QUrl.fromLocalFile(target))

    def closeEvent(self, event) -> None:  # noqa: N802
        # 実行中は QThread 走行中の破棄（クラッシュ要因）を避けるため、即閉じない。
        # キャンセル要求 + 一時停止だけ行い、ワーカー完了(queue_finished)後に閉じる。
        trial_running = (
            self.preview.is_trial_running() if hasattr(self, "preview") else False
        )
        if self._running or trial_running:
            self._closing = True
            self._pause.set()
            for ev in self._cancel_events.values():
                ev.set()
            if hasattr(self, "preview") and self.preview.is_trial_running():
                self.preview.cancel_trial()
            self._flash_hint("終了処理中… 現在の処理を停止しています")
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

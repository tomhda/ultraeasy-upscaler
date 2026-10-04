"""配色とスタイルシート。

Windows のライト/ダーク設定とアクセント色に合わせて配色を作る。
プレビュー枠だけは設定に関係なく暗いままにする（周囲が明るいと画の見え方が変わるため）。
"""
from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QEvent, QObject, Qt, Signal
from PySide6.QtGui import QColor, QPalette

# アクセント色を取得できないときの既定
_FALLBACK_ACCENT = "#21c7d9"


@dataclass(frozen=True)
class Palette:
    """画面で使う色の一式。値は "#rrggbb"。"""

    dark: bool
    bg: str
    panel: str
    panel_soft: str
    input: str
    well: str            # サムネ枠など一段沈んだ面
    border: str
    border_light: str
    text: str
    text_soft: str       # 説明文
    text_dim: str        # 補足・アイコン
    text_mute: str       # 無効状態
    button: str
    button_hover: str
    button_pressed: str
    button_disabled: str
    track: str           # 進捗バーの溝
    scrollbar: str
    danger: str
    accent: str          # 塗りに使うアクセント色
    accent_hi: str       # 塗りのホバー
    on_accent: str       # アクセント色の上に載せる文字
    accent_text: str     # 文字・枠線として背景に載せるアクセント色
    accent_tint: str     # 選択中の行などの薄い塗り
    accent_edge: str
    accent_disabled: str


def _mix(a: str, b: str, t: float) -> str:
    """a から b へ t (0..1) だけ寄せた色。"""
    ca, cb = QColor(a), QColor(b)
    r = round(ca.red() + (cb.red() - ca.red()) * t)
    g = round(ca.green() + (cb.green() - ca.green()) * t)
    bl = round(ca.blue() + (cb.blue() - ca.blue()) * t)
    return QColor(r, g, bl).name()


def _luminance(color: str) -> float:
    c = QColor(color)

    def channel(v: int) -> float:
        x = v / 255
        return x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4

    return 0.2126 * channel(c.red()) + 0.7152 * channel(c.green()) + 0.0722 * channel(c.blue())


def _contrast(a: str, b: str) -> float:
    la, lb = _luminance(a), _luminance(b)
    return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)


def _readable(color: str, bg: str, minimum: float = 4.0) -> str:
    """color を、bg の上で読めるコントラストになるまで白または黒へ寄せる。"""
    toward = "#ffffff" if _luminance(bg) < 0.5 else "#000000"
    out = color
    for step in range(1, 11):
        if _contrast(out, bg) >= minimum:
            break
        out = _mix(color, toward, step / 10)
    return out


def build_palette(dark: bool, accent: str) -> Palette:
    if dark:
        base = dict(
            bg="#0f1318", panel="#171c22", panel_soft="#1b222b", input="#151a20",
            well="#0e141a", border="#2a323b", border_light="#3a4652",
            text="#f3f5f7", text_soft="#c9d0d8", text_dim="#a7adb7", text_mute="#67717e",
            button="#1d2530", button_hover="#252e39", button_pressed="#171f28",
            button_disabled="#171d24", track="#29313b", scrollbar="#4a5664",
            danger="#ef5350",
        )
    else:
        base = dict(
            bg="#f3f4f6", panel="#ffffff", panel_soft="#f7f8fa", input="#ffffff",
            well="#e9ecf0", border="#d5d9e0", border_light="#b6bdc8",
            text="#1b1f24", text_soft="#3f4753", text_dim="#5b6470", text_mute="#9aa3ae",
            button="#f4f5f7", button_hover="#e9ecf0", button_pressed="#dfe3e8",
            button_disabled="#f0f1f3", track="#dde1e7", scrollbar="#b6bdc8",
            danger="#d32f2f",
        )
    panel = base["panel"]
    return Palette(
        dark=dark,
        accent=accent,
        accent_hi=_mix(accent, "#ffffff" if dark else "#000000", 0.14),
        on_accent="#041015" if _contrast(accent, "#041015") >= _contrast(accent, "#ffffff")
        else "#ffffff",
        accent_text=_readable(accent, panel),
        accent_tint=_mix(panel, accent, 0.14),
        accent_edge=_mix(panel, accent, 0.40),
        accent_disabled=_mix(panel, accent, 0.35),
        **base,
    )


def build_qss(p: Palette) -> str:
    return f"""
QWidget {{
    background-color: {p.bg};
    color: {p.text};
    font-family: "Yu Gothic UI", "Meiryo UI", "Segoe UI", sans-serif;
    font-size: 14px;
    letter-spacing: 0px;
}}

QLabel {{
    background-color: transparent;
}}

QFrame#header {{
    background-color: transparent;
    min-height: 44px;
}}
QLabel#appIcon {{
    font-family: "Segoe Fluent Icons";
    font-size: 22px;
    min-width: 32px;
    max-width: 32px;
    min-height: 32px;
    max-height: 32px;
    color: {p.accent_text};
    font-weight: 800;
    border: 2px solid {p.accent_text};
    border-radius: 8px;
}}
QLabel#appTitle {{
    color: {p.text};
    font-size: 20px;
    font-weight: 700;
}}
QPushButton#toolbarButton {{
    background-color: {p.button};
    border: 1px solid {p.border};
    border-radius: 7px;
    padding: 8px 14px;
    color: {p.text};
    font-size: 15px;
    font-weight: 600;
}}
QPushButton#toolbarButton:hover {{
    border-color: {p.border_light};
    background-color: {p.button_hover};
}}
QPushButton#iconButton {{
    background-color: transparent;
    border: none;
    border-radius: 8px;
    min-width: 38px;
    max-width: 38px;
    min-height: 38px;
    max-height: 38px;
}}
QPushButton#iconButton:hover, QPushButton#iconButton[active="true"] {{
    background-color: {p.button_hover};
}}

QFrame#dropZone {{
    background-color: {p.panel};
    border: 2px dashed {p.accent_text};
    border-radius: 8px;
}}
QFrame#dropZone[dragActive="true"] {{
    background-color: {p.accent_tint};
    border-color: {p.accent_hi};
}}
QLabel#dropGlyph {{
    font-family: "Segoe Fluent Icons";
    color: {p.text_dim};
    font-size: 66px;
}}
QLabel#dropTitle {{
    color: {p.text};
    font-size: 30px;
    font-weight: 800;
}}
QLabel#dropHint {{
    color: {p.text_dim};
    font-size: 17px;
    font-weight: 500;
}}

QFrame#dropZone[compact="true"] {{
    background-color: transparent;
    border: 1px dashed {p.border_light};
    border-radius: 7px;
}}
QFrame#dropZone[compact="true"][dragActive="true"] {{
    background-color: {p.accent_tint};
    border-color: {p.accent_hi};
}}
QLabel#dropHintSmall {{
    color: {p.text_dim};
    font-size: 13px;
}}
QFrame#previewPane {{
    background-color: #0b0f13;
    border: 1px solid {p.border};
    border-radius: 8px;
}}
QLabel#previewMessage {{
    color: #c9d0d8;
    font-size: 15px;
}}
QLabel#previewCaption {{
    color: #f3f5f7;
    background-color: rgba(15, 19, 24, 200);
    border-radius: 5px;
    padding: 3px 9px;
    font-size: 13px;
}}

QFrame#controlPanel {{
    background-color: {p.panel};
    border: 1px solid {p.border};
    border-radius: 8px;
}}
QLabel#fieldLabel {{
    color: {p.text};
    font-size: 15px;
    font-weight: 700;
}}
QWidget#fieldLabelWrap, QWidget#checkRow, QWidget#scaleWrap {{
    background-color: transparent;
}}
QLabel#helpIcon {{
    color: {p.accent_text};
    background-color: {p.accent_tint};
    border: 1px solid {p.accent_edge};
    border-radius: 9px;
    font-size: 12px;
    font-weight: 800;
}}
QLabel#helpPopup {{
    background-color: {p.input};
    color: {p.text};
    border: 1px solid {p.border_light};
    border-radius: 6px;
    padding: 9px 11px;
    font-family: "Yu Gothic UI", "Meiryo UI", "Segoe UI", sans-serif;
    font-size: 13px;
    line-height: 1.35;
}}
QLabel#sectionTitle {{
    color: {p.text};
    font-size: 20px;
    font-weight: 800;
}}
QLabel#hint {{
    color: {p.text_soft};
    font-size: 15px;
}}

QPushButton {{
    background-color: {p.button};
    border: 1px solid {p.border};
    border-radius: 7px;
    padding: 7px 13px;
    color: {p.text};
}}
QPushButton:hover {{
    border-color: {p.border_light};
    background-color: {p.button_hover};
}}
QPushButton:pressed {{
    background-color: {p.button_pressed};
}}
QPushButton:disabled {{
    color: {p.text_mute};
    background-color: {p.button_disabled};
    border-color: {p.border};
}}

QPushButton#primary {{
    background-color: {p.accent};
    color: {p.on_accent};
    border: none;
    border-radius: 8px;
    min-height: 58px;
    font-size: 21px;
    font-weight: 800;
}}
QPushButton#primary:hover {{
    background-color: {p.accent_hi};
}}
QPushButton#primary:disabled {{
    background-color: {p.accent_disabled};
    color: {p.text_mute};
}}

QPushButton#kindBtn {{
    background-color: {p.input};
    border: 1px solid {p.border};
    border-radius: 6px;
    min-height: 34px;
    padding: 0 16px;
    font-size: 15px;
    color: {p.text_dim};
}}
QPushButton#kindBtn:checked {{
    background-color: {p.accent_tint};
    border-color: {p.accent_text};
    color: {p.text};
    font-weight: 700;
}}

QPushButton#accent {{
    background-color: {p.accent};
    color: {p.on_accent};
    border: none;
    border-radius: 7px;
    min-height: 26px;
    padding: 7px 18px;
    font-size: 15px;
    font-weight: 700;
}}
QPushButton#accent:hover {{
    background-color: {p.accent_hi};
}}
QPushButton#accent:disabled {{
    background-color: {p.accent_disabled};
    color: {p.text_mute};
}}

QPushButton#scaleBtn {{
    background-color: {p.input};
    border: 1px solid {p.border};
    border-radius: 6px;
    min-width: 76px;
    min-height: 42px;
    padding: 0 16px;
    font-size: 17px;
}}
QPushButton#scaleBtn:checked {{
    background-color: {p.accent};
    color: {p.on_accent};
    font-weight: 800;
    border-color: {p.accent};
}}

QComboBox {{
    background-color: {p.input};
    border: 1px solid {p.border};
    border-radius: 6px;
    color: {p.text};
    min-height: 34px;
    padding: 4px 12px;
    font-size: 16px;
}}
QComboBox:hover {{
    border-color: {p.border_light};
}}
QComboBox::drop-down {{
    border: none;
    width: 28px;
}}
QComboBox QAbstractItemView {{
    background-color: {p.input};
    border: 1px solid {p.border_light};
    selection-background-color: {p.accent};
    selection-color: {p.on_accent};
    outline: none;
}}

QFrame#card {{
    background-color: {p.panel};
    border: 1px solid {p.border};
    border-radius: 8px;
}}
QFrame#card[dragActive="true"] {{
    background-color: {p.accent_tint};
    border: 2px dashed {p.accent_text};
}}
QScrollArea {{
    border: none;
    background-color: transparent;
}}
QScrollBar:vertical {{
    background: transparent;
    width: 9px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {p.scrollbar};
    border-radius: 4px;
    min-height: 26px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
    background: none;
}}

QFrame#queueRow {{
    background-color: {p.panel_soft};
    border: 1px solid {p.border};
    border-radius: 7px;
}}
QFrame#queueRow[selected="true"] {{
    border: 1px solid {p.accent_text};
    background-color: {p.accent_tint};
}}
QLabel#rowName {{
    color: {p.text};
    font-size: 15px;
    font-weight: 700;
}}
QLabel#rowMeta {{
    color: {p.text_dim};
    font-size: 12px;
}}
QLabel#rowStatus {{
    color: {p.text_dim};
    font-size: 12px;
}}
QLabel#rowPercent {{
    color: {p.accent_text};
    font-size: 13px;
    font-weight: 800;
}}
QLabel#thumb {{
    font-family: "Segoe Fluent Icons";
    font-size: 24px;
    background-color: {p.well};
    border: 1px solid {p.border};
    border-radius: 6px;
    color: {p.text_dim};
}}
QPushButton#rowClose {{
    background-color: transparent;
    border: none;
    color: {p.text_dim};
    font-size: 22px;
    padding: 0;
}}
QPushButton#rowClose:hover {{
    color: {p.danger};
}}
QPushButton#rowRetry {{
    background-color: transparent;
    border: none;
    color: {p.text_dim};
    font-size: 16px;
    padding: 0;
}}
QPushButton#rowRetry:hover {{
    color: {p.text};
}}

QProgressBar {{
    background-color: {p.track};
    border: none;
    border-radius: 5px;
    height: 8px;
    text-align: center;
    color: transparent;
}}
QProgressBar::chunk {{
    background-color: {p.accent};
    border-radius: 5px;
}}

QPushButton#link {{
    background-color: transparent;
    border: none;
    color: {p.text_dim};
    padding: 4px 6px;
    font-size: 15px;
}}
QPushButton#link:hover {{
    color: {p.text};
}}

QSpinBox, QLineEdit {{
    background-color: {p.input};
    border: 1px solid {p.border};
    border-radius: 6px;
    padding: 6px 10px;
}}
QCheckBox {{
    background-color: transparent;
    spacing: 8px;
}}
QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border: 1px solid {p.border_light};
    border-radius: 4px;
    background-color: {p.input};
}}
QCheckBox::indicator:checked {{
    background-color: {p.accent};
    border-color: {p.accent};
}}
QWidget#toggleWrap {{
    background-color: transparent;
}}
QCheckBox#clearCheck {{
    background-color: transparent;
    border: none;
    color: {p.text};
    min-height: 34px;
    padding: 0;
    font-size: 15px;
    font-weight: 600;
}}
QCheckBox#clearCheck::indicator {{
    width: 0;
    height: 0;
}}
"""


_current = build_palette(True, _FALLBACK_ACCENT)


def current() -> Palette:
    """いま適用されている配色（自前描画やアイコンの色に使う）。"""
    return _current


class _Notifier(QObject):
    changed = Signal()


# 配色が切り替わったら changed を出す。アイコンなど QSS で塗れないものの更新用。
notifier = _Notifier()


def _system_palette(app) -> Palette:
    dark = app.styleHints().colorScheme() != Qt.ColorScheme.Light
    accent = app.palette().color(QPalette.ColorRole.Accent)
    return build_palette(dark, accent.name() if accent.isValid() else _FALLBACK_ACCENT)


class _Watcher(QObject):
    """Windows 側でライト/ダークやアクセント色が変わったら配色を作り直す。"""

    def __init__(self, app) -> None:
        super().__init__(app)
        self._app = app
        app.styleHints().colorSchemeChanged.connect(lambda _scheme: refresh(app))
        app.installEventFilter(self)

    def eventFilter(self, obj, event) -> bool:  # noqa: N802
        if obj is self._app and event.type() == QEvent.Type.ApplicationPaletteChange:
            refresh(self._app)
        return False


def refresh(app) -> None:
    """システムの設定を読み直し、変わっていれば配色を適用する。"""
    global _current
    palette = _system_palette(app)
    if palette == _current and app.styleSheet():
        return
    _current = palette
    app.setStyleSheet(build_qss(palette))
    notifier.changed.emit()


def apply_theme(app) -> None:
    if app.findChild(_Watcher) is None:
        _Watcher(app)
    refresh(app)

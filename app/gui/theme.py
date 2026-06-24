"""Dark QSS theme matching the v2 mockup direction."""
from __future__ import annotations

BG = "#0f1318"
BG_PANEL = "#171c22"
BG_PANEL_SOFT = "#1b222b"
BG_INPUT = "#151a20"
BORDER = "#2a323b"
BORDER_LIGHT = "#3a4652"
TEXT = "#f3f5f7"
TEXT_DIM = "#a7adb7"
ACCENT = "#21c7d9"
ACCENT_HI = "#38d7e7"
AMBER = "#f6c028"
DANGER = "#ef5350"


QSS = f"""
QWidget {{
    background-color: {BG};
    color: {TEXT};
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
    color: {ACCENT};
    font-weight: 800;
    border: 2px solid {ACCENT};
    border-radius: 8px;
}}
QLabel#appTitle {{
    color: {TEXT};
    font-size: 20px;
    font-weight: 700;
}}
QPushButton#toolbarButton {{
    background-color: #181d24;
    border: 1px solid {BORDER};
    border-radius: 7px;
    padding: 8px 14px;
    color: #d8dde4;
    font-size: 15px;
    font-weight: 600;
}}
QPushButton#toolbarButton:hover {{
    border-color: {BORDER_LIGHT};
    background-color: #20262e;
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
    background-color: #1d242c;
}}

QFrame#dropZone {{
    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                      stop:0 #161b21, stop:0.55 #12171d,
                                      stop:1 #171c22);
    border: 2px dashed {ACCENT};
    border-radius: 8px;
}}
QFrame#dropZone[dragActive="true"] {{
    background-color: #1b2630;
    border-color: {ACCENT_HI};
}}
QLabel#dropGlyph {{
    font-family: "Segoe Fluent Icons";
    color: #a8b0bc;
    font-size: 66px;
}}
QLabel#dropTitle {{
    color: {TEXT};
    font-size: 30px;
    font-weight: 800;
}}
QLabel#dropHint {{
    color: {TEXT_DIM};
    font-size: 17px;
    font-weight: 500;
}}

QFrame#controlPanel {{
    background-color: {BG_PANEL};
    border: 1px solid {BORDER};
    border-radius: 8px;
}}
QLabel#fieldLabel {{
    color: {TEXT};
    font-size: 15px;
    font-weight: 700;
}}
QLabel#sectionTitle {{
    color: {TEXT};
    font-size: 20px;
    font-weight: 800;
}}
QLabel#hint {{
    color: #c9d0d8;
    font-size: 15px;
}}

QPushButton {{
    background-color: #1d2530;
    border: 1px solid {BORDER};
    border-radius: 7px;
    padding: 7px 13px;
    color: {TEXT};
}}
QPushButton:hover {{
    border-color: {BORDER_LIGHT};
    background-color: #252e39;
}}
QPushButton:pressed {{
    background-color: #171f28;
}}
QPushButton:disabled {{
    color: #67717e;
    background-color: #171d24;
    border-color: #252d36;
}}

QPushButton#primary {{
    background-color: {ACCENT};
    color: #041015;
    border: none;
    border-radius: 8px;
    min-width: 278px;
    min-height: 58px;
    font-size: 21px;
    font-weight: 800;
}}
QPushButton#primary:hover {{
    background-color: {ACCENT_HI};
}}
QPushButton#primary:disabled {{
    background-color: #1e535b;
    color: #7ea8ad;
}}

QPushButton#scaleBtn {{
    background-color: {BG_INPUT};
    border: 1px solid {BORDER};
    border-radius: 6px;
    min-width: 76px;
    min-height: 34px;
    padding: 0 16px;
    font-size: 17px;
}}
QPushButton#scaleBtn:checked {{
    background-color: {ACCENT};
    color: #041015;
    font-weight: 800;
    border-color: {ACCENT};
}}

QComboBox {{
    background-color: {BG_INPUT};
    border: 1px solid {BORDER};
    border-radius: 6px;
    color: {TEXT};
    min-height: 34px;
    padding: 4px 12px;
    font-size: 16px;
}}
QComboBox:hover {{
    border-color: {BORDER_LIGHT};
}}
QComboBox::drop-down {{
    border: none;
    width: 28px;
}}
QComboBox QAbstractItemView {{
    background-color: {BG_INPUT};
    border: 1px solid {BORDER_LIGHT};
    selection-background-color: {ACCENT};
    selection-color: #041015;
    outline: none;
}}

QFrame#card {{
    background-color: {BG_PANEL};
    border: 1px solid {BORDER};
    border-radius: 8px;
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
    background: #4a5664;
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
    background-color: {BG_PANEL_SOFT};
    border: 1px solid {BORDER};
    border-radius: 7px;
}}
QWidget#rowText {{
    background-color: transparent;
}}
QLabel#rowName {{
    color: {TEXT};
    font-size: 17px;
    font-weight: 800;
}}
QLabel#rowMeta {{
    color: #a9c7df;
    background-color: #172436;
    border: 1px solid #274058;
    border-radius: 5px;
    padding: 2px 7px;
    font-size: 12px;
}}
QLabel#rowStatus {{
    color: {TEXT_DIM};
    font-size: 12px;
}}
QLabel#rowPercent {{
    color: {ACCENT};
    font-size: 13px;
    font-weight: 800;
}}
QLabel#thumb {{
    font-family: "Segoe Fluent Icons";
    font-size: 24px;
    background-color: #0e141a;
    border: 1px solid {BORDER};
    border-radius: 6px;
    color: #c0c7d1;
}}
QPushButton#rowClose {{
    background-color: transparent;
    border: none;
    color: #b6bdc7;
    font-size: 22px;
    padding: 0;
}}
QPushButton#rowClose:hover {{
    color: {DANGER};
}}

QProgressBar {{
    background-color: #29313b;
    border: none;
    border-radius: 5px;
    height: 8px;
    text-align: center;
    color: transparent;
}}
QProgressBar::chunk {{
    background-color: {ACCENT};
    border-radius: 5px;
}}

QPushButton#link {{
    background-color: transparent;
    border: none;
    color: {TEXT_DIM};
    padding: 4px 6px;
    font-size: 15px;
}}
QPushButton#link:hover {{
    color: {TEXT};
}}

QSpinBox, QLineEdit {{
    background-color: {BG_INPUT};
    border: 1px solid {BORDER};
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
    border: 1px solid {BORDER_LIGHT};
    border-radius: 4px;
    background-color: {BG_INPUT};
}}
QCheckBox::indicator:checked {{
    background-color: {ACCENT};
    border-color: {ACCENT};
}}

QFrame#footer {{
    background-color: transparent;
    border-top: 1px solid #242b33;
}}
QLabel#footerIcon {{
    color: #a8b0bc;
}}
"""


def apply_theme(app) -> None:
    app.setStyleSheet(QSS)

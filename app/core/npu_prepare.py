"""NPU 追加キットの有無とモデルごとの変換状態・変換実行（GUI 非依存）。

NPU を動かす Ryzen AI Software はアプリに同梱できないため、NPU は
別配布のキットを入れた人だけが使える拡張として扱う。このモジュールは
キットの有無・変換済み判定・変換実行だけを持ち、表示は GUI 側が行う。
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass

from ..i18n import t
from . import binaries, helper_backend, jobs
from .helper_backend import HelperBackendUnavailable
from .settings import (
    HELPER_MODEL_ADCSR,
    HELPER_MODEL_AMD_RRDB,
    HELPER_MODEL_ANIME,
    HELPER_MODEL_SPAN,
    HELPER_MODEL_SWINIR,
    UpscaleBackend,
    UpscaleSettings,
    canonical_helper_model,
)

# 変換プロセスの起動優先度。Windows では子プロセスへ継承されるため、
# 長時間の VAIML コンパイル中も PC が使い物になる。1 か所に置き、
# 本処理（open_session の既定）は 0 のまま変えない。
CONVERT_PRIORITY = getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)

# モデルごとの初回変換の目安（分）。Ryzen AI 7 PRO 350・Ryzen AI Software 1.8.0・
# 低い優先度での実測（2026-10-05）: Anime 903 秒、SPAN 850 秒、Real-ESRGAN 1529 秒、
# SwinIR-M 3918 秒。どれも CPU は 1 コアぶん。メモリの最大は SwinIR-M だけ約 25GB、
# ほかは 2GB 未満。AdcSR は 9 月の実測（前半 5201 秒＋後半 1811 秒）で、メモリは未計測。
NPU_CONVERT_MINUTES = {
    HELPER_MODEL_ANIME: 15,
    HELPER_MODEL_SPAN: 14,
    HELPER_MODEL_AMD_RRDB: 25,
    HELPER_MODEL_SWINIR: 65,
    HELPER_MODEL_ADCSR: 120,
}

# 変換画面に並べる順番（右列のモデル選択と同じ並び）。
NPU_MODEL_ORDER = (
    HELPER_MODEL_ANIME,
    HELPER_MODEL_SPAN,
    HELPER_MODEL_AMD_RRDB,
    HELPER_MODEL_SWINIR,
    HELPER_MODEL_ADCSR,
)

# 変換寸法。NPU のタイルは寸法に依らないため、自動切替に落ちない
# 480 以上なら何でもよい（open_session と同じ経路で準備だけ行う）。
_CONVERT_WIDTH = 854
_CONVERT_HEIGHT = 480


@dataclass(frozen=True)
class NpuModel:
    """NPU 変換画面の 1 行分。label_key は表示名を引くためのキー。"""

    key: str
    label_key: str
    has_files: bool
    converted: bool
    minutes: int


def npu_available() -> bool:
    """NPU キットと Ryzen AI の Python の両方があるときだけ True。"""
    kit = binaries.repo_root() / "tools" / "npu-serve" / "npu_serve.py"
    if not kit.is_file():
        return False
    try:
        helper_backend._npu_python()
    except Exception:
        return False
    return True


def is_converted(model_key: str | None) -> bool:
    """そのモデルが NPU 変換済みか（本処理と同じ判定）。"""
    try:
        return bool(helper_backend.npu_compiled(model_key))
    except Exception:
        # 判定に失敗したら未変換扱い（開始前の確認で止めて案内を出す）。
        return False


def npu_models() -> list[NpuModel]:
    """NPU で使えるモデルの一覧。ファイルが揃わないものは出さない。"""
    models: list[NpuModel] = []
    for key in NPU_MODEL_ORDER:
        converted = helper_backend.npu_compiled(key)
        if converted is None:
            continue
        models.append(NpuModel(
            key=key,
            label_key=key,
            has_files=True,
            converted=bool(converted),
            minutes=NPU_CONVERT_MINUTES[key],
        ))
    return models


def not_converted_message(label: str) -> str:
    """未変換のまま開始・試しを押したときの状況行の文言。"""
    return t(
        "{label}は NPU 用の変換がまだです。詳細設定の「NPU の準備」で変換してください。",
        label=label,
    )


def convert(model_key: str, progress=None, cancel=None) -> None:
    """そのモデルを変換する。本処理と同じ open_session の経路で行う。

    準備完了まで待って閉じるだけで、試し実行用の別経路は作らない。
    プロセスは低い優先度で起動する。
    """
    key = canonical_helper_model(model_key)
    if key not in NPU_CONVERT_MINUTES:
        raise ValueError(t("NPU で変換できないモデルです: {model}", model=model_key))
    if helper_backend.npu_compiled(key) is None:
        raise HelperBackendUnavailable(
            t("NPU 用のファイルが揃っていません: {model}", model=model_key))
    if key == HELPER_MODEL_ADCSR and not helper_backend.adcsr_selftest_ready():
        # 無いまま始めると、約 2 時間の変換のあとの自己診断で失敗する。
        raise HelperBackendUnavailable(
            t("NPU 用のファイルが揃っていません: {model}", model=model_key))
    settings = UpscaleSettings(backend=UpscaleBackend.NPU_NATIVE, model=key, scale=4)
    session = helper_backend.open_session(
        settings, _CONVERT_WIDTH, _CONVERT_HEIGHT,
        progress if progress is not None else (lambda _f, _m: None),
        cancel, creationflags=CONVERT_PRIORITY,
    )
    try:
        if cancel is not None and cancel.is_set():
            raise jobs.Cancelled()
    finally:
        session.close(force=cancel is not None and cancel.is_set())

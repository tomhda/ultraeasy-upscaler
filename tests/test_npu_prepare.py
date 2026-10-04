"""NPU 追加キット・変換画面のテスト（オフスクリーン）。

実際の推論・変換（VAIML コンパイル）はしない。`open_session`・`convert`
は偽物に差し替える。実キャッシュ・実モデルには触らず、`UEU_NPU_CACHE` /
`UEU_MODELS_DIR` の差し替えと空の vendor 向けで判定を確かめる。
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import threading
import time
import types
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from app.core import helper_backend, npu_prepare
from app.core.jobs import Cancelled
from app.core.settings import (
    HELPER_MODEL_ADCSR,
    HELPER_MODEL_AMD_RRDB,
    HELPER_MODEL_ANIME,
    HELPER_MODEL_SPAN,
    HELPER_MODEL_SWINIR,
    HELPER_MODEL_FILES,
    HELPER_MODEL_NPU_BACK,
    HELPER_MODEL_NPU_TAIL,
    UpscaleBackend,
    UpscaleSettings,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_IMAGE = REPO_ROOT / "vendor" / "realesrgan" / "input.jpg"

NATIVE_FILES = HELPER_MODEL_FILES[UpscaleBackend.NPU_NATIVE]
ANIME_FULL = NATIVE_FILES[HELPER_MODEL_ANIME][512]
SPAN_FULL = NATIVE_FILES[HELPER_MODEL_SPAN][512]
RRDB_FULL = NATIVE_FILES[HELPER_MODEL_AMD_RRDB][256]
SWINIR_FULL = NATIVE_FILES[HELPER_MODEL_SWINIR][256]
ADCSR_FRONT = NATIVE_FILES[HELPER_MODEL_ADCSR][128]
ADCSR_BACK = HELPER_MODEL_NPU_BACK[HELPER_MODEL_ADCSR]
ANIME_BODY, ANIME_TAIL = HELPER_MODEL_NPU_TAIL[HELPER_MODEL_ANIME][512]
SPAN_BODY, SPAN_TAIL_JSON = HELPER_MODEL_NPU_TAIL[HELPER_MODEL_SPAN][512]

LABELS = {
    HELPER_MODEL_ANIME: "Anime Video v3",
    HELPER_MODEL_SPAN: "4xNomosUni SPAN",
    HELPER_MODEL_AMD_RRDB: "Real-ESRGAN（AMD縮小版）",
    HELPER_MODEL_SWINIR: "SwinIR-M",
    HELPER_MODEL_ADCSR: "AdcSR",
}


def _estimate(model_key: str) -> str:
    """そのモデルの行に出るはずの目安の文言（分数の表が変わっても追従する）。"""
    from app.core import npu_prepare, settings
    from app.gui.settings_drawer import format_convert_estimate

    key = getattr(settings, model_key)
    return format_convert_estimate(npu_prepare.NPU_CONVERT_MINUTES[key])


@pytest.fixture(scope="module")
def app():
    application = QApplication.instance() or QApplication([])
    yield application
    application.processEvents()


def _wait_until(pred, timeout: float = 15.0) -> bool:
    """条件が満たされるまで GUI イベントを回す。"""
    end = time.time() + timeout
    application = QApplication.instance()
    while time.time() < end:
        if pred():
            return True
        if application is not None:
            application.processEvents()
        time.sleep(0.05)
    return pred()


@pytest.fixture
def npu_env(monkeypatch, tmp_path):
    """モデル・キャッシュ・vendor 参照を空の一時フォルダへ向ける。"""
    models = tmp_path / "models"
    models.mkdir()
    cache = tmp_path / "cache"
    cache.mkdir()
    vendor = tmp_path / "vendor-models"
    vendor.mkdir()
    monkeypatch.setenv(helper_backend.MODELS_DIR_ENV, str(models))
    monkeypatch.setenv(helper_backend.NPU_CACHE_ENV, str(cache))
    monkeypatch.setattr(helper_backend, "DEFAULT_VENDOR_MODELS_DIR", vendor)
    monkeypatch.setenv(helper_backend.ADCSR_NPU2_ENV, "1")
    monkeypatch.setenv(helper_backend.NPU_TAILCUT_ENV, "1")
    return models, cache


def _touch(path: Path, data: bytes = b"fake-onnx") -> Path:
    path.write_bytes(data)
    return path


def _make_cache(cache: Path, stem: str) -> Path:
    """単段モデルの変換済みキャッシュ（context.json＋.rai）を作る。"""
    entry = cache / f"modelcachekey_{stem}"
    entry.mkdir(parents=True)
    (entry / "context.json").write_text("{}", encoding="utf-8")
    (entry / "compiled.rai").write_bytes(b"fake-rai")
    return entry


def _stem(filename: str) -> str:
    return Path(filename).stem


def _make_two_stage(models: Path, cache: Path, key: str = "ck"):
    """AdcSR 2 段の前半・後半・マニフェスト・両キャッシュを作る。"""
    front = _touch(models / ADCSR_FRONT, b"front-bytes")
    back = _touch(models / ADCSR_BACK, b"back-bytes")

    def _digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    manifest = {
        "front": {"cache_key": f"{key}-front", "sha256": _digest(front)},
        "back": {"cache_key": f"{key}-back", "sha256": _digest(back)},
    }
    manifest_path = models / "adcsr_npu_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    for side in ("front", "back"):
        entry = cache / manifest[side]["cache_key"]
        entry.mkdir(parents=True)
        (entry / "context.json").write_text("{}", encoding="utf-8")
        (entry / "compiled.rai").write_bytes(b"fake-rai")
    return front, back, manifest_path


class _FakeServeClient:
    """open_session の起動だけを記録する（プロセスは作らない）。"""

    last_creationflags: int = 0

    def __init__(self, command, workdir, env=None, log=None, creationflags=0) -> None:
        type(self).last_creationflags = creationflags

    def connect(self, **kwargs) -> None:
        return None

    def close(self, *, force: bool = False) -> None:
        return None


def _install_fake_npu(monkeypatch, tmp_path) -> None:
    """open_session が実プロセスを作らず NPU 経路まで進むようにする。"""
    python = tmp_path / "python.exe"
    python.touch()
    script = tmp_path / "npu_serve.py"
    script.touch()
    monkeypatch.setattr(helper_backend, "_npu_python", lambda: python)
    monkeypatch.setattr(helper_backend, "_npu_script", lambda: script)
    monkeypatch.setattr(helper_backend, "ServeClient", _FakeServeClient)


def _open_npu(model_key: str):
    settings = UpscaleSettings(backend=UpscaleBackend.NPU_NATIVE, model=model_key)
    return helper_backend.open_session(settings, 854, 480)


def _canned_models(converted=()):
    from app.core.npu_prepare import NpuModel

    return [
        NpuModel(key=key, label_key=key, has_files=True,
                 converted=(key in converted),
                 minutes=npu_prepare.NPU_CONVERT_MINUTES[key])
        for key in npu_prepare.NPU_MODEL_ORDER
    ]


# ------------------------------------------------------- npu_available
def test_npu_available_false_without_kit(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(npu_prepare.binaries, "repo_root", lambda: tmp_path)
    assert npu_prepare.npu_available() is False


def test_npu_available_false_without_python(monkeypatch, tmp_path) -> None:
    kit = tmp_path / "tools" / "npu-serve"
    kit.mkdir(parents=True)
    (kit / "npu_serve.py").touch()
    monkeypatch.setattr(npu_prepare.binaries, "repo_root", lambda: tmp_path)
    monkeypatch.setenv(
        helper_backend.NPU_PYTHON_ENV, str(tmp_path / "none" / "python.exe"))
    assert npu_prepare.npu_available() is False


def test_npu_available_true_with_kit_and_python(monkeypatch, tmp_path) -> None:
    kit = tmp_path / "tools" / "npu-serve"
    kit.mkdir(parents=True)
    (kit / "npu_serve.py").touch()
    python = tmp_path / "python.exe"
    python.touch()
    monkeypatch.setattr(npu_prepare.binaries, "repo_root", lambda: tmp_path)
    monkeypatch.setenv(helper_backend.NPU_PYTHON_ENV, str(python))
    assert npu_prepare.npu_available() is True


# --------------------------------- 変換済み判定と本処理の判定の一致
def test_models_converted_matches_session_without_tail(
    monkeypatch, tmp_path, npu_env
) -> None:
    models, cache = npu_env
    _touch(models / ANIME_FULL)
    _make_cache(cache, _stem(ANIME_FULL))
    _touch(models / RRDB_FULL)  # キャッシュなし
    _install_fake_npu(monkeypatch, tmp_path)

    listed = {model.key: model for model in npu_prepare.npu_models()}
    assert listed[HELPER_MODEL_ANIME].converted is True
    assert listed[HELPER_MODEL_ANIME].has_files is True
    assert listed[HELPER_MODEL_AMD_RRDB].converted is False
    assert HELPER_MODEL_ADCSR not in listed  # 追加ファイルなしは出さない

    assert _open_npu(HELPER_MODEL_ANIME).cache_hit is True
    assert _open_npu(HELPER_MODEL_AMD_RRDB).cache_hit is False


def test_models_converted_follows_tail_body_cache(
    monkeypatch, tmp_path, npu_env
) -> None:
    models, cache = npu_env
    # 全体モデルのキャッシュはあるが body のキャッシュが無い → 未変換。
    _touch(models / ANIME_FULL)
    _make_cache(cache, _stem(ANIME_FULL))
    _touch(models / ANIME_BODY)
    _touch(models / ANIME_TAIL)
    # 全体モデルのキャッシュは無いが body のキャッシュがある → 変換済み。
    _touch(models / SPAN_FULL)
    _touch(models / SPAN_BODY)
    _touch(models / SPAN_TAIL_JSON)
    _make_cache(cache, _stem(SPAN_BODY))
    _install_fake_npu(monkeypatch, tmp_path)

    listed = {model.key: model for model in npu_prepare.npu_models()}
    assert listed[HELPER_MODEL_ANIME].converted is False
    assert listed[HELPER_MODEL_SPAN].converted is True

    assert _open_npu(HELPER_MODEL_ANIME).cache_hit is False
    assert _open_npu(HELPER_MODEL_SPAN).cache_hit is True


def test_models_converted_matches_two_stage_cache(
    monkeypatch, tmp_path, npu_env
) -> None:
    models, cache = npu_env
    _touch(models / RRDB_FULL)
    _make_two_stage(models, cache)
    _install_fake_npu(monkeypatch, tmp_path)

    listed = {model.key: model for model in npu_prepare.npu_models()}
    assert listed[HELPER_MODEL_ADCSR].converted is True
    assert _open_npu(HELPER_MODEL_ADCSR).cache_hit is True

    # 後半を書き換えるとマニフェストと合わなくなる → 未変換。
    (models / ADCSR_BACK).write_bytes(b"tampered-bytes")
    listed = {model.key: model for model in npu_prepare.npu_models()}
    assert listed[HELPER_MODEL_ADCSR].converted is False
    assert _open_npu(HELPER_MODEL_ADCSR).cache_hit is False


def test_models_excludes_adcsr_without_two_stage_files(
    monkeypatch, tmp_path, npu_env
) -> None:
    models, _cache = npu_env
    _touch(models / ADCSR_FRONT)
    _touch(models / ADCSR_BACK)
    _install_fake_npu(monkeypatch, tmp_path)

    listed = {model.key: model for model in npu_prepare.npu_models()}
    assert HELPER_MODEL_ADCSR not in listed


# ------------------------------------------------------------- convert
def test_convert_priority_constant() -> None:
    assert npu_prepare.CONVERT_PRIORITY == subprocess.BELOW_NORMAL_PRIORITY_CLASS


def test_convert_opens_npu_session_and_closes(monkeypatch) -> None:
    calls: dict = {}

    class _Session:
        def close(self, *, force: bool = False) -> None:
            calls["closed"] = force

    def fake_open(settings, width, height, progress=None, cancel=None,
                  *, creationflags=0):
        calls.update(settings=settings, width=width, height=height,
                     progress=progress, cancel=cancel,
                     creationflags=creationflags)
        return _Session()

    monkeypatch.setattr(helper_backend, "open_session", fake_open)
    monkeypatch.setattr(helper_backend, "npu_compiled", lambda _key: True)
    cancel = threading.Event()
    npu_prepare.convert(HELPER_MODEL_ANIME, cancel=cancel)

    assert calls["settings"].backend == UpscaleBackend.NPU_NATIVE
    assert calls["settings"].model == HELPER_MODEL_ANIME
    assert calls["settings"].scale == 4
    assert (calls["width"], calls["height"]) == (854, 480)
    assert calls["creationflags"] == subprocess.BELOW_NORMAL_PRIORITY_CLASS
    assert calls["cancel"] is cancel
    assert calls["closed"] is False


def test_convert_rejects_unknown_model() -> None:
    with pytest.raises(ValueError):
        npu_prepare.convert("unknown-model")


def test_convert_requires_model_files(monkeypatch) -> None:
    monkeypatch.setattr(helper_backend, "npu_compiled", lambda _key: None)
    with pytest.raises(helper_backend.HelperBackendUnavailable):
        npu_prepare.convert(HELPER_MODEL_ANIME)


def test_convert_cancel_propagates(monkeypatch) -> None:
    def fake_open(*_args, **_kwargs):
        raise Cancelled()

    monkeypatch.setattr(helper_backend, "open_session", fake_open)
    monkeypatch.setattr(helper_backend, "npu_compiled", lambda _key: True)
    with pytest.raises(Cancelled):
        npu_prepare.convert(HELPER_MODEL_ANIME, cancel=threading.Event())


def test_open_session_passes_creationflags(monkeypatch, tmp_path, npu_env) -> None:
    models, _cache = npu_env
    _touch(models / RRDB_FULL)
    _install_fake_npu(monkeypatch, tmp_path)

    settings = UpscaleSettings(
        backend=UpscaleBackend.NPU_NATIVE, model=HELPER_MODEL_AMD_RRDB)
    helper_backend.open_session(settings, 854, 480, creationflags=1234)
    assert _FakeServeClient.last_creationflags == 1234
    helper_backend.open_session(settings, 854, 480)
    assert _FakeServeClient.last_creationflags == 0


# --------------------------------------- serve_client の起動フラグ経路
def test_spawn_helper_forwards_creationflags_to_popen(monkeypatch) -> None:
    from app.core import serve_client

    seen: dict = {}

    class _Proc:
        pass

    def fake_popen(command, **kwargs):
        seen.update(kwargs)
        return _Proc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    proc = serve_client._spawn_helper(
        ["fake-helper"], cwd=None, env=None, creationflags=1234)
    assert isinstance(proc, _Proc)
    assert seen["creationflags"] == 1234


def test_spawn_helper_resolves_npu_job_from_repo_root(monkeypatch, tmp_path) -> None:
    from app.core import serve_client

    monkeypatch.setattr(serve_client.sys, "platform", "win32")
    serve_dir = tmp_path / "tools" / "npu-serve"
    serve_dir.mkdir(parents=True)
    monkeypatch.setattr(serve_client.binaries, "repo_root", lambda: tmp_path)
    fake_job = types.ModuleType("npu_job")
    fake_job.job_supported = lambda: True

    def fake_spawn(command, *, cwd=None, env=None, creationflags=0):
        fake_spawn.seen = {"cwd": cwd, "creationflags": creationflags}
        return object()

    fake_job.spawn_in_job = fake_spawn
    monkeypatch.setitem(_sys_modules(), "npu_job", fake_job)
    try:
        serve_client._spawn_helper(
            ["fake-helper"], cwd=None, env=None, creationflags=7)
    finally:
        if str(serve_dir) in serve_client.sys.path:
            serve_client.sys.path.remove(str(serve_dir))
    assert fake_spawn.seen == {"cwd": None, "creationflags": 7}
    assert str(serve_dir) not in serve_client.sys.path


def _sys_modules():
    import sys

    return sys.modules


# ------------------------------------------------- 「NPU の準備」の欄
def _make_drawer(monkeypatch, converted=()):
    from app.gui.settings_drawer import SettingsDrawer

    monkeypatch.setattr(npu_prepare, "npu_available", lambda: True)
    monkeypatch.setattr(
        npu_prepare, "npu_models", lambda: _canned_models(converted))
    return SettingsDrawer(model_label=lambda key: LABELS.get(key, key))


@pytest.mark.parametrize(("minutes", "expected"), [
    (15, "初回変換の目安: 約15分"),
    (13, "初回変換の目安: 約13分"),
    (19, "初回変換の目安: 約19分"),
    (51, "初回変換の目安: 約51分"),
    (120, "初回変換の目安: 約2時間"),
    (90, "初回変換の目安: 約1時間30分"),
    (60, "初回変換の目安: 約1時間"),
])
def test_convert_estimate_format(minutes, expected) -> None:
    from app.gui.settings_drawer import format_convert_estimate

    assert format_convert_estimate(minutes) == expected


@pytest.mark.parametrize(("seconds", "expected"), [
    (0.0, "変換中（経過 0:00）"),
    (754.0, "変換中（経過 12:34）"),
    (3723.0, "変換中（経過 1:02:03）"),
])
def test_convert_elapsed_format(seconds, expected) -> None:
    from app.gui.settings_drawer import format_convert_elapsed

    assert format_convert_elapsed(seconds) == expected


def test_npu_section_texts(app, monkeypatch) -> None:
    drawer = _make_drawer(monkeypatch)
    texts = [w.text() for w in drawer.npu_section.findChildren(
        type(drawer.npu_status))]
    assert "NPU の準備" in texts
    assert ("NPU で使うモデルは、最初に一度だけ変換が必要です。"
            "変換中は PC が重くなります。使うモデルだけ変換してください。") in texts

    anime = drawer.npu_rows[HELPER_MODEL_ANIME]
    assert anime["status"].text() == "未変換"
    assert anime["estimate"].text() == _estimate("HELPER_MODEL_ANIME")
    assert anime["button"].text() == "変換する"

    span = drawer.npu_rows[HELPER_MODEL_SPAN]
    assert span["status"].text() == "未変換"
    assert span["estimate"].text() == _estimate("HELPER_MODEL_SPAN")

    adcsr = drawer.npu_rows[HELPER_MODEL_ADCSR]
    assert adcsr["estimate"].text() == _estimate("HELPER_MODEL_ADCSR")


def test_npu_section_converted_row(app, monkeypatch) -> None:
    drawer = _make_drawer(monkeypatch, converted={HELPER_MODEL_SPAN})
    span = drawer.npu_rows[HELPER_MODEL_SPAN]
    assert span["status"].text() == "変換済み"
    assert span["estimate"].text() == ""
    assert span["button"].isHidden() is True


def _redirect_npu_fs(monkeypatch, tmp_path) -> None:
    """実キャッシュ・実モデルを読まないよう空の一時フォルダへ向ける。"""
    models = tmp_path / "models"
    models.mkdir(exist_ok=True)
    cache = tmp_path / "cache"
    cache.mkdir(exist_ok=True)
    vendor = tmp_path / "vendor-models"
    vendor.mkdir(exist_ok=True)
    monkeypatch.setenv(helper_backend.MODELS_DIR_ENV, str(models))
    monkeypatch.setenv(helper_backend.NPU_CACHE_ENV, str(cache))
    monkeypatch.setattr(helper_backend, "DEFAULT_VENDOR_MODELS_DIR", vendor)


def test_npu_hidden_when_unavailable(app, monkeypatch, tmp_path) -> None:
    from app.gui.main_window import MainWindow

    _redirect_npu_fs(monkeypatch, tmp_path)
    monkeypatch.setattr(npu_prepare, "npu_available", lambda: False)
    win = MainWindow()
    win.show()
    app.processEvents()
    try:
        assert win.backend_combo.findData(UpscaleBackend.NPU_NATIVE.value) == -1
        assert win.drawer.npu_section.isHidden() is True
    finally:
        win.close()
        app.processEvents()


def test_npu_backend_present_when_available(app, monkeypatch, tmp_path) -> None:
    from app.gui.main_window import MainWindow

    _redirect_npu_fs(monkeypatch, tmp_path)
    monkeypatch.setattr(npu_prepare, "npu_available", lambda: True)
    win = MainWindow()
    win.show()
    app.processEvents()
    try:
        assert win.backend_combo.findData(UpscaleBackend.NPU_NATIVE.value) >= 0
        assert win.drawer.npu_section.isHidden() is False
    finally:
        win.close()
        app.processEvents()


def test_conversion_success(app, monkeypatch) -> None:
    drawer = _make_drawer(monkeypatch)
    calls: list = []
    monkeypatch.setattr(
        npu_prepare, "convert",
        lambda key, progress=None, cancel=None: calls.append(key))
    anime = drawer.npu_rows[HELPER_MODEL_ANIME]
    anime["button"].click()
    assert _wait_until(lambda: not drawer.is_converting())
    assert calls == [HELPER_MODEL_ANIME]
    assert anime["status"].text() == "変換済み"
    assert anime["estimate"].text() == ""
    assert anime["button"].isHidden() is True
    assert drawer.npu_status.text() == "変換が終わりました。"


def test_conversion_failure(app, monkeypatch) -> None:
    drawer = _make_drawer(monkeypatch)

    def fake_convert(key, progress=None, cancel=None):
        raise RuntimeError("boom npu_native VAIML")

    monkeypatch.setattr(npu_prepare, "convert", fake_convert)
    anime = drawer.npu_rows[HELPER_MODEL_ANIME]
    anime["button"].click()
    assert _wait_until(lambda: not drawer.is_converting())
    assert anime["status"].text() == "失敗"
    assert anime["estimate"].text() == _estimate("HELPER_MODEL_ANIME")
    assert anime["button"].text() == "変換する"
    assert anime["button"].isHidden() is False
    status = drawer.npu_status.text()
    assert status == "変換できませんでした: boom"
    assert "VAIML" not in status
    assert "npu_native" not in status


def test_conversion_cancel(app, monkeypatch) -> None:
    drawer = _make_drawer(monkeypatch)

    def fake_convert(key, progress=None, cancel=None):
        while not cancel.is_set():
            time.sleep(0.01)
        raise Cancelled()

    monkeypatch.setattr(npu_prepare, "convert", fake_convert)
    anime = drawer.npu_rows[HELPER_MODEL_ANIME]
    anime["button"].click()
    assert _wait_until(drawer.is_converting)
    assert anime["button"].text() == "中止"
    anime["button"].click()
    assert _wait_until(lambda: not drawer.is_converting())
    assert anime["status"].text() == "未変換"
    assert anime["estimate"].text() == _estimate("HELPER_MODEL_ANIME")
    assert anime["button"].text() == "変換する"
    assert drawer.npu_status.text() == "中止しました"


def test_conversion_one_at_a_time(app, monkeypatch) -> None:
    drawer = _make_drawer(monkeypatch)
    release = threading.Event()

    def fake_convert(key, progress=None, cancel=None):
        while not release.is_set():
            if cancel.is_set():
                raise Cancelled()
            time.sleep(0.01)

    monkeypatch.setattr(npu_prepare, "convert", fake_convert)
    anime = drawer.npu_rows[HELPER_MODEL_ANIME]
    span = drawer.npu_rows[HELPER_MODEL_SPAN]
    anime["button"].click()
    assert _wait_until(drawer.is_converting)
    try:
        assert anime["button"].text() == "中止"
        assert anime["button"].isEnabled() is True
        assert span["button"].isEnabled() is False
        assert drawer.npu_rows[HELPER_MODEL_SWINIR]["button"].isEnabled() is False
    finally:
        release.set()
        assert _wait_until(lambda: not drawer.is_converting())


def test_conversion_blocked_while_busy(app, monkeypatch) -> None:
    drawer = _make_drawer(monkeypatch)
    calls: list = []
    monkeypatch.setattr(
        npu_prepare, "convert",
        lambda key, progress=None, cancel=None: calls.append(key))
    drawer.set_external_busy(True)
    drawer.npu_rows[HELPER_MODEL_ANIME]["button"].click()
    app.processEvents()
    assert calls == []
    assert drawer.npu_status.text() == "処理中は変換できません"
    assert drawer.is_converting() is False


# --------------------------------------- メイン画面・試しとのつながり
def _make_window_npu(app, monkeypatch, converted=(), identity_backend=True):
    from app.gui.main_window import MainWindow

    monkeypatch.setattr(npu_prepare, "npu_available", lambda: True)
    monkeypatch.setattr(
        npu_prepare, "npu_models", lambda: _canned_models(converted))
    monkeypatch.setattr(
        npu_prepare, "is_converted", lambda key: key in converted)
    if identity_backend:
        monkeypatch.setattr(
            helper_backend, "effective_backend", lambda b, _w, _h: b)
    win = MainWindow()
    win.show()
    app.processEvents()
    npu = win.backend_combo.findData(UpscaleBackend.NPU_NATIVE.value)
    win.backend_combo.setCurrentIndex(npu)
    app.processEvents()
    return win


def _model_item_texts(win):
    return [win.model_combo.itemText(i) for i in range(win.model_combo.count())]


def test_unconverted_suffix_shown_and_cleared(app, monkeypatch) -> None:
    from app.gui.main_window import _combo_closed_text

    win = _make_window_npu(app, monkeypatch)
    try:
        assert any("（未変換）" in text for text in _model_item_texts(win))
        # 閉じた表示はモデル名だけにする。
        assert _combo_closed_text(
            "Anime Video v3（未変換）｜速度○ 画質◎") == "Anime Video v3"
        assert _combo_closed_text("Anime Video v3｜速度○ 画質◎") == "Anime Video v3"

        # 変換が終わると印が消える。
        monkeypatch.setattr(
            npu_prepare, "is_converted", lambda _key: True)
        win._on_npu_converting_changed(False)
        app.processEvents()
        assert all("（未変換）" not in text for text in _model_item_texts(win))
    finally:
        win.close()
        app.processEvents()


def test_start_blocked_for_unconverted_model(app, monkeypatch) -> None:
    win = _make_window_npu(app, monkeypatch)
    try:
        assert SAMPLE_IMAGE.exists()
        ok, _msg = win.add_path(str(SAMPLE_IMAGE))
        assert ok is True
        win._on_start()
        app.processEvents()
        assert win._running is False
        assert win.status_label.text() == (
            "Anime Video v3は NPU 用の変換がまだです。"
            "詳細設定の「NPU の準備」で変換してください。")
    finally:
        win.close()
        app.processEvents()


def test_start_blocked_for_per_file_unconverted_model(
    app, monkeypatch
) -> None:
    from app.core.settings import HELPER_MODEL_ADCSR

    win = _make_window_npu(app, monkeypatch)
    try:
        ok, _msg = win.add_path(str(SAMPLE_IMAGE))
        assert ok is True
        job = next(iter(win._jobs.values()))
        win.queue.select(job.id)
        win._set_settings_tab("image")
        win._per_file_check.setChecked(True)
        app.processEvents()
        win.model_combo.setCurrentIndex(
            win.model_combo.findData(HELPER_MODEL_ADCSR))
        app.processEvents()
        win._on_start()
        app.processEvents()
        assert win._running is False
        assert win.status_label.text() == (
            "AdcSRは NPU 用の変換がまだです。"
            "詳細設定の「NPU の準備」で変換してください。")
    finally:
        win.close()
        app.processEvents()


def test_converting_blocks_start_and_trial(app, monkeypatch) -> None:
    win = _make_window_npu(app, monkeypatch)
    try:
        ok, _msg = win.add_path(str(SAMPLE_IMAGE))
        assert ok is True
        assert _wait_until(lambda: win.preview._source_image is not None)
        win._on_npu_converting_changed(True)
        app.processEvents()
        assert win.start_btn.isEnabled() is False
        win._on_start()
        assert win.status_label.text() == "NPU の変換中は処理を始められません"
        assert win._running is False
        assert win.preview.trial_btn.isEnabled() is False
        assert win.preview.status_label.text() == "NPU の変換中は試せません"
        win._on_npu_converting_changed(False)
        app.processEvents()
        assert win.start_btn.isEnabled() is True
    finally:
        win.close()
        app.processEvents()


def test_trial_blocked_for_unconverted_model(app, monkeypatch) -> None:
    win = _make_window_npu(app, monkeypatch)
    try:
        ok, _msg = win.add_path(str(SAMPLE_IMAGE))
        assert ok is True
        assert _wait_until(lambda: win.preview._source_image is not None)
        win.preview.trial_btn.click()
        app.processEvents()
        assert win.preview.is_trial_running() is False
        assert win.preview.status_label.text() == (
            "Anime Video v3は NPU 用の変換がまだです。"
            "詳細設定の「NPU の準備」で変換してください。")
    finally:
        win.close()
        app.processEvents()


def test_trial_allowed_when_falling_back_to_gpu(app, monkeypatch) -> None:
    # 短辺 480px 未満（input.jpg は 220x220）は GPU に自動切替するため、
    # 未変換でも試しは止めない。実判定のまま確認する。
    from PIL import Image

    from app.core import upscaler

    win = _make_window_npu(app, monkeypatch, identity_backend=False)
    try:
        ok, _msg = win.add_path(str(SAMPLE_IMAGE))
        assert ok is True
        assert _wait_until(lambda: win.preview._source_image is not None)

        def fake(in_path, out_path, settings, progress=None, cancel=None) -> None:
            with Image.open(in_path) as im:
                im.resize((im.width * 4, im.height * 4),
                          Image.NEAREST).save(out_path, "PNG")

        monkeypatch.setattr(upscaler, "upscale_image", fake)
        win.preview.trial_btn.click()
        assert _wait_until(
            lambda: win.preview.right_combo.count() > 1, timeout=20.0)
        assert win.preview.status_label.text() != (
            "Anime Video v3は NPU 用の変換がまだです。"
            "詳細設定の「NPU の準備」で変換してください。")
    finally:
        win.preview.cancel_trial()
        win.close()
        app.processEvents()


def test_close_while_converting_cancels(app, monkeypatch) -> None:
    win = _make_window_npu(app, monkeypatch)
    try:
        started = threading.Event()

        def fake_convert(key, progress=None, cancel=None):
            started.set()
            while not cancel.is_set():
                time.sleep(0.01)
            raise Cancelled()

        monkeypatch.setattr(npu_prepare, "convert", fake_convert)
        win.drawer.refresh_npu_rows()
        win.drawer.npu_rows[HELPER_MODEL_ANIME]["button"].click()
        assert _wait_until(started.is_set)
        assert _wait_until(win.drawer.is_converting)
        win.close()
        app.processEvents()
        assert win._closing is True
        assert _wait_until(lambda: not win.isVisible(), timeout=20.0)
        assert win.drawer.is_converting() is False
    finally:
        try:
            win.drawer.cancel_conversion()
        except RuntimeError:
            pass
        win.close()
        app.processEvents()


# ------------------------------------------------------- 配布スクリプト
def test_build_portable_excludes_npu_files() -> None:
    text = (REPO_ROOT / "scripts" / "build_portable.ps1").read_text(
        encoding="utf-8-sig")
    assert "*bf16cast*" in text
    assert "*_body_*" in text
    assert '"adcsr*"' in text
    assert "NPU は標準では使いません" in text
    assert "別配布の NPU キット（このフォルダに上書きで展開）" in text
    assert "画像や動画を、ウィンドウにドラッグ＆ドロップします" in text


def test_build_npu_kit_script() -> None:
    raw = (REPO_ROOT / "scripts" / "build_npu_kit.ps1").read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")
    text = raw.decode("utf-8-sig")
    assert "-WithAdcSR" in text
    assert "ultraeasy-upscaler-npu-kit.zip" in text
    assert "ultraeasy-upscaler-npu-kit-adcsr.zip" in text
    assert "NPU ドライバ 32.0.203.329 以降" in text
    assert "UEU_NPU_PYTHON" in text

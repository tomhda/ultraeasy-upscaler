"""試しコアのテスト（GUI 非依存）。

実際の推論はしない。`upscaler.upscale_image` を差し替えた偽物で検証する。
ffmpeg でのコマ取り出しだけ実実行する。
"""
from __future__ import annotations

import threading
from pathlib import Path

import pytest
from PIL import Image

from app.core import trial as trial_core
from app.core.jobs import Cancelled
from app.core.settings import (
    HELPER_MODEL_ADCSR,
    HELPER_MODEL_ANIME,
    UpscaleBackend,
    UpscaleSettings,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_VIDEO = REPO_ROOT / "vendor" / "realesrgan" / "onepiece_demo.mp4"


def _make_png(path: Path, size: tuple[int, int] = (100, 80)) -> Path:
    """単色の PNG を作る。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (10, 120, 200)).save(path, "PNG")
    return path


def test_extract_frame_writes_png(tmp_path: Path) -> None:
    """動画の指定秒の1コマを PNG に書き出す。"""
    assert DEMO_VIDEO.exists(), f"デモ動画がありません: {DEMO_VIDEO}"
    out = tmp_path / "trial-frame.png"
    trial_core.extract_frame(str(DEMO_VIDEO), 1.0, str(out))
    assert out.exists()
    with Image.open(out) as im:
        assert im.width > 0 and im.height > 0


def test_crop_image_clamps_to_bounds(tmp_path: Path) -> None:
    """範囲は画像内に丸める。重ならなければ ValueError。"""
    src = _make_png(tmp_path / "src.png", (100, 80))
    exact = tmp_path / "exact.png"
    trial_core.crop_image(str(src), (10, 10, 50, 40), str(exact))
    with Image.open(exact) as im:
        assert (im.width, im.height) == (50, 40)
    clamped = tmp_path / "clamped.png"
    trial_core.crop_image(str(src), (-10, -10, 30, 30), str(clamped))
    with Image.open(clamped) as im:
        assert (im.width, im.height) == (20, 20)
    with pytest.raises(ValueError):
        trial_core.crop_image(str(src), (200, 200, 10, 10), str(tmp_path / "x.png"))


def test_run_trial_passes_same_settings_and_ignores_interpolation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """本処理と同じ設定・同じ経路で呼び、補間だけ無視する。"""
    from app.core import upscaler

    seen: dict = {}

    def fake(in_path: str, out_path: str, settings: UpscaleSettings,
             progress=None, cancel=None) -> None:
        seen["in"] = in_path
        seen["out"] = out_path
        seen["settings"] = settings
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_bytes(b"fake")

    monkeypatch.setattr(upscaler, "upscale_image", fake)
    src = _make_png(tmp_path / "src.png")
    out = tmp_path / "out.png"
    settings = UpscaleSettings(
        backend=UpscaleBackend.WINML_GPU, scale=4, model=HELPER_MODEL_ANIME,
        interpolation_model="rife-v4.6",
    )
    trial_core.run_trial(str(src), settings, str(out))
    assert seen["in"] == str(src)
    assert seen["out"] == str(out)
    assert seen["settings"].backend == UpscaleBackend.WINML_GPU
    assert seen["settings"].model == HELPER_MODEL_ANIME
    assert seen["settings"].scale == 4
    assert seen["settings"].interpolation_model is None
    # 呼び出し元の設定は変えない。
    assert settings.interpolation_model == "rife-v4.6"


def test_run_trial_allows_adcsr_still(tmp_path: Path, monkeypatch) -> None:
    """動画に使えないモデルでも静止画1コマの試しは実行する。"""
    from app.core import upscaler

    called: list = []

    def fake(in_path: str, out_path: str, settings, progress=None, cancel=None) -> None:
        called.append(settings.model)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_bytes(b"fake")

    monkeypatch.setattr(upscaler, "upscale_image", fake)
    src = _make_png(tmp_path / "src.png")
    trial_core.run_trial(
        str(src),
        UpscaleSettings(backend=UpscaleBackend.WINML_GPU, model=HELPER_MODEL_ADCSR),
        str(tmp_path / "out.png"),
    )
    assert called == [HELPER_MODEL_ADCSR]


def test_run_trial_cancel(tmp_path: Path, monkeypatch) -> None:
    """中止イベントで Cancelled になる。"""
    from app.core import upscaler

    def fake(in_path, out_path, settings, progress=None, cancel=None) -> None:
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        raise AssertionError("中止済みのはず")

    monkeypatch.setattr(upscaler, "upscale_image", fake)
    src = _make_png(tmp_path / "src.png")
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(Cancelled):
        trial_core.run_trial(
            str(src),
            UpscaleSettings(backend=UpscaleBackend.WINML_GPU, model=HELPER_MODEL_ANIME),
            str(tmp_path / "out.png"),
            cancel=cancel,
        )


def test_format_time_position() -> None:
    """位置の時刻表示の書式（短時間は M:SS、1時間以上は H:MM:SS）。"""
    assert trial_core.format_time_position(42, 130) == "0:42 / 2:10"
    assert trial_core.format_time_position(0, 0) == "0:00 / 0:00"
    assert trial_core.format_time_position(3723, 7200) == "1:02:03 / 2:00:00"


def test_work_dir_roundtrip() -> None:
    """作業フォルダは ueu-trial- 接頭辞で作り、消せる。"""
    workdir = trial_core.create_work_dir()
    try:
        assert workdir.name.startswith("ueu-trial-")
        assert workdir.is_dir()
    finally:
        trial_core.cleanup_work_dir(workdir)
    assert not workdir.exists()

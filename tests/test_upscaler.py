"""upscaler モジュールの実機テスト（実 GPU 実行・小さく高速）。

プロジェクトルートから実行すること:
  .venv\\Scripts\\python.exe -m pytest tests/test_upscaler.py -v
"""
from __future__ import annotations

import shutil
import threading
from pathlib import Path

import pytest
from PIL import Image

from app.core import binaries, jobs, upscaler
from app.core.settings import UpscaleBackend, UpscaleSettings

REPO = Path(__file__).resolve().parents[1]
INPUT_JPG = REPO / "vendor" / "realesrgan" / "input.jpg"
INPUT2_JPG = REPO / "vendor" / "realesrgan" / "input2.jpg"


def _require_assets() -> None:
    binaries.realesrgan_exe()  # exe が無ければ BinaryError
    if not INPUT_JPG.exists() or not INPUT2_JPG.exists():
        pytest.skip("テスト用入力画像が見つかりません")


class _ProgressRecorder:
    def __init__(self) -> None:
        self.fractions: list[float] = []
        self.messages: list[str] = []

    def __call__(self, fraction: float, message: str) -> None:
        self.fractions.append(fraction)
        self.messages.append(message)


def test_upscale_image(tmp_path: Path) -> None:
    _require_assets()
    out = tmp_path / "out.png"
    rec = _ProgressRecorder()
    settings = UpscaleSettings(scale=4, model="realesrgan-x4plus", image_format="png")

    upscaler.upscale_image(str(INPUT_JPG), str(out), settings, progress=rec)

    # 出力が存在し 4x（220 -> 880）であること。
    assert out.exists()
    with Image.open(out) as im:
        assert im.size == (880, 880)

    # 進捗は単調非減少で 1.0 付近に到達していること。
    assert rec.fractions, "進捗コールバックが呼ばれていない"
    assert rec.fractions == sorted(rec.fractions)
    assert rec.fractions[-1] == pytest.approx(1.0)
    # 中間進捗（0 < f < 1）も観測できていること。
    assert any(0.0 < f < 1.0 for f in rec.fractions)


def test_upscale_image_jpg_format(tmp_path: Path) -> None:
    """出力拡張子に応じて -f が選ばれ、jpg でも書き出せること。"""
    _require_assets()
    out = tmp_path / "out.jpg"
    settings = UpscaleSettings(scale=4, model="realesrgan-x4plus", image_format="jpg")

    upscaler.upscale_image(str(INPUT_JPG), str(out), settings)

    assert out.exists()
    with Image.open(out) as im:
        assert im.format == "JPEG"
        assert im.size == (880, 880)


def test_upscale_folder(tmp_path: Path) -> None:
    _require_assets()
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    shutil.copy(INPUT_JPG, in_dir / "input.jpg")
    shutil.copy(INPUT2_JPG, in_dir / "input2.jpg")

    rec = _ProgressRecorder()
    settings = UpscaleSettings(scale=4, model="realesrgan-x4plus", image_format="png")

    upscaler.upscale_folder(str(in_dir), str(out_dir), settings, progress=rec)

    # 2 枚とも出力されていること（拡張子は -f=png）。
    assert (out_dir / "input.png").exists()
    assert (out_dir / "input2.png").exists()
    # 進捗は最後に 1.0 へ到達。
    assert rec.fractions
    assert rec.fractions[-1] == pytest.approx(1.0)
    assert rec.messages[-1].endswith("枚")


def test_upscale_image_animevideov3_scale2(tmp_path: Path) -> None:
    """multi-scale モデルでスケール接尾辞選択が効くこと（animevideov3 x2）。"""
    _require_assets()
    out = tmp_path / "av3_x2.png"
    settings = UpscaleSettings(scale=2, model="realesr-animevideov3", image_format="png")

    upscaler.upscale_image(str(INPUT_JPG), str(out), settings)

    # 220 -> 440（2x）。-x2 param が選ばれている証拠。
    assert out.exists()
    with Image.open(out) as im:
        assert im.size == (440, 440)


def test_build_cmd_basic() -> None:
    """_build_cmd が想定どおりの引数列を作ること。"""
    settings = UpscaleSettings(
        scale=4, model="realesrgan-x4plus",
        tile_size=0, gpu_id=-1, threads="1:2:2", tta_mode=False,
    )
    cmd = upscaler._build_cmd("in.png", "out.png", settings)

    assert Path(cmd[0]).name.startswith("realesrgan-ncnn-vulkan")
    # 主要オプションが含まれる。
    for flag, val in [("-i", "in.png"), ("-o", "out.png"),
                      ("-n", "realesrgan-x4plus"), ("-s", "4"), ("-j", "1:2:2")]:
        assert flag in cmd
        assert cmd[cmd.index(flag) + 1] == val
    # -m はモデルディレクトリ。
    assert cmd[cmd.index("-m") + 1] == str(binaries.models_dir())
    # 既定では -t / -g / -x は付かない。
    assert "-t" not in cmd
    assert "-g" not in cmd
    assert "-x" not in cmd


def test_build_cmd_optional_flags() -> None:
    """tile/gpu/tta が設定されたとき対応フラグが付くこと。"""
    settings = UpscaleSettings(
        scale=4, model="realesrgan-x4plus",
        tile_size=128, gpu_id=0, threads="2:4:4", tta_mode=True,
    )
    cmd = upscaler._build_cmd("in.png", "out.png", settings)
    assert cmd[cmd.index("-t") + 1] == "128"
    assert cmd[cmd.index("-g") + 1] == "0"
    assert cmd[cmd.index("-j") + 1] == "2:4:4"
    assert "-x" in cmd


def test_build_cmd_unsupported_scale_raises() -> None:
    """対応 param の無い (model, scale) は ValueError。"""
    settings = UpscaleSettings(scale=5, model="realesr-animevideov3")
    with pytest.raises(ValueError):
        upscaler._build_cmd("in.png", "out.png", settings)


def test_upscale_image_cancel(tmp_path: Path) -> None:
    """事前セットされた cancel で即座に Cancelled が送出されること。"""
    _require_assets()
    out = tmp_path / "cancel.png"
    settings = UpscaleSettings(scale=4, model="realesrgan-x4plus", image_format="png")

    cancel = threading.Event()
    cancel.set()  # 開始前からキャンセル状態

    with pytest.raises(jobs.Cancelled):
        upscaler.upscale_image(str(INPUT_JPG), str(out), settings, cancel=cancel)


def test_upscale_image_npu_dispatch(monkeypatch, tmp_path: Path) -> None:
    """backend=npu なら Vulkan exe ではなく NPU backend に委譲する。"""
    calls: list[tuple[str, str, UpscaleSettings]] = []
    out = tmp_path / "out.png"

    def fake_npu(in_path, out_path, settings, progress=None, cancel=None):
        calls.append((in_path, out_path, settings))
        out.write_bytes(b"ok")
        if progress:
            progress(1.0, "完了")

    from app.core import npu_backend

    monkeypatch.setattr(npu_backend, "upscale_image", fake_npu)
    settings = UpscaleSettings(backend=UpscaleBackend.NPU, scale=4)
    upscaler.upscale_image("in.png", str(out), settings)

    assert len(calls) == 1
    assert calls[0][0] == "in.png"
    assert calls[0][1] == str(out)
    assert calls[0][2].backend == UpscaleBackend.NPU


def test_upscale_folder_npu_dispatch(monkeypatch, tmp_path: Path) -> None:
    """フォルダも backend=npu なら NPU backend に委譲する。"""
    calls: list[tuple[str, str, UpscaleSettings]] = []

    def fake_npu(in_dir, out_dir, settings, progress=None, cancel=None):
        calls.append((in_dir, out_dir, settings))
        if progress:
            progress(1.0, "1/1 枚")

    from app.core import npu_backend

    monkeypatch.setattr(npu_backend, "upscale_folder", fake_npu)
    settings = UpscaleSettings(backend=UpscaleBackend.NPU, scale=4)
    upscaler.upscale_folder("in", str(tmp_path / "out"), settings)

    assert len(calls) == 1
    assert calls[0][2].backend == UpscaleBackend.NPU


def test_npu_rejects_unsupported_model(tmp_path: Path) -> None:
    """NPU非対応モデルは conda 起動前に ValueError で弾かれる。"""
    from app.core import npu_backend

    settings = UpscaleSettings(
        backend=UpscaleBackend.NPU, scale=4, model="realesr-general-x4v3"
    )
    with pytest.raises(ValueError, match="NPU"):
        npu_backend.upscale_image(
            str(tmp_path / "a.png"), str(tmp_path / "b.png"), settings
        )

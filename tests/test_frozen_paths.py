"""exe（PyInstaller）版でのパス解決。"""
from __future__ import annotations

import sys
from pathlib import Path


def test_app_root_is_next_to_the_exe_when_frozen(monkeypatch, tmp_path):
    """exe 版では、同梱の models/ や vendor/ を exe の隣から探す。

    __file__ 基準のままだと _internal/ の中を指し、同梱した GPU 用モデルが
    見つからずに Vulkan へ落ちてしまう。
    """
    from app.core import settings

    exe = tmp_path / "ultraeasy-upscaler.exe"
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(exe))

    assert settings._app_root() == tmp_path.resolve()


def test_app_root_is_the_repository_when_run_from_source():
    from app.core import binaries, settings

    assert settings._app_root() == Path(settings.__file__).resolve().parents[2]
    assert settings._app_root() == binaries.repo_root()

# -*- mode: python ; coding: utf-8 -*-
"""Portable build: GUI + CLI exes sharing one onedir folder.

Build with scripts/build_portable.ps1 (do not call PyInstaller by hand).
Both exes go into one COLLECT, so they share the same _internal folder.
Paths here are exe-relative at runtime: settings._app_root() and
binaries.repo_root() both resolve to the exe folder, so the CLI exe finds
models/ and vendor/ next to itself just like the GUI exe.
"""

import os

REPO = SPECPATH  # type: ignore[name-defined]  # folder holding this spec
ENTRY_GUI = os.path.join(REPO, "app", "main.py")
ENTRY_CLI = os.path.join(REPO, "app", "cli_main.py")

block_cipher = None

a_gui = Analysis(  # noqa: F821
    [ENTRY_GUI],
    pathex=[REPO],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    cipher=block_cipher,
    noarchive=False,
)
a_cli = Analysis(  # noqa: F821
    [ENTRY_CLI],
    pathex=[REPO],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    cipher=block_cipher,
    noarchive=False,
)
pyz_gui = PYZ(a_gui.pure, a_gui.zipped_data, cipher=block_cipher)  # noqa: F821
exe_gui = EXE(  # noqa: F821
    pyz_gui,
    a_gui.scripts,
    [],
    exclude_binaries=True,
    name="ultraeasy-upscaler",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

pyz_cli = PYZ(a_cli.pure, a_cli.zipped_data, cipher=block_cipher)  # noqa: F821
exe_cli = EXE(  # noqa: F821
    pyz_cli,
    a_cli.scripts,
    [],
    exclude_binaries=True,
    name="ultraeasy-upscaler-cli",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(  # noqa: F821
    exe_gui,
    exe_cli,
    a_gui.binaries,
    a_gui.datas,
    a_cli.binaries,
    a_cli.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="ultraeasy-upscaler",
)

"""Console entry for the frozen build (ultraeasy-upscaler-cli.exe).

PyInstaller entry script; run from source with ``python -m app.cli`` instead.
"""
from app.cli import main

if __name__ == "__main__":
    raise SystemExit(main())

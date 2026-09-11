"""Download the official SwinIR-M real-world x4 weight into ignored tmp/."""
from __future__ import annotations

import argparse
import hashlib
import shutil
import urllib.request
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth"
MODEL_URL = f"https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/{MODEL_NAME}"
EXPECTED_SHA256 = "b9afb61e65e04eb7f8aba5095d070bbe9af28df76acd0c9405aeb33b814bcfc6"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "tmp" / "swinir-models")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / MODEL_NAME
    if target.is_file() and _sha256(target) == EXPECTED_SHA256:
        print(f"verified {target} sha256={EXPECTED_SHA256}")
        return
    partial = target.with_suffix(target.suffix + ".part")
    partial.unlink(missing_ok=True)
    print(f"downloading {MODEL_URL}")
    with urllib.request.urlopen(MODEL_URL) as response, partial.open("wb") as stream:
        shutil.copyfileobj(response, stream)
    digest = _sha256(partial)
    if digest != EXPECTED_SHA256:
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            f"SwinIR model checksum mismatch: expected={EXPECTED_SHA256} actual={digest}"
        )
    partial.replace(target)
    print(f"saved {target} sha256={digest}")


if __name__ == "__main__":
    main()

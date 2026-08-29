"""新AIモデルの取得・変換手順を再現する補助スクリプト。

重み・ONNX・NPUキャッシュは公開リポジトリへ含めない。既定ではネットワーク
アクセスを行わず、モデル情報と変換コマンドだけを表示する。明示的に
``--download`` を付けたときだけ重みを ``tmp/npu-anime/span`` へ取得し、
SHA-256を検証する。

例::

    .venv\\Scripts\\python.exe scripts\\get_ai_models.py --list
    .venv\\Scripts\\python.exe scripts\\get_ai_models.py --download purephoto
    .venv\\Scripts\\python.exe scripts\\get_ai_models.py --pipeline purephoto --tile 512

ONNX化は既存の ``scripts/npu/export_spandrel.py``、NPU用bf16cast化は
Ryzen AI環境の ``quark.onnx.tools.convert_fp32_to_bf16`` を使用する。
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPAN_DIR = ROOT / "tmp" / "npu-anime" / "span"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    filename: str
    url: str
    sha256: str
    license: str
    attribution: str


MODELS = {
    "purephoto": ModelSpec(
        name="purephoto",
        filename="4xNomosUni_span_multijpg.safetensors",
        url=(
            "https://huggingface.co/Phips/4xNomosUni_span_multijpg/resolve/main/"
            "4xNomosUni_span_multijpg.safetensors"
        ),
        sha256="3BEDFF643A1BA51B12E0174EBCA62649A930AE3E7B0868BE9706D8659D4D32A2",
        license="CC-BY-4.0",
        attribution="Philip Hofmann/Phips",
    ),
    "modernspan": ModelSpec(
        name="modernspan",
        filename="2x_ModernSpanimationV1.pth",
        url=(
            "https://github.com/TNTwise/Models/releases/download/"
            "2x_ModernSpanimationV1/2x_ModernSpanimationV1.pth"
        ),
        sha256="BC6CA08AB1EEB9884A5C43C025EDA97A5AA9CCFAA567734F426AEDF55CF78327",
        license="MIT",
        attribution="TNTwise",
    ),
    "swinir": ModelSpec(
        name="swinir",
        filename="003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth",
        url=(
            "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/"
            "003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth"
        ),
        sha256="B9AFB61E65E04EB7F8ABA5095D070BBE9AF28DF76ACD0C9405AEB33B814BCFC6",
        license="Apache-2.0",
        attribution="Jingyun Liang (SwinIR)",
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _download(spec: ModelSpec) -> Path:
    SPAN_DIR.mkdir(parents=True, exist_ok=True)
    destination = SPAN_DIR / spec.filename
    if destination.is_file():
        actual = _sha256(destination)
        if actual == spec.sha256:
            print(f"取得済み・SHA-256一致: {destination}")
            return destination
        print(f"SHA-256不一致のため再取得: {destination}")

    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(
        spec.url,
        headers={"User-Agent": "ultraeasy-upscaler/get_ai_models"},
    )
    print(f"ダウンロード: {spec.url}")
    try:
        with urllib.request.urlopen(request, timeout=300) as response, temporary.open("wb") as out:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
        actual = _sha256(temporary)
        if actual != spec.sha256:
            raise RuntimeError(
                f"SHA-256不一致: expected={spec.sha256}, actual={actual}"
            )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"取得・検証完了: {destination} ({destination.stat().st_size} bytes)")
    return destination


def _pipeline(spec: ModelSpec, tile: int) -> None:
    weights = SPAN_DIR / spec.filename
    fp32 = SPAN_DIR / f"{spec.name}_nchw_{tile}x{tile}_fp32.onnx"
    bf16 = SPAN_DIR / f"{spec.name}_nchw_{tile}x{tile}_bf16cast.onnx"
    python = ".venv\\Scripts\\python.exe"
    print("\n--- 変換手順（重み・生成物はtmp配下でgit管理外） ---")
    print(
        f"{python} scripts/npu/export_spandrel.py "
        f"--weights {weights.relative_to(ROOT)} --name {spec.name} --tile {tile}"
    )
    print(
        "python -m quark.onnx.tools.convert_fp32_to_bf16 "
        f"--input {fp32.relative_to(ROOT)} --output {bf16.relative_to(ROOT)} "
        "--format with_cast"
    )
    print(f"GPU用fp32: {fp32}")
    print(f"NPU用bf16cast: {bf16}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="登録済みモデルを表示")
    parser.add_argument(
        "--download", nargs="*", metavar="MODEL", choices=sorted(MODELS),
        help="指定モデルを取得してSHA-256検証（省略時は全モデル）",
    )
    parser.add_argument(
        "--pipeline", choices=sorted(MODELS),
        help="指定モデルのspandrel/bf16変換コマンドを表示",
    )
    parser.add_argument("--tile", type=int, choices=(256, 512), default=512)
    return parser


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    args = build_parser().parse_args()

    if args.list or (args.download is None and args.pipeline is None):
        for spec in MODELS.values():
            print(
                f"{spec.name}: {spec.filename}, {spec.license}, "
                f"SHA-256={spec.sha256}, {spec.attribution}"
            )

    if args.download is not None:
        names = args.download or list(MODELS)
        for name in names:
            _download(MODELS[name])

    if args.pipeline:
        _pipeline(MODELS[args.pipeline], args.tile)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""fp32 ONNX を Quark で BF16 化する（キャリブレーション不要）。

Ryzen AI conda 環境で実行:
  python scripts/npu/convert_bf16.py --prefix x4plus_anime
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TILE = int(sys.argv[sys.argv.index("--tile") + 1]) if "--tile" in sys.argv else 256
PREFIX = (sys.argv[sys.argv.index("--prefix") + 1]
          if "--prefix" in sys.argv else "x4plus_anime")
SRC = REPO / "tmp" / "npu-anime" / f"{PREFIX}_nchw_{TILE}x{TILE}_fp32.onnx"
DST = REPO / "tmp" / "npu-anime" / f"{PREFIX}_nchw_{TILE}x{TILE}_bf16.onnx"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    if not SRC.exists():
        print(f"fp32 ONNX がありません: {SRC}")
        return 1

    from quark.onnx import ModelQuantizer
    from quark.onnx.quantization.config import Config, get_default_config

    quant_config = get_default_config("BF16")
    # BF16 は型変換のみでキャリブレーション不要だが、APIがreaderを要求するため
    # ランダムデータで満たす（結果には影響しない）
    quant_config.extra_options["UseRandomData"] = True
    config = Config(global_quant_config=quant_config)
    quantizer = ModelQuantizer(config)
    quantizer.quantize_model(
        model_input=str(SRC),
        model_output=str(DST),
        calibration_data_reader=None,
    )
    if not DST.exists():
        print("BF16出力が生成されませんでした")
        return 1
    print(f"BF16変換完了: {DST.name} ({DST.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

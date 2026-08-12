"""animevideov3 fp32 ONNX を Quark で XINT8 (u8s8) 量子化する。

Ryzen AI conda 環境で実行:
  %USERPROFILE%\\miniforge3\\envs\\ryzen-ai-1.7.1\\python.exe scripts/npu/quantize_animevideov3.py

入力: tmp/npu-anime/animevideov3_nchw_256x256_fp32.onnx
      tmp/npu-anime/calib/*.png（make_calib_patches.py の出力）
出力: tmp/npu-anime/animevideov3_nchw_256x256_u8s8.onnx

XINT8 = activation QUInt8(対称・2のべき乗スケール) + weight QInt8。
既存の realesrgan NPU モデル（producer=quark.onnx）と同じ流儀。
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from onnxruntime.quantization.calibrate import CalibrationDataReader

REPO = Path(__file__).resolve().parents[2]
TILE = int(sys.argv[sys.argv.index("--tile") + 1]) if "--tile" in sys.argv else 256
PREFIX = (sys.argv[sys.argv.index("--prefix") + 1]
          if "--prefix" in sys.argv else "animevideov3")
SRC = REPO / "tmp" / "npu-anime" / f"{PREFIX}_nchw_{TILE}x{TILE}_fp32.onnx"
DST = REPO / "tmp" / "npu-anime" / f"{PREFIX}_nchw_{TILE}x{TILE}_u8s8.onnx"
if "--calib" in sys.argv:
    CALIB_DIR = REPO / "tmp" / "npu-anime" / sys.argv[sys.argv.index("--calib") + 1]
else:
    CALIB_DIR = REPO / "tmp" / "npu-anime" / ("calib" if TILE == 256 else f"calib{TILE}")
CALIB_N = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 100


class TileDataReader(CalibrationDataReader):
    """256x256 パッチを既存ランナーと同一の前処理 (RGB, /255, NCHW) で供給。"""

    def __init__(self, tile_dir: Path, input_name: str = "input", n: int = CALIB_N):
        self.data = []
        for p in sorted(tile_dir.glob("*.png"))[:n]:
            img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            x = img[:, :, ::-1].astype("float32") / 255.0
            x = np.ascontiguousarray(x.transpose(2, 0, 1))[None]
            self.data.append({input_name: x})
        self.it = iter(self.data)

    def get_next(self):
        return next(self.it, None)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    if not SRC.exists():
        print(f"fp32 ONNX がありません: {SRC}（先に export_animevideov3.py を実行）")
        return 1
    reader = TileDataReader(CALIB_DIR)
    if len(reader.data) < min(CALIB_N, 50):
        print(f"キャリブレーション画像不足: {len(reader.data)} 枚"
              "（先に make_calib_patches.py を実行）")
        return 1
    print(f"キャリブレーション: {len(reader.data)} 枚")

    from quark.onnx import ModelQuantizer
    from quark.onnx.quantization.config import Config, get_default_config

    quant_config = get_default_config("XINT8")
    quant_config.enable_npu_cnn = True  # PRelu/DepthToSpace の量子化に必須
    config = Config(global_quant_config=quant_config)

    quantizer = ModelQuantizer(config)
    quantizer.quantize_model(
        model_input=str(SRC),
        model_output=str(DST),
        calibration_data_reader=reader,
    )
    if not DST.exists():
        print("量子化出力が生成されませんでした")
        return 1
    print(f"量子化完了: {DST.name} ({DST.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

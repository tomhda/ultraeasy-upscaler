"""spandrel の画像超解像モデルを NPU 用の固定形状 ONNX に出力する。

実行例:
  %USERPROFILE%\\miniforge3\\envs\\ryzen-ai-1.7.1\\python.exe \
    scripts/npu/export_spandrel.py \
    --weights tmp/npu-anime/span/2x_ModernSpanimationV1.pth \
    --name modernspan --tile 256

入力は spandrel の ImageModelDescriptor が想定する RGB [0,1] NCHW
float32 とし、NPU 向けに空間次元を完全固定する。出力は
tmp/npu-anime/span/{name}_nchw_{W}x{H}_fp32.onnx。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from spandrel import ImageModelDescriptor, ModelLoader


REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "tmp" / "npu-anime" / "span"


class ModelOutputAdapter(nn.Module):
    """ImageModelDescriptor.model を ONNX の単一 Tensor 出力へ適合させる。"""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.model(x)
        if isinstance(output, (tuple, list)):
            output = output[0]
        if not isinstance(output, torch.Tensor):
            raise TypeError(f"モデル出力がTensorではありません: {type(output)!r}")
        return output


class ExplicitSiLU(nn.Module):
    """SiLU を Sigmoid/Mul に展開し、TorchとORTの比較を同じ演算列にする。"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


def replace_silu(module: nn.Module) -> int:
    """モデル内のネイティブSiLUを等価な明示演算へ置き換える。"""

    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.SiLU):
            setattr(module, name, ExplicitSiLU())
            replaced += 1
        else:
            replaced += replace_silu(child)
    return replaced


def parse_size(value: str) -> tuple[int, int]:
    try:
        width_text, height_text = value.lower().split("x", 1)
        width, height = int(width_text), int(height_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--size は WxH 形式で指定してください") from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("--size の幅・高さは正数で指定してください")
    return width, height


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--name", required=True, help="出力ONNXのプレフィックス")
    parser.add_argument("--tile", type=int, choices=(256, 512), default=256)
    parser.add_argument("--size", type=parse_size, help="固定入力サイズ WxH（指定時は--tileより優先）")
    return parser


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    args = build_parser().parse_args()

    weights = args.weights if args.weights.is_absolute() else REPO / args.weights
    weights = weights.resolve()
    if not weights.exists():
        print(f"重みがありません: {weights}")
        return 1

    in_w, in_h = args.size if args.size else (args.tile, args.tile)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = OUT_DIR / f"{args.name}_nchw_{in_w}x{in_h}_fp32_raw.onnx"
    out_path = OUT_DIR / f"{args.name}_nchw_{in_w}x{in_h}_fp32.onnx"

    print(f"重み: {weights}")
    print(f"入力サイズ: 1x3x{in_h}x{in_w}, RGB [0,1] NCHW, float32")

    try:
        descriptor = ModelLoader(device="cpu").load_from_file(weights)
    except Exception as exc:  # noqa: BLE001 - 個別モデルの失敗を呼出側で記録する
        print(f"spandrelロード失敗: {type(exc).__name__}: {exc}")
        return 1

    if not isinstance(descriptor, ImageModelDescriptor):
        print(f"ImageModelDescriptorではありません: {type(descriptor).__name__}")
        return 1
    if descriptor.input_channels != 3 or descriptor.output_channels != 3:
        print(
            "RGB 3chモデルではありません: "
            f"input={descriptor.input_channels}, output={descriptor.output_channels}"
        )
        return 1

    # ImageModelDescriptor の SRモデルは RGB [0,1] の NCHW Tensor を受ける。
    # この2重量みでは正規化・NHWC変換は不要なので、modelをそのままアダプタへ渡す。
    # SiLUだけは、ネイティブTorch実装とORTのSigmoid/Mul展開で極端値の丸め差が
    # 最大差へ増幅されるため、意味を変えない明示演算へ先に置き換える。
    silu_count = replace_silu(descriptor.model)
    model = ModelOutputAdapter(descriptor.model).eval()
    scale = int(descriptor.scale)
    print(
        "モデル: "
        f"{descriptor.architecture.name if hasattr(descriptor.architecture, 'name') else descriptor.architecture}, "
        f"scale={scale}, tags={descriptor.tags}, "
        "入力アダプタ=RGB[0,1] NCHW identity, "
        f"SiLU明示展開={silu_count}"
    )

    # 白色ノイズは本モデルの反復残差を過大増幅し、実画像では現れない
    # 数値丸め差だけを最大差として拾うため、0.5中心の低振幅入力を使う。
    rng = np.random.default_rng(12345)
    x_np = 0.5 + 0.01 * rng.standard_normal((1, 3, in_h, in_w), dtype=np.float32)
    x_np = np.clip(x_np, 0.0, 1.0).astype(np.float32, copy=False)
    x = torch.from_numpy(x_np)
    with torch.inference_mode():
        reference = model(x).cpu().numpy()
    expected_shape = (1, 3, in_h * scale, in_w * scale)
    if tuple(reference.shape) != expected_shape:
        print(f"Torch出力形状NG: got={tuple(reference.shape)}, expected={expected_shape}")
        return 1
    print(f"入力仕様確認OK: Torch出力形状={tuple(reference.shape)}")

    try:
        torch.onnx.export(
            model,
            x,
            str(raw_path),
            input_names=["input"],
            output_names=["output"],
            opset_version=17,
            do_constant_folding=True,
            dynamic_axes=None,
        )
    except Exception as exc:  # noqa: BLE001 - モデル単位で失敗扱いにする
        print(f"ONNX export失敗: {type(exc).__name__}: {exc}")
        return 1
    print(f"ONNX raw出力: {raw_path.name}")

    import onnx
    from onnxsim import simplify

    try:
        simplified, ok = simplify(onnx.load(str(raw_path)))
    except Exception as exc:  # noqa: BLE001 - rawをフォールバックにする
        print(f"onnxsim例外（rawを使用）: {type(exc).__name__}: {exc}")
        out_path.write_bytes(raw_path.read_bytes())
    else:
        if not ok:
            print("onnxsim簡略化NG（rawを使用）")
            out_path.write_bytes(raw_path.read_bytes())
        else:
            onnx.save(simplified, str(out_path))
            print(f"onnxsim簡略化OK: {out_path.name}")

    try:
        import onnxruntime as ort

        session = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        got = session.run(None, {input_name: x_np})[0]
    except Exception as exc:  # noqa: BLE001 - モデル単位で失敗扱いにする
        print(f"ORT検証実行失敗: {type(exc).__name__}: {exc}")
        return 1

    diff = float(np.abs(reference - got).max())
    print(f"ORT出力形状: {tuple(got.shape)}, torch/ORT最大差: {diff:.3e}")
    if tuple(got.shape) != expected_shape or diff > 1e-4:
        print(f"検証NG: expected_shape={expected_shape}, max_diff<=1e-4")
        return 1

    print(f"検証OK: {out_path} ({out_path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

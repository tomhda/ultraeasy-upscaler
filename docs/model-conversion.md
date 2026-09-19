# モデルの取得と変換

配布物（`setup.ps1` が取得する zip）を使わず、重みの取得から ONNX 化・NPU 用の変換までを自分で行う手順。

機械固有の絶対パスはソースへ埋め込まない。重み・ONNX・NPU キャッシュは git へ追加しない。

## Anime Video v3 / 4xNomosUni SPAN / Real-ESRGAN（AMD縮小版）

```powershell
.venv\Scripts\python.exe scripts\get_ai_models.py --list
.venv\Scripts\python.exe scripts\get_ai_models.py --download purephoto
.venv\Scripts\python.exe scripts\get_ai_models.py --pipeline purephoto --tile 512
```

`get_ai_models.py` は重み URL と SHA-256 を検証し、`scripts/npu/export_spandrel.py` による固定形状 fp32 ONNX 化と、
Ryzen AI 環境での bf16cast 変換コマンドを表示する。

```text
python -m quark.onnx.tools.convert_fp32_to_bf16 --input <fp32.onnx> --output <bf16cast.onnx> --format with_cast
```

## SwinIR-M

`scripts/get_ai_models.py --download swinir` で重みを取得し、`scripts/npu/export_spandrel.py --tile 256` で ONNX を生成する。
エクスポート時に、VAIML コンパイラが負の Slice 境界でクラッシュする問題（[amd/RyzenAI-SW#397](https://github.com/amd/RyzenAI-SW/issues/397)）の回避書き換えを自動適用する。
NPU 用は上の bf16cast 変換を行う。CUDA 経路は `scripts\setup_swinir.ps1` で PyTorch CUDA 環境と重みを `tmp/` に導入する（[swinir-experimental.md](swinir-experimental.md)）。

## AdcSR

AdcSR リポジトリを clone し、Hugging Face の Guaishou74851/AdcSR から `net_params_200.pkl` と `halfDecoder.ckpt` を取得し、
Stable Diffusion 2.1-base の diffusers 形式ローカルディレクトリを用意する（公式リポジトリは gated のため、公開ミラー `sd2-community/stable-diffusion-2-1-base` などを使う）。

```powershell
git clone https://github.com/Guaishou74851/AdcSR.git <AdcSR clone>
.venv\Scripts\python.exe scripts/adcsr/export_adcsr.py --repo <AdcSR clone> --sd <SD2.1-base ディレクトリ> --weights <net_params_200.pkl> --half-decoder <halfDecoder.ckpt> --size 128 --out <models>/adcsr_nchw_128x128_fp32.onnx
```

これで GPU（DirectML）経路が使える。NPU 経路は前半（UNet）と後半（VAE デコーダ）の 2 プロセス構成で動かすため、
fp32 の export 後に、正規化の 4D 化 → bf16cast → 前半/後半への切断、の順で用意する。

```powershell
.venv\Scripts\python.exe scripts/adcsr/rewrite_in_to_n5.py --input <fp32 onnx> --output <N5 fp32 onnx>
<Ryzen AI環境のpython> -m quark.onnx.tools.convert_fp32_to_bf16 --input <N5 fp32 onnx> --output <N5 bf16 onnx> --format with_cast
.venv\Scripts\python.exe scripts/adcsr/split_adcsr_npu.py --input <N5 bf16 onnx> --out-dir <models> --cache-key-front <前半cache_key> --cache-key-back <後半cache_key> --fp32 <N5 fp32 onnx> --ref-image <参照png>
```

切断後は前半・後半の ONNX と `adcsr_npu_manifest.json` を models ディレクトリに置く。
初回起動時に前半・後半を順に VAIML コンパイルする（この検証機で前半約 93 分 + 後半約 30 分。次回はキャッシュを利用）。
2 プロセス構成は VitisAI EP の不具合（同一プロセスで後半を実行すると前半の以後の出力が NaN になる。[amd/RyzenAI-SW#402](https://github.com/amd/RyzenAI-SW/issues/402)）の回避で、
SDK 更新後は起動時のセルフテストで検知する。`UEU_ADCSR_NPU2=0` で GPU 実行に戻る。

NPU 側の技術的な背景は [ryzen-ai-npu-super-resolution-notes](https://github.com/tomhda/ryzen-ai-npu-super-resolution-notes) を参照。

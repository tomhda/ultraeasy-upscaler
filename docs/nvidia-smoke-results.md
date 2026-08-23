# NVIDIA GeForce RTX 5060 Ti 実機スモーク

測定日: 2026-08-23（JST）

## 環境

- GPU: NVIDIA GeForce RTX 5060 Ti、ドライバ 610.62、VRAM 16311 MiB
- CPU: AMD Ryzen 7 9700X 8-Core Processor
- 併設GPU: AMD Radeon(TM) Graphics（内蔵、報告VRAM 485 MB）
- Python: 3.13.15
- .NET SDK: 8.0.424
- ffmpeg / ffprobe: 9.0.1 essentials build
- モデル: `animevideov3_nchw_256x256_fp32.onnx`（固定256x256、fp32）
- タイル: overlap 16、ウォームアップ1回

## Windows ML EP列挙

`winml-sr list --download` で次を確認した。

- `NvTensorRTRTXExecutionProvider`: カタログから取得、`Ready`
- `DmlExecutionProvider`: NVIDIA GPUとAMD iGPUの2デバイス
- `NvTensorRTRTXExecutionProvider`: NVIDIA GPU

## 画像1枚

入力は `vendor/realesrgan/input.jpg`（220x220）。

| EP | セッション生成 | タイル処理 | pure-run | wall total |
|---|---:|---:|---:|---:|
| DirectML（NVIDIA GPU） | 1.5秒 | 7.4 ms | 5.8 ms | 2.28秒 |
| NvTensorRTRTXExecutionProvider | 0.7秒 | 7.6 ms | 6.1 ms | 1.67秒 |

DirectMLとTensorRT RTXの出力比較は PSNR **63.87 dB** だった。

初回のDirectML実行ではDMLデバイス2件を同時に渡して
`DML EP factory currently only supports one device at a time` が発生した。
`tools/winml-sr/Program.cs` を修正し、DMLではNVIDIA（Vendor ID `0x10DE`）を優先して1台だけ選択するようにした。
必要な場合は `--device-index` で選択を上書きできる。

## 動画3秒

640x480、24fps、音声付きの3秒クリップ（72フレーム）を使用した。

- AI経路: Python `upscale_video_piped` → WinML常駐ヘルパー → DirectML（NVIDIA GPU）
- タイル: 256x256、9タイル/フレーム
- 結果: 72/72フレーム成功、エラー0
- 出力: 2560x1920、3.000秒、H.264 High、音声AAC
- 実エンコーダ: `h264_nvenc`
- 定常フレームログ: 約155〜159 ms/フレーム
- 推論: 648タイル、pure-run中央値6.0 ms/タイル、合計3.96秒

`video.detect_hw_encoder("h264")` は、NVIDIAドライバ（`nvidia-smi`）が利用できる環境で
NVENCをAMFより先に検証するよう変更した。NVIDIA専用機ではNVENCを優先し、AMD専用機では従来どおりAMFを優先する。

## 現時点の扱い

GUIのGPU経路は引き続きDirectMLを使用する。TensorRT RTXはCLIで実行できることを確認したが、今回の1枚測定ではDirectMLのpure-runが僅かに速く、GUIの既定EPへは追加していない。

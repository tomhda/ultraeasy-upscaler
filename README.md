# ultraeasy-upscaler

Windows ローカル専用の、画像・動画のアップスケールとフレーム補間ツール。
超解像モデルを DirectML GPU・AMD Ryzen AI NPU・NVIDIA CUDA・Vulkan のいずれかで実行する。

> **English summary.** Local image/video upscaler for Windows. Runs Real-ESRGAN, SPAN, SwinIR and AdcSR (one-step diffusion SR)
> on DirectML GPU, AMD Ryzen AI NPU (VitisAI EP, XDNA2), NVIDIA CUDA or Vulkan.
> Technical notes on the NPU side (SwinIR / AdcSR workarounds, DepthToSpace tail-cut, measurements) are in
> [ryzen-ai-npu-super-resolution-notes](https://github.com/tomhda/ryzen-ai-npu-super-resolution-notes).

- 入力: 画像 1 枚、フォルダ、動画（音声保持・H.264 出力）
- 拡大: 4 倍固定の超解像モデル（下表）と、従来の realesrgan-ncnn-vulkan（2x/4x）
- フレーム補間: RIFE v4.6（NCNN/Vulkan）
- 試し: 選んだ 1 コマ（またはその一部）だけを先に拡大し、処理前と左右に並べて見比べる
- NPU で動かすための技術的な記録（コンパイラの回避策、高速化、測定）は別リポジトリ [ryzen-ai-npu-super-resolution-notes](https://github.com/tomhda/ryzen-ai-npu-super-resolution-notes)

## 起動と必要物

### ポータブル版（exe）

[Releases](https://github.com/tomhda/ultraeasy-upscaler/releases/latest) の
`ultraeasy-upscaler-portable-win64.zip` を展開し、`ultraeasy-upscaler.exe` を起動する。
Python・ffmpeg・realesrgan-ncnn-vulkan・RIFE v4.6・DirectML 用ヘルパー（`winml-sr`）と
GPU 用モデル（AdcSR を除く）を同梱している。DirectML GPU と Vulkan で動作する。

NPU は標準では使わない。AMD Ryzen AI 搭載機で NPU を使う場合は、次の 2 つを追加する。

- AMD の Ryzen AI Software 1.8.0（AMD のサイトから入手して導入。NPU ドライバ 32.0.203.329 以降）
- `ultraeasy-upscaler-npu-kit.zip`（NPU キット。exe のフォルダに上書きで展開する）。
  AdcSR を NPU で使う場合は `ultraeasy-upscaler-npu-kit-adcsr.zip` も展開する

NPU 用のモデルは、最初に一度だけ変換が必要になる。詳細設定の「NPU の準備」で、使うモデルだけを選んで変換する
（所要時間は「[実測](#実測)」の表を参照）。

### ソースから起動する場合（`setup.ps1`）

GitHub Release の配布物（ビルド済み `winml-sr` と変換済み ONNX）を使う手順。
`dotnet` SDK とモデルの変換作業は不要。

```
powershell -ExecutionPolicy Bypass -File setup.ps1
```

`-WithAdcSR` で AdcSR（約 1.8GB）を追加取得、`-WithNpu` で NPU 用モデルを追加取得する。
取得後は `run.bat` をダブルクリック、または `.venv\Scripts\python.exe -m app.main` で起動する。
`setup.ps1` は前提の確認、SHA-256 の検証、`vendor/winml-sr/` と `models/ai/` への展開、
サンプル画像での動作確認まで行う。NPU は Ryzen AI Software 1.8.0 の導入案内のみ表示する。
取得元は v0.9.1 の配布物（`winml-sr` とモデルは v0.9.1 から変更なし）。

### 自分でビルド・変換する場合

| 必要物 | 用途 | 必須 |
|---|---|---|
| Python 3.13（同梱の `.venv`） | 本体 | 必須 |
| ffmpeg / ffprobe（PATH 上） | 動画の抽出・再結合・エンコード | 必須 |
| `vendor/realesrgan/`（realesrgan-ncnn-vulkan 一式 exe + models） | Vulkan 経路・フォールバック | 必須 |
| `vendor/rife/`（rife-ncnn-vulkan.exe + rife-v4.6） | フレーム補間 | 任意 |
| `dotnet` 8 SDK で `tools/winml-sr` をビルド | DirectML GPU 経路（自分でビルドする場合のみ） | 任意 |
| Ryzen AI Software 1.8.0 相当の Python 環境と VitisAI EP | NPU 経路 | 任意 |
| PyTorch CUDA 環境（`scripts/setup_swinir.ps1` で `tmp/` に導入） | SwinIR-M CUDA 経路 | 任意 |

Vulkan / RIFE の資材は `.venv\Scripts\python.exe scripts\get_models.py` で取得する。
超解像モデルの取得と変換は「[モデルの取得と変換](#モデルの取得と変換)」を参照。
`setup.ps1` が取得する zip を作り直すには `scripts\build_release.ps1`、ポータブル版と NPU キットは
「[ポータブル版と NPU キットの作成](#ポータブル版と-npu-キットの作成)」を参照。

## 画面と使い方

画面は 3 列。左がメディアの一覧、中央がプレビュー、右が設定と「開始」。

- **追加**: ファイルやフォルダを、ウィンドウのどこにでもドラッグ＆ドロップする。
- **設定**: 右列の「画像」「動画」で、種類ごとに既定のモデルと倍率を決める（動画はフレーム補間モデルも）。
  特定のファイルだけ変える場合は、一覧でそのファイルを選び「このファイルだけ別の設定にする」を入れる。
  一覧の各行に、そのファイルが何で処理されるか（モデル・倍率・補間・個別設定の有無）を表示する。
- **試す**: 中央の「試す」（動画は「このコマを試す」）で、選択中の 1 コマだけを本処理と同じ経路で拡大し、
  境界線の左右に処理前と処理後を並べる。動画はスライダーでコマを選ぶ（初期位置は 2 秒）。
  「範囲を選ぶ」で一部だけを試せる。試したモデルは覚えており、「左」「右」で比べる相手を選べる。
  マウスホイールで拡大、ドラッグで移動、「等倍」で出力の 1 ピクセルを画面の 1 ピクセルに合わせる。
- **やり直す**: 完了した行のやり直しボタン、または「すべてやり直す」で待機中に戻し、設定を変えて再処理する。
- AI実行先・保存形式・出力フォルダなどは、右上の歯車（詳細設定）にある。
- 配色は Windows のライト/ダーク設定とアクセント色に従う（プレビュー枠は常に暗い）。

## AI 実行先

詳細設定の「AI実行先」で実行方式を選ぶ。既定の「自動（GPU優先）」は DirectML GPU に正規化され、
ヘルパーの起動に失敗した画像・フォルダ・動画は Vulkan へフォールバックする。
DirectML / NPU / CUDA は 4 倍固定の常駐ヘルパー（別プロセス）で推論し、
Vulkan を選んだ場合だけ従来の Real-ESRGAN モデル一覧（2x/4x）を表示する。

| AI実行先 | 実行方式 | 必要環境 | 備考 |
|---|---|---|---|
| 自動（GPU優先） | DirectML GPU（`tools/winml-sr`） | ビルド済み `winml-sr.exe` | 起動失敗時は Vulkan |
| GPU（DirectML） | 同上 | 同上 | 明示的に GPU を選ぶ |
| NPU | VitisAI EP（`tools/npu-serve`） | Ryzen AI Software 1.8.0 と `tools/npu-serve`（ポータブル版では NPU キット） | 両方がある PC でだけ選択肢に出る。モデルごとに初回の変換（VAIML コンパイル）が必要 |
| SwinIR-M（CUDA・超低速） | PyTorch CUDA（`tools/swinir`） | CUDA 環境と SwinIR-M 重み | NVIDIA 専用。起動できない場合に別モデルへ自動変更はしない |
| Vulkan | realesrgan-ncnn-vulkan | `vendor/realesrgan` | 従来経路。フォールバック兼用 |

NPU の入力が短辺 480px 未満のときは GPU へ自動切替する。
NPU 用モデルの変換は、詳細設定の「NPU の準備」でモデルごとに実行する。未変換のモデルを選んだまま
「開始」や「試す」を押した場合は、処理を始めずに「NPU の準備」へ案内する（処理の途中で変換は始めない）。
NPU 経路は GPU をほぼ占有しない（推論中の iGPU 3D エンジンは idle 水準、CPU 2〜7%）。

## モデル

DirectML / NPU では具体的なモデル名で選ぶ。GPU と NPU で対応する ONNX とタイルが異なる。

| GUIのモデルキー | 表示名 | 実体モデル | アーキテクチャ | 用途 | 実行先 | 既定タイル (GPU / NPU) |
|---|---|---|---|---|---|---|
| `animevideov3` | Anime Video v3 | realesr-animevideov3（NPU は PReLU 分解版 `dp`） | SRVGGNetCompact | アニメ・線画・CG | GPU / NPU | 256〜512 自動 / 512 |
| `4xNomosUni` | 4xNomosUni SPAN | 4xNomosUni_span_multijpg | SPAN（48nf） | 実写の毛・肌・背景の質感を残す | GPU / NPU | 256〜512 自動 / 512 |
| `AMD-RRDB` | Real-ESRGAN（AMD縮小版） | AMD 縮小 RRDB 版 | RRDB | 輪郭を強く見せたい実写 | GPU / NPU | 256 / 256 |
| `SwinIR` | SwinIR-M | 003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN | SwinIR（window attention） | 静止画の最高画質（低速） | GPU / NPU / CUDA | 256 / 256 |
| `AdcSR` | AdcSR | AdcSR net_params_200（SD2.1-base 派生・1 ステップ） | 生成型 1 ステップ拡散（UNet + VAE デコーダ） | 実写の静止画専用（動画不可） | GPU / NPU | 128 / 128（マージン 32） |

選び方の目安:

- 迷ったら「自動（GPU優先）」。アニメ・CG は Anime Video v3、実写は 4xNomosUni SPAN から。
- 静止画を時間をかけて最高画質にするなら SwinIR-M か AdcSR。AdcSR は生成型なので実写向きで、テクスチャを作り足す。
- GPU を他の作業に使いたいときは NPU。GPU を使わずに回せる（4xNomosUni SPAN は NPU が GPU の約 2 倍速い。Anime Video v3 は静止画で同等、動画は GPU が速い）。
- AdcSR はタイルの継ぎ目に低周波の暗い格子が出るため、マージン 32 とクロスフェード合成に加え、平坦領域限定の固定テンプレート補正を合成時に適用する（[docs/adcsr-tile-diagnosis.md](docs/adcsr-tile-diagnosis.md)）。

## 実測

### AMD Ryzen AI 7 PRO 350（Radeon 860M / XDNA2 NPU）

入力 854x480 → 4 倍（3416x1920）、タイル分割・結合・色変換込みの 1 枚あたり（常駐セッションの定常値、3 回の最良値）。
GPU は fp32 ONNX を DirectML で、NPU は bf16cast を Ryzen AI SW 1.8.0 の VitisAI EP（VAIML コンパイル）で実行。
ドライバ 32.0.203.329、32GB LPDDR5-8000。

| モデル | GPU (DirectML, fp32) | NPU (VitisAI, bf16) | NPU bf16 忠実度* |
|---|---|---|---|
| Anime Video v3 | **0.46 秒** | 0.47 秒 | 49.4 dB |
| 4xNomosUni SPAN | 0.51 秒 | **0.25 秒** | 46.9 dB |
| Real-ESRGAN（AMD縮小版） | 2.78 秒 | 2.07 秒 | 37.9 dB** |
| SwinIR-M | 59 秒 | 82 秒 | 38.5 dB |
| AdcSR（112 タイル） | 108 秒 | 247 秒（1 タイル 2.05 秒 = 前半 0.7 + 後半 1.3） | 45.4 dB*** |

\* 同一モデルの fp32 出力との PSNR。40 dB 前後は目視でほぼ判別不能の水準。
\*\* Ryzen AI 1.7.1 時点の測定値（1.8.0 では速度のみ再測定）。
\*\*\* 1280x534 の写真 1 枚（180 タイル）を GPU 版と比較した値。AdcSR は 1280x534 で GPU 約 4.5 分、NPU 約 6.5 分。
NPU の値は NPU 電源モード Default での測定。`xrt-smi configure --pmode turbo`（AC 電源時）では同じキャッシュのまま
Anime Video v3 0.35 秒、4xNomosUni SPAN 0.18 秒、Real-ESRGAN（AMD縮小版）1.23 秒、SwinIR-M 49 秒、AdcSR 131 秒（1280x534 で約 3.1 分）、
動画は 4xNomosUni SPAN 4.89 fps、Anime Video v3 2.77 fps。出力は Default と同一。
測定条件と高速化の内容は [ryzen-ai-npu-super-resolution-notes](https://github.com/tomhda/ryzen-ai-npu-super-resolution-notes) を参照。

NPU 用モデルの初回変換（VAIML コンパイル）。アプリの「NPU の準備」と同じ条件（プロセス優先度 BelowNormal）で、
空のキャッシュから変換が終わるまでを 2026-10-05 に測定。CPU 使用率は 16 論理コアに対する変換プロセスの値、
メモリは変換プロセスのコミット（private bytes）の最大値。

| モデル（変換する ONNX） | 所要時間 | CPU | メモリの最大 |
|---|---|---|---|
| Anime Video v3（tail-cut body, 512 タイル, bf16cast） | 903 秒（約 15 分） | 約 6%（1 コア相当） | 約 1.9GB |
| 4xNomosUni SPAN（tail-cut body, 512 タイル, fp32 直接） | 850 秒（約 14 分） | 約 6% | 約 1.2GB |
| Real-ESRGAN（AMD縮小版）（256 タイル, bf16cast） | 1529 秒（約 25 分） | 約 6% | 約 1.3GB |
| SwinIR-M（256 タイル, bf16cast） | 3918 秒（約 65 分） | 約 6% | 約 25GB |
| AdcSR（128 タイル, 前半＋後半） | 前半 5201 秒 + 後半 1811 秒（2026-09-12 の測定） | 未測定 | 未測定 |

変換中に並行して実行した CPU 処理の所要時間は、変換なしのときの約 1.07 倍。優先度を Normal にしても所要時間はほぼ同じ（850 秒と 792 秒）。
SwinIR-M の変換中は、物理メモリ 32GB の本機で空きメモリが 124MB まで下がった。

動画（rawvideo パイプライン・音声保持・3 秒クリップの E2E）:

| 経路 | 実効 fps | 1 フレームあたり |
|---|---|---|
| GPU (DirectML) × Anime Video v3 | **2.48 fps** | 0.40 秒 |
| NPU (VitisAI) × Anime Video v3 | 1.95 fps | 0.51 秒 |
| NPU (VitisAI) × 4xNomosUni SPAN | **3.61 fps** | 0.28 秒 |

動画の 1 フレーム値が静止画より速いのは、デコード・変換と推論を重ねて隠すため。
Vulkan 経路（realesrgan-ncnn-vulkan）の animevideov3 は実効約 0.7 秒 / 枚。

### NVIDIA GeForce RTX 5060 Ti（Ryzen 7 9700X）

2026-08-23 の実機確認。`animevideov3` 256 タイル、220x220 入力、overlap 16。

| 経路 | セッション生成 | タイル処理 | wall total |
|---|---:|---:|---:|
| DirectML（NVIDIA GPU） | 1.5 秒 | 7.4 ms | 2.28 秒 |
| NvTensorRTRTXExecutionProvider | 0.7 秒 | 7.6 ms | 1.67 秒 |

両経路の出力 PSNR は 63.87 dB。854x480 入力の wall total は DirectML 1.50 秒、TensorRT 1.60 秒、Vulkan 1.888 秒（AMD 内蔵 GPU の DirectML は 3.11 秒）。
動画（640x480→2560x1920、NVENC）では 12 秒クリップで DirectML 25.2 秒 / TensorRT 25.0 秒と差は約 0.8%、TensorRT の優位はモデル依存。
SwinIR-M CUDA は実写 1 秒の動画で E2E 約 19 秒、640x480 アニメ 1 秒で約 57 秒。
詳細は [docs/nvidia-smoke-results.md](docs/nvidia-smoke-results.md)、[docs/gpu-benchmark-2026-08-24.md](docs/gpu-benchmark-2026-08-24.md)、[docs/swinir-experimental.md](docs/swinir-experimental.md)。

## 動画の処理

- 「アップスケーラーモデル」と「フレーム補間モデル」は独立して選べ、双方に「なし」がある（アップスケールのみ／補間のみ／両方）。
- 両方を選んだ場合の順序は詳細設定「処理の順番」で選ぶ。既定は「アプコン→補間」（重いモデルの対象フレーム数を補間前に抑えられる）。高解像度出力でメモリが厳しい場合のみ「補間→アプコン」。
- 出力は再生互換性を優先して H.264、元音声を保持する。最大出力サイズは `UEU_MAX_VIDEO_DIM`（既定 3840x2160）。
- RIFE 補間が有効な動画は PNG フレーム経路を使う。補間なしの動画は rawvideo 3 スレッドパイプライン（ffmpeg デコード → AI 推論 → ffmpeg エンコード）で、PNG の中間書き出しを省略する。
- SwinIR-M CUDA の動画は 150 フレーム単位で H.264 チャンクを確定し、中止・異常終了後に同じ入力と設定で再実行すると完了済みチャンクを飛ばして再開する（再開データは出力先の `.＜出力名＞.swinir-work-*`、完成後に自動削除）。RIFE との併用と HDR 動画には未対応。
- AdcSR は静止画専用で、動画には使えない。HDR 動画（PQ/HLG）は未対応。
- 「一時停止」は現在のジョブ完了後に停止する。実行中ジョブを今すぐ中止するには一覧の行の × を押す。
- 設定は「開始」を押した時点の値が待機中の各ファイルへ適用される。モデル・倍率・フレーム補間モデルは
  画像用と動画用の既定、またはファイルごとの個別設定から決まり、AI実行先・出力先・詳細設定は全体で共通。

## モデルの取得と変換

`setup.ps1` を使う場合、変換済みのモデルは Release から取得されるので、この作業は不要。
重みの取得から ONNX 化・NPU 用の変換までを自分で行う手順は [docs/model-conversion.md](docs/model-conversion.md)。

## パス設定（環境変数で上書き可能）

| 環境変数 | 既定値 | 用途 |
|---|---|---|
| `UEU_WINML_HELPER` | `vendor/winml-sr/winml-sr.exe`、次に `tools/winml-sr/bin/Release/net*/win-x64/winml-sr.exe` の自動探索 | WinML ヘルパーの明示指定 |
| `UEU_MODELS_DIR` | `models/ai` | GPU fp32 / NPU bf16cast モデルの探索先 |
| `UEU_NPU_PYTHON` | `%USERPROFILE%\miniforge3\envs\ryzen-ai-1.8.0\python.exe` | NPU 常駐サーバーを起動する Python |
| `UEU_NPU_CACHE` | `vendor/amd-npu-1.8` | NPU 用モデルの変換結果（キャッシュ）の置き場所 |
| `UEU_ADCSR_NPU2` | `1` | `0` で AdcSR の NPU 2 プロセス構成を無効化（GPU 実行へ） |
| `UEU_NPU_TAILCUT` | `1` | `0` で NPU tail-cut を無効化（全体モデルで実行） |
| `UEU_SWINIR_PYTHON` | `tmp/swinir-venv/Scripts/python.exe` | SwinIR CUDA 環境の Python |
| `UEU_SWINIR_MODEL` | `tmp/swinir-models/003_*.pth` | SwinIR-M 重みの明示指定 |
| `UEU_SWINIR_STARTUP_TIMEOUT` | `1800` 秒 | CUDA worker の起動待ち（30〜86400 秒） |
| `UEU_SWINIR_CHUNK_FRAMES` | `150` | 動画チェックポイント間隔（100〜300 フレーム） |
| `UEU_MAX_VIDEO_DIM` | `3840x2160` | H.264 出力の最大幅×高さ（例: `1920x1080`） |

## モデルの画質比較

列は左から（すべて GPU/DirectML・fp32 で実行）:

1. オリジナル（lanczos 4x・AI なし）
2. Anime Video v3 (`animevideov3`) = realesr-animevideov3
3. 4xNomosUni SPAN (`4xNomosUni`) = 4xNomosUni_span_multijpg
4. Real-ESRGAN（AMD縮小版） (`AMD-RRDB`) = AMD 縮小 RRDB 版

トゥーン CG — Big Buck Bunny (480p):

![Big Buck Bunny](docs/benchmarks/model_guide_bbb.png)

セル画アニメ — Superman (1941, 320x240):

![Superman 1941](docs/benchmarks/model_guide_sup.png)

実写 — Tears of Steel (720p):

![Tears of Steel](docs/benchmarks/model_guide_tos.png)

- Anime Video v3: 細部を整理してなめらかに。劣化した古い素材に最も強い
- 4xNomosUni SPAN: 原本の質感・粒状感を尊重する忠実系。綺麗なソースで真価
- Real-ESRGAN（AMD縮小版）: 輪郭や毛の 1 本 1 本を立てる知覚系。加工感は強め

## NPU に関する技術メモ

NPU で動かすための回避策、高速化、測定の記録は別リポジトリにまとめている:
[ryzen-ai-npu-super-resolution-notes](https://github.com/tomhda/ryzen-ai-npu-super-resolution-notes)（英語のページと、日本語の研究ノート全文）。

- 報告済みの不具合: [amd/RyzenAI-SW#397](https://github.com/amd/RyzenAI-SW/issues/397)（負の Slice 境界での assertion）、[#398](https://github.com/amd/RyzenAI-SW/issues/398)（長い NPU カーネルでの TDR ライブダンプ）、[#402](https://github.com/amd/RyzenAI-SW/issues/402)（同一プロセスのセッション間で出力が NaN になる）

## ポータブル版と NPU キットの作成

PowerShell 7（`pwsh`）で実行する。

```
pwsh -File scripts\build_portable.ps1 -WithHelper
pwsh -File scripts\build_npu_kit.ps1 -WithAdcSR
```

- `build_portable.ps1` は PyInstaller で `portable_dist/ultraeasy-upscaler/` と `ultraeasy-upscaler-portable-win64.zip` を作る。
  Python・ffmpeg・realesrgan-ncnn-vulkan・RIFE v4.6 を同梱し、`-WithHelper` で `vendor/winml-sr/` と GPU 用モデル（AdcSR を除く）も同梱する。
  同梱物のライセンス文書（`THIRD-PARTY-NOTICES.txt` ほか）も入れる。NPU 専用のファイルは入れない。
  作成後に exe の自己テスト（同梱バイナリとモデルが exe の隣から見つかるか）を実行する。
- `build_npu_kit.ps1` は `tools/npu-serve/` と NPU 用モデルを `ultraeasy-upscaler-npu-kit.zip` にまとめる。
  `-WithAdcSR` で AdcSR の前半・後半とマニフェストを `ultraeasy-upscaler-npu-kit-adcsr.zip`（無圧縮）にまとめる。
  どちらも exe のフォルダに上書きで展開する構造。

## アーキテクチャ

- **GUI**: PySide6。`app/gui`
  - `main_window.py` 3 列の画面、画像用・動画用・個別の設定、やり直し
  - `queue_view.py` メディアの一覧、`compare_view.py` 左右比較と試し、`settings_drawer.py` 詳細設定と「NPU の準備」
  - `theme.py` 配色（Windows のライト/ダークとアクセント色に連動）
- **コア（GUI 非依存・単体テスト可）**: `app/core`
  - `binaries.py` 外部バイナリ / モデル探索
  - `media.py` 種別判定・ffprobe メタ取得
  - `upscaler.py` 常駐ヘルパーと realesrgan-ncnn-vulkan の統合ラッパ（画像 / フォルダ）
  - `helper_backend.py` / `serve_client.py` DirectML / NPU / CUDA ヘルパーのモデル解決・常駐セッション・バイナリプロトコル
  - `video.py` ffmpeg 抽出 / 再結合 / HW エンコード、rawvideo パイプライン、SwinIR CUDA のチャンク再開
  - `interpolator.py` RIFE NCNN/Vulkan フレーム補間
  - `engine.py` ジョブのオーケストレーション、`jobs.py` / `settings.py` データモデル
  - `trial.py` 試し（コマの取り出し・切り出し・本処理と同じ経路での 1 枚の拡大）
  - `npu_prepare.py` NPU キットの有無、モデルごとの変換状態、変換の実行
  - `npu_backend.py` / `npu_worker.py` 旧 NPU API（スクリプト互換のため残置、GUI では使用しない）
- **ヘルパー（常駐プロセス）**: `tools/winml-sr/`（C#、DirectML。クロスフェード合成と AdcSR の格子補正を含む）、
  `tools/npu-serve/`（Python、VitisAI EP。AdcSR は `npu_worker.py` × 2 と `npu_twostage.py` の 2 プロセス構成）、
  `tools/swinir/`（Python、PyTorch CUDA）

## 開発

```
.venv\Scripts\python.exe -m pytest        # コアの単体テスト
```

## ライセンスと帰属

- 本リポジトリのコード: [MIT](LICENSE)
- `vendor/amd-npu/` の Real-ESRGAN NPU モデル: AMD 公式モデル由来のため [Research-only RAIL-MS](vendor/amd-npu/LICENSE)（研究用途限定）
- Anime Video v3 / Real-ESRGAN Anime の NPU モデル: BSD-3-Clause の [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) 重みから変換
- 4xNomosUni_span_multijpg: CC-BY-4.0, by Philip Hofmann/Phips（取得元と SHA-256 は [docs/span-bench-results.md](docs/span-bench-results.md)）
- 003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN: Apache-2.0, by Jingyun Liang（SwinIR）。CUDA 経路の実装由来は `tools/swinir/NOTICE.md`
- AdcSR: Apache-2.0, Guaishou74851（CVPR 2025）。基盤の Stable Diffusion 2.1-base は CreativeML Open RAIL++-M の適用対象で、使用前に利用者自身が使用条件を確認すること（全文は Hugging Face の配布ページにあり、ログインが必要）
- Release 配布物の zip には上記のライセンス文（`scripts/release-licenses/` と同一内容）と各モデルの NOTICE を同梱する。`winml-sr-win-x64.zip` には Microsoft Windows ML Runtime の license.txt と ThirdPartyNotices.txt を同梱し、再配布条件（license.txt §3）を NOTICE-winml-sr.txt に記載する。AdcSR の zip には同系統の CreativeML Open RAIL-M 全文（使用制限 Attachment A を含む）を同梱し、再配布時は同じ使用制限を利用者に課す
- ポータブル版には FFmpeg（gyan.dev の full build、GPL v3）、Qt 6 / PySide6（LGPL v3）、realesrgan-ncnn-vulkan（MIT）、rife-ncnn-vulkan（MIT）などを
  それぞれのライセンスのまま同梱する。一覧と入手元は同梱の `THIRD-PARTY-NOTICES.txt`（原本は `scripts/portable-notices/`）。
  上流の配布物に含まれるサンプル画像・動画は同梱しない
- ベンチマーク画像の素材: [Big Buck Bunny](https://peach.blender.org) / [Tears of Steel](https://mango.blender.org)（© Blender Foundation, CC-BY 3.0）、Superman (1941) はパブリックドメイン

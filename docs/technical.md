# 技術的な詳細

使い方は [README](../README.md) を参照。このページは、ソースからの起動、実行方式、実測、内部の構成をまとめる。
NPU で動かすための回避策・高速化・測定の記録は、別リポジトリ
[ryzen-ai-npu-super-resolution-notes](https://github.com/tomhda/ryzen-ai-npu-super-resolution-notes) にある。

## ソースから起動する

### Release の配布物を使う（`setup.ps1`）

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
「開始」や「クイック確認」を押した場合は、処理を始めずに「NPU の準備」へ案内する（処理の途中で変換は始めない）。
NPU 経路は GPU をほぼ占有しない（推論中の iGPU 3D エンジンは idle 水準、CPU 2〜7%）。

## モデルの実体

DirectML / NPU では具体的なモデル名で選ぶ。GPU と NPU で対応する ONNX とタイルが異なる。

| GUIのモデルキー | 表示名 | 実体モデル | アーキテクチャ | 用途 | 実行先 | 既定タイル (GPU / NPU) |
|---|---|---|---|---|---|---|
| `animevideov3` | Anime Video v3 | realesr-animevideov3（NPU は PReLU 分解版 `dp`） | SRVGGNetCompact | アニメ・線画・CG | GPU / NPU | 256〜512 自動 / 512 |
| `4xNomosUni` | 4xNomosUni SPAN | 4xNomosUni_span_multijpg | SPAN（48nf） | 実写の毛・肌・背景の質感を残す | GPU / NPU | 256〜512 自動 / 512 |
| `AMD-RRDB` | Real-ESRGAN（AMD縮小版） | AMD 縮小 RRDB 版 | RRDB | 輪郭を強く見せたい実写 | GPU / NPU | 256 / 256 |
| `SwinIR` | SwinIR-M | 003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN | SwinIR（window attention） | 静止画の最高画質（低速） | GPU / NPU / CUDA | 256 / 256 |
| `AdcSR` | AdcSR | AdcSR net_params_200（SD2.1-base 派生・1 ステップ） | 生成型 1 ステップ拡散（UNet + VAE デコーダ） | 実写の静止画専用（動画不可） | GPU / NPU | 128 / 128（マージン 32） |

- AdcSR はタイルの継ぎ目に低周波の暗い格子が出るため、マージン 32 とクロスフェード合成に加え、平坦領域限定の固定テンプレート補正を合成時に適用する（[adcsr-tile-diagnosis.md](adcsr-tile-diagnosis.md)）。

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
詳細は [docs/nvidia-smoke-results.md](nvidia-smoke-results.md)、[docs/gpu-benchmark-2026-08-24.md](gpu-benchmark-2026-08-24.md)、[docs/swinir-experimental.md](swinir-experimental.md)。

## 動画の処理

- 「アップスケーラーモデル」と「フレーム補間モデル」は独立して選べ、双方に「なし」がある（アップスケールのみ／補間のみ／両方）。
- 両方を選んだ場合の順序は詳細設定「処理の順番」で選ぶ。既定は「アプコン→補間」（重いモデルの対象フレーム数を補間前に抑えられる）。高解像度出力でメモリが厳しい場合のみ「補間→アプコン」。
- 出力は再生互換性を優先して H.264、元音声を保持する。最大出力サイズは `UEU_MAX_VIDEO_DIM`（既定 3840x2160）。
- フレーム補間は RIFE v4.6（NCNN/Vulkan）と FILM (Style)（Windows ML/DirectML、追加キット）。補間後の fps は既定で元の 2 倍。
  FILM (Style) は中間時刻 0.5 のフレームを再帰的に作るので、元の 2・4・8 倍のみ。拡大と併用するときは、
  拡大後の絵を渡すとメモリが足りなくなるため、「処理の順番」にかかわらず常に補間してから拡大する。
- RIFE / FILM 補間が有効な動画は PNG フレーム経路を使う。補間なしの動画は rawvideo 3 スレッドパイプライン（ffmpeg デコード → AI 推論 → ffmpeg エンコード）で、PNG の中間書き出しを省略する。
- SwinIR-M CUDA の動画は 150 フレーム単位で H.264 チャンクを確定し、中止・異常終了後に同じ入力と設定で再実行すると完了済みチャンクを飛ばして再開する（再開データは出力先の `.＜出力名＞.swinir-work-*`、完成後に自動削除）。RIFE との併用と HDR 動画には未対応。
- AdcSR は静止画専用で、動画には使えない。HDR 動画（PQ/HLG）は未対応。
- 「一時停止」は現在のジョブ完了後に停止する。実行中ジョブを今すぐ中止するには一覧の行の × を押す。
- 設定は「開始」を押した時点の値が待機中の各ファイルへ適用される。モデル・倍率・フレーム補間モデルは
  画像用と動画用の既定、またはファイルごとの個別設定から決まり、AI実行先・出力先・詳細設定は全体で共通。

## モデルの取得と変換

`setup.ps1` を使う場合、変換済みのモデルは Release から取得されるので、この作業は不要。
重みの取得から ONNX 化・NPU 用の変換までを自分で行う手順は [docs/model-conversion.md](model-conversion.md)。

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
| `UEU_FILM_HELPER` | `vendor/winml-film/winml-film.exe`、次に `tools/winml-film/bin/Release/net*/win-x64/winml-film.exe` | FILM ヘルパーの明示指定 |
| `UEU_FILM_MODEL` | `models/film/film_style_fp32.onnx` | FILM の ONNX の明示指定 |
| `UEU_LANG` | Windows の表示言語 | 表示言語（`ja` / `en`）。コマンドラインは既定で `en` |

## ポータブル版と追加キットの作成

PowerShell 7（`pwsh`）で実行する。

```
pwsh -File scripts\build_portable.ps1 -WithHelper
pwsh -File scripts\build_npu_kit.ps1 -WithAdcSR
pwsh -File scripts\build_film_kit.ps1
pwsh -File scripts\build_adcsr_kit.ps1
```

- `build_portable.ps1` は PyInstaller で `portable_dist/togu-scaler/` と `togu-scaler-portable-win64.zip` を作る。
  Python・ffmpeg・realesrgan-ncnn-vulkan・RIFE v4.6 を同梱し、`-WithHelper` で `vendor/winml-sr/` と GPU 用モデル（AdcSR を除く）も同梱する。
  同梱物のライセンス文書（`THIRD-PARTY-NOTICES.txt` ほか）も入れる。NPU 専用のファイルは入れない。
  画面用の `togu-scaler.exe` とコマンドライン用の `togu-scaler-cli.exe` を、同じ `_internal` を共有する形で作る
  （定義は `togu-scaler.spec`）。
  作成後に exe の自己テスト（同梱バイナリとモデルが exe の隣から見つかるか）と、`togu-scaler-cli.exe status` を実行する。
- `build_npu_kit.ps1` は `tools/npu-serve/` と NPU 用モデルを `togu-scaler-npu-kit.zip` にまとめる。
  `-WithAdcSR` で AdcSR の前半・後半とマニフェストを `togu-scaler-npu-kit-adcsr.zip`（無圧縮）にまとめる。
- `build_film_kit.ps1` は `vendor/winml-film/` と `models/film/` を `togu-scaler-film-kit.zip` にまとめる。
  FILM の ONNX の作り方は [scripts/film/README.md](../scripts/film/README.md)。
- `build_adcsr_kit.ps1` は GPU 用の AdcSR モデルを `togu-scaler-adcsr-kit.zip` にまとめる。
- どのキットも exe のフォルダに上書きで展開する構造。

## アーキテクチャ

- **GUI**: PySide6。`app/gui`
  - `main_window.py` ヘッダーの一括設定と 3 列の画面、ファイルごとの個別設定、やり直し
  - `queue_view.py` メディアの一覧、`compare_view.py` 左右比較とクイック確認、`settings_drawer.py` 詳細設定・「追加キット」・「NPU の準備」
  - `theme.py` 配色（Windows のライト/ダークとアクセント色に連動）
- **コマンドライン**: `app/cli.py`（Qt に依存しない。コアを呼ぶだけ）。使い方は [AGENTS.md](../AGENTS.md)
- **コア（GUI 非依存・単体テスト可）**: `app/core`
  - `catalog.py` モデルの表示名・説明・実行先ごとの一覧（画面とコマンドラインで共有）
  - `binaries.py` 外部バイナリ / モデル探索
  - `media.py` 種別判定・ffprobe メタ取得
  - `upscaler.py` 常駐ヘルパーと realesrgan-ncnn-vulkan の統合ラッパ（画像 / フォルダ）
  - `helper_backend.py` / `serve_client.py` DirectML / NPU / CUDA ヘルパーのモデル解決・常駐セッション・バイナリプロトコル
  - `video.py` ffmpeg 抽出 / 再結合 / HW エンコード、rawvideo パイプライン、SwinIR CUDA のチャンク再開
  - `interpolator.py` RIFE NCNN/Vulkan フレーム補間と FILM への振り分け、`film.py` FILM ヘルパーの実行
  - `addon_kits.py` 追加キットの導入判定とダウンロード先
  - `engine.py` ジョブのオーケストレーション、`jobs.py` / `settings.py` データモデル
  - `trial.py` クイック確認（コマの取り出し・切り出し・本処理と同じ経路での 1 枚の拡大）
  - `npu_prepare.py` NPU キットの有無、モデルごとの変換状態、変換の実行
  - `npu_backend.py` / `npu_worker.py` 旧 NPU API（スクリプト互換のため残置、GUI では使用しない）
- **ヘルパー（常駐プロセス）**: `tools/winml-sr/`（C#、DirectML。クロスフェード合成と AdcSR の格子補正を含む）、
  `tools/npu-serve/`（Python、VitisAI EP。AdcSR は `npu_worker.py` × 2 と `npu_twostage.py` の 2 プロセス構成）、
  `tools/swinir/`（Python、PyTorch CUDA）、`tools/winml-film/`（C#、DirectML。FILM の補間。常駐せず、フレームのフォルダ単位で起動）

## 開発

```
.venv\Scripts\python.exe -m pytest        # コアの単体テスト
```

README の「画質の比較」の画像は `scripts/make_model_guide.py` で作り直せる（各素材をアプリと同じ経路で 5 つのモデルに通す）。

## ライセンスと、配布物への同梱

- 本リポジトリのコード: [MIT](../LICENSE)
- `vendor/amd-npu/` の Real-ESRGAN NPU モデル: AMD 公式モデル由来のため [Research-only RAIL-MS](../vendor/amd-npu/LICENSE)（研究用途限定）
- Anime Video v3 / Real-ESRGAN Anime の NPU モデル: BSD-3-Clause の [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) 重みから変換
- 4xNomosUni_span_multijpg: CC-BY-4.0, by Philip Hofmann/Phips（取得元と SHA-256 は [docs/span-bench-results.md](span-bench-results.md)）
- 003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN: Apache-2.0, by Jingyun Liang（SwinIR）。CUDA 経路の実装由来は `tools/swinir/NOTICE.md`
- AdcSR: Apache-2.0, Guaishou74851（CVPR 2025）。基盤の Stable Diffusion 2.1-base は CreativeML Open RAIL++-M の適用対象で、使用前に利用者自身が使用条件を確認すること（全文は Hugging Face の配布ページにあり、ログインが必要）
- Release 配布物の zip には上記のライセンス文（`scripts/release-licenses/` と同一内容）と各モデルの NOTICE を同梱する。`winml-sr-win-x64.zip` には Microsoft Windows ML Runtime の license.txt と ThirdPartyNotices.txt を同梱し、再配布条件（license.txt §3）を NOTICE-winml-sr.txt に記載する。AdcSR の zip には同系統の CreativeML Open RAIL-M 全文（使用制限 Attachment A を含む）を同梱し、再配布時は同じ使用制限を利用者に課す
- ポータブル版には FFmpeg（gyan.dev の full build、GPL v3）、Qt 6 / PySide6（LGPL v3）、realesrgan-ncnn-vulkan（MIT）、rife-ncnn-vulkan（MIT）などを
  それぞれのライセンスのまま同梱する。一覧と入手元は同梱の `THIRD-PARTY-NOTICES.txt`（原本は `scripts/portable-notices/`）。
  上流の配布物に含まれるサンプル画像・動画は同梱しない
- ベンチマーク画像の素材: [Big Buck Bunny](https://peach.blender.org) / [Tears of Steel](https://mango.blender.org)（© Blender Foundation, CC-BY 3.0）、Superman (1941) はパブリックドメイン

# ultraeasy-upscaler

**[⬇ ultraeasy-upscaler v0.10.0 をダウンロード](https://github.com/tomhda/ultraeasy-upscaler/releases/download/v0.10.0/ultraeasy-upscaler-portable-win64.zip)**（Windows x64 用の zip、376 MB）。手順は「[導入](#導入)」を参照。

ultraeasy-upscaler は、画像と動画を AI で 4 倍に拡大する Windows 用のアプリ。

処理はすべて手元の PC で行い、ファイルを外部に送らない。

動画の拡大は時間がかかるので、先に 1 コマだけ拡大して、処理前と左右に並べて確かめられる。

動画のコマ数を増やしてなめらかにするフレーム補間（RIFE v4.6）もできる。

> **English summary.** Local image/video upscaler for Windows. Runs Real-ESRGAN, SPAN, SwinIR and AdcSR (one-step diffusion SR)
> on DirectML GPU, AMD Ryzen AI NPU (VitisAI EP, XDNA2), NVIDIA CUDA or Vulkan. Download the portable zip above.
> Technical notes on the NPU side are in
> [ryzen-ai-npu-super-resolution-notes](https://github.com/tomhda/ryzen-ai-npu-super-resolution-notes).

![1 コマを試して、処理前と処理後を左右に並べた画面](docs/images/main.jpg)

## できること

- 画像、フォルダ、動画を 4 倍に拡大する。動画は元の音声を残し、H.264 で保存する。
- 本処理の前に、選んだ 1 コマ（またはその一部）だけを拡大して、処理前と見比べる。試したモデル同士も比べられる。
- アニメ向け・実写向けなど、5 つのモデルから選ぶ。画像用と動画用で別のモデルを決めておける。
- 特定のファイルだけ、別のモデルで処理する。
- 終わったファイルを、一覧に残したまま別のモデルでやり直す。
- 動画のフレーム補間（RIFE v4.6）を、拡大と組み合わせて、または単独で行う。
- GPU（DirectX 12 対応のもの）で動作する。AMD Ryzen AI 搭載機では、追加のキットを入れると NPU でも動作する。
- 配色は Windows のライト/ダーク設定とアクセント色に従う。

## 画面

▼ 左がメディアの一覧、中央がプレビュー、右が設定と「開始」。一覧の各行に、そのファイルを何で処理するかが出る。
中央は「このコマを試す」を押した後の状態で、境界線の左が処理前、右が処理後。境界線はドラッグで動かせる。

![メイン画面](docs/images/main.jpg)

▼ 「等倍」にすると、出力の 1 ピクセルを画面の 1 ピクセルに合わせて細部を比べられる。
この画像は「このファイルだけ別の設定にする」を入れて、1 枚だけ別のモデルにした状態。

![等倍での比較](docs/images/compare-zoom.jpg)

画面の素材は [Big Buck Bunny](https://peach.blender.org) と [Tears of Steel](https://mango.blender.org)（© Blender Foundation, CC BY 3.0）、Superman (1941) はパブリックドメイン。

## 導入

Windows 11（x64）と、DirectX 12 対応の GPU で動作を確認している。

1. [`ultraeasy-upscaler-portable-win64.zip`](https://github.com/tomhda/ultraeasy-upscaler/releases/download/v0.10.0/ultraeasy-upscaler-portable-win64.zip) をダウンロードする。
2. 好きな場所に展開する。インストールは不要。
3. `ultraeasy-upscaler.exe` を起動する。署名のない実行ファイルなので、Windows の SmartScreen が「Windows によって PC が保護されました」と表示することがある。その場合は「詳細情報」→「実行」を選ぶ。

zip には動作に必要なものがすべて入っている。フォルダの中のファイルは移動や削除をしないこと。
削除するときは、展開したフォルダごと消す。

## 使い方

### 追加する

- 画像、動画、フォルダを、ウィンドウのどこにでもドラッグ＆ドロップする。左下の枠をクリックして選ぶこともできる。

### モデルを選ぶ

- 右側の「画像」「動画」で、種類ごとに使うモデルを選ぶ。モデルの下に、向き不向きと速さの説明が出る。
- 一覧でファイルを選んで「このファイルだけ別の設定にする」を入れると、そのファイルだけ別のモデルにできる。
- どれを選ぶかは「[モデルの選び方](#モデルの選び方)」を参照。

### 試す

- 一覧でファイルを選び、中央の「試す」（動画は「このコマを試す」）を押す。そのコマだけを本処理と同じ方法で拡大し、処理前と並べて表示する。
- 動画は、プレビューの下のスライダーで試すコマを選ぶ。最初は 2 秒の位置になっている。
- 「範囲を選ぶ」で、画の一部だけを試せる。時間のかかるモデルを短時間で確かめるときに使う。
- モデルを変えてもう一度試すと、結果が増える。「左」「右」で、処理前と各モデルの結果から比べる相手を選ぶ。
- マウスホイールで拡大、ドラッグで移動。「全体表示」「等倍」で表示倍率を切り替える。

### 開始する

- 「開始」を押すと、待機中のファイルを上から順に処理する。出力は元のファイルと同じ場所の `upscaled` フォルダに保存する（「出力先」と詳細設定で変更できる）。
- 「一時停止」は、処理中のファイルが終わってから止まる。今すぐ止めるには、その行の × を押す。
- 既にあるファイルは上書きせず、`(1)` のように番号を付ける。

### やり直す

- 終わった行の丸い矢印、または「すべてやり直す」で待機中に戻す。モデルを変えてから「開始」を押すと、もう一度処理する。

### 詳細設定

- 右上の歯車を押すと、AI の実行先、保存形式、動画の画質、出力フォルダ名、フレーム補間後の fps などを変更できる。

## モデルの選び方

時間は Radeon 860M（Ryzen AI 7 PRO 350 の内蔵 GPU）で 854×480 の 1 枚を 4 倍にしたときの値。

| モデル | 向いているもの | 1 枚の時間 | 補足 |
|---|---|---|---|
| Anime Video v3 | アニメ・線画・CG | 0.46 秒 | 細部を整理してなめらかにする。劣化した古い素材に強い。実写には向かない |
| 4xNomosUni SPAN | 実写 | 0.51 秒 | 元の質感や粒状感を残す。きれいな素材に向く |
| Real-ESRGAN（AMD縮小版） | 実写 | 2.78 秒 | 輪郭や毛を 1 本ずつ立てる。加工感は強め。研究用途限定のライセンス |
| SwinIR-M | 実写の静止画 | 59 秒 | 時間をかけて細部まで復元する |
| AdcSR | 実写の静止画 | 108 秒 | 生成型で、質感を作り足す。動画には使えない。ポータブル版には含まない（下記） |

- 迷ったら、アニメ・CG は Anime Video v3、実写は 4xNomosUni SPAN から試す。
- AdcSR を使うには、[v0.9.1 の `models-adcsr-gpu-fp32.zip`](https://github.com/tomhda/ultraeasy-upscaler/releases/tag/v0.9.1)（約 1.8GB）を
  `models\ai` フォルダに展開する。使用条件は zip 内の NOTICE を参照。
- 詳細設定の「AI実行先」で Vulkan を選ぶと、従来の realesrgan-ncnn-vulkan のモデル（2 倍・4 倍）を使える。

### 画質の比較

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

## NPU を使う場合

AMD Ryzen AI 搭載機では、GPU を空けたまま NPU で処理できる。標準では使わず、次の 2 つを入れた PC でだけ選べるようになる。

- AMD の Ryzen AI Software 1.8.0（AMD のサイトから入手して導入。NPU ドライバ 32.0.203.329 以降）
- [`ultraeasy-upscaler-npu-kit.zip`](https://github.com/tomhda/ultraeasy-upscaler/releases/download/v0.10.0/ultraeasy-upscaler-npu-kit.zip)（56 MB）。
  AdcSR も NPU で使う場合は [`ultraeasy-upscaler-npu-kit-adcsr.zip`](https://github.com/tomhda/ultraeasy-upscaler/releases/download/v0.10.0/ultraeasy-upscaler-npu-kit-adcsr.zip)（1.7 GB）も

手順:

1. ultraeasy-upscaler を終了し、キットの zip の中身を `ultraeasy-upscaler.exe` があるフォルダに上書きで展開する。
2. 起動して歯車を押し、「NPU の準備」で使うモデルの「変換する」を押す。変換はモデルごとに最初の一度だけ必要。
3. 変換が終わったら、「AI実行先」で NPU を選ぶ。

▼ 「NPU の準備」。モデルごとに、変換の目安の時間が出る。

![NPU の準備](docs/images/npu-prepare.jpg)

変換にかかる時間とメモリ（Ryzen AI 7 PRO 350 での実測）:

| モデル | 時間 | メモリの最大 |
|---|---|---|
| Anime Video v3 | 約 15 分 | 約 1.9GB |
| 4xNomosUni SPAN | 約 14 分 | 約 1.2GB |
| Real-ESRGAN（AMD縮小版） | 約 25 分 | 約 1.3GB |
| SwinIR-M | 約 65 分 | 約 25GB |
| AdcSR | 約 2 時間 | 未測定 |

変換中の CPU 使用は 1 コア分で、PC はそのまま使える。SwinIR-M はメモリを多く使うので、ほかのアプリを閉じてから実行する。
NPU ドライバや Ryzen AI Software を更新すると、変換のやり直しが必要になることがある。

NPU で動かすための技術的な記録（コンパイラの回避策、高速化、測定）は、別のリポジトリ
[ryzen-ai-npu-super-resolution-notes](https://github.com/tomhda/ryzen-ai-npu-super-resolution-notes) にまとめている。

## 制限

- 拡大は 4 倍固定。2 倍は、AI実行先を Vulkan にしたときの一部のモデルだけ。
- 動画の出力は H.264 で、最大 3840×2160。これを超える場合は縮小して保存する。
- HDR 動画（PQ / HLG）は処理できない。
- AdcSR は静止画専用で、動画には使えない。
- 「試す」は拡大だけを確かめる。フレーム補間の結果は確かめられない。
- モデルなどの設定は、アプリを終了すると既定に戻る。
- NPU は、Ryzen AI Software を入れた AMD Ryzen AI 搭載機でだけ使える。

## 技術的な詳細

ソースからの起動、実行方式ごとの必要環境、実測、環境変数、内部の構成、ポータブル版の作り方は
[docs/technical.md](docs/technical.md) を参照。

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

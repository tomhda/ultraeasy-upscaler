# ultraeasy-upscaler

[English](README.en.md)

**[⬇ ultraeasy-upscaler v0.11.0 をダウンロード](https://github.com/tomhda/ultraeasy-upscaler/releases/download/v0.11.0/ultraeasy-upscaler-portable-win64.zip)**
（Windows x64 用の zip、376 MB）。手順は「[導入](#導入)」を参照。

ultraeasy-upscaler は、画像や動画を かんたん操作でアップスケール/フレーム補間ができる Windows 用のアプリです。
処理は全てローカルで動作し、情報収集なし、広告なし・オープンソースです。
Real-ESRGAN など、最大5つの超解像モデルと、RIFEv4.6のフレーム補間モデルに対応しています。
ドラッグ＆ドロップでの一括変換に対応。GPU/一部NPU 対応。
複数のモデルでの処理結果を見比べたり、動画の場合は先に指定の 1 コマだけを処理して確認することもできます。

![写真を AdcSR で試し、処理前と処理後を等倍で左右に並べた画面](docs/images/compare-zoom.jpg)

## できること

- 超解像モデルを用いて画像、動画を キレイに拡大。動画はH.264 で保存。
- 超解像モデル：Anime Video v3、 4xNomosUni SPAN、Real-ESRGAN（AMD縮小版）※、SwinIR-M、AdcSR※

- 動画処理は時間がかかるため、本処理の前に 1 コマ（またはその一部）を選んで処理し、処理前と見比べることができます。複数のモデル同士も比較可能です。
- モデルを指定して一括処理の他、特定のファイルだけ別のモデルで処理することも可能。
- DirectX 12 対応GPUで動作。AMD Ryzen AI 搭載機では、追加の拡張キットを入れると NPU でも動作可能。

- フレーム補間モデルを用いて、動画をなめらかにすることができます。超解像との組み合わせ実行も可能。
- フレーム補間モデル：RIFE v4.6

※Real-ESRGAN（AMD縮小版）は研究用途限定のライセンス
※AdcSRは静止画専用・別途ダウンロード必須

## 画面

▼ 動作画面。左がメディアの一覧、中央がプレビュー、右で設定を操作します。
中央は写真を AdcSR で試した後の状態で、境界線の左右で処理前後を比較できます。

![メイン画面](docs/images/main.jpg)

▼ 等倍での比較表示も可能

![等倍での比較](docs/images/compare-zoom.jpg)

使用素材：
[Big Buck Bunny](https://peach.blender.org)、[Tears of Steel](https://mango.blender.org)（どちらも © Blender Foundation, CC BY 3.0）
Superman (1941)（パブリックドメイン）

## 導入

Windows 11（x64）と、DirectX 12 対応の GPU で動作確認済み

1. [`ultraeasy-upscaler-portable-win64.zip`](https://github.com/tomhda/ultraeasy-upscaler/releases/download/v0.11.0/ultraeasy-upscaler-portable-win64.zip) をダウンロード
2. zip を好きな場所に展開（ポータブル版のためインストール不要）。
3. `ultraeasy-upscaler.exe` を起動。署名のない実行ファイルなので、Windows の SmartScreen が「Windows によって PC が保護されました」と表示することがある。その場合は「詳細情報」→「実行」。

zip には動作に必要な全ファイルが入っているため、フォルダの中のファイルを移動・削除すると動作しなくなる可能性が非常に高いです。
アンインストールは展開フォルダごと削除ください。

## 使い方

### メディア追加

- ドラッグ＆ドロップまたは左下の枠をクリックしてメディアを追加します。

### モデル選択

- 右側の「画像」「動画」で、メディア種類ごとに処理に使うモデルを選びます。
- 一覧でファイルを選んで「このファイルだけ別の設定にする」と、そのファイルだけ別のモデルで処理可能です。
- どれを選ぶかは「[モデルの選び方](#モデルの選び方)」を参照。

### 試す

- 中央の「試す」（動画は「このコマを試す」）を押すと、そのフレームだけを本処理と同じ方法で拡大し、処理前と並べて比較できます。
- 動画は、プレビューの下のスライダーで試すフレームを選択可能です。デフォルトは開始 2 秒地点。
- 「範囲を選ぶ」では、画像の一部だけを切り取って処理し、比較できます。
- モデルを変えてもう一度処理すると、比較できる結果が増えていきます。「左」「右」で、処理前と各モデルの結果から比べる相手を選択可能

### 開始

- 「開始」を押すと、待機中のファイルを上から順に処理します。出力は元のファイルと同じ場所の `upscaled` フォルダに保存。（「出力先」と詳細設定で変更可能）
- 「一時停止」は、処理中のファイルが終わってから停止するボタンです。
- 「 × 」今すぐ処理を中止します。
- 処理後の同ファイル名が同名になる場合は`(1)` のように番号付きで保存されます。

### やり直す

- 終わった行の丸い矢印、または「すべてやり直す」で処理前に戻します。

### 詳細設定

- 右上の歯車を押すと、AI の実行先、保存形式、動画の画質、出力フォルダ名、フレーム補間後の fps、表示言語などが変更可能です。
- 画面は日本語と英語に対応しています。最初は Windows の表示言語に合わせて決まり、「表示言語」で変更できます（次回の起動から切り替わります）。

## モデルの選び方

時間は Radeon 860M（Ryzen AI 7 PRO 350 の内蔵 GPU）で 854×480 の 1 枚を 4 倍にしたときの値。

| モデル | 向いているもの | 1 枚の時間 | 補足 |
|---|---|---|---|
| Anime Video v3 | アニメ・線画・CG | 0.46 秒 | 細部を整理してなめらかにする。劣化した古い素材に強い。実写には加工感が強く出る |
| 4xNomosUni SPAN | 実写 | 0.51 秒 | 元の質感や粒状感を残す実写向きの素材。補正は強くないが元の雰囲気が残り、きれいな素材に向く |
| Real-ESRGAN（AMD縮小版） | 実写 | 2.78 秒 | 輪郭や毛を 1 本ずつ立てる。加工感は強め。研究用途限定のライセンス |
| SwinIR-M | 実写の静止画 | 59 秒 | Transformerモデル。時間をかけて細部まで復元。 |
| AdcSR | 実写の静止画 | 108 秒 | AI生成型で、質感を作り足すモデル。とても重く書き足しも強いため静止画専用。本体には含まれず別途DLが必要（下記） |

- 迷ったら、アニメ・CG は Anime Video v3、実写は 4xNomosUni SPAN がオススメ
- AdcSR を使うには、[v0.9.1 の `models-adcsr-gpu-fp32.zip`](https://github.com/tomhda/ultraeasy-upscaler/releases/tag/v0.9.1)（約 1.8GB）を
  ダウンロードし、`ultraeasy-upscaler.exe` があるフォルダの中の `models\ai` フォルダに展開してください。使用条件は zip 内の NOTICE を参照。
- 詳細設定の「AI実行先」を Vulkan にすると、別の方式（realesrgan-ncnn-vulkan）で処理します。選べるモデルが Real-ESRGAN 系の 5 種類に変わり、Anime Video v3 では 2 倍の拡大も選べます。GPU での処理を開始できなかった場合は、自動でこの方式に切り替わります。

### 画質の比較

4倍への拡大結果
上段は左から オリジナル（Lanczos で 4 倍・AI なし）、Anime Video v3、4xNomosUni SPAN。
下段は左から Real-ESRGAN（AMD縮小版）、SwinIR-M、AdcSR。
すべて GPU / DirectML で実行。

トゥーン CG — Big Buck Bunny (480p):

![Big Buck Bunny](docs/benchmarks/model_guide_bbb.png)

セル画アニメ — Superman (1941, 320x240):

![Superman 1941](docs/benchmarks/model_guide_sup.png)

実写 — Tears of Steel (720p):

![Tears of Steel](docs/benchmarks/model_guide_tos.png)

- Anime Video v3: 細部を整理してなめらかに。劣化した古い素材に最も強い
- 4xNomosUni SPAN: 元の質感や粒状感を残す。加工は控えめで、きれいな素材ほど向く
- Real-ESRGAN（AMD縮小版）: 輪郭や毛の 1 本 1 本をくっきり立てる。加工感は強め
- SwinIR-M: 輪郭を崩さずに細部を締める。4xNomosUni SPAN より時間がかかる
- AdcSR: 肌のしわ、毛、布の織り目などを AI が描き足す。実写では最も精細。セル画では元のざらつきまで質感として描くので向かない

## NPU 拡張キット

AMD Ryzen AI 搭載機では、追加キットをダウンロードすることで NPU 処理が可能。次の 2 つが導入済みの場合、選択可能になる。

- AMD の [Ryzen AI Software 1.8.0](https://ryzenai.docs.amd.com/en/latest/inst.html)（AMD のサイトから入手して導入。NPU ドライバ 32.0.203.329 以降）
- [`ultraeasy-upscaler-npu-kit.zip`](https://github.com/tomhda/ultraeasy-upscaler/releases/download/v0.11.0/ultraeasy-upscaler-npu-kit.zip)（56 MB）。

AdcSR を NPU で使う場合は、追加でNPU専用モデル [`ultraeasy-upscaler-npu-kit-adcsr.zip`](https://github.com/tomhda/ultraeasy-upscaler/releases/download/v0.11.0/ultraeasy-upscaler-npu-kit-adcsr.zip)（1.7 GB）も必要。

手順:

1. ultraeasy-upscaler を終了し、キットの zip の中身を `ultraeasy-upscaler.exe` があるフォルダに上書きで展開する。
2. 起動して歯車を押し、「NPU の準備」で、使うモデルを NPU 用に変換する（モデルの行にある「NPU 用に変換」を押す）。NPU ではモデルをそのままでは動かせないため、モデルごとに最初の一度だけこの作業が必要。変換作業は重くないがメモリ一部を占有し、数分～数時間かかるため注意。
3. NPU 用への変換が終わったら、「AI実行先」で NPU を選択。

▼ 「NPU の準備」。モデルごとに、NPU 用への変換にかかる時間の目安が出る

![NPU の準備](docs/images/npu-prepare.jpg)

モデルを NPU 用に変換するときの時間と占有メモリ（Ryzen AI 7 PRO 350 での実測）:

| モデル | 時間 | メモリの最大 |
|---|---|---|
| Anime Video v3 | 約 15 分 | 約 1.9GB |
| 4xNomosUni SPAN | 約 14 分 | 約 1.2GB |
| Real-ESRGAN（AMD縮小版） | 約 25 分 | 約 1.3GB |
| SwinIR-M | 約 65 分 | 約 25GB |
| AdcSR | 約 2 時間 | 未測定 |

変換中の CPU 使用は 1 コア分のため、CPU占有は少ない。
SwinIR-M 変換はメモリを多く使うので、ほかのアプリを閉じてからの実行を推奨。
NPU ドライバや Ryzen AI Software を更新した場合、NPU 用への変換のやり直しが必要となる可能性があるため注意。

NPU で動かすための技術的な記録（コンパイラの回避策、高速化、測定）は、別のリポジトリ
[ryzen-ai-npu-super-resolution-notes](https://github.com/tomhda/ryzen-ai-npu-super-resolution-notes) 

## 制限

- 拡大倍率は 4 倍のみです。2 倍は、AI実行先を Vulkan にしたときに一部のモデルでのみ対応します。
- 動画の出力は最大 3840×2160 で、これを超える場合は同サイズまで縮小されます。
- HDR 動画（PQ / HLG）は処理不可。
- AdcSR は静止画専用です。
- モデルなどの設定はアプリを終了時にデフォルトに戻ります。
- NPU は、Ryzen AI Software を入れた AMD Ryzen AI 搭載機にて、追加キットを入れた場合のみ使用可能です。

## 技術的な詳細

ソースからの起動、実行方式ごとの必要環境、実測、環境変数、内部の構成、ポータブル版の作り方は
[docs/technical.md](docs/technical.md) を参照。

## ライセンス

- ultraeasy-upscaler 本体は [MIT License](LICENSE) です。
- モデルには、それぞれ別のライセンスがあります。使う前に確認してください。

| モデル | 作者 | ライセンス |
|---|---|---|
| Anime Video v3（realesr-animevideov3） | xinntao（[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)） | BSD 3-Clause |
| 4xNomosUni SPAN（4xNomosUni_span_multijpg） | Philip Hofmann（Phips） | CC BY 4.0（作者の表示が必要） |
| Real-ESRGAN（AMD縮小版） | AMD | Research-only RAIL-MS（研究用途のみ） |
| SwinIR-M（003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN） | Jingyun Liang | Apache 2.0 |
| AdcSR | Guaishou74851（CVPR 2025） | Apache 2.0。基になった Stable Diffusion 2.1-base の使用条件（CreativeML Open RAIL++-M）も適用されるため、使う前に利用者自身で確認すること |
| RIFE v4.6 | hzwer（[Practical-RIFE](https://github.com/hzwer/Practical-RIFE)） | MIT |

- ポータブル版には、FFmpeg（GPL v3）、Qt 6 / PySide6（LGPL v3）、realesrgan-ncnn-vulkan（MIT）、rife-ncnn-vulkan（MIT）、Microsoft Windows ML ランタイムなどを同梱しています。
  一覧と入手元は zip 内の `THIRD-PARTY-NOTICES.txt`、モデルのライセンス全文は `models\ai` フォルダにあります。
- 画面写真と比較画像の素材: [Big Buck Bunny](https://peach.blender.org)、[Tears of Steel](https://mango.blender.org)（© Blender Foundation, CC BY 3.0）、Superman (1941)（パブリックドメイン）

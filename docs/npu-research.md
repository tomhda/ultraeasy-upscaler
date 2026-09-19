# NPU超解像 研究ノート（2026-08）

Ryzen AI NPU で Real-ESRGAN 系モデルを動かし、GPU (Vulkan/NCNN) 経路と
速度・画質で比較した記録。結論から: **bf16 + VAIML コンパイルにより、
GPU とほぼ同画質の 4x アップスケールを GPU 占有ゼロで実行できる**。

## 環境

- AMD Ryzen AI 7 PRO 350（NPU: XDNA2）/ Radeon 860M (iGPU) / 32GB LPDDR5-8000
- Ryzen AI Software 1.7.1（conda env `ryzen-ai-1.7.1`: onnxruntime VitisAI EP / Quark 0.11rc1）
- GPU側: realesrgan-ncnn-vulkan（NCNN fp16）
- 検証素材: Big Buck Bunny / Tears of Steel（CC-BY）、Superman 1941（PD）

## 最終ベンチマーク（→4x、1フレームあたり実測）

実効値は 854x480→4x・12フレームフォルダ一括の平均（動画実運用相当）。

| 構成 | 精度 | タイル速度 | 実効/枚 |
|---|---|---|---|
| GPU + Anime Video v3 | fp16 | — | **0.7秒** |
| NPU + Anime Video v3 | bf16(分解) | **151ms** | 約3.1秒 |
| NPU + Real-ESRGAN | bf16 | 168ms | 約3.7秒 |
| NPU + Real-ESRGAN Anime | bf16 | 262ms | 約4.5秒 |
| GPU + General Video v3（強/弱） | fp16 | — | 参考: 720p単発 約3秒 |
| GPU + Real-ESRGAN | fp16 | — | 参考: 単発 17秒 |

NPU実効はタイル推論の理論値（1.8〜3.1秒/枚）に対し、Pythonワーカーの
フレーム入出力（PNG読み書き・タイル分割結合）が約1.2〜1.4秒/枚を上乗せする。
軽量モデルではGPU実行系（C++・パイプライン化済み）が実効で明確に速い。
NPUの利点は速度ではなく、GPUをほぼ占有せずに実行できる点にある（他のGPU作業と並走可能）。

## 主要な発見

### 1. int8/XIR フローは「見かけの帯域天井」を作る

int8 (XINT8/u8s8, XIR フロー) では、演算量が約10倍違う RRDB と
SRVGGNetCompact が同速（335 vs 323ms/タイル）、タイルを 256→512 に
4倍化しても MP/s 不変（1.02x）で、「約4.2MP/s の帯域律速」と見えた。
**実際は Q/DQ 変換と多サブグラフ分割（realesrgan で11個）のフロー
オーバーヘッドが支配していた。**

### 2. bf16 + VAIML が本命（キャリブレーション不要で int8 より速く高画質）

`quark.onnx.tools.convert_fp32_to_bf16 --format with_cast` で fp32 ONNX を
Cast ベースの bf16 に変換すると、VitisAI EP が VAIML フローで
**単一サブグラフ**にコンパイルする。

| モデル | int8 → bf16 タイル速度 | fp32忠実度 (PSNR mean/min) |
|---|---|---|
| Real-ESRGAN (縮小RRDB) | 335 → **168ms**（2.0x） | 37.75/34.06 → **37.87/36.37 dB** |
| Real-ESRGAN Anime (RRDB 6B) | 351 → **262ms**（1.3x） | 35.89/31.83 → **39.41/37.24 dB** |

目視でも GPU fp16 と NPU bf16 は区別不能（領域差分 35dB 超）。
キャリブレーション画像・巨大一時ファイル・長時間探索がすべて不要になる。

注意: `ModelQuantizer` の BF16 config は `com.amd.quark:ExtendedQuantizeLinear`
（カスタムop）を出力し、環境にカスタムopライブラリが無く動かない。
tools の変換スクリプト（標準opのみ）を使うこと。

### 3. VAIML bf16 の無音誤コンパイル: 犯人は「実学習済み重みのPReLU」

animevideov3（SRVGGNetCompact）の bf16 はエラーなくコンパイル・高速実行
されるが**出力が数値爆発する（PSNR −40dB、max|diff|>1500、決定論的）**。
4段階の二分探索で原因を特定した:

| phase | 実験 | 結果 |
|---|---|---|
| 1 | 単体op 5種（PReLU/PixelShuffle/Resize+Add等） | 全てOK → 単体opはシロ |
| 2 | 同構造を深さ2/8/16・活性化3種（slope=0.25） | 全てOK → 構造・深さもシロ |
| 3 | 実重み vs onnxsim有無 vs 合成+onnxsim | **実重みのみBROKEN** → onnxsimはシロ |
| 4 | 合成+負slope / 実重み+PReLU分解 | 負slopeで忠実度59→35dBに劣化・コンパイル38→1005秒に爆増。**分解版はOK（38.4dB）** |

結論: 実モデルの PReLU slope（min −1.38 / max +1.69、負値・1超えを含む）が
VAIML の PReLU 処理を異常経路に追い込む。完全な爆発には実重みのもう一要素
（巨大バイアス等との複合）が関与するとみられる。

**回避策（採用済み）**: fp32 段階で `PReLU(x) = ReLU(x) − w⊙ReLU(−x)` に等価
分解してから bf16 変換（`export_animevideov3.py --decompose-prelu`）。
分解版 animevideov3 bf16 は **151ms/タイル（7.0MP/s）で NPU 最速**、
fp32忠実度 38.4dB。再現実験は `scripts/npu/bisect_vaiml_bf16*.py`。

### 4. int8 キャリブレーションの知見（bf16 移行前の記録）

- キャリブ素材のドメイン一致とクリーンさが効く: グレイン入り素材を除き
  日本のアニメ+BBB の32枚に変えるだけで +0.4dB（min +0.6dB）
- Quark MinMSE は %TEMP% に巨大な中間テンソルを書く
  （RRDB 256px: 約2.5GB/枚。512px×100枚では116GB）
- 64枚×RRDB は 32GB RAM で失敗（探索フェーズのメモリはほぼ枚数比例）

### 5. 画質の傾向（目視評）

- アニメ/CG: GPU+AnimeVideoV3 が最良。NPU+Real-ESRGAN Anime (bf16) が肉薄し
  「好みの差」の範囲
- 実写: AnimeVideoV3 系はのっぺり（美肌フィルタ化）。Real-ESRGAN 系が上
- General Video v3（ノイズ除去弱, wdn）は原本の質感を残す忠実系で自然。
  Real-ESRGAN は輪郭強調の知覚系で「加工感」。忠実 vs 知覚の好みで選ぶ
- int8 の劣化はエッジのギザギザとして現れる（bf16 で解消）

## Windows ML 経由の実行（2026-08-14〜15 追記）

Ryzen AI SW 直叩きの代わりに、OS標準の Windows ML（ONNX Runtime +
EPカタログ）経由で同じモデルを実行する検証。実装は姉妹リポジトリ
`tools/winml-sr`（C#コンソール、MSIX不要・
unpackaged動作）。**fp32 ONNX を投げるだけで、量子化・bf16変換・
キャッシュ管理は全て EP 側が自動化**する。

実測（854x480→4x、PSNRは同モデルCPU fp32比）:

| 構成 | ms/タイル | 1枚(推論) | PSNR |
|---|---|---|---|
| Anime Video v3 × DML (GPU) | 52ms/256 | **0.63秒** | 101dB |
| Anime Video v3 dp × VitisAI (NPU) 512タイル | 512ms/512 | **1.02秒** | 51.6dB |
| Real-ESRGAN (RRDB) × VitisAI (NPU) | 170ms/256 | 2.06秒 | 47.2dB |
| Real-ESRGAN (RRDB) × DML (GPU) | 231ms/256 | 2.78秒 | 94.9dB |

主要な発見:

- **WU配信の VitisAI EP 1.8 では PReLU bf16 誤コンパイル（発見4）が
  発生しない**（非分解モデルで PSNR 59.7dB）。ただし分解版(dp)の方が
  依然約2倍速い（256タイルで 133ms vs 274ms）ため分解は高速化技として有効
- **NPU時間は処理ピクセル数にほぼ線形**（dp bf16 で約2.0µs/px、
  タイル毎固定費は約7msのみ）。よってタイルサイズは「パディング＋
  オーバーラップの捨てピクセル最小」で選ぶ。854x480 は core 480 の
  512タイル（縦ピッタリ・2枚）で捨て28%まで減り 1.02秒/枚
- **フレームぴったり（854x480一発）は不成立**: VAIML コンパイルは通るが
  実行時に XRT の単一ディスパッチ時間制限超過（ERT_CMD_STATE_TIMEOUT）。
  タイル分割はランタイム制約としても必須
- DML (DirectML) は同じ 860M を使う realesrgan-ncnn-vulkan より約2倍速い。
  MIGraphX (69ms) / RyzenAILight (227ms) は DML (52ms) に劣り不採用
- EP は Microsoft Store でなく **Windows Update 経由で配信**されるため、
  WU 一時停止中はダウンロード不可（公式既知事項）。取得後は停止中も動作
- 初回の VAIML コンパイル（1〜9分）は EP が自動で永続キャッシュし、
  以後のセッション生成は 0.1〜1.2秒
- 常駐 serve モード（stdin/stdout 生ピクセルパイプ）で、フレームI/O
  オーバーヘッドを 1.2〜1.4秒 → 約50〜60ms/枚 に短縮

## Ryzen AI SW 1.8.0 への更新（2026-08-15 追記）

WinML EPカタログ停止（NPUドライバ.329がVitisAI EP 1.8.68の対応上限.297を超過、
詳細は winai リポジトリ docs/span-bench-results.md 追記4）を受け、ネイティブ側を
1.8.0 に更新した（conda env `ryzen-ai-1.8.0`・1.7.1と並存・キャッシュは
`vendor/amd-npu-1.8/` に分離。1.8.0はドライバ.280〜対応、推奨.376）。

512タイル bf16、853x480→4x、タイル処理込み実測:

| モデル | 1.7.1 | 1.8.0 | 画質(vs fp32) |
|---|---|---|---|
| Anime Video v3 dp | 1.305s/枚 | **1.139s/枚** | 48.3dB（両版同値） |
| purephoto (SPAN 4x) | 0.707s/枚 | **0.602s/枚** | 43.1dB（両版同値） |

- コンパイラ世代更新で一律13〜15%高速化。WinML EP 1.8.68時代の純推論計
  （1.02s/0.48s）との差はPython側タイル処理オーバーヘッド込み計測でほぼ説明可能
- VAIMLコンパイル時間は増加（av3dp512: 389→914s）。初回のみなので実害小
- SPAN系（purephoto）は1.7.1/1.8.0どちらのVAIMLでも問題なくコンパイル・実行

## SwinIR-M（attention系）のNPU対応（2026-08-29 追記）

*English summary for searchability: [swinir-npu.md](swinir-npu.md)
(Running SwinIR on an AMD Ryzen AI NPU).*

静止画最高画質枠として SwinIR-M（`003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN`、
real-world SR、Apache-2.0）を追加した。`export_spandrel.py` で 256x256 固定 fp32 ONNX 化
（torch/ORT 最大差 7.7e-07）。attention 系（MatMul/Softmax/LayerNormalization/roll）を
このスタックで NPU コンパイルした初のモデル。

実測（853x480/1920x1080 → 4x、Radeon 860M / Ryzen AI 7 PRO 350）:

| 経路 | 精度 | 256タイル/枚 | 480p 1枚 | 1080p 1枚 | 忠実度 |
|---|---|---|---|---|---|
| GPU (DirectML) | fp32 | 約4.5s | 約60s (12タイル) | 213s (45タイル) | - |
| NPU (VitisAI 1.8.0) | bf16 | 6.6s | 約90s | 推定297s | 38.5dB (vs fp32) |

### VAIMLコンパイラのassertion crashと切り分け

bf16cast モデルの VitisAI セッション生成が、ONNX→ONNX-MLIR lowering 段の
native assertion（`"Iteratees do not have equal length"`、exit 0x80000003）で
2.8秒で停止した。20超の切り分け試行（部分グラフ抽出＋合成最小グラフ）の結果:

- 単体では全て成功: LayerNormalization / Softmax / 4D MatMul / roll等価のSlice+Concat /
  window分割のTranspose・Reshape連鎖 / マスクAdd付きwindow attention / 非シフト完全ブロック
- 落ちるのは「**負の starts/ends リテラルを持つ Slice 2本 + Concat**」の組み合わせのみ
  （`torch.roll(x, shifts=+4)` の逆roll export形）。最小再現器は3ノード。
  同一形状・同一構成で符号を非負にしただけの順rollは成功する
- スタックは `lowerONNXToONNXMLIR` → `ResultNamesUpdater` → `zip_equal<SmallVector<string>>`

### 回避と結果

負のSlice境界リテラルを正の等価値へ書き換えるだけで回避できる（数値はbit-exact、
ノード数・op種は不変。フルモデルで72箇所）。書き換え後は1697ノードの全体が
コンパイルを通り、VAIMLが 4971 op / 2969 GOPs を100%・単一サブグラフで受理した
（初回コンパイル51分、以後はキャッシュ）。この書き換えは `export_spandrel.py` に
恒久組込み済み（`_rewrite_slice_nonneg`、export時に常時適用）。

### 採用判断はリソース占有率で行った

SwinIR-M は 256タイル1枚で 2969 GOPs あり、NPU は GPU より遅い。採用理由は速度では
なく占有率（2秒間隔サンプリング、アイドル差分）:

| 実行中 | iGPU 3Dエンジン | CPU全体 | NPU | メモリコミット増 |
|---|---|---|---|---|
| NPU (VitisAI) | 2.4%（アイドル同等） | +0.7pt | 100%（時間積分） | +7.65GB |
| GPU (DirectML) | 99.1% | +0.8pt | 0% | +3.19GB |

NPU実行中はiGPUが空いたままなので、他の作業と並行して裏で回せる。既知の注意点:
NPU実行中のみ約38秒周期（タイル6枚ごと）で最大+7GBの過渡的なコミットスパイクを観測。
発生源は特定済みで、**WindowsのTDR検知によるカーネルライブダンプ採取**
（WERバケット `LKD_0x141_Tdr:6_IMAGE_ipustack.sys`。SwinIR-Mの1推論6.6秒が
タイムアウト閾値を超えるため。プールタグ vTDR/dxgkrnl → Ldmp/ntoskrnl の
シーケンスがスパイクと完全同期、NPUアダプタのDXGI共有メモリは不変）。
採取中は約0.7秒システムが停止するが、推論自体はリセット・エラー・速度低下なし。
1推論が閾値未満の realesr-animevideov3 / purephoto では発生しない。

計測ノウハウ: NPU専用のパフォーマンスカウンタは無く、`GPU Engine` カウンタセットに
別LUIDのデバイスとして現れる。値は推論1回につき1サンプルのバースト報告
（約650%）なので、中央値ではなく時間積分の平均で読む。xrt-smi でHWコンテキストの
Active状態と使用カラム数の裏取りができる。

### 軽量版 SwinIR-S の検討（2026-09-05〜06、不採用）

SwinIR-S（002_lightweightSR_DIV2K_s64w8_SwinIR-S_x4、878Kパラメータ、bicubic劣化で学習）を
同じ手順（export_spandrel → quark bf16cast → VAIML）で NPU 化した。
VAIML は 3317 op / 272.9 GOPs を100%受理し、コンパイル36.6分。負のSlice境界の書換は0件だった。

| 項目 | SwinIR-M | SwinIR-S |
|---|---|---|
| 演算量（256タイル） | 2969 GOPs | 273 GOPs |
| NPU 1タイル | 6.6 s | 2.98 s |
| GPU(DML) 1タイル | 約4.5 s | 1.29 s |
| NPU bf16 と fp32 の一致 | 38.5 dB | 47.7 dB |

演算量が1/11でもNPUは2.2倍速にしかならない。実効92 GOPS/s で、
XDNA2 + VAIML の transformer 実行は op 数に比例した固定コスト（約0.9 ms/op）が支配的と読める。
チャネル幅の削減は効かず、層数（op数）を減らす以外に速度は出ない。3.0 s/タイルでも
TDR ライブダンプは発生した（閾値は3秒未満）。

画質は目視で不採用。bicubic / x4plus / SwinIR-M / SwinIR-S の4列比較
（`tmp/swinir-eval/sheets/`）で、SwinIR-S は小さな正方形のタイルを敷き詰めたような
格子状の破綻が全体に出て、4者の中で明確に最も悪かった。bicubic 劣化で学習した軽量モデルは
実写・アニメの実素材には向かない。GUI には登録しない。

## AdcSR の NPU 対応（2026-09 追記）

*English summary for searchability: [adcsr-npu.md](adcsr-npu.md)*

AdcSR（net_params_200、SD2.1-base派生・1ステップ）は 1 グラフで NPU 常駐すると
2 回目以降全画素 NaN になる。切り分けの経緯: 1 グラフで 2 回目以降 NaN →
73 分割では正常 → N5 書換えで 1 サブグラフ化しても NaN →
UNet 出力直後で前半 F / 後半 G に切断すると F 単独・G 単独は何回でも正常だが、
同一プロセスで G を 1 回実行すると F が恒久 NaN（一方向 G→F 汚染）→
別プロセスに分けると 100/100 回正常（tmp/adcsr-npu/round5/b1/RESULT.md §3〜§4、
tmp/adcsr-npu/round5/b2/RESULT.md）。

条件: N5 書換え済み bf16cast、境界 3 テンソル（main [1,256,64,64]、
mean/std [1,3,1,1]、いずれも float32。名前で照合）。
F の VAIML 受理は 4786/4797 op・498.809 GOPs（単一サブグラフ、約93分）、
G は 1520/1520 op・658.343 GOPs（約30分）。
反復は逐次で 2.05 s/タイル（F 0.71＋転送 2.5 ms＋G 1.33）。
重ね実行は 2.20 s/タイルと遅くなるため逐次に固定。

実装は tools/npu-serve/npu_worker.py（役割別テンソル実行器）と
tools/npu-serve/npu_twostage.py（TwoStageSession: 起動時セルフテスト、
タイル単位の健全性検査、1 画像 1 回の内部復旧、復旧不能時は
TWO_STAGE_FATAL を 1 件。画像単位で同じ AdcSR の DirectML へ再処理）。
`UEU_ADCSR_NPU2=0` で従来の GPU 実行に戻る。

## SinSR の下見（2026-09-12、不採用）

AdcSR と同じ「1 ステップの拡散系超解像」である SinSR（ResShift の蒸留、CVPR 2024）が
AdcSR より速いかを、NPU に載せる前の下見として DirectML（Radeon 860M）で確かめた。
公式コードの推論経路（LR 前処理 → VQGAN encode → 1 ステップ UNet → VQGAN decode）を
1 モジュールに包んで固定形状 ONNX（64→256 と 128→512、opset 17）にし、公式推論と一致を確認した。

| | AdcSR 128→512 | SinSR 64→256 | SinSR 128→512 |
|---|---|---|---|
| ONNX ノード数 | 1510 | 1717 | 1729 |
| GOPs/タイル | 1099 | 1111 | 5267 |
| 出力 1MP あたり GOPs | 4193 | 16948 | 20093 |
| DirectML 1 タイル | 1.1〜1.6 秒 | 0.73 秒 | GPU ハング（計測不可） |
| tos 1280x534 1 枚 | 約 100 秒（84 タイル） | 502 秒（680 タイル） | — |
| ライセンス | Apache-2.0 / OpenRAIL-M | CC BY-NC-SA 4.0 / S-Lab 1.0（非商用） | 同左 |

- 出力画素あたりでは AdcSR の約 2.2 倍遅い。主因は VQGAN の encode/decode が出力サイズ（512 角）で動くことと
  Swin attention（MatMul 40・Softmax 20・Erf 18）。
- 128→512 版は VQ の距離計算 `Einsum`（16384×8192、約 537MB の中間テンソル）が iGPU のメモリを超えて
  DXGI 887A0006 でハングする（64 版の同テンソルは約 134MB で完走。サイズ依存と整合）。
- 画質は tos（実写）で AdcSR が明確に上（SinSR は平滑化が強い）。bbb（アニメ）は健闘するが背景に生成ノイズのまだらが出る。
- NPU に載せた場合の見込みは op 数換算で約 6.5 秒/タイル（本機の約 0.9 ms/op 律速から。推測）で、AdcSR の約 3 倍。
- 以上から不採用。作業物は `tmp/sinsr/`（git 管理外）。

副産物として、DirectML の ONNX Runtime は **shape に `-1` を含む Reshape でセッション初期化に失敗する**
（`MLOperatorAuthorImpl.cpp(2879)`、E_INVALIDARG。1 ノードの最小プローブで再現）。
shape inference で求めた具体値に書き換えると通る（SinSR 128 版で 196 箇所）。
同じ shape 定数を形状の異なる Reshape が共有している場合は形状ごとに定数を複製する必要がある。
Erf・Einsum・ArgMin・cubic/nearest Resize・3D InstanceNormalization・Softmax（4D/5D）・
負境界の Slice＋Concat は DML では問題なかった。

## 設定スイープと AI Analyzer の実測（2026-09 追記）

測定記録は `tmp/npu-perf/RESULT.md`（round8）。条件は Ryzen AI SW 1.8.0、
VitisAI EP、電源モード Default（変更なし）。

- `xrt-smi validate` は 3 種すべて PASSED:
  gemm 51.3 TOPS（公称 INT8 50 TOPS と一致）、latency 平均 67.5 us、
  throughput 平均 60889.2 ops。SR 系の実効（約 0.5〜1 GOPS相当）は
  ピークの約 1/100 で、律速は演算器のピークではない（事実）。
- 設定スイープ（SPAN 512 タイル、bf16cast。基準 0.2457 s）:
  O1 は 1.36〜1.38 s（約 5.6 倍遅い。コンパイルは速いが実行が遅い）。
  O3・vectorized・unvectorized は基準と同等（出力が既存 cache とビット一致する
  ものもあり、格納指定は SPAN に無効）。既定 O2/auto が最良のまま。
- FP32 直接入力はモデル依存。SPAN は bf16cast と同速（0.256 s）で
  fp32 参照比 PSNR が 44.51 dB → 46.94 dB に上がる（Quark 変換も不要）。
  一方 AdcSR 後半 G の FP32 直接版は metaDef 6 個・計 76 ノードしか NPU に載らず、
  残りが CPU で動いた（`vendor/amd-npu-1.8/modelcachekey_r8_g_fp32/context.json` で確認）。
  新しい body を NPU 化したら metaDef が 1 個・全ノード NPU かを必ず確認する。
- AI Analyzer（SPAN 512 タイル約 240 ms）:
  中盤約 20 レイヤが一律 2.0〜2.3 ms（計 42 ms。サイズによらない固定費と推測）、
  終盤レイヤ 43〜46（upsampler の DepthToSpace・L3 spill 系）が 17/42/20/18 ms
  （計 97 ms）、id 50 区間 71 ms。終盤＋id 50 区間で約 7 割を占める。
  実効 238.8 GOPs / 0.25 s = 約 950 GOPS。
- AI Analyzer（AdcSR 後半 G、約 1400 ms/run）:
  AIE レイヤ時間の 92%（1023 ms / 1111 ms）が `L3_OFM_Buffer_spill` で、
  attention・Softmax・MatMul は 2 レイヤ 2.6 ms のみ。
  G の時間の 9 割は大活性の搬送（spill）で、attention 等は律速ではない（事実）。

## tail-cut: 末尾 DepthToSpace 以降の CPU 実行（2026-09 追記）

上記の「終盤 upsampler 系が約 7 割」から、DepthToSpace の直前でグラフを切り、
NPU は `[1,48,512,512]` を出力、PixelShuffle（Anime Video v3 は入力の最近傍
4 倍加算を追加）を CPU の numpy で行う。要素数は同じなので転送量は変わらない。
CPU 後処理は次タイルの NPU 実行と 2 スレッドで重ねる（キュー深さ 2）。
実験記録は `tmp/npu-perf/tailcut/RESULT.md`、製品記録は
`tmp/npu-perf/impl-tailcut/RESULT.md`。

製品計測（UEU serve 経路、512 タイル overlap 16。2 回測定の小さい方。
`--tail` なしの 1 モデル経路は変更前とビット一致を確認済み）:

| モデル・入力 | 全体 NPU | tail-cut（NPU body＋CPU 後処理） | 全体版との PSNR |
|---|---|---|---|
| SPAN、bbb 853x480 | 0.608 秒 | **0.358 秒** | 46.91 dB |
| SPAN、tos 1280x534 | 1.692 秒 | **0.914 秒** | 46.48 dB |
| Anime Video v3、bbb 853x480 | 1.211 秒 | **0.615 秒** | 53.35 dB |
| Anime Video v3、tos 1280x534 | 3.504 秒 | **1.635 秒** | 54.56 dB |

- SPAN body は fp32 直接入力（73 ノード、238.776 GOPs、VAIML 対応 100%、
  サブグラフ 1）。fp32 参照比 PSNR は bf16cast 全体版の 44.51 dB → 46.94 dB。
  全体版との相互 PSNR 46.9 dB は bf16 量子化の差で、劣化ではない。
- Anime body は PReLU 分解版の bf16cast（onnx 417 ノードのうち Cast 融合で
  meta 311 ノード、328.087 GOPs、サブグラフ 1）。PReLU を含む fp32 版は
  VAIML bf16 で無音の誤コンパイルを起こす既知問題があるため使わない。
- CPU 後処理（SPAN 約 19 ms/タイル、Anime 約 57 ms/タイル）は NPU 実行より
  短いためパイプラインで隠れる。`[timing] tail-post` に合計を出す。
- アプリ経路（NPU_NATIVE、3 秒 72 フレーム動画の E2E）:
  SPAN 1.57 fps → **2.44 fps**、Anime Video v3 0.79 fps → **1.46 fps**。
  新旧出力の PSNR は 43〜53 dB。NPU 実行中の CPU 使用率（全体、5 秒ごと）は
  新旧で同水準（10% 台前半。ffmpeg の伸縮・符号化が支配的）。
  `UEU_NPU_TAILCUT=0` で従来の全体モデルに戻せる。
- body とマニフェスト（`*.tail.json`）が models に無い環境では
  従来の全体モデルへフォールバックする（v0.9.0 の配布物でも動作する）。


## NPU の電源モード（2026-09-19 追記）

このページの他の実測値はすべて NPU 電源モード Default（`xrt-smi examine` の
Power Mode）での値。`xrt-smi configure --pmode turbo`（AC 電源必須）に切り替えて
同じキャッシュ・同じスクリプト（warmup 2 + 10 run、SwinIR-M は 3 run）で測り直した。
再コンパイルは不要。出力の有限性と fp32 参照比 PSNR は Default と同じ。

| モデル（1 run 中央値） | Default | Turbo | 比 |
|---|---|---|---|
| SPAN 512 タイル（全体モデル） | 0.246 秒 | 0.130 秒 | 1.89 倍 |
| Anime Video v3 512 タイル（全体モデル） | 0.524 秒 | 0.257 秒 | 2.04 倍 |
| AdcSR 前半 F（128 タイル） | 0.738 秒 | 0.361 秒 | 2.04 倍 |
| AdcSR 後半 G | 1.359 秒 | 0.628 秒 | 2.16 倍 |
| SwinIR-M 256 タイル | 6.73 秒 | 3.39 秒 | 1.99 倍 |

tail-cut 経路（`npu_serve.py --tail`、1 枚あたり、3 回の小さい方 2 回の範囲）:

| 条件 | Default | Turbo | 同機 GPU（DirectML） |
|---|---|---|---|
| SPAN、bbb 853x480 | 0.362 秒 | 0.275〜0.282 秒 | 0.51 秒 |
| SPAN、tos 1280x534 | 0.914 秒 | 0.678〜0.681 秒 | |
| Anime Video v3、bbb 853x480 | 0.615 秒 | 0.447〜0.455 秒 | 0.46 秒 |
| Anime Video v3、tos 1280x534 | 1.635 秒 | 1.128〜1.131 秒 | |

AdcSR の 2 プロセス構成（`npu_serve.py --model-back`、overlap 32、`--require-cache`）:

| 条件 | Default | Turbo |
|---|---|---|
| tos 中央 256x256（16 タイル） | 35.5 秒（2.2 秒/タイル） | 16.9 秒（1.06 秒/タイル） |
| tos 1280x534（180 タイル） | 未計測（2.05〜2.1 秒/タイルから約 370 秒の見込み） | 188.7 秒（1.05 秒/タイル） |

256x256 の出力は Default の出力とビット一致（sha256 先頭 `5ad7ddd9d3705d39`）。

- NPU 実行部分は約 2 倍速くなるが、SPAN / Anime Video v3 の 1 枚あたりでは 1.3〜1.45 倍にとどまる。
  SPAN bbb では NPU 実行の合計が約 0.145 秒で、残り約 0.13 秒はタイル分割・
  結合・uint8 化・パイプ転送と CPU 後処理（Turbo で変わらない部分）。
- この残りを削った結果は次節。
- Default は Windows の電源モードに追従する設定（AMD 文書）。Default のときの
  実クロックは未計測。消費電力・発熱の比較も未計測。
- 電源モードは OS 側の設定で、アプリからは変更しない。再起動後に保持されるかは未確認。


## npu_serve の CPU 側の削減（2026-09-19 追記）

Turbo で NPU 実行が縮むと、1 枚あたりの時間の 4〜5 割が CPU 側になった。
SPAN tail-cut・bbb 853x480（2 タイル）の内訳（変更前、Turbo）:
NPU 実行 計 約 145 ms、タイル結合 約 45 ms、`clip(x*255).astype(uint8)` 約 50 ms、
CHW→HWC 約 20 ms（いずれも全タイルの NPU 実行が終わった後に直列で実行）。
Anime Video v3 は `np.repeat` 2 回による最近傍拡大＋加算が 1 タイル約 120 ms で、
NPU 実行（約 140 ms/タイル）とほぼ並んでいた。

変更:

- 量子化までをタイル単位にし、tail の後処理スレッドで実行する（次タイルの NPU 実行と重なる）。
- 先に有効コア（overlap と右端・下端のパディングを除いた領域）を低解像度側で切り出し、
  拡大後の float 全体と結合用の float バッファを作らない。
- pixel shuffle の並べ替えと uint8 化を、最終出力バッファ（HWC uint8）への 1 回の書き込みにまとめる。
  最近傍拡大は放送（`[h,1,w,1,c]`）で加算し、拡大済み配列を作らない。
- 全体モデル経路（tail なし）も同じくタイル単位で量子化して直接書き込む。
- 応答は `tobytes()` を作らずに書き、クライアントは受信先の配列へ `readinto` する（中間コピー 3 回を削除）。

演算は要素ごとに従来と同じで、出力は SPAN / Anime Video v3（tail-cut）と
animevideov3dp 全体モデルの bbb・tos で変更前とビット一致（sha256 で確認）。

| 条件（Turbo、1 枚あたり、クライアント往復） | 変更前 | 変更後 | 同機 GPU（DirectML） |
|---|---|---|---|
| SPAN、bbb 853x480 | 0.275〜0.282 秒 | 0.173〜0.181 秒 | 0.51 秒 |
| SPAN、tos 1280x534 | 0.678〜0.681 秒 | 0.448〜0.471 秒 | |
| Anime Video v3、bbb 853x480 | 0.447〜0.455 秒 | 0.346〜0.351 秒 | 0.46 秒 |
| Anime Video v3、tos 1280x534 | 1.128〜1.131 秒 | 0.879〜0.881 秒 | |

アプリ経路（NPU_NATIVE、3 秒 72 フレーム動画の E2E、Turbo）:
SPAN 4.89 fps（14.73 秒）、Anime Video v3 2.77 fps（25.99 秒）。
同日朝の Default・変更前は 2.44 fps / 1.46 fps、GPU（DirectML）× Anime Video v3 は 2.48 fps。

SPAN bbb の残りは NPU 実行 約 145 ms に対し約 30 ms（入力の float 化と分割、
最後のタイルの後処理、約 20 MB のパイプ書き込み）。

Default 電源モードに戻して同じ条件で測り直した値（README の実測表はこの値）:

| 条件（Default、1 枚あたり、3 回） | 変更前 | 変更後 | 同機 GPU（DirectML） |
|---|---|---|---|
| SPAN、bbb 853x480 | 0.362 秒 | 0.250〜0.274 秒 | 0.51 秒 |
| SPAN、tos 1280x534 | 0.914 秒 | 0.677〜0.687 秒 | |
| Anime Video v3、bbb 853x480 | 0.615 秒 | 0.466〜0.494 秒 | 0.46 秒 |
| Anime Video v3、tos 1280x534 | 1.635 秒 | 1.324〜1.334 秒 | |

アプリ経路の 3 秒 72 フレーム動画（Default）: SPAN 2.44 → 3.61 fps（19.96 秒）、
Anime Video v3 1.46 → 1.95 fps（36.91 秒）。出力の sha256 は Turbo・変更前と同一。


## 旧構成の比較画像（2026-08 上旬・旧5列マトリクス）

列は左から: オリジナル(bicubic) / GPU+AnimeVideoV3 / NPU+AnimeVideoV3 / NPU+Real-ESRGAN / GPU+Real-ESRGAN。

![Big Buck Bunny](benchmarks/quality_matrix_bigbuckbunny.png)

![Superman 1941](benchmarks/quality_matrix_superman1941.png)

![Tears of Steel](benchmarks/quality_matrix_tearsofsteel.png)

## 用途別の推奨

| 用途 | 構成 |
|---|---|
| 実写・汎用の動画 | NPU + Real-ESRGAN (bf16) |
| アニメ（画質優先） | NPU + Real-ESRGAN Anime (bf16) |
| アニメ（速度優先） | GPU + Anime Video v3 |
| 実写（質感重視） | GPU + General Video v3（ノイズ除去弱） |
| 静止画（最高画質） | GPU + SwinIR-M |
| 静止画（最高画質・GPUをほぼ占有しない） | NPU + SwinIR-M (bf16) |

## 再現手順（scripts/npu/）

```text
export_animevideov3.py   SRVGGNetCompact → 固定256x256 fp32 ONNX（--tile で512可・--size WxH で非正方形可）
export_x4plus_anime.py   RRDBNet(6B) → 同上
make_calib_patches.py    フレーム → 256pxキャリブパッチ（int8用・--src/--out/--count）
quantize_animevideov3.py Quark XINT8 量子化（int8用・--prefix/--calib/--n）
verify_animevideov3_npu.py  NPUコンパイル・割当・PSNR・速度の一括検証
split_tail.py  末尾 DepthToSpace 以降の切断（body＋tail マニフェスト生成・CPU 一致検証）
```

bf16 の実務手順は2行:

```text
python -m quark.onnx.tools.convert_fp32_to_bf16 --input <fp32.onnx> --output <bf16.onnx> --format with_cast
初回セッション生成で vendor/amd-npu/modelcachekey_<stem>/ にVAIMLキャッシュが生成される（数分〜15分）
```

キャッシュキーは ONNX ファイル名 stem から自動導出されるため、
`onnx-models/<name>.onnx` と `modelcachekey_<name>/` は必ず対にする。

## 運用メモ

- 一時停止はジョブ完了後に効く。実行中ジョブの即時中止はキュー行の ×
- VAIML 初回コンパイルは数分〜1時間弱（SwinIR-M 256は約51分。小型モデルのキャッシュのみ git に同梱）
- 比較画像の生成スクリプトは tmp/npu-anime/（素材は自前で用意すること）

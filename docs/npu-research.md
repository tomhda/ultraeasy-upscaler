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
NPUの価値は速度ではなく「GPU完全フリー・低発熱での並走」。

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
`ultraeasy-upscaler-winai/tools/winml-sr-poc`（C#コンソール、MSIX不要・
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

## 用途別の推奨

| 用途 | 構成 |
|---|---|
| 普段の動画（実写・汎用） | NPU + Real-ESRGAN (bf16) |
| アニメを放置で高画質に | NPU + Real-ESRGAN Anime (bf16) |
| アニメを今すぐ最速で | GPU + Anime Video v3 |
| 実写で原本の質感重視 | GPU + General Video v3（ノイズ除去弱） |
| 静止画を最高画質で | GPU + Real-ESRGAN |

## 再現手順（scripts/npu/）

```text
export_animevideov3.py   SRVGGNetCompact → 固定256x256 fp32 ONNX（--tile で512可・--size WxH で非正方形可）
export_x4plus_anime.py   RRDBNet(6B) → 同上
make_calib_patches.py    フレーム → 256pxキャリブパッチ（int8用・--src/--out/--count）
quantize_animevideov3.py Quark XINT8 量子化（int8用・--prefix/--calib/--n）
verify_animevideov3_npu.py  NPUコンパイル・割当・PSNR・速度の一括検証
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
- VAIML 初回コンパイルは数分〜15分（キャッシュを git に同梱して回避）
- 比較画像の生成スクリプトは tmp/npu-anime/（素材は自前で用意すること）

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

| 構成 | 精度 | タイル速度 | 480p/枚 | 720p/枚 |
|---|---|---|---|---|
| GPU + Anime Video v3 | fp16 | — | **1.2秒** | 2.5秒 |
| NPU + Real-ESRGAN | bf16 | 168ms | 約1.5秒 | 約4秒 |
| NPU + Real-ESRGAN Anime | bf16 | 262ms | 約2.4秒 | 約6秒 |
| NPU + Anime Video v3 | int8 | 323ms | 約3秒 | 約9秒 |
| GPU + General Video v3（強/弱） | fp16 | — | — | 約3秒 |
| GPU + Real-ESRGAN | fp16 | — | 17秒 | 47秒 |

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

### 3. VAIML bf16 は SRVGGNetCompact を無音で誤コンパイルする

animevideov3（PReLU + DepthToSpace 構成）の bf16 は正常にコンパイル・
高速実行（98ms/タイル）されるが**出力が崩壊する（PSNR 5dB）**。
エラーは一切出ない。このため animevideov3 のみ int8 を維持。
将来の回避案: fp32 段階で PReLU を `ReLU(x) − w⊙ReLU(−x)` に分解してから
bf16 変換（数学的に等価、未検証）。

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
export_animevideov3.py   SRVGGNetCompact → 固定256x256 fp32 ONNX（--tile で512可）
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

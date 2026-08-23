# AGENTS.md — ultraeasy-upscaler 作業規約と引き継ぎ

このファイルはコーディングエージェント（Codex / Claude Code）向けの共通コンテキスト。
人間向けの説明は README.md、測定・調査の履歴は docs/ にある。

## プロジェクトの現状（2026-08）

Windows ローカル専用の画像/動画アップスケーラー（PySide6 GUI）。AI実行バックエンドは3系統:

| AI実行先 | 実体 | 対応環境 |
|---|---|---|
| GPU (DirectML) | `tools/winml-sr/` の C# 常駐ヘルパー（Windows ML, fp32 ONNX） | DirectX 12 GPU 全般（AMD/NVIDIA/Intel） |
| NPU | `tools/npu-serve/npu_serve.py`（Ryzen AI SW 1.8.0 の VitisAI EP, bf16） | **AMD Ryzen AI 搭載機のみ** |
| Vulkan | realesrgan-ncnn-vulkan（従来経路・フォールバック） | Vulkan GPU 全般 |

GUI と各ヘルパーは同一のバイナリプロトコル（UEU: stdin/stdout, int32 LE, RGB24。
`app/core/serve_client.py`）で通信する。バックエンド解決・モデル解決は `app/core/helper_backend.py`。

動画: RIFE補間が有効なジョブは従来のPNGフレーム経路、補間なし×新バックエンドは
rawvideo 3スレッドパイプライン（`video.upscale_video_piped`）。出力が 3840x2160 を超える
場合はエンコーダ側で自動縮小する（H.264/HWエンコーダ上限のため）。

モデルは GUI で**実モデル名**を選ぶ（`app/gui/main_window.py` の `_MODEL_LABELS` /
`_MODEL_INFO` / `_BACKEND_DESC`）。現行: Anime Video v3（realesr-animevideov3）、
4xNomosUni SPAN（4xNomosUni_span_multijpg, CC-BY-4.0）、Real-ESRGAN（AMD縮小RRDB版）、
および Vulkan 用の従来モデル群。

## セットアップ（新しいマシン）

1. `git clone` 後、Python 3.13 で `.venv` を作り `requirements.txt` を入れる
2. ffmpeg / ffprobe を PATH に
3. モデル ONNX（git管理外）を `tmp/npu-anime/` に配置（`ueu-models-onnx.zip` を展開。
   または `UEU_MODELS_DIR` で場所を指定）。再生成は `scripts/get_ai_models.py` と
   `scripts/npu/export_*.py`
4. .NET 8 SDK で `cd tools/winml-sr && dotnet build -c Release`（DirectML ヘルパー）
5. Vulkan/RIFE 資材は `scripts/get_models.py`
6. `run.bat` で起動。`.venv\Scripts\python -m pytest` が全通過すること（64 passed 時点）

環境変数（README の表も参照）: `UEU_WINML_HELPER` / `UEU_MODELS_DIR` /
`UEU_NPU_PYTHON` / `UEU_NPU_CACHE` / `UEU_MAX_VIDEO_DIM`

## NVIDIA GPU 機での確認・実験タスク

1. DirectML 経路が動くことの確認（画像1枚・動画3秒）。AMD機の基準値は下表
2. **NVIDIA 専用 EP の評価**: `tools/winml-sr` で `winml-sr list --download` を実行すると
   Windows ML カタログから `NvTensorRtRtxExecutionProvider`（RTX 30xx 以降）が取得できる
   はず。`winml-sr run --ep-name NvTensorRtRtxExecutionProvider ...` で DirectML と
   速度・PSNR を比較する（EPのダウンロードは Windows Update が有効な状態で行う）。
   有望なら GUI の AI実行先に追加する（`helper_backend` の解決表と `_BACKEND_DESC` に行を足す）
3. HWエンコーダ: `video.detect_hw_encoder` が h264_nvenc を実エンコードで検証して選ぶ
   はず。動作確認し、失敗するなら原因を調べる
4. AMD NPU が無い環境では「NPU」を選んでも起動失敗→Vulkanフォールバックになる。
   非対応環境では選択肢を無効化または非表示にする改善が未実装
5. 上記の結果は docs/（新規 md）に実測値で記録し、README のベンチ表には
   「NVIDIA機」列または別表として追加する

AMD機（Ryzen AI 7 PRO 350 / Radeon 860M）の基準値 @854x480→4x, 1枚:
DirectML: Anime Video v3 0.46s / 4xNomosUni SPAN 0.51s / Real-ESRGAN 2.78s。
動画 rawパイプ: GPU 2.48fps。

## 作業規約（必ず守る）

- **既存のUI表示・文書の設計を指示なく別の抽象化に置き換えない**。変更前に原文
  （必要なら git 履歴）を読み、その設計の中に追加する
- 公開ドキュメント（README / docs）は宣伝調ラベル禁止、モデルは正式名、測定条件・
  精度・タイル・dB・時間を具体的に書く。README は現在の姿のみ、履歴は docs/npu-research.md
- モデル重み・ONNX・NPUキャッシュ・ヘルパーのビルド成果物は git に追加しない
- マシン固有の絶対パスをソースに埋め込まない（環境変数で上書き可能な既定値にする）
- 変更後は `pytest` 全通過＋該当経路の実機スモーク（画像1枚/動画3秒）を必ず行う。
  GUI 変更は `scripts/preview_shot.py` でスクショ確認
- git commit / push はユーザーの指示があってから

## 既知の罠（docs に詳細あり）

- SPAN 系の ONNX エクスポート: export 前の forward を `torch.inference_mode` で
  行わない／`nn.SiLU` をモジュール置換しない／検証は別インスタンスの参照と比較
  （`scripts/npu/export_spandrel.py` のコメント参照）
- 4x固定の新バックエンドに 1080p 以上を入れると 8K 出力になる → 4K 自動フィットで対処済み
- 奇数幅の動画は H.264 の偶数制約に当たる → 偶数パディング済み
- AMD NPU の Windows ML カタログ EP はドライバ版数で適格性が変わる（AMD機固有の事情、
  docs/span-bench-results.md 追記4）。NVIDIA 機には関係ない

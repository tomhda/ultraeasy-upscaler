# ultraeasy-upscaler

real-esrgan-gui の不便を解消する、Windows ローカル専用の「かんたん神アプコン」ツール。

## 解決する課題
- **① D&D 非対応・複数枚はフォルダ指定のみ** → ドラッグ&ドロップ対応、複数ファイル/フォルダをキューに一括投入。
- **② 静止画のみ（動画は手動で 分解→拡大→結合）** → 動画を自動で **分解 → 拡大 → 再結合（音声維持）**。
- **③ NPU未活用** → Ryzen AI NPU（VitisAI EP）で画像 x4 アップスケールに対応。
- **④ フレーム補間なし** → フェーズ2で対応予定（RIFE / rife-ncnn-vulkan）。

## アーキテクチャ
- **GUI**: PySide6（ダークテーマ・ネイティブ D&D）。`app/gui`
- **コア（GUI 非依存・単体テスト可）**: `app/core`
  - `binaries.py` 外部バイナリ/モデル探索
  - `media.py` 種別判定・ffprobe メタ取得
  - `upscaler.py` realesrgan-ncnn-vulkan ラッパ（画像/フォルダ）
  - `npu_backend.py` / `npu_worker.py` Ryzen AI conda 環境経由の NPU ラッパ（画像/フォルダ）
  - `video.py` ffmpeg 抽出/再結合/HWエンコード
  - `engine.py` ジョブのオーケストレーション
  - `jobs.py` / `settings.py` データモデル

## 必要物
- Python 3.13（同梱の `.venv` を使用）
- ffmpeg / ffprobe（PATH 上）
- `vendor/realesrgan/` に realesrgan-ncnn-vulkan 一式（exe + models）

## NPU バックエンド
- 画像/画像フォルダの **4x** アップスケールのみ対応。
- 動画は当面 `GPU (Vulkan)` 経路を使う。
- Ryzen AI Software 1.7.1 の conda 環境 `ryzen-ai-1.7.1` が必要。
- NPU モデルと VitisAI キャッシュは `vendor/amd-npu/` に配置済み。
- 通常の GUI は `.venv` で起動し、NPU処理だけ `conda run -n ryzen-ai-1.7.1` のサブプロセスで実行する。
  - `vendor/` は git 管理外。初回・再クローン時は `.venv\Scripts\python.exe scripts\get_models.py` で本体＋追加モデル（汎用動画モデル含む）を一括取得。

## 起動
`run.bat` をダブルクリック、または:
```
.venv\Scripts\python.exe -m app.main
```

## 開発
```
.venv\Scripts\python.exe -m pytest        # コアの単体テスト
```

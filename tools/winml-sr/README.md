# winml-sr — Windows ML で自前ONNXを実行するヘルパー

Windows ML（スタンドアロン `Microsoft.Windows.AI.MachineLearning` NuGet）で
Real-ESRGAN系の固定256x256タイルONNXを実行する検証用コンソールアプリ。
**MSIX不要・unpackagedでそのまま動く**（ImageScaler系と違いパッケージidentity不要）。

タイル分割/結合は `app/core/npu_runner.py` と同一ロジック
（reflectパディング → オーバーラップ付き切り出し → コア領域のみ合成）。

## ビルド・実行

```powershell
dotnet build -c Release
.\bin\Release\net8.0-windows10.0.22621.0\win-x64\winml-sr.exe list
```

```powershell
# EP列挙（--download でカタログEPの取得も試行）
winml-sr list [--download]

# 実行（--ep-policy か --ep-name のどちらかで指定。既定 npu ポリシー）
winml-sr run --model <onnx> --input <img> --output <img>
             [--ep-policy npu|gpu|cpu|default|power|perf|efficiency]
             [--ep-name DmlExecutionProvider [--device-type GPU]]
             [--overlap 16] [--compile] [--download] [--warmup 2]

# 画質比較
winml-sr psnr --a ref.png --b out.png
```

## 検証結果（2026-08-14, Radeon 860M / Ryzen AI KRK NPU）

入力 854x480 → x4、タイル256・overlap16 = 12タイル。モデルは
`tmp/npu-anime/*_nchw_256x256_fp32.onnx`（全部fp32を投入、
NPU側の精度変換はEP任せ）。PSNRは同モデルのCPU fp32出力比。

| モデル | EP | ms/タイル(中央値) | 推論合計 | PSNR | セッション生成(初回→キャッシュ後) |
|---|---|---|---|---|---|
| animevideov3 (SRVGG) | **DML (GPU)** | **52ms** | **0.63s** | 101dB | 0.6s |
| animevideov3 | MIGraphX (GPU) | 69ms | 0.83s | 98.5dB | 90s→未計測 |
| animevideov3 | RyzenAILight | 227ms | 2.73s | 99.0dB | 0.1s |
| animevideov3 | VitisAI (NPU) | 274ms | 3.3s | 59.7dB | 60s→0.1s |
| **animevideov3dp** (PReLU分解) | **VitisAI (NPU)** | **133ms** | **1.6s** | 51.6dB | 275s→0.4s |
| animevideov3 | CPU | 275ms | 3.45s | — | 0s |
| realesrgan (RRDB) | DML (GPU) | 231ms | 2.78s | 94.9dB | 0.5s |
| **realesrgan (RRDB)** | **VitisAI (NPU)** | **170ms** | **2.06s** | 47.2dB | 540s→0.3s |
| realesrgan (RRDB) | CPU | 1000ms | 12.0s | — | 0s |

既存手段との比較（同一ワークロード）:
- GPU Vulkan exe (animevideov3): 1.2s/枚 → **WinML DML 0.63s で約2倍高速**
- ネイティブNPU bf16 dp (RyzenAI 1.7.1 VAIML): 151ms/タイル → **WinML dp 133ms でNPU新記録**
- ネイティブNPU bf16 realesrgan: 約1.5s/枚 → WinML 2.06s（近いが僅かに劣後）

### 512タイル実験（dp・NPU）

| タイル | タイル数@480p | ms/タイル | µs/px | 1枚合計 | PSNR |
|---|---|---|---|---|---|
| 256 (core 224) | 12 | 133ms | 2.03 | 1.6s | 51.6dB |
| 512 (core 480) | 2 | 512ms | 1.95 | **1.02s** | 51.6dB |

- タイル毎の固定コストは**約7msのみ**（T=F+C·px にフィットさせると F≈7ms）。NPU時間は処理ピクセル数にほぼ線形
- 512タイルが速い理由は**パディング無駄の削減**（256タイルは480行の画像に672行分の処理＝約4割が捨てピクセル。512はcore480が縦にピッタリ）
- → **タイルサイズは「無駄が最小になるように」選ぶのが正解**。854x480ならdp 512タイルで1.02s/枚＝Vulkan exe(1.2s)を超える
- タイルサイズ毎に別のVAIMLコンパイル＆キャッシュが必要（512版は初回400s→キャッシュ後1.2s）

わかったこと:
- **軽量モデルはDML、重量モデル/GPUフリー運用はVitisAI NPU** の住み分けが WinML 一本で成立
- **ネイティブVAIMLのPReLU bf16数値爆発バグはWinML EP(1.8)では発生しない**
  （非分解animevideov3でPSNR 59.7dB。WU配信EPは `WindowsWorkload.EP.AMD.VitisAI.1.8` =
  RyzenAI SW 1.7.1 より新しい。amd/RyzenAI-SW#390 は EP 1.8 で修正済みの可能性大）
- **VitisAI EPは初回コンパイル結果を自動で永続キャッシュ**する（2回目以降のセッション生成は0.1〜0.4s）。
  アプリ側でのキャッシュ管理は不要
- `OrtModelCompilationOptions`（EPContextコンパイル）はVitisAIでは現状動作しない
  （0.2sで144KBのスタブを吐き、そのモデルでのセッション生成は失敗）。自動キャッシュがあるので実害なし
- RRDBのVAIMLコンパイル中に `Invalid NodeIndex` エラーが多数出るが結果は正常（非致命）
- WebGpu EP はDLできるが TryRegister=False（未調査）

## serve モード（常駐ヘルパー、2026-08-14実装）

```powershell
winml-sr serve --model <onnx> --ep-name <EP> [--overlap 16] [--warmup 2]
```

stdout/stdin のバイナリプロトコル（int32 LE、RGB24生ピクセル）で常駐処理。
`UEUH`(ready: scale/tileW/tileH) / `UEUF`(要求: w/h/RGB) / `UEUD`(応答) / `UEUE`(エラー)。
ログは全て stderr。stdin EOF で終了。Pythonクライアント+ベンチ: `serve_client.py`。

本家アプリからは `app/core/helper_backend.py` が次の場所を探索する。

```text
tools/winml-sr/bin/Release/net*/win-x64/winml-sr.exe
```

任意の場所へ配置した場合は `UEU_WINML_HELPER` で実行ファイルを指定できる。

実測（480p→4x、20フレーム連続、クライアント壁時計）:

| 構成 | ms/frame | オーバーヘッド(推論以外) |
|---|---|---|
| DML GPU + av3 256 | **534ms** | 約60ms |
| VitisAI NPU + av3dp 512 | **1100ms** | 約52ms |

**旧方式（exe都度起動+PNG受け渡し）の+1.2〜1.4s/枚 → 約50〜60msに短縮**。
出力は単発runとビット一致（PSNR 99dB=mse0）。準備完了まで GPU 1.5s / NPU 3.2s。

### I/O Binding + 3段パイプライン化（2026-08-15）

出力テンソルの`ToArray()`コピー（512タイルで48MiB/枚）を事前確保OrtValueの
出力バインディングで排除し、serveを「受信変換/推論/変換送信」の3段パイプライン
（各段仕掛かり1、順序保持）に変更。クライアントは深さ2の先行送信対応。

| 構成 | 旧同期 | 新同期 | パイプライン | 純粋Run中央値 |
|---|---|---|---|---|
| DML GPU + av3 256 | 604ms | 481ms | **459ms/frame** | **34.5ms/タイル** |
| VitisAI NPU + av3dp 512 | 1078ms | 1130ms | 1103ms/frame | 543ms/タイル(揺らぎ509〜548) |

- **GPUは-24%**: 52ms/タイルと思っていたうち約18msが出力コピーだった。真のGPU推論は34.5ms/タイル
- **NPUはほぼ不変**: 純粋Runが543ms/タイルでコピー影響は約4%のみ。**約2µs/pxはNPUの実力値と確定**
  （タイル全体565ms＝Run 543+入力充填1.4+マージ12.4）。NPU実行時間自体に±4%程度の揺らぎあり
- 全構成で出力はビット一致（PSNR 99dB）を維持

## VitisAI / MIGraphX EP の取得

EPカタログからのダウンロードは **Windows Updateの一時停止中は失敗する**
（EPはWU経由配信・公式ドキュメント記載の既知事項）。
エラー: 「製品が適用できないか、見つかりません」。WU再開後に `winml-sr list --download`。
NPUドライバ要件（32.0.203.280+）は 32.0.203.314 で充足済み。
取得済みEPはシステム共有・自動更新（`%LOCALAPPDATA%\Packages\WindowsWorkload.EP.AMD.*`）。
一度入ればWU停止中も動作する。

参考:
- https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/execution-provider-errors
- https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/initialize-execution-providers
- https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/tutorial

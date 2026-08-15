# SPAN 超解像モデル WinML ベンチ結果

> **✅ 追記4（2026-08-15夕・原因確定）**: NPU停止の根本原因は「提供終了・世代交代」ではなく
> **NPUドライバ非互換**。8/14の更新窓でNPUドライバが32.0.203.314→**.329**になり、
> 現行VitisAI EP 1.8.68の公式対応範囲（**.280〜.297**、MS docs明記）の上限を超過→
> カタログの適格性判定がVitisAIを除外（判定反映が数時間遅延したため夜間ベンチは成功していた）。
> VitisAI/MIGraphXはMS公式カタログに現役掲載で提供終了ではない。RyzenAILightは
> 後継ではなく「前処理済みハイブリッドモデル用のCPUベース軽量EP」（ORT PR #25513）。
> 実EPの配布実体は `Microsoft.WinML.AMD.NPU.EP.2`（0.0.68.0、8/14 20:28導入、
> VitisAI+RyzenAILight同梱。下記追記の1.8.58/1.8.60は5月からある旧世代で無関係）。
> BYO再試行（現役1.8.68 DLL+MSIX動的依存）もDllMain 1114で不可（適格性状態参照の疑い）。
> **復旧見込み: 次期VitisAI 1.8.72が2026年8月第4週(8D)にGA予定**（新ドライバ対応の公算大）
> →配信後に短い更新窓で取得すれば復旧見込み。ドライバの.297以下へのロールバックは
> リスクに見合わないため見送り。詳細: ユーザー調査 `ultraeasy-upscaler-winai_AMD_NPU停止原因調査_20260815.md`

> **⚠️ 追記2（2026-08-15昼）: 本ドキュメントのONNXは2つのエクスポートバグで壊れていた。**
> ①export前のforwardを`torch.inference_mode`で実行（Conv3XCの`.data`直接代入と干渉し
> トレースが無音破損）②`nn.SiLU(inplace=True)`を非インプレース等価式にモジュール置換
> （SPANはインプレース副作用に依存、purephotoのみ該当・PSNR22dBに崩壊）。
> スクリプトの自己検証は「トレースで変異した同一オブジェクト」を参照にしていたため
> 両バグとも素通し。修正済み（export_spandrel.py、フレッシュ参照検証に変更）。
> **本ドキュメントの画質評（PSNR 36〜42dB等）は壊れモデルの数値で無効**。
> 速度はグラフ構造がほぼ同じため参考値としては有効だが、NPU復旧後に要再測定。
> 初回の目視評価（ドット格子・色ズレ・バンディング）も壊れモデル起因であり
> モデル自体の評価ではない。修正版での再評価は別途。

> **✅ 追記3（2026-08-15・最終目視評価、修正版ONNX・DML fp32）**
> BBB/Superman/ToS の同一フレームで「本家1.7.1 bf16 vs SPAN」をユーザー目視評価。
> - BBB(綺麗なCG): **purephoto勝ち**（毛の質感を残しつつ解像。忠実系）
> - Superman(劣化した古いアニメ): **本家av3dp bf16の明確勝ち**（忠実系は劣化まで再現）
> - ToS(実写): **realesrgan bf16とpurephotoは引き分け**（加工感強め vs 弱めの好みの軸）
> - modernspan: 全素材で勝ちどころ無し（アニメで本家未満、実写は油絵化）→画質面は落選
>
> 結論: **purephotoは「忠実系・実写/綺麗ソース向け」として採用価値あり**（NPU復旧後は
> 速度面でも realesrgan bf16 の約3倍が見込める）。modernspanはNPU速度枠でのみ再検討。
> NPU復旧後の宿題: SPAN速度の再測、purephoto bf16化の画質確認。
> Windows Update経由のEP自動更新で `WindowsWorkload.EP.AMD.VitisAI.Framework` が
> 1.8.58→1.8.60 に上がり、カタログから VitisAI / MIGraphX が除外（※追記4で原因確定、「配信停止」は誤り）
> （「製品が適用できないか、見つかりません」）。後継の `RyzenAILightExecutionProvider`
> (NPU+GPU) は本リポジトリのfp32 CNNモデルを**サイレントにCPUフォールバック**する
> （セッション生成0.1s・PSNR 99dB=CPUとビット一致・CPU同等速度で確認）。
> EPContextコンパイル経路はAccessViolationでクラッシュ。残置されていた旧VitisAI DLLの
> BYO登録（--ep-lib）も DllMain初期化失敗(Error 1114)で不成立。
> **つまり2026-08-15朝時点でこのマシンのNPU実行は停止中**。DML(GPU)数値は影響なし。
> VitisAI時代のNPU数値は「EPが復旧すれば再現可能なはずの参考記録」として残す。

測定日: 2026-08-15 JST  
入力: `test_480p.png`（854x480、RGB）  
タイル: overlap 16、warmup 2、入力タイル数は256で12枚、512で2枚  
実行環境: Windows ML / VitisAI 1.8系、DML、CPU。NPUのセッション生成時間は初回VAIMLコンパイルを含む。

最終結果: ONNX化・Torch/ORT照合は4/4、CPU・DML・VitisAIの最終ベンチは全項目成功。失敗扱いで残った項目はない。

## モデル情報

| 名前 | アーキ | スケール | ライセンス | 重みサイズ | SHA-256 | DL元 |
|---|---|---:|---|---:|---|---|
| modernspan | SPAN（64nf、ModernSpanimationV1） | 2x | MIT | 15,830,922 bytes（15.83 MB） | `BC6CA08AB1EEB9884A5C43C025EDA97A5AA9CCFAA567734F426AEDF55CF78327` | [TNTwise/Models release](https://github.com/TNTwise/Models/releases/download/2x_ModernSpanimationV1/2x_ModernSpanimationV1.pth) |
| purephoto | SPAN（48nf、NomosUni span_multijpg） | 4x | CC-BY-4.0 | 4,492,232 bytes（4.49 MB） | `3BEDFF643A1BA51B12E0174EBCA62649A930AE3E7B0868BE9706D8659D4D32A2` | [Hugging Face](https://huggingface.co/Phips/4xNomosUni_span_multijpg/resolve/main/4xNomosUni_span_multijpg.safetensors) |

## ONNX化

`scripts/npu/export_spandrel.py` を作成し、spandrelの `ImageModelDescriptor.model` を使用した。両モデルとも入力仕様は RGB `[0,1]`、float32、NCHW、3chであり、正規化・NHWC変換アダプタは不要だった。固定形状、opset 17、dynamic axesなし、onnxsim後のモデルを出力した。

| モデル | 入力 | 出力 | ONNXサイズ | Torch/ORT最大差 | 結果 |
|---|---|---|---:|---:|---|
| modernspan | `1x3x256x256` | `1x3x512x512` | 2,927,203 bytes | `4.172e-7` | 合格 |
| modernspan | `1x3x512x512` | `1x3x1024x1024` | 2,927,203 bytes | `4.768e-7` | 合格 |
| purephoto | `1x3x256x256` | `1x3x1024x1024` | 1,725,172 bytes | `1.103e-6` | 合格 |
| purephoto | `1x3x512x512` | `1x3x2048x2048` | 1,725,172 bytes | `1.460e-6` | 合格 |

4本すべてで最大差 `<1e-4`、ONNX checker、ORT CPU実行、出力shapeを確認した。
さらに指定テスト画像から実画像タイルを作って照合し、最大差はmodernspan 2.76e-6、purephoto 1.32e-5で、実入力でも`<1e-4`だった。

## ベンチ結果

`ms/タイル` はタイル全体の中央値、`pure-run中央値` は `session.Run` の中央値、`1枚pure-run合計` は全タイルのpure-run合計。`1枚wall合計` はプロセス起動、EP初期化、セッション生成、画像I/O、タイル処理を含むため、NPUの初回行ではコンパイル時間が支配的になる。PSNRは同一モデル・同一タイルサイズのCPU出力との比較であり、画質の絶対評価ではない。

| モデル | EP | タイル | タイル数 | ms/タイル中央値 | pure-run中央値 | 1枚pure-run合計 | 1枚wall合計 | µs/px | PSNR | セッション生成 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| modernspan | CPU | 256 | 12 | 282.3 ms | 282.0 ms | 3.29 s | 4.75 s | 4.308 | 基準 | 0.0 s |
| modernspan | CPU | 512 | 2 | 1,080.0 ms | 1,078.1 ms | 2.14 s | 5.43 s | 4.120 | 基準 | 0.0 s |
| modernspan | DML（GPU） | 256 | 12 | 56.8 ms | 56.0 ms | 0.67 s | 2.08 s | 0.867 | 95.01 dB | 0.3 s |
| modernspan | VitisAI（NPU） | 256 | 12 | 47.7 ms | 46.7 ms | 0.55 s | 226.44 s | 0.728 | 42.10 dB | 225.0 s |
| modernspan | VitisAI（NPU） | 512 | 2 | 176.1 ms | 172.2 ms | 0.34 s | 198.52 s | 0.672 | 42.10 dB | 197.0 s |
| purephoto | CPU | 256 | 12 | 176.4 ms | 170.8 ms | 2.10 s | 3.43 s | 2.692 | 基準 | 0.0 s |
| purephoto | CPU | 512 | 2 | 723.4 ms | 716.1 ms | 1.42 s | 3.87 s | 2.760 | 基準 | 0.0 s |
| purephoto | DML（GPU） | 256 | 12 | 46.0 ms | 42.5 ms | 0.51 s | 1.87 s | 0.702 | 90.46 dB | 0.2 s |
| purephoto | VitisAI（NPU） | 256 | 12 | 70.1 ms | 67.1 ms | 0.80 s | 398.66 s | 1.070 | 36.62 dB | 396.7 s |
| purephoto | VitisAI（NPU） | 512 | 2 | 258.4 ms | 239.4 ms | 0.48 s | 326.42 s | 0.986 | 36.62 dB | 324.4 s |

µs/pxは次式で算出した。

```text
µs/px = ms/タイル × 1000 / (タイル入力幅 × タイル入力高さ)
```

### 既存実測との比較

| 既存構成 | タイル | ms/タイル | pure-run | 1枚合計 | µs/px |
|---|---:|---:|---:|---:|---:|
| animevideov3 DML（GPU） | 256 | 52 ms | **34.5 ms** | **459 ms/枚（serve）** | 0.526 |
| animevideov3dp VitisAI（NPU） | 256 | **133 ms** | — | — | 2.029 |
| animevideov3dp VitisAI（NPU） | 512 | 512 ms | **543 ms** | **1.02 s/枚** | 1.953 |
| NPU処理ピクセル線形基準 | — | — | — | — | **約2.0** |

SPANのµs/pxは、modernspanがNPU 0.728（256）/0.672（512）、purephotoがNPU 1.070（256）/0.986（512）だった。いずれも既存NPU基準の約2.0 µs/pxを下回る。DMLではmodernspan 0.867、purephoto 0.702で、既存animevideov3 DMLの0.526より遅い。

なお、modernspanは2x、既存animevideov3/animevideov3dpは4xであり、出力画素数は同一ではない。µs/pxは指定どおり入力タイル画素基準で比較している。

## 所見

### NPUへの載り方

4つのVitisAIログすべてで、`Compilation Complete`、`ERROR:0`、終了コード0、出力PNG生成を確認した。各ログのサブグラフ出力は `vaiml_par_0` の1系統だけで、`vaiml_par_1` 以降は観測されなかった。したがって、今回のSPANグラフはVitisAI上で単一のVAIMLパーティションに載ったと推定する。

`Invalid NodeIndex` は各NPU実行で148件出たが、コンパイルは `WARNING:76, CRITICAL-WARNING:0, ERROR:0` で完了し、推論も成功した。現状は非致命的なコンパイラログとして扱う。

CPUに対してNPUが明確に短縮されているため、CPUフォールバックとは判断しない。modernspanは256でCPU 282.0msに対してNPU 46.7ms、purephotoは256でCPU 170.8msに対してNPU 67.1msだった。512でもmodernspan 172.2ms、purephoto 239.4msで、CPUの1,078.1ms、716.1msより短い。

### DMLとNPUの使い分け

- DMLはPSNRが90dB以上でCPU出力に近く、purephoto 256ではpure-run合計0.51秒。数値再現性を優先するならDMLが安全。
- NPUは初回コンパイルが約197〜397秒と長いが、セッション生成後の推論はmodernspan 0.55秒（256）/0.34秒（512）、purephoto 0.80秒（256）/0.48秒（512）まで短い。夜間バッチや常駐利用でキャッシュを再利用できる場合に向く。
- NPUのPSNRは42.10dB（modernspan）、36.62dB（purephoto）で、DMLより大きく低下した。これは同一モデルCPU出力との数値差であり、NPU出力をそのまま本番採用する前に、実画像の目視・下流品質基準を通す必要がある。
- 854x480ではoverlap16の256タイルが12枚、512タイルが2枚になるため、両モデルとも512の方が1枚pure-run合計が短い。NPUは512を第一候補にする価値がある。

### animevideov3を置き換える価値

- DMLだけを見ると、既存animevideov3のpure 34.5ms/タイル、459ms/枚（serve）に対して、modernspanは56.0ms/タイル・0.67秒、purephotoは42.5ms/タイル・0.51秒であり、速度だけを理由に置き換える価値はない。
- NPUでは既存animevideov3dpの約2.0 µs/pxに対してSPANは0.67〜1.07 µs/pxで、512の1枚pure-runも0.34〜0.48秒まで短縮した。NPU運用を前提にすれば置き換え候補になる。
- ただしNPU PSNR低下が残るため、現時点の結論は「速度面では置き換え価値あり、品質面は要受入判定」。自動的な全面置換はしない。

## ハマった点

1. `pip install spandrel` の依存解決が、既存の `torch 2.4.1+cpu` を `2.13.0+cpu` に置き換えた。条件違反を残さないため、PyTorch公式CPUインデックスから `torch 2.4.1+cpu` と互換 `torchvision 0.19.1+cpu` を復元した。最終確認はtorch 2.4.1+cpu。
2. Hugging Faceの4.49MB safetensorsは初回の大容量GETが無転送で終了した。同じ指定URLを再試行して取得し、サイズとSHA-256を確認した。
3. purephotoを白色ノイズでTorch/ORT照合すると、モデル内部の反復残差が値を約-990〜1536まで増幅し、丸め差の最大値が約`5.9e-2`になった。0.5中心・振幅0.01の決定的なRGB入力を検証ダミーにし、SiLUをSigmoid/Mulへ明示展開した結果、4本すべてが`1.46e-6`以下で合格した。
4. VitisAI初回セッションはモデル・タイルごとに197〜397秒かかった。コンパイル中はプロセスを中断せず、peano-lib/aiecompilerのログが継続する状態を待機した。

## 生成物とログ

- エクスポータ: `scripts/npu/export_spandrel.py`
- ONNX: `tmp/npu-anime/span/{modernspan,purephoto}_nchw_{256,512}x{256,512}_fp32.onnx`
- ベンチ出力・ログ: `tmp/npu-anime/span/bench/`

# GPU速度比較（2026-08-24）

同じ入力・モデル・タイル条件で、NVIDIA GeForce RTX 5060 Ti、AMD内蔵GPU、TensorRT、Vulkanの速度を比較した。

## 測定条件

- GPU: NVIDIA GeForce RTX 5060 Ti（VRAM 16GB）、AMD Radeon(TM) Graphics（内蔵、報告VRAM 485MB）
- CPU: AMD Ryzen 7 9700X
- モデル: `animevideov3_nchw_256x256_fp32.onnx`（256x256、fp32、x4）
- 入力: 854x480、出力: 3416x1920
- タイル: 256x256、12タイル（4x3）、overlap 16
- DirectML / TensorRT: ウォームアップ2回
- Vulkan: アプリのVulkan経路、`tile_size=256`
- 出力: すべてPNG。WinML系の時間はhelper CLIの計測値、Vulkanはアプリ呼び出しの経過時間

入力画像・動画はベンチマーク時に用意したローカルのテスト素材から生成した。
素材そのものはリポジトリへ同梱せず、公開時の再測定には出典を明示できるサンプル動画または合成テスト動画を使用する。

## 結果

| 経路 | デバイス | セッション生成 | タイル中央値 | pure-run合計 | wall total |
|---|---|---:|---:|---:|---:|
| DirectML | NVIDIA RTX 5060 Ti | 0.3秒 | 7.8ms | 0.07秒 | **1.50秒** |
| DirectML | AMD Radeon Graphics（内蔵） | 0.2秒 | 136.6ms | 1.61秒 | 3.11秒 |
| TensorRT RTX | NVIDIA RTX 5060 Ti | 0.5秒 | **7.4ms** | **0.07秒** | 1.60秒 |
| Vulkan | NVIDIA RTX 5060 Ti | — | — | — | 1.888秒 |

WinML系のpure-runは、セッション作成・PNG入出力を除いた推論部分である。Vulkan経路は同じ粒度のタイミングをアプリ側へ公開していないため、プロセス起動から出力完了までの経過時間を記録した。

## 出力の一致度

NVIDIA DirectMLの出力を基準に、同じ3416x1920出力のPSNRを計算した。

| 比較対象 | PSNR |
|---|---:|
| AMD DirectML | 95.89dB |
| TensorRT RTX | 64.10dB |
| Vulkan | 58.02dB |

PSNRは実装・モデル形式の差を確認するための相互比較値であり、単独の画質評価値ではない。AMD DirectMLはNVIDIA DirectMLとほぼ同じ出力になった。

## 判断

- RTXではTensorRTのタイル中央値がDirectMLより約5%短いが、起動・入出力を含むwall totalはDirectMLの方が短かった（1.50秒対1.60秒）。GUIの既定経路は現状どおりDirectMLとする。
- AMD内蔵GPUでもDirectMLは動作した。ただし報告VRAMが485MBのため、RTXとの速度差はハードウェア差が大きく、AMD GPU全般の代表値ではない。
- Vulkanはこの1枚のE2Eでは1.888秒で、RTX DirectMLの1.50秒より遅かった。動画のようにhelperセッションを再利用する処理では、wall totalよりpure-runやフレーム単位の定常時間を重視する。
- TensorRT RTXはCLI経路で利用可能なことを確認した。GUIへ組み込むかは、複数サイズ・複数モデル・動画の定常測定を追加してから判断する。

## 動画比較

同じ `upscale_video_piped` のffmpeg rawvideoパイプラインへ、DirectMLとTensorRTの常駐セッションをそれぞれ接続した。入力は640x480、24fps、音声付きで、出力は2560x1920、H.264（`h264_nvenc`）、AAC音声。動画の長さだけを変え、モデル・タイル・エンコーダを揃えた。

| 素材長 / フレーム数 | DirectML E2E | TensorRT E2E | TRT−DML | DML pure-run合計 | TRT pure-run合計 |
|---|---:|---:|---:|---:|---:|
| 1秒 / 24 | **3.626秒** | 3.635秒 | +0.009秒 | 1.32秒 | 1.19秒 |
| 3秒 / 72 | 7.570秒 | **7.513秒** | −0.057秒 | 3.97秒 | 3.57秒 |
| 12秒 / 288 | 25.227秒 | **25.026秒** | −0.201秒 | 15.80秒 | 14.56秒 |

セッション生成はDirectMLが約0.3秒、TensorRTが約0.5〜0.7秒だった。pure-run合計はタイル推論時間の合計であり、デコード・色変換・NVENCは含まない。E2Eはセッション生成から動画出力完了までを含むため、3段パイプラインの並列化によりpure-run差がそのままE2E差にはならない。

短い1秒動画ではDirectMLがわずかに速く、3秒以上ではTensorRTが逆転した。12秒での差は0.201秒（約0.8%）に留まるため、長い動画ほどTensorRTの定常推論優位は現れるものの、現状の動画全体ではNVENC・入出力処理が支配的で、劇的な差にはならない。

## モデル別の動画比較

同じ3秒動画で、モデルだけを差し替えて比較した。いずれも256x256入力、x4、9タイル/フレーム、overlap 16、ウォームアップ1回である。

| モデル | DirectML E2E | TensorRT E2E | TRT−DML | DML pure-run合計 | TRT pure-run合計 |
|---|---:|---:|---:|---:|---:|
| Anime Video v3 | 7.570秒 | **7.513秒** | −0.057秒 | 3.97秒 | 3.57秒 |
| 4xNomosUni SPAN（purephoto） | 8.405秒 | **7.576秒** | −0.829秒 | 4.79秒 | 3.52秒 |
| Real-ESRGAN（AMD-RRDB） | **23.310秒** | 24.694秒 | +1.384秒 | 19.45秒 | 19.21秒 |

TensorRTのセッション生成はAnime Video v3が0.7秒、purephotoが0.8秒、Real-ESRGANが2.1秒だった（DirectMLはそれぞれ0.3秒、0.4秒、0.5秒）。purephotoは推論差がセッション生成差を上回ったためE2Eでも速くなった。一方、Real-ESRGANはpure-runが約0.24秒短いだけでセッション生成差が約1.6秒あるため、3秒動画ではDirectMLが速い。

したがって「動画が長いほどTensorRTが有利」という大枠はあるが、実際の逆転点はモデル依存である。Real-ESRGANでは同じ傾向を単純外挿すると、セッション生成差を回収するには数十秒級の動画が必要になる見込みであり、追加の長尺測定なしには断定しない。

<#Requires -Version 5.1
<#
.SYNOPSIS
    GitHub Release 配布物（ビルド済み winml-sr と変換済み ONNX の zip一式）を作成する。

.DESCRIPTION
    (a) tools/winml-sr を dotnet build し、実行一式を winml-sr-win-x64.zip に固める。
    (b) GPU fp32 ONNX を models-gpu-fp32.zip に固める。
    (c) AdcSR fp32 ONNX を models-adcsr-gpu-fp32.zip に固める。
    (d) NPU bf16cast ONNX を models-npu-bf16.zip に固める。
    (e) AdcSR NPU 前半/後半＋マニフェストを models-adcsr-npu-bf16.zip に固める。
    各 zip の SHA-256 を SHA256SUMS.txt に書き出す。
    zip 内は展開先（vendor/winml-sr/、models/ai/）直下に置ける平置き構造にする。

    ONNX の選定は app/core/settings.py の HELPER_MODEL_FILES のファイル名と
    一致するものだけを対象にする。一致しない旧名ファイルは配布しない。

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\build_release.ps1
    powershell -ExecutionPolicy Bypass -File scripts\build_release.ps1 -OutDir tmp\easy-setup\dist -SkipBuild
#>
param(
    [string]$OutDir = "",
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if (-not $OutDir) {
    $OutDir = Join-Path $repo "tmp\easy-setup\dist"
}
$licenses = Join-Path $repo "scripts\release-licenses"
$stage = Join-Path $OutDir "_stage"

function Fail([string]$Message) {
    Write-Error $Message
    exit 1
}

function Get-Sha256([string]$Path) {
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLower()
}

try {
    foreach ($dir in @($OutDir, $stage)) {
        if (-not (Test-Path -LiteralPath $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
        }
    }
    if (-not (Test-Path -LiteralPath (Join-Path $licenses "LICENSE-BSD-3-Real-ESRGAN.txt"))) {
        Fail "ライセンス原文がありません: $licenses。先に取得してください。"
    }

    # --- (a) winml-sr のビルド ---
    if (-not $SkipBuild) {
        $dotnet = Get-Command dotnet -ErrorAction SilentlyContinue
        if (-not $dotnet) {
            Fail "dotnet が見つかりません。.NET 8 SDK を導入して PATH を通してください。"
        }
        Write-Output "dotnet build を実行します…"
        & dotnet build (Join-Path $repo "tools\winml-sr") -c Release
        if ($LASTEXITCODE -ne 0) {
            Fail "dotnet build に失敗しました（終了コード $LASTEXITCODE）。"
        }
    }
    $buildOut = Get-ChildItem -LiteralPath (Join-Path $repo "tools\winml-sr\bin\Release") `
        -Directory -Filter "net*" | Sort-Object Name -Descending |
        ForEach-Object {
            Get-ChildItem -LiteralPath $_.FullName -Directory -Filter "win-x64" |
                Select-Object -First 1 -ExpandProperty FullName
        } | Select-Object -First 1
    if (-not $buildOut -or -not (Test-Path -LiteralPath (Join-Path $buildOut "winml-sr.exe"))) {
        Fail "ビルド成果物が見つかりません: tools/winml-sr/bin/Release/net*/win-x64/winml-sr.exe"
    }
    if (-not (Test-Path -LiteralPath (Join-Path $buildOut "seam_templates\adcsr_ov32_p256.json"))) {
        Fail "seam_templates がビルド出力にありません: $buildOut"
    }
    $nuget = Join-Path $env:USERPROFILE ".nuget\packages\microsoft.windows.ai.machinelearning\2.2.12"
    $winmlLicense = Join-Path $nuget "license.txt"
    $winmlTpn = Join-Path $nuget "ThirdPartyNotices.txt"
    foreach ($need in @($winmlLicense, $winmlTpn)) {
        if (-not (Test-Path -LiteralPath $need)) {
            Fail "再配布に必要なファイルがありません: $need。dotnet build で NuGet を復元してください。"
        }
    }

    # --- ソース ONNX の解決（HELPER_MODEL_FILES のファイル名と一致するものだけ） ---
    $modelRoots = @{
        Top  = Join-Path $repo "tmp\npu-anime"
        Span = Join-Path $repo "tmp\npu-anime\span"
        Adc  = Join-Path $repo "tmp\adcsr\onnx"
    }
    function Resolve-Model([string]$Key, [string]$Path) {
        if (-not (Test-Path -LiteralPath $Path)) {
            Fail "配布対象のモデルがありません ($Key): $Path"
        }
        return $Path
    }

    $gpuModels = @(
        @{ Name = "animevideov3_nchw_256x256_fp32.onnx"; Source = (Resolve-Model "animevideov3/256" (Join-Path $modelRoots.Top "animevideov3_nchw_256x256_fp32.onnx")) }
        @{ Name = "animevideov3_nchw_512x512_fp32.onnx"; Source = (Resolve-Model "animevideov3/512" (Join-Path $modelRoots.Top "animevideov3_nchw_512x512_fp32.onnx")) }
        @{ Name = "purephoto_nchw_256x256_fp32.onnx"; Source = (Resolve-Model "4xNomosUni/256" (Join-Path $modelRoots.Span "purephoto_nchw_256x256_fp32.onnx")) }
        @{ Name = "purephoto_nchw_512x512_fp32.onnx"; Source = (Resolve-Model "4xNomosUni/512" (Join-Path $modelRoots.Span "purephoto_nchw_512x512_fp32.onnx")) }
        @{ Name = "realesrgan_nchw_256x256_fp32.onnx"; Source = (Resolve-Model "AMD-RRDB/256" (Join-Path $modelRoots.Top "realesrgan_nchw_256x256_fp32.onnx")) }
        @{ Name = "swinir_nchw_256x256_fp32.onnx"; Source = (Resolve-Model "SwinIR/256" (Join-Path $modelRoots.Span "swinir_nchw_256x256_fp32.onnx")) }
    )
    $adcsrGpu = @(
        @{ Name = "adcsr_nchw_128x128_fp32.onnx"; Source = (Resolve-Model "AdcSR/128" (Join-Path $modelRoots.Adc "adcsr_nchw_128x128_fp32.onnx")) }
    )
    $npuModels = @(
        @{ Name = "animevideov3dp_nchw_512x512_bf16cast.onnx"; Source = (Resolve-Model "animevideov3/NPU512" (Join-Path $modelRoots.Top "animevideov3dp_nchw_512x512_bf16cast.onnx")) }
        @{ Name = "purephoto_nchw_512x512_bf16cast.onnx"; Source = (Resolve-Model "4xNomosUni/NPU512" (Join-Path $modelRoots.Span "purephoto_nchw_512x512_bf16cast.onnx")) }
        @{ Name = "realesrgan_nchw_256x256_bf16cast.onnx"; Source = (Resolve-Model "AMD-RRDB/NPU256" (Join-Path $modelRoots.Top "realesrgan_nchw_256x256_bf16cast.onnx")) }
        @{ Name = "swinir_nchw_256x256_bf16cast.onnx"; Source = (Resolve-Model "SwinIR/NPU256" (Join-Path $modelRoots.Span "swinir_nchw_256x256_bf16cast.onnx")) }
    )
    $adcsrNpu = @(
        @{ Name = "adcsr_front_nchw_128x128_bf16cast.onnx"; Source = (Resolve-Model "AdcSR/NPU前半" (Join-Path $modelRoots.Top "adcsr_front_nchw_128x128_bf16cast.onnx")) }
        @{ Name = "adcsr_back_nchw_128x128_bf16cast.onnx"; Source = (Resolve-Model "AdcSR/NPU後半" (Join-Path $modelRoots.Top "adcsr_back_nchw_128x128_bf16cast.onnx")) }
        @{ Name = "adcsr_npu_manifest.json"; Source = (Resolve-Model "AdcSR/NPUマニフェスト" (Join-Path $modelRoots.Top "adcsr_npu_manifest.json")) }
    )

    # --- NOTICE 文の雛形 ---
    $noticeGpuModels = @'
NOTICE-models-gpu-fp32.txt — GPU（DirectML）用 fp32 ONNX モデル
展開先: models/ai/（UEU_MODELS_DIR 未設定時の既定探索先）

| zip 内ファイル | GUI のモデルキー | 実体・帰属 | ライセンス（同梱ファイル） |
|---|---|---|---|
| animevideov3_nchw_256x256_fp32.onnx | animevideov3（Anime Video v3） | realesr-animevideov3（xinntao/Real-ESRGAN 由来） | BSD-3-Clause（LICENSE-BSD-3-Real-ESRGAN.txt） |
| animevideov3_nchw_512x512_fp32.onnx | animevideov3（Anime Video v3） | 同上 | 同上 |
| purephoto_nchw_256x256_fp32.onnx | 4xNomosUni（4xNomosUni SPAN） | 4xNomosUni_span_multijpg（Philip Hofmann/Phips） | CC-BY-4.0（LICENSE-CC-BY-4.0.txt。表示の保持が必要） |
| purephoto_nchw_512x512_fp32.onnx | 4xNomosUni（4xNomosUni SPAN） | 同上 | 同上 |
| realesrgan_nchw_256x256_fp32.onnx | AMD-RRDB（Real-ESRGAN（AMD縮小版）） | AMD 縮小 RRDB 版 | Research-only RAIL-MS（LICENSE-RAIL-MS-AMD-RRDB.txt。研究用途限定） |
| swinir_nchw_256x256_fp32.onnx | SwinIR（SwinIR-M） | 003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN（Jingyun Liang） | Apache-2.0（LICENSE-Apache-2.0-SwinIR.txt） |

取得元と SHA-256 はリポジトリの docs/span-bench-results.md を参照。
'@

    $noticeAdcsrGpu = @'
NOTICE-models-adcsr-gpu-fp32.txt — AdcSR GPU（DirectML）用 fp32 ONNX モデル
展開先: models/ai/（UEU_MODELS_DIR 未設定時の既定探索先）

| zip 内ファイル | GUI のモデルキー | 実体・帰属 | ライセンス（同梱ファイル） |
|---|---|---|---|
| adcsr_nchw_128x128_fp32.onnx | AdcSR（AdcSR） | AdcSR net_params_200（Guaishou74851、CVPR 2025。SD2.1-base 派生・1 ステップ） | Apache-2.0（LICENSE-Apache-2.0-AdcSR.txt）＋基盤モデルの使用制限（下記） |

基盤モデル Stable Diffusion 2.1-base は CreativeML Open RAIL++-M の適用対象で、
利用前に利用者自身が使用条件を確認すること。
全文（要ログイン）: https://huggingface.co/stabilityai/stable-diffusion-2/blob/main/LICENSE-MODEL
同系統の CreativeML Open RAIL-M 全文（使用制限 Attachment A を含む）を
LICENSE-OpenRAIL-M-CompVis-SD1.txt に同梱する。使用制限の要点:
法令遵守、未成年への危害禁止、虚偽情報・個人情報の悪用禁止、差別・嫌がらせ用途の禁止、
医療助言・法執行などへの利用禁止。再配布時は同じ使用制限を利用者に課すこと。
'@

    $noticeNpuModels = @'
NOTICE-models-npu-bf16.txt — NPU（Ryzen AI VitisAI EP）用 bf16cast ONNX モデル
展開先: models/ai/（UEU_MODELS_DIR 未設定時の既定探索先）
初回起動時に VAIML コンパイルが必要（次回以降はキャッシュを利用）。

| zip 内ファイル | GUI のモデルキー | 実体・帰属 | ライセンス（同梱ファイル） |
|---|---|---|---|
| animevideov3dp_nchw_512x512_bf16cast.onnx | animevideov3（Anime Video v3） | realesr-animevideov3 の PReLU 分解版（xinntao/Real-ESRGAN 由来） | BSD-3-Clause（LICENSE-BSD-3-Real-ESRGAN.txt） |
| purephoto_nchw_512x512_bf16cast.onnx | 4xNomosUni（4xNomosUni SPAN） | 4xNomosUni_span_multijpg（Philip Hofmann/Phips） | CC-BY-4.0（LICENSE-CC-BY-4.0.txt。表示の保持が必要） |
| realesrgan_nchw_256x256_bf16cast.onnx | AMD-RRDB（Real-ESRGAN（AMD縮小版）） | AMD 縮小 RRDB 版 | Research-only RAIL-MS（LICENSE-RAIL-MS-AMD-RRDB.txt。研究用途限定） |
| swinir_nchw_256x256_bf16cast.onnx | SwinIR（SwinIR-M） | 003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN（Jingyun Liang） | Apache-2.0（LICENSE-Apache-2.0-SwinIR.txt） |
'@

    $noticeAdcsrNpu = @'
NOTICE-models-adcsr-npu-bf16.txt — AdcSR NPU（2 段モード）用 bf16cast ONNX モデル
展開先: models/ai/（UEU_MODELS_DIR 未設定時の既定探索先）
前半（UNet）・後半（VAE デコーダ）・adcsr_npu_manifest.json の 3 点がそろった
場合だけ NPU 2 段モードで実行する。初回起動時に前半・後半を順に VAIML
コンパイルする（次回以降はキャッシュを利用）。

帰属: AdcSR net_params_200（Guaishou74851、CVPR 2025。SD2.1-base 派生）。
コード由来の Apache-2.0（LICENSE-Apache-2.0-AdcSR.txt）に加え、基盤モデル
Stable Diffusion 2.1-base は CreativeML Open RAIL++-M の適用対象で、
利用前に利用者自身が使用条件を確認すること。
全文（要ログイン）: https://huggingface.co/stabilityai/stable-diffusion-2/blob/main/LICENSE-MODEL
同系統の CreativeML Open RAIL-M 全文（使用制限 Attachment A を含む）を
LICENSE-OpenRAIL-M-CompVis-SD1.txt に同梱する。再配布時は同じ使用制限を
利用者に課すこと。
'@

    $noticeWinml = @'
NOTICE-winml-sr.txt — ビルド済み DirectML ヘルパー winml-sr
展開先: vendor/winml-sr/（UEU_WINML_HELPER 未設定時の既定探索先）

内容: tools/winml-sr の Release ビルド一式（winml-sr.exe、依存 DLL、
seam_templates/ の AdcSR 格子補正テンプレート）。
このリポジトリのコードは MIT License（Copyright (c) 2026 tomhda）。
利用はリポジトリ直下の LICENSE に従う。

同梱の license.txt と ThirdPartyNotices.txt は Microsoft Windows ML Runtime
（Microsoft.Windows.AI.MachineLearning NuGet）の配布条件で、license.txt §3
（DISTRIBUTABLE CODE）によりアプリと一緒に再配布する。再配布の条件:
配布先に本合意と同等の保護条項への同意を求めること、Microsoft を免責する
こと、再配布コードを GPL 化しないこと、Microsoft の商標を使わないこと。
'@

    function New-Package(
        [string]$ZipName,
        [array]$Files,
        [string]$NoticeName,
        [string]$NoticeBody,
        [array]$LicenseFiles,
        [string]$Compression
    ) {
        $work = Join-Path $stage ([IO.Path]::GetFileNameWithoutExtension($ZipName))
        if (Test-Path -LiteralPath $work) {
            Remove-Item -LiteralPath $work -Recurse -Force
        }
        New-Item -ItemType Directory -Path $work -Force | Out-Null
        $rows = @()
        foreach ($entry in $Files) {
            Copy-Item -LiteralPath $entry.Source -Destination (Join-Path $work $entry.Name) -Force
            $rows += "{0}  {1}  {2}" -f $entry.Name, (Get-Sha256 (Join-Path $work $entry.Name)), `
                ((Get-Item -LiteralPath $entry.Source).Length)
        }
        foreach ($lic in $LicenseFiles) {
            Copy-Item -LiteralPath (Join-Path $licenses $lic) -Destination (Join-Path $work $lic) -Force
        }
        $body = $NoticeBody + "`r`n`r`n--- 内容物の SHA-256 とサイズ（bytes） ---`r`n" + ($rows -join "`r`n") + "`r`n"
        # UTF-8（BOM なし）で NOTICE を書き出す。
        [IO.File]::WriteAllText((Join-Path $work $NoticeName), $body, [Text.UTF8Encoding]::new($false))
        $zipPath = Join-Path $OutDir $ZipName
        if (Test-Path -LiteralPath $zipPath) {
            Remove-Item -LiteralPath $zipPath -Force
        }
        $level = [IO.Compression.CompressionLevel]::Optimal
        if ($Compression -eq "NoCompression") {
            $level = [IO.Compression.CompressionLevel]::NoCompression
        }
        Add-Type -AssemblyName System.IO.Compression.FileSystem
        [IO.Compression.ZipFile]::CreateFromDirectory($work, $zipPath, $level, $false)
        # Write-Output だと関数の戻り値が汚れるため進捗は Write-Host に出す。
        Write-Host ("作成: {0} ({1} bytes)" -f $zipPath, (Get-Item -LiteralPath $zipPath).Length)
        return $zipPath
    }

    $built = @()

    # --- (a) winml-sr-win-x64.zip（実行一式＋Microsoft 配布条件＋NOTICE） ---
    $helperWork = Join-Path $stage "winml-sr-win-x64"
    if (Test-Path -LiteralPath $helperWork) {
        Remove-Item -LiteralPath $helperWork -Recurse -Force
    }
    New-Item -ItemType Directory -Path $helperWork -Force | Out-Null
    Copy-Item -Path (Join-Path $buildOut "*") -Destination $helperWork -Recurse -Force
    Copy-Item -LiteralPath $winmlLicense -Destination (Join-Path $helperWork "license.txt") -Force
    Copy-Item -LiteralPath $winmlTpn -Destination (Join-Path $helperWork "ThirdPartyNotices.txt") -Force
    [IO.File]::WriteAllText((Join-Path $helperWork "NOTICE-winml-sr.txt"), $noticeWinml, [Text.UTF8Encoding]::new($false))
    $winmlZip = Join-Path $OutDir "winml-sr-win-x64.zip"
    if (Test-Path -LiteralPath $winmlZip) {
        Remove-Item -LiteralPath $winmlZip -Force
    }
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    [IO.Compression.ZipFile]::CreateFromDirectory(
        $helperWork, $winmlZip, [IO.Compression.CompressionLevel]::Optimal, $false)
    Write-Output ("作成: {0} ({1} bytes)" -f $winmlZip, (Get-Item -LiteralPath $winmlZip).Length)
    $built += $winmlZip

    $modelLicenses = @(
        "LICENSE-BSD-3-Real-ESRGAN.txt",
        "LICENSE-CC-BY-4.0.txt",
        "LICENSE-RAIL-MS-AMD-RRDB.txt",
        "LICENSE-Apache-2.0-SwinIR.txt"
    )
    $adcsrLicenses = @(
        "LICENSE-Apache-2.0-AdcSR.txt",
        "LICENSE-OpenRAIL-M-CompVis-SD1.txt"
    )

    # --- (b)(c)(d)(e) モデル zip（AdcSR の 1.8GB 級は無圧縮で格納する） ---
    $built += New-Package "models-gpu-fp32.zip" $gpuModels `
        "NOTICE-models-gpu-fp32.txt" $noticeGpuModels $modelLicenses "Optimal"
    $built += New-Package "models-adcsr-gpu-fp32.zip" $adcsrGpu `
        "NOTICE-models-adcsr-gpu-fp32.txt" $noticeAdcsrGpu $adcsrLicenses "NoCompression"
    $built += New-Package "models-npu-bf16.zip" $npuModels `
        "NOTICE-models-npu-bf16.txt" $noticeNpuModels $modelLicenses "Optimal"
    $built += New-Package "models-adcsr-npu-bf16.zip" $adcsrNpu `
        "NOTICE-models-adcsr-npu-bf16.txt" $noticeAdcsrNpu $adcsrLicenses "NoCompression"

    # --- SHA256SUMS.txt（sha256sum 形式: "<hash>  <zip名>"） ---
    $sumPath = Join-Path $OutDir "SHA256SUMS.txt"
    $lines = foreach ($zip in $built) {
        "{0}  {1}" -f (Get-Sha256 $zip), [IO.Path]::GetFileName($zip)
    }
    [IO.File]::WriteAllLines($sumPath, $lines, [Text.UTF8Encoding]::new($false))
    Write-Output "作成: $sumPath"
    Write-Output "完了: $($built.Count) 個の zip と SHA256SUMS.txt を $OutDir に出力しました。"
}
finally {
    if (Test-Path -LiteralPath $stage) {
        Remove-Item -LiteralPath $stage -Recurse -Force
    }
}

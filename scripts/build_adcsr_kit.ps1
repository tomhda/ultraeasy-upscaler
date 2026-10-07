<#Requires -Version 5.1
<#
.SYNOPSIS
    AdcSR（GPU 用）追加キットの zip（別配布）を作成する。

.DESCRIPTION
    ultraeasy-upscaler-adcsr-kit.zip を作る。exe のフォルダに
    上書き展開する前提の構造（models/ai/ の GPU 用 AdcSR 一式、
    AdcSRキットの使い方.txt）にする。

    対象ファイルの選定:
    - ONNX は app/core/settings.py の HELPER_MODEL_FILES[WINML_GPU][AdcSR][128]
     （scripts/build_release.ps1 の models-adcsr-gpu-fp32.zip と同じ）。
    - NOTICE / LICENSE は scripts/build_release.ps1 の AdcSR GPU 用と同じ
      （NOTICE-models-adcsr-gpu-fp32.txt、LICENSE-Apache-2.0-AdcSR.txt、
      LICENSE-OpenRAIL-M-CompVis-SD1.txt）。
    - 本体 zip は AdcSR を除く（scripts/build_portable.ps1 の除外条件を参照）。

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\build_adcsr_kit.ps1
#>

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$modelsSource = Join-Path $repo "models\ai"
$licensesDir = Join-Path $repo "scripts\release-licenses"

function Fail([string]$Message) {
    throw $Message
}

$zipName = "ultraeasy-upscaler-adcsr-kit.zip"
$zipPath = Join-Path $repo $zipName
if (Test-Path -LiteralPath $zipPath) {
    Fail "既にあります（上書きしません）: $zipPath"
}

# GPU 用の AdcSR モデル一式（models/ai/ 直下に置く前提のファイル名）。
$modelFiles = @(
    "adcsr_nchw_128x128_fp32.onnx",
    "NOTICE-models-adcsr-gpu-fp32.txt",
    "LICENSE-Apache-2.0-AdcSR.txt",
    "LICENSE-OpenRAIL-M-CompVis-SD1.txt"
)
foreach ($name in $modelFiles) {
    $source = Join-Path $modelsSource $name
    if (-not (Test-Path -LiteralPath $source)) {
        Fail "配布対象のモデルがありません: $source"
    }
}
$usageJa = Join-Path $PSScriptRoot "AdcSRキットの使い方.txt"
$usageEn = Join-Path $PSScriptRoot "How to use the AdcSR kit.txt"
foreach ($doc in @($usageJa, $usageEn)) {
    if (-not (Test-Path -LiteralPath $doc)) {
        Fail "説明書がありません: $doc"
    }
}

$stage = Join-Path ([IO.Path]::GetTempPath()) "ueu-adcsr-kit"
if (Test-Path -LiteralPath $stage) {
    Remove-Item -LiteralPath $stage -Recurse -Force
}
try {
    $modelsOut = Join-Path $stage "models\ai"
    New-Item -ItemType Directory -Path $modelsOut -Force | Out-Null

    foreach ($name in $modelFiles) {
        Copy-Item -LiteralPath (Join-Path $modelsSource $name) -Destination $modelsOut -Force
    }
    Copy-Item -LiteralPath $usageJa -Destination $stage -Force
    Copy-Item -LiteralPath $usageEn -Destination $stage -Force

    # AdcSR の 1.8GB 級は無圧縮で格納する（build_release.ps1 と同じ）。
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    [IO.Compression.ZipFile]::CreateFromDirectory(
        $stage, $zipPath, [IO.Compression.CompressionLevel]::NoCompression, $false)
}
finally {
    if (Test-Path -LiteralPath $stage) {
        Remove-Item -LiteralPath $stage -Recurse -Force
    }
}

# 中身の一覧とサイズを表示する。
Add-Type -AssemblyName System.IO.Compression.FileSystem
$zip = [IO.Compression.ZipFile]::OpenRead($zipPath)
try {
    foreach ($entry in $zip.Entries) {
        Write-Output ("  {0} ({1} bytes)" -f $entry.FullName, $entry.Length)
    }
}
finally {
    $zip.Dispose()
}
Write-Output ("作成: {0} ({1} bytes)" -f $zipPath, (Get-Item -LiteralPath $zipPath).Length)

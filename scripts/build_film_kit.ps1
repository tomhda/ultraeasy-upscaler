<#Requires -Version 5.1
<#
.SYNOPSIS
    FILM 追加キットの zip（別配布）を作成する。

.DESCRIPTION
    togu-scaler-film-kit.zip を作る。exe のフォルダに
    上書き展開する前提の構造（vendor/winml-film/ 一式、models/film/ 一式、
    FILMキットの使い方.txt）にする。
    LICENSE-Apache-2.0-FILM.txt は本体 zip には入れず、このキットに入れる
    （配布物の作りは scripts/build_portable.ps1 の注記を参照）。

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\build_film_kit.ps1
#>

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$helperSource = Join-Path $repo "vendor\winml-film"
$modelsSource = Join-Path $repo "models\film"

function Fail([string]$Message) {
    throw $Message
}

$zipName = "togu-scaler-film-kit.zip"
$zipPath = Join-Path $repo $zipName
if (Test-Path -LiteralPath $zipPath) {
    Fail "既にあります（上書きしません）: $zipPath"
}

$exe = Join-Path $helperSource "winml-film.exe"
if (-not (Test-Path -LiteralPath $exe)) {
    Fail "FILM ヘルパーがありません: $exe"
}
$model = Join-Path $modelsSource "film_style_fp32.onnx"
if (-not (Test-Path -LiteralPath $model)) {
    Fail "FILM モデルがありません: $model"
}
$usageJa = Join-Path $PSScriptRoot "FILMキットの使い方.txt"
$usageEn = Join-Path $PSScriptRoot "How to use the FILM kit.txt"
foreach ($doc in @($usageJa, $usageEn)) {
    if (-not (Test-Path -LiteralPath $doc)) {
        Fail "説明書がありません: $doc"
    }
}

$stage = Join-Path ([IO.Path]::GetTempPath()) "togu-film-kit"
if (Test-Path -LiteralPath $stage) {
    Remove-Item -LiteralPath $stage -Recurse -Force
}
try {
    $helperOut = Join-Path $stage "vendor\winml-film"
    $modelsOut = Join-Path $stage "models\film"
    New-Item -ItemType Directory -Path $helperOut, $modelsOut -Force | Out-Null

    Copy-Item -Path (Join-Path $helperSource "*") -Destination $helperOut -Recurse -Force
    Copy-Item -Path (Join-Path $modelsSource "*") -Destination $modelsOut -Recurse -Force
    Copy-Item -LiteralPath $usageJa -Destination $stage -Force
    Copy-Item -LiteralPath $usageEn -Destination $stage -Force

    Add-Type -AssemblyName System.IO.Compression.FileSystem
    [IO.Compression.ZipFile]::CreateFromDirectory(
        $stage, $zipPath, [IO.Compression.CompressionLevel]::Optimal, $false)
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

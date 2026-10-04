<#Requires -Version 5.1
<#
.SYNOPSIS
    NPU 追加キットの zip（別配布）を作成する。

.DESCRIPTION
    ultraeasy-upscaler-npu-kit.zip（AdcSR 以外）を作る。exe のフォルダに
    上書き展開する前提の構造（tools/npu-serve/*.py、models/ai/ の NPU 用
    モデル、NPUキットの使い方.txt）にする。
    -WithAdcSR を付けたときだけ ultraeasy-upscaler-npu-kit-adcsr.zip
    （無圧縮）も作る。

    モデルの選定は app/core/settings.py の表（HELPER_MODEL_FILES の
    NPU_NATIVE、HELPER_MODEL_NPU_TAIL、AdcSR の前半・後半・マニフェスト）
    に従う。実ファイルは models/ai/ から集めるため、このスクリプトの
    実行はモデルが models/ai/ に揃っている PC でのみ可能。

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\build_npu_kit.ps1
    powershell -ExecutionPolicy Bypass -File scripts\build_npu_kit.ps1 -WithAdcSR
#>
param(
    [switch]$WithAdcSR
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $repo ".venv\Scripts\python.exe"
$modelsSource = Join-Path $repo "models\ai"
$serveSource = Join-Path $repo "tools\npu-serve"
$licensesDir = Join-Path $repo "scripts\release-licenses"

function Fail([string]$Message) {
    throw $Message
}

function Get-ModelLists {
    if (-not (Test-Path -LiteralPath $python)) {
        Fail "Python 環境がありません: $python"
    }
    Push-Location -LiteralPath $repo
    try {
        $code = (
            "from app.core.settings import " +
            "HELPER_MODEL_FILES, HELPER_MODEL_NPU_TAIL, " +
            "HELPER_MODEL_NPU_BACK, ADCSR_NPU_MANIFEST, " +
            "HELPER_MODEL_ADCSR, UpscaleBackend; " +
            "native = HELPER_MODEL_FILES[UpscaleBackend.NPU_NATIVE]; " +
            "base = [f for k, t in native.items() " +
            "if k != HELPER_MODEL_ADCSR for f in t.values()]; " +
            "base += [n for t in HELPER_MODEL_NPU_TAIL.values() " +
            "for pair in t.values() for n in pair]; " +
            "print('BASE'); " +
            "print(chr(10).join(base)); " +
            "print('ADCSR'); " +
            "print(native[HELPER_MODEL_ADCSR][128]); " +
            "print(HELPER_MODEL_NPU_BACK[HELPER_MODEL_ADCSR]); " +
            "print(ADCSR_NPU_MANIFEST)"
        )
        $lines = & $python -c $code
    }
    finally {
        Pop-Location
    }
    if ($LASTEXITCODE -ne 0) {
        Fail "モデル一覧の取得に失敗しました（終了コード $LASTEXITCODE）。"
    }
    $mode = ""
    $base = @()
    $adcsr = @()
    foreach ($line in $lines) {
        if ($line -eq "BASE") { $mode = "base"; continue }
        if ($line -eq "ADCSR") { $mode = "adcsr"; continue }
        if ($line -eq "") { continue }
        if ($mode -eq "base") { $base += $line }
        elseif ($mode -eq "adcsr") { $adcsr += $line }
    }
    if ($base.Count -eq 0 -or $adcsr.Count -ne 3) {
        Fail "モデル一覧の取得に失敗しました。"
    }
    return @{ Base = $base; Adcsr = $adcsr }
}

$usageText = @"
ultraeasy-upscaler NPU キット

AMD Ryzen AI 搭載の PC で、NPU を使って処理するための追加ファイルです。

必要なもの
- ultraeasy-upscaler ポータブル版
- AMD の Ryzen AI Software 1.8.0（AMD のサイトから入手して導入）
- NPU ドライバ 32.0.203.329 以降

入れ方
1. ultraeasy-upscaler を終了します。
2. この zip の中身を、ultraeasy-upscaler.exe があるフォルダに上書きで展開します。
3. ultraeasy-upscaler を起動し、右上の歯車から詳細設定を開きます。
4. 「NPU の準備」で、使うモデルの「変換する」を押します。
5. 変換が終わったら、「AI実行先」で NPU を選びます。

変換について
- 変換はモデルごとに最初の一度だけ必要です。画面に目安の時間が出ます。
- 変換中も PC は使えます。CPU は 1 コアぶんしか使いません。
- SwinIR-M は変換中にメモリを最大で約 25GB 使います（AdcSR は未計測）。
  メモリが 32GB 未満の PC では、ほかのアプリを閉じてから実行してください。
- NPU ドライバや Ryzen AI Software を更新すると、変換のやり直しが必要になることがあります。

Ryzen AI Software を標準と違う場所に入れた場合は、環境変数 UEU_NPU_PYTHON に
その python.exe の場所を指定してください。
"@

function New-NpuKit(
    [string]$ZipName,
    [array]$ModelFiles,
    [string]$NoticeName,
    [array]$LicenseFiles,
    [string]$Compression
) {
    $zipPath = Join-Path $repo $ZipName
    if (Test-Path -LiteralPath $zipPath) {
        Fail "既にあります（上書きしません）: $zipPath"
    }
    $stage = Join-Path ([IO.Path]::GetTempPath()) "ueu-npu-kit"
    if (Test-Path -LiteralPath $stage) {
        Remove-Item -LiteralPath $stage -Recurse -Force
    }
    try {
        $serveOut = Join-Path $stage "tools\npu-serve"
        $modelsOut = Join-Path $stage "models\ai"
        New-Item -ItemType Directory -Path $serveOut, $modelsOut -Force | Out-Null

        # NPU 常駐ヘルパー（直下の .py だけ。selftest/ と __pycache__ は入れない）。
        $serveFiles = Get-ChildItem -LiteralPath $serveSource -Filter "*.py" -File
        if ($serveFiles.Count -eq 0) {
            Fail "NPU ヘルパーがありません: $serveSource"
        }
        foreach ($file in $serveFiles) {
            Copy-Item -LiteralPath $file.FullName -Destination $serveOut -Force
        }

        foreach ($name in $ModelFiles) {
            $source = Join-Path $modelsSource $name
            if (-not (Test-Path -LiteralPath $source)) {
                Fail "配布対象のモデルがありません: $source"
            }
            Copy-Item -LiteralPath $source -Destination $modelsOut -Force
        }

        $noticeSource = Join-Path $modelsSource $NoticeName
        if (-not (Test-Path -LiteralPath $noticeSource)) {
            Fail "NOTICE がありません: $noticeSource"
        }
        Copy-Item -LiteralPath $noticeSource -Destination $modelsOut -Force

        foreach ($lic in $LicenseFiles) {
            $source = Join-Path $licensesDir $lic
            if (-not (Test-Path -LiteralPath $source)) {
                Fail "ライセンス原文がありません: $source"
            }
            Copy-Item -LiteralPath $source -Destination $modelsOut -Force
        }

        if ($ZipName -eq "ultraeasy-upscaler-npu-kit.zip") {
            [IO.File]::WriteAllText(
                (Join-Path $stage "NPUキットの使い方.txt"),
                $usageText, [Text.UTF8Encoding]::new($true))
        }

        $level = [IO.Compression.CompressionLevel]::Optimal
        if ($Compression -eq "NoCompression") {
            $level = [IO.Compression.CompressionLevel]::NoCompression
        }
        Add-Type -AssemblyName System.IO.Compression.FileSystem
        [IO.Compression.ZipFile]::CreateFromDirectory($stage, $zipPath, $level, $false)
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
}

$lists = Get-ModelLists
$baseLicenses = @(
    "LICENSE-BSD-3-Real-ESRGAN.txt",
    "LICENSE-CC-BY-4.0.txt",
    "LICENSE-RAIL-MS-AMD-RRDB.txt",
    "LICENSE-Apache-2.0-SwinIR.txt"
)
$adcsrLicenses = @(
    "LICENSE-Apache-2.0-AdcSR.txt",
    "LICENSE-OpenRAIL-M-CompVis-SD1.txt"
)

New-NpuKit "ultraeasy-upscaler-npu-kit.zip" $lists.Base `
    "NOTICE-models-npu-bf16.txt" $baseLicenses "Optimal"
if ($WithAdcSR) {
    New-NpuKit "ultraeasy-upscaler-npu-kit-adcsr.zip" $lists.Adcsr `
        "NOTICE-models-adcsr-npu-bf16.txt" $adcsrLicenses "NoCompression"
}

#Requires -Version 5.1
<#
.SYNOPSIS
    GitHub Release からビルド済みヘルパーと変換済みモデルを取得し、導入する。

.DESCRIPTION
    (i) 前提（.venv の Python、ffmpeg、vendor/realesrgan）を確認する。
        realesrgan が無ければ scripts/get_models.py を実行する。
    (ii) GitHub Release から必要な zip と SHA256SUMS.txt を取得し、検証する。
    (iii) vendor/winml-sr/ と models/ai/ へ展開する。
    (iv) 動作確認として assets/sample/superman_src.png（320x240）を
        winml-sr で処理し、出力が 1280x960 であることを確認する。
    (v) NPU（-WithNpu 指定時）は Ryzen AI Software の導入案内を表示するだけ。

    -LocalDist は検証用: Release から取得せず、指定ディレクトリの zip を使う。

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File setup.ps1
    powershell -ExecutionPolicy Bypass -File setup.ps1 -WithAdcSR -WithNpu
    powershell -ExecutionPolicy Bypass -File setup.ps1 -LocalDist tmp\easy-setup\dist -WithAdcSR
#>
param(
    [string]$Release = "latest",
    [switch]$WithAdcSR,
    [switch]$WithNpu,
    [string]$LocalDist = ""
)

$ErrorActionPreference = "Stop"
$repo = $PSScriptRoot
$ownerRepo = "tomhda/ultraeasy-upscaler"

function Fail([string]$Message) {
    Write-Error $Message
    exit 1
}

# --- (i) 前提確認 ---
$python = Join-Path $repo ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    Fail "Python 環境が見つかりません: $python。先に .venv を用意してください。"
}
foreach ($tool in @("ffmpeg", "ffprobe")) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
        Fail "$tool が見つかりません。ffmpeg を導入して PATH を通してください。"
    }
}
$realesrganExe = Join-Path $repo "vendor\realesrgan\realesrgan-ncnn-vulkan.exe"
if (-not (Test-Path -LiteralPath $realesrganExe)) {
    Write-Output "vendor/realesrgan が無いため scripts/get_models.py を実行します…"
    & $python (Join-Path $repo "scripts\get_models.py")
    if (($LASTEXITCODE -ne 0) -or (-not (Test-Path -LiteralPath $realesrganExe))) {
        Fail "Vulkan 資材の取得に失敗しました。ネットワークを確認して再実行してください。"
    }
}

# --- (ii) 必要な zip の決定 ---
$wanted = @("winml-sr-win-x64.zip", "models-gpu-fp32.zip")
if ($WithAdcSR) {
    $wanted += "models-adcsr-gpu-fp32.zip"
}
if ($WithNpu) {
    $wanted += "models-npu-bf16.zip"
    if ($WithAdcSR) {
        $wanted += "models-adcsr-npu-bf16.zip"
    }
}

$fetchDir = ""
if ($LocalDist) {
    $fetchDir = (Resolve-Path $LocalDist -ErrorAction SilentlyContinue)
    if (-not $fetchDir) {
        Fail "ローカル配布物ディレクトリが見つかりません: $LocalDist"
    }
    $fetchDir = $fetchDir.Path
    Write-Output "ローカル配布物を使います: $fetchDir"
}
else {
    if ($Release -eq "latest") {
        $baseUrl = "https://github.com/$ownerRepo/releases/latest/download"
    }
    else {
        $baseUrl = "https://github.com/$ownerRepo/releases/download/$Release"
    }
    $fetchDir = Join-Path $repo "tmp\easy-setup\downloads"
    if (-not (Test-Path -LiteralPath $fetchDir)) {
        New-Item -ItemType Directory -Path $fetchDir -Force | Out-Null
    }
    try {
        Write-Output "SHA256SUMS.txt を取得します…"
        Invoke-WebRequest -Uri "$baseUrl/SHA256SUMS.txt" -OutFile (Join-Path $fetchDir "SHA256SUMS.txt")
        foreach ($name in $wanted) {
            $dest = Join-Path $fetchDir $name
            if (-not (Test-Path -LiteralPath $dest)) {
                Write-Output "$name を取得します…"
                Invoke-WebRequest -Uri "$baseUrl/$name" -OutFile $dest
            }
            else {
                Write-Output "$name は取得済みのため再利用します。"
            }
        }
    }
    catch {
        Fail "ダウンロードに失敗しました: $($_.Exception.Message)。ネットワークと Release の有無を確認してください。"
    }
}

# --- SHA-256 検証 ---
$sumFile = Join-Path $fetchDir "SHA256SUMS.txt"
if (-not (Test-Path -LiteralPath $sumFile)) {
    Fail "SHA256SUMS.txt がありません: $sumFile"
}
$sums = @{}
foreach ($line in [IO.File]::ReadAllLines($sumFile)) {
    $m = [regex]::Match($line, "^\s*([0-9a-fA-F]{64})\s+(.+?)\s*$")
    if ($m.Success) {
        $sums[$m.Groups[2].Value.Trim().TrimStart("*")] = $m.Groups[1].Value.ToLower()
    }
}
foreach ($name in $wanted) {
    $path = Join-Path $fetchDir $name
    if (-not (Test-Path -LiteralPath $path)) {
        Fail "必要な配布物がありません: $path"
    }
    if (-not $sums.ContainsKey($name)) {
        Fail "SHA256SUMS.txt に $name の記録がありません。"
    }
    $actual = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLower()
    if ($actual -ne $sums[$name]) {
        Fail "$name の SHA-256 が一致しません。配布物が壊れている可能性があります。"
    }
    Write-Output "$name の SHA-256 を確認しました。"
}

# --- (iii) 展開 ---
$helperDir = Join-Path $repo "vendor\winml-sr"
$modelsDir = Join-Path $repo "models\ai"
foreach ($dir in @($helperDir, $modelsDir)) {
    if (-not (Test-Path -LiteralPath $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
}
foreach ($name in $wanted) {
    $dest = if ($name -eq "winml-sr-win-x64.zip") { $helperDir } else { $modelsDir }
    Write-Output "$name を $dest へ展開します…"
    Expand-Archive -LiteralPath (Join-Path $fetchDir $name) -DestinationPath $dest -Force
}
$helperExe = Join-Path $helperDir "winml-sr.exe"
if (-not (Test-Path -LiteralPath $helperExe)) {
    Fail "展開後に winml-sr.exe がありません: $helperExe"
}

# --- (iv) 動作確認（320x240 → 1280x960） ---
$sample = Join-Path $repo "assets\sample\superman_src.png"
if (-not (Test-Path -LiteralPath $sample)) {
    Fail "動作確認用の画像がありません: $sample"
}
$smokeDir = Join-Path $repo "tmp\easy-setup\smoke"
if (-not (Test-Path -LiteralPath $smokeDir)) {
    New-Item -ItemType Directory -Path $smokeDir -Force | Out-Null
}
$smokeOut = Join-Path $smokeDir "superman_out.png"
$smokeModel = Join-Path $modelsDir "animevideov3_nchw_256x256_fp32.onnx"
if (-not (Test-Path -LiteralPath $smokeModel)) {
    Fail "動作確認用のモデルがありません: $smokeModel"
}
Write-Output "動作確認を実行します（DirectML、Anime Video v3、256 タイル）…"
& $helperExe run --model $smokeModel --input $sample --output $smokeOut --ep-name DmlExecutionProvider
if ($LASTEXITCODE -ne 0) {
    Fail "動作確認に失敗しました（終了コード $LASTEXITCODE）。GPU と DirectX 12 対応を確認してください。"
}
$size = & $python -c "from PIL import Image; im = Image.open(r'$smokeOut'); print(f'{im.size[0]}x{im.size[1]}')"
if ($LASTEXITCODE -ne 0) {
    Fail "出力画像の寸法を確認できませんでした: $smokeOut"
}
$size = $size.Trim()
if ($size -ne "1280x960") {
    Fail "出力寸法が 1280x960 ではありません（実際: $size）。"
}
Write-Output "動作確認に成功しました: $smokeOut（$size）"

# --- (v) NPU は案内のみ ---
if ($WithNpu) {
    Write-Output ""
    Write-Output "NPU を使うには Ryzen AI Software 1.8.0 の導入が必要です（配布対象外）。"
    Write-Output "手順: https://ryzenai.docs.amd.com/en/1.8/inst.html"
    Write-Output "導入後、NPU 常駐サーバーを起動する Python を UEU_NPU_PYTHON に設定します。"
    Write-Output ("例: setx UEU_NPU_PYTHON `"%USERPROFILE%\miniforge3\envs\ryzen-ai-1.8.0\python.exe`"")
    Write-Output "初回起動時は VAIML コンパイルが数分〜1時間かかります（次回以降はキャッシュを利用）。"
}

Write-Output ""
Write-Output "導入が完了しました。"
Write-Output "ヘルパー: $helperExe"
Write-Output "モデル: $modelsDir"
Write-Output "dotnet SDK は不要です。run.bat で起動できます。"

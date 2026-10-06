param(
    [string]$Destination = "",
    [switch]$WithHelper
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $repo ".venv\Scripts\python.exe"
$work = Join-Path $repo "portable_build"
$dist = Join-Path $repo "portable_dist"
$app = Join-Path $dist "ultraeasy-upscaler"

function Assert-UnderRepo([string]$Path) {
    $full = [System.IO.Path]::GetFullPath($Path)
    if (-not $full.StartsWith($repo + [System.IO.Path]::DirectorySeparatorChar)) {
        throw "Unsafe build path: $full"
    }
}

foreach ($path in @($work, $dist)) {
    Assert-UnderRepo $path
    if (Test-Path -LiteralPath $path) {
        Remove-Item -LiteralPath $path -Recurse -Force
    }
}

if (-not (Test-Path -LiteralPath $python)) {
    throw "Python environment not found: $python"
}

& $python -m PyInstaller `
    --name ultraeasy-upscaler `
    --windowed `
    --onedir `
    --clean `
    --noconfirm `
    --distpath $dist `
    --workpath $work `
    --specpath $work `
    (Join-Path $repo "app\main.py")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$vendorOut = Join-Path $app "vendor"
$realesrganOut = Join-Path $vendorOut "realesrgan"
$rifeOut = Join-Path $vendorOut "rife"
$ffmpegOut = Join-Path $vendorOut "ffmpeg\bin"
New-Item -ItemType Directory -Path $realesrganOut, $rifeOut, $ffmpegOut -Force | Out-Null

$realesrganSource = Join-Path $repo "vendor\realesrgan"
if (-not (Test-Path -LiteralPath (Join-Path $realesrganSource "realesrgan-ncnn-vulkan.exe"))) {
    throw "Real-ESRGAN assets are missing"
}
# 上流の配布物に入っているサンプル（画像・アニメの動画）は再配布しない。
$realesrganSamples = @("input.jpg", "input2.jpg", "onepiece_demo.mp4")
Get-ChildItem -LiteralPath $realesrganSource | Where-Object { $realesrganSamples -notcontains $_.Name } |
    ForEach-Object { Copy-Item -LiteralPath $_.FullName -Destination $realesrganOut -Recurse -Force }

$rifeBase = Get-ChildItem -LiteralPath (Join-Path $repo "vendor\rife") -Recurse `
    -Filter "rife-ncnn-vulkan.exe" | Select-Object -First 1 -ExpandProperty DirectoryName
if (-not $rifeBase -or -not (Test-Path -LiteralPath (Join-Path $rifeBase "rife-v4.6"))) {
    throw "RIFE v4.6 assets are missing"
}
foreach ($name in @("rife-ncnn-vulkan.exe", "vcomp140.dll", "LICENSE", "README.md", "rife-v4.6")) {
    $source = Join-Path $rifeBase $name
    if (Test-Path -LiteralPath $source) {
        Copy-Item -LiteralPath $source -Destination $rifeOut -Recurse -Force
    }
}

$ffmpeg = (Get-Command ffmpeg -ErrorAction Stop).Source
$ffprobe = (Get-Command ffprobe -ErrorAction Stop).Source
Copy-Item -LiteralPath $ffmpeg -Destination (Join-Path $ffmpegOut "ffmpeg.exe") -Force
Copy-Item -LiteralPath $ffprobe -Destination (Join-Path $ffmpegOut "ffprobe.exe") -Force
# ffmpeg は GPL のビルドを同梱することがあるため、ライセンスとソースの入手元を必ず添える。
$ffmpegRoot = Split-Path -Parent (Split-Path -Parent $ffmpeg)
foreach ($name in @("LICENSE", "README.txt")) {
    $doc = Join-Path $ffmpegRoot $name
    if (-not (Test-Path -LiteralPath $doc)) {
        throw "ffmpeg の $name が見つかりません（ライセンス文書を同梱できません）: $doc"
    }
    Copy-Item -LiteralPath $doc -Destination (Join-Path $vendorOut "ffmpeg\$name") -Force
}

if ($WithHelper) {
    $helperSource = Join-Path $repo "vendor\winml-sr"
    if (-not (Test-Path -LiteralPath (Join-Path $helperSource "winml-sr.exe"))) {
        throw "winml-sr helper is missing (run setup.ps1 first)"
    }
    $helperOut = Join-Path $vendorOut "winml-sr"
    New-Item -ItemType Directory -Path $helperOut -Force | Out-Null
    Copy-Item -Path (Join-Path $helperSource "*") -Destination $helperOut -Recurse -Force

    # GPU 用だけを同梱する。NPU 専用（bf16cast / tail-cut body / AdcSR）は
    # 別配布の NPU キットに入れるため除く。
    $modelsSource = Join-Path $repo "models\ai"
    $modelsOut = Join-Path $app "models\ai"
    New-Item -ItemType Directory -Path $modelsOut -Force | Out-Null
    $onnxFiles = Get-ChildItem -LiteralPath $modelsSource -Filter "*.onnx" -File `
        -ErrorAction SilentlyContinue | Where-Object {
            ($_.Name -notlike "*bf16cast*") -and
            ($_.Name -notlike "*_body_*") -and
            ($_.Name -notlike "adcsr*")
        }
    if (-not $onnxFiles) {
        throw "Helper models are missing in models/ai (run setup.ps1 first)"
    }
    foreach ($file in $onnxFiles) {
        Copy-Item -LiteralPath $file.FullName -Destination $modelsOut -Force
    }
    # モデルの帰属とライセンス（CC-BY の表示、研究用途限定の条件などを含む）
    $modelDocs = @(
        "NOTICE-models-gpu-fp32.txt", "LICENSE-BSD-3-Real-ESRGAN.txt", "LICENSE-CC-BY-4.0.txt",
        "LICENSE-RAIL-MS-AMD-RRDB.txt", "LICENSE-Apache-2.0-SwinIR.txt"
    )
    foreach ($name in $modelDocs) {
        $doc = Join-Path $modelsSource $name
        if (-not (Test-Path -LiteralPath $doc)) { throw "モデルのライセンス文書がありません: $doc" }
        Copy-Item -LiteralPath $doc -Destination $modelsOut -Force
    }
}

# 本体と同梱物のライセンス
$notices = Join-Path $repo "scripts\portable-notices"
Copy-Item -LiteralPath (Join-Path $repo "LICENSE") -Destination (Join-Path $app "LICENSE.txt") -Force
Copy-Item -LiteralPath (Join-Path $notices "THIRD-PARTY-NOTICES.txt") -Destination $app -Force
Copy-Item -LiteralPath (Join-Path $notices "THIRD-PARTY-NOTICES.en.txt") -Destination $app -Force
$qtLicenses = Join-Path $app "licenses\PySide6"
New-Item -ItemType Directory -Path $qtLicenses -Force | Out-Null
Copy-Item -LiteralPath (Join-Path $notices "LGPL-3.0.txt") -Destination $qtLicenses -Force
Copy-Item -LiteralPath (Join-Path $ffmpegRoot "LICENSE") -Destination (Join-Path $qtLicenses "GPL-3.0.txt") -Force

Copy-Item -LiteralPath (Join-Path $repo "README.md") -Destination (Join-Path $app "README.md") -Force
Copy-Item -LiteralPath (Join-Path $repo "README.en.md") -Destination (Join-Path $app "README.en.md") -Force
@"
ultraeasy-upscaler ポータブル版

1. ultraeasy-upscaler.exe をダブルクリックします。
2. 画像や動画を、ウィンドウにドラッグ＆ドロップします。
3. 右側でモデルを選びます。「試す」を押すと、1 コマだけ先に結果を確認できます。
4. 「開始」を押します。

フォルダの中のファイルは、移動や削除をしないでください。

NPU について
NPU は標準では使いません。AMD Ryzen AI 搭載の PC で NPU を使うには、次の 2 つが必要です。
- AMD の Ryzen AI Software 1.8.0（AMD のサイトから入手して導入）
- 別配布の NPU キット（このフォルダに上書きで展開）
"@ | Set-Content -LiteralPath (Join-Path $app "はじめに.txt") -Encoding UTF8
@"
ultraeasy-upscaler portable build

1. Double-click ultraeasy-upscaler.exe.
2. Drag and drop images or videos onto the window.
3. Choose a model on the right. Select Try to check the result on one frame first.
4. Select Start.

Do not move or delete the files inside this folder.
The interface follows the Windows display language. You can change it under More settings > Display language.

About the NPU
The NPU is not used by default. To use the NPU on an AMD Ryzen AI PC, you need both of the following.
- AMD Ryzen AI Software 1.8.0 (get it from AMD and install it)
- The NPU kit, a separate download (extract it into this folder, overwriting files)
"@ | Set-Content -LiteralPath (Join-Path $app "Getting started.txt") -Encoding UTF8

& (Join-Path $app "ultraeasy-upscaler.exe") --portable-self-test
if ($LASTEXITCODE -ne 0) {
    throw "Portable self-test failed: $LASTEXITCODE"
}

if (-not $Destination) {
    $Destination = Join-Path $repo "ultraeasy-upscaler-portable-win64.zip"
}
$destinationFull = [System.IO.Path]::GetFullPath($Destination)
if (Test-Path -LiteralPath $destinationFull) {
    throw "Destination already exists: $destinationFull"
}
Compress-Archive -LiteralPath $app -DestinationPath $destinationFull -CompressionLevel Optimal
Write-Output "PORTABLE_FOLDER=$app"
Write-Output "PORTABLE_ZIP=$destinationFull"

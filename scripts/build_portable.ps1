param(
    [string]$Destination = ""
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
Copy-Item -Path (Join-Path $realesrganSource "*") -Destination $realesrganOut -Recurse -Force

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

Copy-Item -LiteralPath (Join-Path $repo "README.md") -Destination (Join-Path $app "README.md") -Force
@"
ultraeasy-upscaler ポータブル版

1. ultraeasy-upscaler.exe をダブルクリックします。
2. 動画または画像をドロップします。
3. アップスケーラーモデルとフレーム補間モデルを選びます。
4. 「開始」を押します。

フォルダ内のファイルは移動・削除しないでください。
AMD Radeon / NVIDIA GeForce RTX は GPU (Vulkan) を選択します。
動画はWindows標準プレーヤー対応のH.264で出力します。
"@ | Set-Content -LiteralPath (Join-Path $app "はじめに.txt") -Encoding UTF8

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

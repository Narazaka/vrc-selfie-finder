<#
.SYNOPSIS
    vsf-gui.exe ランチャービルドスクリプト
.DESCRIPTION
    黒窓を出さずに `uv run vsf-gui` を起動する軽量な exe を生成する。
    .NET の csc.exe を使うため追加インストール不要。
.EXAMPLE
    .\build-launcher.ps1
#>

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$csFile = Join-Path $scriptDir "launcher.cs"
$exeFile = Join-Path $scriptDir "vsf-gui.exe"

if (-not (Test-Path $csFile)) {
    Write-Error "$csFile が見つかりません。"
    exit 1
}

# .NET Framework の csc.exe を探す
$csc = Join-Path ([System.Runtime.InteropServices.RuntimeEnvironment]::GetRuntimeDirectory()) "csc.exe"
if (-not (Test-Path $csc)) {
    # フォールバック: 最新の .NET Framework csc を探す
    $csc = Get-ChildItem "C:\Windows\Microsoft.NET\Framework64\v*\csc.exe" -ErrorAction SilentlyContinue |
           Sort-Object { $_.Directory.Name } -Descending |
           Select-Object -First 1 -ExpandProperty FullName
}

if (-not $csc -or -not (Test-Path $csc)) {
    Write-Error "csc.exe が見つかりません。.NET Framework がインストールされていません。"
    exit 1
}

Write-Host "Using compiler: $csc"
Write-Host "Building vsf-gui.exe ..."

# コンパイル: /target:winexe でコンソール非表示の Windows アプリとして生成
& $csc /target:winexe /out:$exeFile /reference:System.Windows.Forms.dll $csFile

if ($LASTEXITCODE -eq 0) {
    $size = (Get-Item $exeFile).Length / 1KB
    Write-Host "`nビルド成功: $exeFile ($([math]::Round($size, 0)) KB)" -ForegroundColor Green
} else {
    Write-Error "ビルド失敗"
    exit 1
}

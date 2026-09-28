# Run as administrator. Never restarts Windows automatically.
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$outputDirectory = Join-Path $projectRoot 'outputs'
New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null
$resultPath = Join-Path $outputDirectory 'wsl-install-exit.txt'
$logPath = Join-Path $outputDirectory 'wsl-install.log'
Start-Transcript -LiteralPath $logPath -Force
try {
    & wsl.exe --install -d Ubuntu-24.04 --no-launch
    $installExit = $LASTEXITCODE
    Set-Content -LiteralPath $resultPath -Value $installExit -Encoding ascii
    if ($installExit -ne 0) { throw "WSL installer exited with $installExit; see $logPath" }
} finally {
    Stop-Transcript
}

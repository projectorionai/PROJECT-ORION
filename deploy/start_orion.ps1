param(
    [ValidateSet('quick-tunnel', 'serve', 'funnel')][string]$Route = 'quick-tunnel',
    [switch]$Headless,
    [switch]$Background,
    [switch]$CheckOnly
)
$ErrorActionPreference = 'Stop'
$repository = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot 'resolve_orion_python.ps1')
$pythonExecutable = Get-OrionPython -Repository $repository
Write-Host "Checked ORION runtime: $pythonExecutable"
if ($CheckOnly) { return }
$env:PYTHONUTF8 = '1'
$env:ORION_REMOTE_ACCESS = '1'
$env:ORION_REMOTE_HOST = '127.0.0.1'
$env:ORION_REMOTE_PORT = '8000'
$env:ORION_REMOTE_TLS = '0'
$env:ORION_REMOTE_EXPOSURE = $Route
$env:ORION_REMOTE_PAIR_ON_START = '1'
$arguments = @(('"' + (Join-Path $repository 'orion.py') + '"'))
if ($Headless) { $arguments += '--headless' }
if ($Background) {
    $logDirectory = Join-Path $repository '.runtime-cache'
    New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
    $process = Start-Process -FilePath $pythonExecutable -ArgumentList $arguments -WorkingDirectory $repository `
        -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $logDirectory 'orion-stdout.log') `
        -RedirectStandardError (Join-Path $logDirectory 'orion-stderr.log')
    Write-Host "ORION launcher PID: $($process.Id)"
} else {
    Set-Location -LiteralPath $repository
    if ($Headless) { & $pythonExecutable orion.py --headless }
    else { & $pythonExecutable orion.py }
}

param(
    [int]$Port = 8001,
    [string]$CloudflaredPath = (Join-Path $env:TEMP 'orion-cloudflared-2026.9.3.exe')
)

$ErrorActionPreference = 'Stop'
$repository = Split-Path -Parent $PSScriptRoot
$apk = Join-Path $repository 'releases\android\orion-companion.apk'
if (-not (Test-Path -LiteralPath $apk -PathType Leaf)) {
    throw 'Build and sign the Android installer before starting its download tunnel.'
}
if (-not (Test-Path -LiteralPath $CloudflaredPath -PathType Leaf)) {
    throw 'Run deploy/download_cloudflared_quick.ps1 first.'
}
$expected = 'f096265ec2fcbe9bb6e2d64268db167ced3fcbb83d894bdb9e2fcdb26f2ea7e2'
if ((Get-FileHash -LiteralPath $CloudflaredPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expected) {
    throw 'The Cloudflare tunnel client failed its pinned checksum check.'
}

. (Join-Path $PSScriptRoot 'resolve_orion_python.ps1')
$pythonExecutable = Get-OrionPython -Repository $repository
$logDirectory = Join-Path $repository '.runtime-cache'
New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
$server = Start-Process -FilePath $pythonExecutable `
    -ArgumentList @('deploy\apk_download_server.py', '--port', [string]$Port) `
    -WorkingDirectory $repository -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $logDirectory 'apk-server-stdout.log') `
    -RedirectStandardError (Join-Path $logDirectory 'apk-server-stderr.log')

try {
    $ready = $false
    for ($attempt = 0; $attempt -lt 10; $attempt++) {
        Start-Sleep -Milliseconds 500
        if ($server.HasExited) { throw 'The installer-only server stopped unexpectedly.' }
        try {
            $download = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/download/orion-companion.apk" `
                -Method Head -TimeoutSec 2 -UseBasicParsing
            $control = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/api/health" `
                -TimeoutSec 2 -UseBasicParsing -SkipHttpErrorCheck
            $ready = $download.StatusCode -eq 200 -and $control.StatusCode -eq 404
            if ($ready) { break }
        } catch { }
    }
    if (-not $ready) { throw 'The installer-only server did not pass its route checks.' }

    Write-Host 'Cloudflare will print a temporary HTTPS address. Add /download/orion-companion.apk to download the signed installer.'
    Write-Host 'Only the installer is served; ORION controls are not on this tunnel.'
    & $CloudflaredPath tunnel --url "http://127.0.0.1:$Port" 2>&1 |
        Tee-Object -FilePath (Join-Path $logDirectory 'apk-tunnel.log')
    if ($LASTEXITCODE -ne 0) { throw "cloudflared exited with code $LASTEXITCODE" }
} finally {
    if (-not $server.HasExited) { Stop-Process -Id $server.Id -ErrorAction SilentlyContinue }
}

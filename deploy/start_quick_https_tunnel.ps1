param(
    [string]$CloudflaredPath = (Join-Path $env:TEMP 'orion-cloudflared-2026.9.3.exe')
)

$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $CloudflaredPath -PathType Leaf)) {
    throw 'cloudflared is missing. Run deploy/download_cloudflared_quick.ps1 first.'
}
$expectedSha256 = 'f096265ec2fcbe9bb6e2d64268db167ced3fcbb83d894bdb9e2fcdb26f2ea7e2'
$actual = (Get-FileHash -LiteralPath $CloudflaredPath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actual -ne $expectedSha256) {
    throw 'cloudflared failed the pinned SHA-256 check. Do not run this file.'
}
if (Test-Path -LiteralPath (Join-Path $env:USERPROFILE '.cloudflared\config.yaml')) {
    throw 'Cloudflare Quick Tunnel cannot run while ~/.cloudflared/config.yaml exists. Use a clean temporary profile.'
}

try {
    $health = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/api/health' -TimeoutSec 5
    $page = Invoke-WebRequest -Uri 'http://127.0.0.1:8000/' -TimeoutSec 5 -UseBasicParsing
} catch {
    throw 'Start the ORION desktop with ORION_REMOTE_EXPOSURE=quick-tunnel and the loopback uplink on port 8000 first.'
}
if (-not $health.ok -or $page.Content -notmatch 'const QUICK_TUNNEL=true;') {
    throw 'The process on port 8000 is not ORION in quick-tunnel mode. Restart ORION with the documented environment settings.'
}

Write-Host 'Cloudflare will print a temporary https://...trycloudflare.com URL below.'
Write-Host 'Enter that URL in the Android app. It changes whenever this tunnel restarts.'
Write-Host 'Keep this PowerShell window open while using the temporary URL.'
& $CloudflaredPath tunnel --url http://127.0.0.1:8000
if ($LASTEXITCODE -ne 0) { throw "cloudflared exited with code $LASTEXITCODE" }

param(
    [switch]$Public,
    [switch]$Headless,
    [switch]$DirectTailnetHttps,
    [switch]$CheckOnly,
    [string]$MagicDnsHost = ''
)

$ErrorActionPreference = 'Stop'
if ($Public -and $DirectTailnetHttps) {
    throw 'Choose either -Public or -DirectTailnetHttps.'
}
$repo = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repo

if (-not (Get-Command tailscale -ErrorAction SilentlyContinue)) {
    throw 'Tailscale is not installed or tailscale.exe is not on PATH.'
}
. (Join-Path $PSScriptRoot 'resolve_orion_python.ps1')
$pythonExe = Get-OrionPython -Repository $repo
$env:PYTHONUTF8 = '1'

$env:ORION_REMOTE_ACCESS = '1'
if ($DirectTailnetHttps) {
    # This route works when the tailnet administrator has not enabled Serve.
    # Only the Tailscale address is bound; the application never listens on LAN.
    $tailIp = ([string](& tailscale ip -4 2>$null | Select-Object -First 1)).Trim()
    if ($LASTEXITCODE -ne 0 -or $tailIp -notmatch '^100\.(?:[6-9]\d|1[01]\d|12[0-7])\.\d+\.\d+$') {
        throw "Could not read this PC's Tailscale IPv4 address. Check that Tailscale is running and this PowerShell session can access its CLI."
    }
    if (-not $MagicDnsHost) {
        $statusText = & tailscale status --json 2>$null
        if ($LASTEXITCODE -eq 0) {
            $MagicDnsHost = ([string](($statusText | Out-String | ConvertFrom-Json).Self.DNSName)).TrimEnd('.')
        }
    }
    if ($MagicDnsHost -notmatch '^[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)+\.ts\.net$') {
        throw "Supply this PC's full MagicDNS name with -MagicDnsHost, for example orion-host.example.ts.net."
    }
    $env:ORION_REMOTE_HOST = $tailIp
    $env:ORION_REMOTE_PORT = '443'
    $env:ORION_REMOTE_TLS = '1'
    $env:ORION_REMOTE_EXPOSURE = 'tailnet-direct'
    $env:ORION_REMOTE_DNS_NAME = $MagicDnsHost
    $env:ORION_REMOTE_FIREWALL = '0'
    Write-Host "ORION direct tailnet HTTPS: https://$MagicDnsHost/"
    Write-Host "Both the PC and phone must have Tailscale connected. The phone will ask you to trust this PC's certificate once."
    Write-Host 'Run deploy/configure_tailscale_firewall.ps1 -AllowDirectTailnetHttps once in an elevated PowerShell if the phone times out.'
} else {
    $env:ORION_REMOTE_HOST = '127.0.0.1'
    $env:ORION_REMOTE_PORT = '8000'
    $env:ORION_REMOTE_TLS = '0'
    $env:ORION_REMOTE_EXPOSURE = if ($Public) { 'funnel' } else { 'serve' }
    Remove-Item Env:ORION_REMOTE_DNS_NAME -ErrorAction SilentlyContinue
}

if ($CheckOnly) {
    if (-not $DirectTailnetHttps) {
        throw '-CheckOnly is currently supported with -DirectTailnetHttps only.'
    }
    Write-Host "Python: $pythonExe"
    return
}

$target = 'http://127.0.0.1:8000'
if ($DirectTailnetHttps) {
    # ORION itself provides TLS on the tailnet interface.
} elseif ($Public) {
    # Funnel exposes the paired gateway to the public internet on HTTPS 443.
    tailscale funnel --bg --https=443 $target
    if ($LASTEXITCODE -ne 0) { throw 'Tailscale Funnel setup failed.' }
    tailscale funnel status
} else {
    # Serve reaches the gateway through the tailnet only.
    tailscale serve --bg --https=443 $target
    if ($LASTEXITCODE -ne 0) { throw 'Tailscale Serve setup failed.' }
    tailscale serve status
}

if ($Headless) { & $pythonExe orion.py --headless }
else { & $pythonExe orion.py }


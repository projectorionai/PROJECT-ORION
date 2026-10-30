# Download the pinned Cloudflare client for a temporary HTTPS tunnel.
$ErrorActionPreference = 'Stop'
$version = '2026.9.3'
$expectedSha256 = 'f096265ec2fcbe9bb6e2d64268db167ced3fcbb83d894bdb9e2fcdb26f2ea7e2'
$destination = Join-Path $env:TEMP "orion-cloudflared-$version.exe"
$url = "https://github.com/cloudflare/cloudflared/releases/download/$version/cloudflared-windows-amd64.exe"

if (Test-Path -LiteralPath $destination) {
    $actual = (Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -eq $expectedSha256) {
        Write-Output $destination
        return
    }
    throw "An existing cloudflared file at $destination has an unexpected SHA-256. Inspect it before retrying."
}

Invoke-WebRequest -Uri $url -OutFile $destination
$actual = (Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actual -ne $expectedSha256) {
    throw "Downloaded cloudflared did not match Cloudflare's published SHA-256 ($actual). Do not run $destination."
}
Write-Output $destination

# Run in an elevated PowerShell. Serve mode needs no inbound ORION port.
param(
    [switch]$AllowDirectPeerUdp,
    [switch]$AllowDirectLanTcp,
    [switch]$AllowDirectTailnetHttps,
    [int]$DirectLanPort = 8000,
    [int]$TailscaleUdpPort = 41641
)

$ErrorActionPreference = 'Stop'
$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = [Security.Principal.WindowsPrincipal]::new($identity)
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw 'Run this script in an elevated PowerShell window.'
}

$adapters = @(Get-NetAdapter | Where-Object {
    $_.Name -match 'Tailscale' -or $_.InterfaceDescription -match 'Tailscale'
})
if ($adapters.Count -eq 0) { throw 'No Tailscale virtual adapter was found.' }
foreach ($adapter in $adapters) {
    $profile = Get-NetConnectionProfile -InterfaceIndex $adapter.ifIndex -ErrorAction SilentlyContinue
    if (-not $profile) {
        Write-Warning "Windows has no connection profile for $($adapter.Name); Serve still works without one."
    } elseif ($profile.NetworkCategory -ne 'Private') {
        Set-NetConnectionProfile -InterfaceIndex $adapter.ifIndex -NetworkCategory Private
    }
}

# Direct HTTPS is limited to the virtual adapter, this PC's tailnet address,
# and peer addresses. Serve mode still needs no inbound application rule.
if ($AllowDirectTailnetHttps) {
    foreach ($adapter in $adapters) {
        $tailAddresses = @(Get-NetIPAddress -InterfaceIndex $adapter.ifIndex `
            -AddressFamily IPv4 -ErrorAction Stop | Where-Object {
                $_.IPAddress -match '^100\.(?:[6-9]\d|1[01]\d|12[0-7])\.\d+\.\d+$'
            })
        if ($tailAddresses.Count -eq 0) {
            throw "No Tailscale IPv4 address was found on $($adapter.Name)."
        }
        foreach ($address in $tailAddresses) {
            $name = "ORION direct tailnet HTTPS ($($adapter.Name), $($address.IPAddress))"
            if (-not (Get-NetFirewallRule -DisplayName $name -ErrorAction SilentlyContinue)) {
                New-NetFirewallRule -DisplayName $name -Direction Inbound -Action Allow `
                    -Protocol TCP -LocalPort 443 -LocalAddress $address.IPAddress `
                    -RemoteAddress '100.64.0.0/10' -InterfaceAlias $adapter.Name `
                    -Profile Any | Out-Null
            }
        }
    }
}

# Optional direct WireGuard peer traffic. Tailscale can relay when this is shut.
if ($AllowDirectPeerUdp) {
    $name = 'ORION Tailscale peer UDP'
    if (-not (Get-NetFirewallRule -DisplayName $name -ErrorAction SilentlyContinue)) {
        New-NetFirewallRule -DisplayName $name -Direction Inbound -Action Allow `
            -Protocol UDP -LocalPort $TailscaleUdpPort -Profile Any | Out-Null
    }
}

# Only for ORION_REMOTE_HOST=0.0.0.0 on a trusted LAN. Loopback Serve mode
# cannot be reached directly through this rule and does not require it.
if ($AllowDirectLanTcp) {
    $name = 'ORION direct LAN TCP'
    if (-not (Get-NetFirewallRule -DisplayName $name -ErrorAction SilentlyContinue)) {
        New-NetFirewallRule -DisplayName $name -Direction Inbound -Action Allow `
            -Protocol TCP -LocalPort $DirectLanPort -RemoteAddress LocalSubnet `
            -Profile Private | Out-Null
    }
}

Get-NetConnectionProfile | Where-Object {
    $_.InterfaceIndex -in $adapters.ifIndex
} | Select-Object InterfaceAlias, NetworkCategory

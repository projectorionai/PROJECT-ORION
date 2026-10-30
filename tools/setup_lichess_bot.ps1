# Check a Lichess account, or explicitly convert a new dedicated account to BOT.
# Tokens are entered securely and are never saved or included in command history.
[CmdletBinding()]
param([switch]$Upgrade)

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

function Invoke-OrionLichessBotSetup {
    $secureToken = $null
    $tokenPointer = [IntPtr]::Zero
    $plainToken = $null
    $requestHeaders = $null
    $upgradeSubmitted = $false
    $originalTls = [Net.ServicePointManager]::SecurityProtocol
    try {
        if ($PSVersionTable.PSVersion.Major -lt 6) {
            [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        }
        $secureToken = Read-Host 'Paste the Lichess token with Play bot moves (bot:play)' -AsSecureString
        if ($secureToken -isnot [Security.SecureString]) {
            throw 'Secure token input is unavailable.'
        }
        $tokenPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureToken)
        $plainToken = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($tokenPointer)
        if ([string]::IsNullOrWhiteSpace($plainToken) -or $plainToken.Length -gt 512 -or
            $plainToken -notmatch '^[A-Za-z0-9._~-]+$') {
            throw 'The token format is invalid.'
        }
        $requestHeaders = @{ Authorization = "Bearer $plainToken" }

        # The host and paths are fixed. A redirect must never receive the token.
        $account = Invoke-RestMethod -Method Get -Uri 'https://lichess.org/api/account' `
            -Headers $requestHeaders -TimeoutSec 15 -MaximumRedirection 0 `
            -Verbose:$false -Debug:$false -ErrorAction Stop
        if ($null -eq $account -or $null -eq $account.PSObject.Properties['username']) {
            throw 'The authenticated account was not identified.'
        }
        $username = [string]$account.username
        if ($username -notmatch '^[A-Za-z0-9][A-Za-z0-9_-]{1,29}$') {
            throw 'The authenticated account name is invalid.'
        }
        if ($null -ne $account.PSObject.Properties['title'] -and $account.title -ceq 'BOT') {
            Write-Host "@$username is already a BOT account. No account changes were made."
            return 0
        }
        if (-not $Upgrade) {
            Write-Host "@$username is not a BOT account. Verification only; no account changes were made."
            Write-Host 'Use a new dedicated account that has never played before running with -Upgrade.'
            return 0
        }

        if ($null -ne $account.PSObject.Properties['count'] -and $null -ne $account.count) {
            foreach ($countName in @('all', 'rated', 'ai', 'playing')) {
                $countProperty = $account.count.PSObject.Properties[$countName]
                if ($null -eq $countProperty) { continue }
                $gameCount = [long]0
                if (-not [long]::TryParse([string]$countProperty.Value, [ref]$gameCount) -or $gameCount -lt 0) {
                    Write-Host 'The account game count could not be verified. No account changes were made.'
                    return 1
                }
                if ($gameCount -gt 0) {
                    Write-Host 'This account has recorded games and cannot be converted. Create a new dedicated account.'
                    return 1
                }
            }
        }
        Write-Host "WARNING: This permanently converts @$username to a BOT account. It cannot be undone."
        Write-Host 'Use a dedicated account that has never played. Lichess also checks eligibility before converting it.'
        $confirmation = Read-Host "Type the exact authenticated username ($username) to convert it, or press Enter to cancel"
        if (-not [string]::Equals($confirmation, $username, [StringComparison]::Ordinal)) {
            Write-Host 'Cancelled; no account changes were made.'
            return 0
        }

        $upgradeSubmitted = $true
        $null = Invoke-RestMethod -Method Post -Uri 'https://lichess.org/api/bot/account/upgrade' `
            -Headers $requestHeaders -TimeoutSec 15 -MaximumRedirection 0 `
            -Verbose:$false -Debug:$false -ErrorAction Stop
        $verified = Invoke-RestMethod -Method Get -Uri 'https://lichess.org/api/account' `
            -Headers $requestHeaders -TimeoutSec 15 -MaximumRedirection 0 `
            -Verbose:$false -Debug:$false -ErrorAction Stop
        if ($null -eq $verified -or $null -eq $verified.PSObject.Properties['username'] -or
            $null -eq $verified.PSObject.Properties['title'] -or
            $verified.username -cne $username -or $verified.title -cne 'BOT') {
            throw 'The BOT account could not be verified.'
        }
        Write-Host "Verified: @$username is now a BOT account. The token has not been saved."
        return 0
    }
    catch {
        # Never print exception messages, response bodies or request headers.
        if ($upgradeSubmitted) {
            Write-Host 'The upgrade result could not be verified. It may have completed; run verification again before retrying.'
        }
        else {
            Write-Host 'Lichess verification failed. Check the token, bot:play permission and internet connection; no upgrade was requested.'
        }
        return 1
    }
    finally {
        if ($tokenPointer -ne [IntPtr]::Zero) {
            [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($tokenPointer)
        }
        if ($null -ne $secureToken -and $secureToken -is [Security.SecureString]) {
            $secureToken.Dispose()
        }
        if ($null -ne $requestHeaders) { $requestHeaders.Clear() }
        $plainToken = $null
        [Net.ServicePointManager]::SecurityProtocol = $originalTls
    }
}

$result = Invoke-OrionLichessBotSetup
exit $result

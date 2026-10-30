"""Test the account helper with isolated PowerShell mocks and no network access."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "setup_lichess_bot.ps1"
TOKEN = "lip_SYNTHETIC_TEST_TOKEN_NOT_A_REAL_CREDENTIAL"
WRAPPER = r"""
param([string]$HelperPath, [string]$OptionsPath, [string]$ResultsPath)
$ErrorActionPreference = 'Stop'
$global:mockLichessOptions = Get-Content -LiteralPath $OptionsPath -Raw | ConvertFrom-Json
$global:mockLichessRequests = New-Object 'System.Collections.Generic.List[object]'
$global:mockLichessPrompts = New-Object 'System.Collections.Generic.List[object]'
$global:mockLichessGetCount = 0
function Read-Host {
    [CmdletBinding()]
    param([string]$Prompt, [switch]$AsSecureString)
    $global:mockLichessPrompts.Add([pscustomobject]@{ secure = [bool]$AsSecureString; prompt = $Prompt })
    if ($AsSecureString) {
        $secure = New-Object Security.SecureString
        foreach ($character in $global:mockLichessOptions.token.ToCharArray()) { $secure.AppendChar($character) }
        $secure.MakeReadOnly()
        return $secure
    }
    Write-Host 'CONFIRMATION_PROMPT_SEEN'
    return $global:mockLichessOptions.confirmation
}
function Invoke-RestMethod {
    [CmdletBinding()]
    param([string]$Method, [string]$Uri, [hashtable]$Headers, [int]$TimeoutSec, [int]$MaximumRedirection)
    $global:mockLichessRequests.Add([pscustomobject]@{
        method = $Method; uri = $Uri; timeout = $TimeoutSec; redirects = $MaximumRedirection
        authenticated = $Headers.Authorization -ceq ('Bearer ' + $global:mockLichessOptions.token)
    })
    if ($global:mockLichessOptions.failMethod -eq $Method) { throw ('Hidden response body: ' + $global:mockLichessOptions.token) }
    if ($Method -eq 'Get') {
        $accounts = @($global:mockLichessOptions.accounts)
        $index = [Math]::Min($global:mockLichessGetCount, $accounts.Count - 1)
        $global:mockLichessGetCount += 1
        return $accounts[$index]
    }
    return [pscustomobject]@{ ok = $true }
}
$LASTEXITCODE = 0
try {
    if ($global:mockLichessOptions.upgrade) { & $HelperPath -Upgrade } else { & $HelperPath }
}
finally {
    [pscustomobject]@{ requests = $global:mockLichessRequests.ToArray(); prompts = $global:mockLichessPrompts.ToArray(); code = $LASTEXITCODE } |
        ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $ResultsPath -Encoding UTF8
}
exit $LASTEXITCODE
"""


@pytest.fixture
def run_helper(tmp_path):
    powershell = shutil.which("pwsh") or shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is unavailable")

    def run(*, accounts=None, confirmation="OrionBot", upgrade=False, fail_method="", token=TOKEN, runtime=None):
        wrapper = tmp_path / "mock_lichess.ps1"
        options_path = tmp_path / "options.json"
        results_path = tmp_path / "results.json"
        wrapper.write_text(WRAPPER, encoding="utf-8")
        options_path.write_text(json.dumps({
            "accounts": accounts or [{"username": "OrionBot", "count": {"all": 0}}],
            "confirmation": confirmation, "upgrade": upgrade, "failMethod": fail_method, "token": token,
        }), encoding="utf-8")
        # Each runtime must use its own modules, rather than inherit incompatible
        # PowerShell 7 modules when Windows PowerShell 5.1 runs the documented command.
        child_env = os.environ.copy()
        child_env.pop("PSModulePath", None)
        process = subprocess.run(
            [runtime or powershell, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", str(wrapper),
             "-HelperPath", str(SCRIPT), "-OptionsPath", str(options_path), "-ResultsPath", str(results_path)],
            capture_output=True, text=True, timeout=20, check=False, env=child_env,
        )
        output = process.stdout + process.stderr
        assert token not in output
        assert results_path.exists(), output
        state = json.loads(results_path.read_text(encoding="utf-8-sig"))
        return process, state, output

    return run


def test_default_is_authenticated_verification_only(run_helper):
    process, state, output = run_helper()
    assert process.returncode == 0
    assert [request["method"] for request in state["requests"]] == ["Get"]
    assert state["prompts"][0]["secure"] is True
    assert len(state["prompts"]) == 1
    assert "not a BOT" in output and "Verification only" in output


def test_upgrade_requires_confirmation_then_verifies_bot_title(run_helper):
    process, state, output = run_helper(
        upgrade=True,
        accounts=[{"username": "OrionBot", "count": {"all": 0}}, {"username": "OrionBot", "title": "BOT"}],
    )
    assert process.returncode == 0
    assert [request["method"] for request in state["requests"]] == ["Get", "Post", "Get"]
    assert state["requests"][1]["uri"] == "https://lichess.org/api/bot/account/upgrade"
    assert output.index("cannot be undone") < output.index("CONFIRMATION_PROMPT_SEEN")
    assert "Verified:" in output
    assert all(request["timeout"] == 15 and request["redirects"] == 0 for request in state["requests"])
    assert all(request["authenticated"] for request in state["requests"])


@pytest.mark.parametrize("confirmation", ["", "OtherBot", "orionbot", " OrionBot", "OrionBot "])
def test_mismatched_or_cancelled_confirmation_never_posts(run_helper, confirmation):
    process, state, output = run_helper(upgrade=True, confirmation=confirmation)
    assert process.returncode == 0
    assert [request["method"] for request in state["requests"]] == ["Get"]
    assert "Cancelled" in output


@pytest.mark.parametrize("upgrade", [False, True])
def test_existing_bot_needs_no_upgrade_or_confirmation(run_helper, upgrade):
    process, state, output = run_helper(
        upgrade=upgrade, accounts=[{"username": "OrionBot", "title": "BOT", "count": {"all": 123}}],
    )
    assert process.returncode == 0
    assert [request["method"] for request in state["requests"]] == ["Get"]
    assert len(state["prompts"]) == 1
    assert "already a BOT" in output


@pytest.mark.parametrize("count", [1, 123, -1, "invalid"])
def test_ineligible_or_unverifiable_game_count_refuses_upgrade(run_helper, count):
    process, state, output = run_helper(upgrade=True, accounts=[{"username": "OrionBot", "count": {"all": count}}])
    assert process.returncode == 1
    assert [request["method"] for request in state["requests"]] == ["Get"]
    assert len(state["prompts"]) == 1
    assert "cannot be converted" in output or "could not be verified" in output


def test_missing_game_count_leaves_eligibility_to_lichess_after_confirmation(run_helper):
    process, state, _output = run_helper(
        upgrade=True, accounts=[{"username": "OrionBot"}, {"username": "OrionBot", "title": "BOT"}],
    )
    assert process.returncode == 0
    assert [request["method"] for request in state["requests"]] == ["Get", "Post", "Get"]


@pytest.mark.parametrize("count", [{"rated": 1}, {"all": 0, "ai": 1}, {"playing": 1}])
def test_any_known_nonzero_game_count_refuses_upgrade(run_helper, count):
    process, state, output = run_helper(upgrade=True, accounts=[{"username": "OrionBot", "count": count}])
    assert process.returncode == 1
    assert [request["method"] for request in state["requests"]] == ["Get"]
    assert "cannot be converted" in output


@pytest.mark.parametrize("method", ["Get", "Post"])
def test_network_error_never_exposes_response_exception_or_token(run_helper, method):
    process, state, output = run_helper(upgrade=True, fail_method=method)
    assert process.returncode == 1
    assert "Hidden response body" not in output
    assert "may have completed" in output if method == "Post" else "no upgrade was requested" in output
    assert [request["method"] for request in state["requests"]] == (["Get", "Post"] if method == "Post" else ["Get"])


@pytest.mark.parametrize("account", [{}, {"username": "malicious\naccount"}])
def test_unidentified_account_cannot_reach_upgrade(run_helper, account):
    process, state, _output = run_helper(upgrade=True, accounts=[account])
    assert process.returncode == 1
    assert [request["method"] for request in state["requests"]] == ["Get"]
    assert len(state["prompts"]) == 1


@pytest.mark.parametrize("verified", [{"username": "OrionBot"}, {"username": "OtherBot", "title": "BOT"}])
def test_success_is_not_claimed_until_same_account_has_bot_title(run_helper, verified):
    process, state, output = run_helper(upgrade=True, accounts=[{"username": "OrionBot"}, verified])
    assert process.returncode == 1
    assert [request["method"] for request in state["requests"]] == ["Get", "Post", "Get"]
    assert "Verified:" not in output
    assert "may have completed" in output


def test_invalid_token_never_reaches_network(run_helper):
    process, state, _output = run_helper(token="invalid token\nInjected: header", upgrade=True)
    assert process.returncode == 1
    assert state["requests"] == []


def test_documented_windows_powershell_command_is_compatible(run_helper):
    runtime = shutil.which("powershell")
    if runtime is None:
        pytest.skip("Windows PowerShell is unavailable")
    process, state, output = run_helper(
        runtime=runtime, upgrade=True,
        accounts=[{"username": "OrionBot", "count": {"all": 0}}, {"username": "OrionBot", "title": "BOT"}],
    )
    assert process.returncode == 0
    assert [request["method"] for request in state["requests"]] == ["Get", "Post", "Get"]
    assert "Verified:" in output

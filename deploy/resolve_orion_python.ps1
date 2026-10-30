function Get-OrionPython {
    param([string]$Repository = (Split-Path -Parent $PSScriptRoot))
    $candidates = @()
    if ($env:ORION_PYTHON) { $candidates += $env:ORION_PYTHON }
    $candidates += (Join-Path $Repository '.neural-debug-env\Scripts\python.exe')
    $candidates += (Join-Path $Repository '.venv\Scripts\python.exe')
    foreach ($name in @('python', 'py')) {
        $command = Get-Command $name -ErrorAction SilentlyContinue
        if ($command) { $candidates += $command.Source }
    }
    foreach ($candidate in ($candidates | Select-Object -Unique)) {
        if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) { continue }
        try {
            & $candidate -c "import importlib.util; assert all(importlib.util.find_spec(m) for m in ('PyQt6.QtWidgets','PyQt6.QtWebEngineWidgets','qasync','aiohttp','sounddevice','google.genai','numpy','psutil'))" 2>$null | Out-Null
            if ($LASTEXITCODE -eq 0) { return $candidate }
        } catch { }
    }
    throw 'No working ORION Python environment was found. Install requirements/core.txt into a working virtual environment, then set ORION_PYTHON to its python.exe.'
}

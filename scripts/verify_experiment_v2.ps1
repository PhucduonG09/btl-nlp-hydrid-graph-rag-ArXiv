param([switch]$CheckEngine)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
$taskPython = Join-Path (Get-Location) '.venv-kg/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) { throw 'Missing .venv-kg. See README.md.' }
foreach ($taskScript in @('audit_reference_artifacts.py', 'summarize_generation_v2.py', 'audit_judge_witnesses_v2.py')) {
    & $taskPython -X utf8 (Join-Path $PSScriptRoot $taskScript)
    if ($LASTEXITCODE -ne 0) { throw "Failed: $taskScript" }
}
if ($CheckEngine) {
    & $taskPython -X utf8 (Join-Path $PSScriptRoot 'run_locked_test_v2.py') check-dev
    if ($LASTEXITCODE -ne 0) { throw 'Deployment engine differs from saved DEV results.' }
}
$taskTestRun = 'verify-' + (Get-Date -Format 'yyyyMMdd-HHmmss')
& $taskPython -X utf8 -m pytest tests -q "--basetemp=.pytest_cache/$taskTestRun" -o "cache_dir=.pytest_cache/$taskTestRun-cache"
if ($LASTEXITCODE -ne 0) { throw 'Tests failed.' }
Write-Host 'Verified local results. No API calls, no test evaluation, no workbook edits.'

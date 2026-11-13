$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
$taskPython = Join-Path (Get-Location) '.venv-kg/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) { throw 'Missing .venv-kg. See README.md for setup.' }
foreach ($taskScript in @('build_reference_graph.py', 'run_graph_dev.py', 'analyze_graph_dev.py', 'audit_reference_artifacts.py')) {
    & $taskPython -X utf8 (Join-Path $PSScriptRoot $taskScript)
    if ($LASTEXITCODE -ne 0) { throw "Failed: $taskScript" }
}
Write-Host 'Reference DEV run completed. No API calls and no workbook changes.'

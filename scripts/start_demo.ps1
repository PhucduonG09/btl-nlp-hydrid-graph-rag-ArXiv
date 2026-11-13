param([switch]$AgentApi)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
$taskPython = Join-Path (Get-Location) '.venv-kg/Scripts/python.exe'
$taskArguments = @('run', 'app.py', '--server.headless=true', '--server.address=127.0.0.1', '--server.port=8501', '--server.fileWatcherType=none', '--browser.gatherUsageStats=false')
if ($AgentApi) { $taskArguments += '--server.enableAgentApi=true' }
& $taskPython -m streamlit @taskArguments

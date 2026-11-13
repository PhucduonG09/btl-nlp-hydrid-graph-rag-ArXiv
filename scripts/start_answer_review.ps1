param([switch]$OpenBrowser)
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$taskPython = Join-Path $taskRoot '.venv-kg/Scripts/python.exe'
$taskApp = Join-Path $taskRoot 'review_answers.py'
$taskUrl = 'http://127.0.0.1:8503'
$taskLogs = Join-Path $taskRoot 'outputs'
if (-not (Test-Path -LiteralPath $taskPython)) { throw 'Missing .venv-kg/Scripts/python.exe' }
New-Item -ItemType Directory -Path $taskLogs -Force | Out-Null
function Test-ReviewHealth {
    try {
        $taskResponse = Invoke-WebRequest -Uri "$taskUrl/_stcore/health" -UseBasicParsing -TimeoutSec 2
        return ($taskResponse.StatusCode -eq 200 -and $taskResponse.Content.Trim() -eq 'ok')
    } catch { return $false }
}
if (-not (Test-ReviewHealth)) {
    $taskArguments = @('-X', 'utf8', '-m', 'streamlit', 'run', ('"' + $taskApp + '"'),
        '--server.address=127.0.0.1', '--server.port=8503', '--server.headless=true',
        '--server.fileWatcherType=none', '--browser.gatherUsageStats=false', '--server.enableAgentApi=true')
    $taskProcess = Start-Process -FilePath $taskPython -ArgumentList $taskArguments -WorkingDirectory $taskRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $taskLogs 'answer-review.stdout.log') -RedirectStandardError (Join-Path $taskLogs 'answer-review.stderr.log')
    $taskProcess.Id | Set-Content -LiteralPath (Join-Path $taskLogs 'answer-review.pid')
    $taskReady = $false
    for ($taskAttempt = 0; $taskAttempt -lt 20; $taskAttempt++) {
        if (Test-ReviewHealth) { $taskReady = $true; break }
        Start-Sleep -Milliseconds 500
    }
    if (-not $taskReady) { throw 'Review server did not start. See outputs/answer-review.stderr.log' }
}
Write-Host "Ready: $taskUrl"
if ($OpenBrowser) { Start-Process $taskUrl }

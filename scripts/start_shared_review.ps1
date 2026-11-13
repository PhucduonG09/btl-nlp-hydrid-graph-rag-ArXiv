param([string[]]$AllowedEmails=@(),[switch]$OpenBrowser)
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$taskPython = Join-Path $taskRoot '.venv-kg/Scripts/python.exe'
$taskTunnel = Join-Path $taskRoot 'data/tools/cloudflared/cloudflared.exe'
$taskLogs = Join-Path $taskRoot 'outputs'
$taskMeta = Join-Path $taskLogs 'review-share.json'
$taskEmailFile = Join-Path $taskLogs 'review-share-allowed-emails.json'
if ($AllowedEmails.Count -eq 0 -and (Test-Path -LiteralPath $taskEmailFile)) {
    $AllowedEmails = [string[]](Get-Content -LiteralPath $taskEmailFile -Raw | ConvertFrom-Json)
}
if ($AllowedEmails.Count -eq 0) { throw 'Supply the explicitly approved reviewer email addresses before starting a shared link.' }
foreach ($taskEmail in $AllowedEmails) {
    if ($taskEmail -notmatch '^[A-Za-z0-9.!#$%&+?_=~-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$') { throw 'Use individual email addresses, not wildcard domains.' }
}
function Test-ShareHealth {
    try { return ((Invoke-WebRequest 'http://127.0.0.1:8504/_stcore/health' -UseBasicParsing -TimeoutSec 2).Content.Trim() -eq 'ok') }
    catch { return $false }
}
if (Test-Path -LiteralPath $taskMeta) {
    $taskPrevious = Get-Content -LiteralPath $taskMeta -Raw | ConvertFrom-Json
    $taskExisting = Get-Process -Id $taskPrevious.tunnel_pid -ErrorAction SilentlyContinue
    $taskSameEmails = (@(Compare-Object @($taskPrevious.allowed_emails) $AllowedEmails).Count -eq 0)
    if ($taskExisting -and $taskExisting.ProcessName -eq 'cloudflared' -and $taskPrevious.protected -and $taskSameEmails -and (Test-ShareHealth)) {
        Write-Host $taskPrevious.url
        if ($OpenBrowser) { Start-Process $taskPrevious.url }
        exit 0
    }
}
if (-not (Test-Path -LiteralPath $taskTunnel)) { throw 'Missing verified cloudflared executable in data/tools/cloudflared.' }
if (Test-ShareHealth) { throw 'Port 8504 already serves an app; inspect before replacing it.' }
New-Item -ItemType Directory -Path $taskLogs -Force | Out-Null
$taskTunnelLog = Join-Path $taskLogs 'review-share-tunnel.stderr.log'
$taskTunnelArguments = @('tunnel','--url','http://127.0.0.1:8504','--protocol','http2','--no-autoupdate')
foreach ($taskEmail in $AllowedEmails) { $taskTunnelArguments += @('--allowed-mail',$taskEmail) }
$taskTunnelProcess = $null
$taskPidFile = Join-Path $taskLogs 'review-share-tunnel.pid'
if (Test-Path -LiteralPath $taskPidFile) {
    $taskOldId = [int](Get-Content -LiteralPath $taskPidFile)
    $taskOld = Get-CimInstance Win32_Process -Filter "ProcessId = $taskOldId"
    if ($taskOld -and $taskOld.Name -eq 'cloudflared.exe' -and $taskOld.CommandLine.Contains('http://127.0.0.1:8504')) {
        $taskApproved = $taskOld.CommandLine.Contains('--allowed-mail')
        foreach ($taskEmail in $AllowedEmails) { $taskApproved = $taskApproved -and $taskOld.CommandLine.Contains($taskEmail) }
        if (-not $taskApproved) { throw 'Existing tunnel does not match the approved protected email list.' }
        $taskTunnelProcess = Get-Process -Id $taskOldId
    }
}
if (-not $taskTunnelProcess) {
    $taskTunnelProcess = Start-Process -FilePath $taskTunnel -ArgumentList $taskTunnelArguments -WorkingDirectory $taskRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $taskLogs 'review-share-tunnel.stdout.log') -RedirectStandardError $taskTunnelLog
}
$taskTunnelProcess.Id | Set-Content -LiteralPath (Join-Path $taskLogs 'review-share-tunnel.pid')
$taskPublicUrl = $null
for ($taskAttempt=0; $taskAttempt -lt 45; $taskAttempt++) {
    if (Test-Path -LiteralPath $taskTunnelLog) {
        $taskMatch = [regex]::Match([string](Get-Content -LiteralPath $taskTunnelLog -Raw),'https://[a-z0-9-]+\.trycloudflare\.com')
        if ($taskMatch.Success) { $taskPublicUrl = $taskMatch.Value; break }
    }
    if ($taskTunnelProcess.HasExited) { throw 'Tunnel exited. See outputs/review-share-tunnel.stderr.log' }
    Start-Sleep -Milliseconds 500
}
if (-not $taskPublicUrl) { throw 'No tunnel URL yet. See outputs/review-share-tunnel.stderr.log' }
$taskHostname = ([Uri]$taskPublicUrl).Host
$taskApp = Join-Path $taskRoot 'share_review_answers.py'
$taskArguments = @('-X','utf8','-m','streamlit','run',('"'+$taskApp+'"'),
    '--server.address=127.0.0.1','--server.port=8504','--server.headless=true',
    '--server.fileWatcherType=none','--browser.gatherUsageStats=false','--server.enableAgentApi=false',
    '--server.enableStaticServing=false','--server.enableCORS=true','--server.enableXsrfProtection=true',
    '--client.showErrorDetails=none',("--browser.serverAddress=$taskHostname"),
    ("--server.corsAllowedOrigins=$taskPublicUrl"),'--server.allowedHosts=localhost',
    '--server.allowedHosts=127.0.0.1',("--server.allowedHosts=$taskHostname"))
$taskServer = Start-Process -FilePath $taskPython -ArgumentList $taskArguments -WorkingDirectory $taskRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $taskLogs 'review-share-app.stdout.log') -RedirectStandardError (Join-Path $taskLogs 'review-share-app.stderr.log')
$taskServer.Id | Set-Content -LiteralPath (Join-Path $taskLogs 'review-share-app.pid')
$taskReady=$false
for ($taskAttempt=0; $taskAttempt -lt 25; $taskAttempt++) {
    if (Test-ShareHealth) { $taskReady=$true; break }
    Start-Sleep -Milliseconds 500
}
if (-not $taskReady) { throw 'Review server did not start. See outputs/review-share-app.stderr.log' }
@{url=$taskPublicUrl;tunnel_pid=$taskTunnelProcess.Id;server_pid=$taskServer.Id;port=8504;protected=$true;allowed_emails=$AllowedEmails;mode='shared_blind_human_review';created_at=(Get-Date -Format o)} | ConvertTo-Json | Set-Content -LiteralPath $taskMeta -Encoding utf8
$taskPublicUrl | Set-Content -LiteralPath (Join-Path $taskLogs 'review-share-link.txt') -Encoding utf8
Write-Host $taskPublicUrl
if ($OpenBrowser) { Start-Process $taskPublicUrl }

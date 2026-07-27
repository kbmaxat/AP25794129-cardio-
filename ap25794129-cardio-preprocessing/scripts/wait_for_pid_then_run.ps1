param(
    [Parameter(Mandatory = $true)]
    [int]$WaitPid,

    [Parameter(Mandatory = $true)]
    [string]$ScriptPath,

    [string]$WorkingDirectory = "",
    [string]$StdoutLog = "",
    [string]$StderrLog = "",
    [int]$PollSeconds = 30
)

if ([string]::IsNullOrWhiteSpace($WorkingDirectory)) {
    $WorkingDirectory = Split-Path -Parent $ScriptPath
}

while (Get-Process -Id $WaitPid -ErrorAction SilentlyContinue) {
    Start-Sleep -Seconds $PollSeconds
}

$startArgs = @{
    FilePath = "powershell.exe"
    ArgumentList = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $ScriptPath)
    WorkingDirectory = $WorkingDirectory
    WindowStyle = "Hidden"
    Wait = $true
}

if (-not [string]::IsNullOrWhiteSpace($StdoutLog)) {
    $startArgs.RedirectStandardOutput = $StdoutLog
}

if (-not [string]::IsNullOrWhiteSpace($StderrLog)) {
    $startArgs.RedirectStandardError = $StderrLog
}

Start-Process @startArgs

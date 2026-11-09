# Daily resume of the Stage 9 Groq run (Windows Task Scheduler "Stage9_Groq_Daily").
# Reads GROQ_API_KEY from the user environment; never prints it. Removes its own
# scheduled task once no rows remain.
$ErrorActionPreference = "Stop"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Out = Join-Path (Split-Path $Here -Parent) "results\stage9_frontier"
$Python = Join-Path (Split-Path (Split-Path $Here -Parent) -Parent) "finetune_env\Scripts\python.exe"
$Log = Join-Path $Out ("run_" + (Get-Date -Format "yyyyMMdd_HHmm") + ".log")

$env:GROQ_API_KEY = [Environment]::GetEnvironmentVariable('GROQ_API_KEY', 'User')
if (-not $env:GROQ_API_KEY) { "GROQ_API_KEY not set at user scope" | Out-File $Log -Encoding utf8; exit 1 }

Set-Location $Here
cmd /c "set PYTHONIOENCODING=utf-8&& set PYTHONUNBUFFERED=1&& ""$Python"" run_frontier.py --confirm > ""$Log"" 2>&1"

if (Select-String -Path $Log -Pattern "Rows remaining after this run: 0" -Encoding utf8 -Quiet) {
    Unregister-ScheduledTask -TaskName "Stage9_Groq_Daily" -Confirm:$false -ErrorAction SilentlyContinue
}

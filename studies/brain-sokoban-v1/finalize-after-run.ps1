param([Parameter(Mandatory=$true)][int]$RunnerPid)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$studyPython = Join-Path $projectRoot 'flybrain\.venv\Scripts\python.exe'
$studyRunner = Get-Process -Id $RunnerPid -ErrorAction SilentlyContinue
if ($null -ne $studyRunner) { $studyRunner.WaitForExit() }
Set-Location -LiteralPath $projectRoot
# A fresh interpreter reads every analysis module and all completed artifacts.
# If the study stopped early, the report retains the incomplete status.
& $studyPython -B -X utf8 -u workflow.py study analyze
exit $LASTEXITCODE

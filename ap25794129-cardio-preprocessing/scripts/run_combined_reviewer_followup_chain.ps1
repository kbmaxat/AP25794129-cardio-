param(
    [string]$RepoRoot = ""
)

if ([string]::IsNullOrWhiteSpace($RepoRoot)) {
    $RepoRoot = Split-Path -Parent $PSScriptRoot
}

Set-Location $RepoRoot

Write-Host "Starting combined multi-seed top-3 follow-up..."
& (Join-Path $PSScriptRoot "run_unet_combined_multiseed_top3.ps1")
if ($LASTEXITCODE -ne 0) {
    throw "Combined multi-seed top-3 follow-up failed."
}

Write-Host "Starting combined long-schedule top-3 follow-up..."
& (Join-Path $PSScriptRoot "run_unet_combined_longschedule_top3.ps1")
if ($LASTEXITCODE -ne 0) {
    throw "Combined long-schedule top-3 follow-up failed."
}

Write-Host "Combined reviewer follow-up chain completed."

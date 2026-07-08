param(
    [int]$AfterProcessId = 0,
    [string]$Manifest = "data/manifests_local/segmentation_public_combined.csv",
    [string]$OutputRoot = "outputs/unet_mode_grid",
    [int]$Epochs = 10,
    [int]$BatchSize = 8,
    [int]$ImageHeight = 256,
    [int]$ImageWidth = 256,
    [string]$BaseTag = ""
)

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

if ([string]::IsNullOrWhiteSpace($BaseTag)) {
    $BaseTag = (Get-Date -Format "yyyyMMdd_HHmmss") + "_public_segmentation_suite"
}

if ($AfterProcessId -gt 0) {
    $existing = Get-Process -Id $AfterProcessId -ErrorAction SilentlyContinue
    if ($null -ne $existing) {
        Write-Host "Waiting for process $AfterProcessId to complete before starting CAMUS and combined runs..."
        Wait-Process -Id $AfterProcessId
    } else {
        Write-Host "Process $AfterProcessId was not found; starting follow-up suite immediately."
    }
}

Write-Host "Starting CAMUS follow-up run..."
& .\scripts\run_unet_mode_grid.ps1 `
    -Manifest $Manifest `
    -OutputRoot $OutputRoot `
    -Epochs $Epochs `
    -BatchSize $BatchSize `
    -ImageHeight $ImageHeight `
    -ImageWidth $ImageWidth `
    -DatasetFilter CAMUS `
    -RunTag ($BaseTag + "_camus")

Write-Host "Starting combined ACDC+CAMUS follow-up run..."
& .\scripts\run_unet_mode_grid.ps1 `
    -Manifest $Manifest `
    -OutputRoot $OutputRoot `
    -Epochs $Epochs `
    -BatchSize $BatchSize `
    -ImageHeight $ImageHeight `
    -ImageWidth $ImageWidth `
    -RunTag ($BaseTag + "_combined")

Write-Host "Public segmentation suite completed."

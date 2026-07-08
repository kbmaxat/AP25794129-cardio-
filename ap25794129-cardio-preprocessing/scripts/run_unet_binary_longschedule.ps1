param(
    [string]$Manifest = "data/manifests_local/segmentation_public_combined.csv",
    [string]$OutputRoot = "outputs/unet_binary_longschedule",
    [int]$Epochs = 50,
    [int]$BatchSize = 8,
    [int]$Seed = 42,
    [double]$LearningRate = 0.001,
    [double]$WeightDecay = 0.0001,
    [double]$Threshold = 0.5,
    [int]$EarlyStoppingPatience = 10,
    [int]$EarlyStoppingMinEpochs = 15,
    [double]$EarlyStoppingMinDelta = 0.0005,
    [string[]]$Modes = @("none", "wavelet", "nlm"),
    [string[]]$DatasetFilter = @(),
    [int]$ImageHeight = 256,
    [int]$ImageWidth = 256,
    [string]$RunTag = ""
)

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

if ([string]::IsNullOrWhiteSpace($RunTag)) {
    $RunTag = Get-Date -Format "yyyyMMdd_HHmmss"
}

Write-Host ("Running long-schedule binary benchmark -> {0}" -f (Join-Path $OutputRoot $RunTag))

& (Join-Path $PSScriptRoot "run_unet_mode_grid.ps1") `
    -Manifest $Manifest `
    -OutputRoot $OutputRoot `
    -Epochs $Epochs `
    -BatchSize $BatchSize `
    -Seed $Seed `
    -LearningRate $LearningRate `
    -WeightDecay $WeightDecay `
    -Threshold $Threshold `
    -EarlyStoppingPatience $EarlyStoppingPatience `
    -EarlyStoppingMinEpochs $EarlyStoppingMinEpochs `
    -EarlyStoppingMinDelta $EarlyStoppingMinDelta `
    -Modes $Modes `
    -DatasetFilter $DatasetFilter `
    -ImageHeight $ImageHeight `
    -ImageWidth $ImageWidth `
    -RunTag $RunTag

if ($LASTEXITCODE -ne 0) {
    throw "Long-schedule binary benchmark failed."
}

Write-Host "Long-schedule binary benchmark completed."

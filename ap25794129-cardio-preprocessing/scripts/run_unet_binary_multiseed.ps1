param(
    [string]$Manifest = "data/manifests_local/segmentation_public_combined.csv",
    [string]$OutputRoot = "outputs/unet_binary_multiseed",
    [int]$Epochs = 10,
    [int]$BatchSize = 8,
    [int[]]$Seeds = @(11, 22, 33, 44, 55),
    [double]$LearningRate = 0.001,
    [double]$WeightDecay = 0.0001,
    [double]$Threshold = 0.5,
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

$sessionRoot = Join-Path $OutputRoot $RunTag
New-Item -ItemType Directory -Force -Path $sessionRoot | Out-Null
Write-Host "Session root: $sessionRoot"

foreach ($seed in $Seeds) {
    $seedRoot = Join-Path $sessionRoot ("seed_{0}" -f $seed)
    Write-Host ("Running seed sweep entry for seed {0} -> {1}" -f $seed, $seedRoot)

    & (Join-Path $PSScriptRoot "run_unet_mode_grid.ps1") `
        -Manifest $Manifest `
        -OutputRoot $seedRoot `
        -Epochs $Epochs `
        -BatchSize $BatchSize `
        -Seed $seed `
        -LearningRate $LearningRate `
        -WeightDecay $WeightDecay `
        -Threshold $Threshold `
        -Modes $Modes `
        -DatasetFilter $DatasetFilter `
        -ImageHeight $ImageHeight `
        -ImageWidth $ImageWidth `
        -RunTag "modes"

    if ($LASTEXITCODE -ne 0) {
        throw "Multi-seed sweep failed for seed '$seed'"
    }
}

Write-Host "All requested multi-seed runs completed."

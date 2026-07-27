param(
    [string]$Manifest = "data/manifests_local/segmentation_public_combined.csv",
    [string]$OutputRoot = "outputs/unet_multiclass_acdc_top3_extended",
    [int]$Epochs = 50,
    [int]$BatchSize = 8,
    [int]$EarlyStoppingPatience = 10,
    [int]$EarlyStoppingMinEpochs = 15,
    [double]$EarlyStoppingMinDelta = 0.0005,
    [string[]]$Modes = @("none", "wavelet", "nlm"),
    [string[]]$DatasetFilter = @("ACDC"),
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

foreach ($mode in $Modes) {
    $runName = "unet_multiclass_extended_" + (($DatasetFilter -join "_").ToLower()) + "_" + $mode
    $outputDir = Join-Path $sessionRoot $runName
    $args = @(
        "-m", "cardiac_image_system.experiments.train_unet_multiclass",
        "--manifest", $Manifest,
        "--output-dir", $outputDir,
        "--mode", $mode,
        "--epochs", $Epochs,
        "--batch-size", $BatchSize,
        "--image-height", $ImageHeight,
        "--image-width", $ImageWidth,
        "--early-stopping-patience", $EarlyStoppingPatience,
        "--early-stopping-min-epochs", $EarlyStoppingMinEpochs,
        "--early-stopping-min-delta", $EarlyStoppingMinDelta
    )

    if ($DatasetFilter.Count -gt 0) {
        $args += "--dataset-filter"
        $args += $DatasetFilter
    }

    $modeStart = Get-Date
    Write-Host "Running extended multiclass mode '$mode' -> $outputDir"
    py -3 @args
    if ($LASTEXITCODE -ne 0) {
        throw "Extended experiment failed for mode '$mode'"
    }
    $elapsed = (Get-Date) - $modeStart
    Write-Host ("Completed extended multiclass mode '{0}' in {1}" -f $mode, $elapsed.ToString("hh\:mm\:ss"))
}

Write-Host "All requested extended multiclass modes completed."

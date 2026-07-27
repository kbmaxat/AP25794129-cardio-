param(
    [string]$Manifest = "data/manifests_local/segmentation_public_combined.csv",
    [string]$OutputRoot = "outputs/unet_mode_grid",
    [int]$Epochs = 10,
    [int]$BatchSize = 8,
    [int]$Seed = 42,
    [double]$LearningRate = 0.001,
    [double]$WeightDecay = 0.0001,
    [double]$Threshold = 0.5,
    [int]$EarlyStoppingPatience = 0,
    [int]$EarlyStoppingMinEpochs = 0,
    [double]$EarlyStoppingMinDelta = 0.0,
    [string[]]$Modes = @("none", "gaussian", "wavelet", "nlm", "clahe", "hybrid"),
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

foreach ($mode in $Modes) {
    $runName = if ($DatasetFilter.Count -gt 0) {
        "unet_" + (($DatasetFilter -join "_").ToLower()) + "_" + $mode
    } else {
        "unet_combined_" + $mode
    }

    $outputDir = Join-Path $sessionRoot $runName
    $args = @(
        "-m", "cardiac_image_system.experiments.train_unet_baseline",
        "--manifest", $Manifest,
        "--output-dir", $outputDir,
        "--mode", $mode,
        "--epochs", $Epochs,
        "--batch-size", $BatchSize,
        "--seed", $Seed,
        "--learning-rate", $LearningRate,
        "--weight-decay", $WeightDecay,
        "--threshold", $Threshold,
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
    Write-Host "Running mode '$mode' -> $outputDir"
    py -3 @args
    if ($LASTEXITCODE -ne 0) {
        throw "Experiment failed for mode '$mode'"
    }
    $elapsed = (Get-Date) - $modeStart
    Write-Host ("Completed mode '{0}' in {1}" -f $mode, $elapsed.ToString("hh\:mm\:ss"))
}

Write-Host "All requested modes completed."

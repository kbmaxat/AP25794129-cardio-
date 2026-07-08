param(
    [string]$Manifest = "data/manifests_local/segmentation_public_combined.csv",
    [string]$OutputRoot = "outputs/unet_binary_longschedule_camus_top3",
    [int]$Epochs = 50,
    [int]$BatchSize = 8,
    [int]$Seed = 42,
    [int]$EarlyStoppingPatience = 10,
    [int]$EarlyStoppingMinEpochs = 15,
    [double]$EarlyStoppingMinDelta = 0.0005,
    [string[]]$Modes = @("none", "wavelet", "nlm"),
    [string]$RunTag = ""
)

& (Join-Path $PSScriptRoot "run_unet_binary_longschedule.ps1") `
    -Manifest $Manifest `
    -OutputRoot $OutputRoot `
    -Epochs $Epochs `
    -BatchSize $BatchSize `
    -Seed $Seed `
    -EarlyStoppingPatience $EarlyStoppingPatience `
    -EarlyStoppingMinEpochs $EarlyStoppingMinEpochs `
    -EarlyStoppingMinDelta $EarlyStoppingMinDelta `
    -Modes $Modes `
    -DatasetFilter @("CAMUS") `
    -RunTag $RunTag

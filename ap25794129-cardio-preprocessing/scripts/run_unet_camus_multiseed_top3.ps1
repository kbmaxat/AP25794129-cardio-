param(
    [string]$Manifest = "data/manifests_local/segmentation_public_combined.csv",
    [string]$OutputRoot = "outputs/unet_binary_multiseed_camus_top3",
    [int]$Epochs = 10,
    [int]$BatchSize = 8,
    [int[]]$Seeds = @(11, 22, 33, 44, 55),
    [string[]]$Modes = @("none", "wavelet", "nlm"),
    [string]$RunTag = ""
)

& (Join-Path $PSScriptRoot "run_unet_binary_multiseed.ps1") `
    -Manifest $Manifest `
    -OutputRoot $OutputRoot `
    -Epochs $Epochs `
    -BatchSize $BatchSize `
    -Seeds $Seeds `
    -Modes $Modes `
    -DatasetFilter @("CAMUS") `
    -RunTag $RunTag

# Experiment Protocol

## Main comparison

Modes:

1. none
2. gaussian
3. wavelet
4. nlm
5. clahe
6. hybrid

## Main outputs

- `metrics_slice_level.csv`
- `metrics_patient_level.csv`
- processed PNG images
- predicted proxy masks
- runtime log

## Interpretation

PSNR/SSIM are image-level metrics.  
Dice/IoU/HD95 are structure-level proxy metrics.  
They do not prove clinical diagnostic accuracy.

## Required next extension

Add U-Net locked inference and mode-specific training scenarios.

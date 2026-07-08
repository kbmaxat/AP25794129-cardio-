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

Add statistical comparison across preprocessing modes after the baseline U-Net runs are complete.

## Downstream baseline now implemented

Experiment entrypoint:

```bash
python -m cardiac_image_system.experiments.train_unet_baseline --manifest data/manifests_local/segmentation_public_combined.csv --output-dir outputs/unet_baseline_combined_none --mode none
```

Operational logic:

1. Load a CSV manifest with patient-level identifiers.
2. Resolve train/validation/test splits from the `subset` column when all three are available.
3. Fall back to patient-level random splitting when the manifest does not contain a complete validation topology.
4. Apply one preprocessing mode consistently to all splits.
5. Train a compact binary U-Net baseline against foreground-vs-background masks.
6. Export slice-level and patient-level Dice/IoU/HD95 summaries.

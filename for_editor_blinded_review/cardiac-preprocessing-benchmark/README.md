# Cardiac Preprocessing Benchmark (Anonymous Review Package)

This repository contains an anonymized code package for blinded peer review.

## Scope

Research prototype for benchmarking hand-crafted preprocessing strategies in cardiac image segmentation workflows.

- No patient-identifying data are included.
- No author-identifying metadata are intentionally included in this package.
- This is research software and not a clinical diagnostic product.

## Included components

- `cardiac_image_system/core`: data I/O, preprocessing, proxy segmentation, metrics, split validation.
- `cardiac_image_system/models`: compact 2D U-Net implementation.
- `cardiac_image_system/experiments`: preprocessing comparison and ablation entrypoints.
- `tests`: unit tests for core functionality.
- `scripts/create_synthetic_sample.py`: synthetic demo-data generator.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
```

## Run preprocessing comparison

```bash
python -m cardiac_image_system.experiments.run_preprocessing_comparison \
  --manifest data/sample/manifest.csv \
  --output-dir outputs/experiment_001 \
  --modes none gaussian wavelet nlm clahe hybrid
```

## Note for editors

If strict double-blind handling is required, please distribute a source ZIP exported from this repository rather than Git metadata.

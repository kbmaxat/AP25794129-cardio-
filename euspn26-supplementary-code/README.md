# Verifiable Cardiac MRI Data Preparation: Supplementary Code

This directory contains the public supplementary code and non-image audit artifacts for the EUSPN 2026 conference submission:

> **A Verifiable Framework for Reproducible Data Preparation in 3D+t Cardiac MRI**

The release includes source code, tests, pinned dependencies, split definitions, and non-image
result manifests. It deliberately excludes the original and derived medical images.

## Scope of this release

This repository is intended to support auditability and partial reproducibility of the published
pipeline. It contains:

- the dataset parser and preparation workflow,
- deterministic split logic,
- transform and QC implementation,
- tests,
- split and provenance manifests,
- agreement and five-fold QC summaries.

It does **not** contain the original or derived image volumes.

## Source data

Download the Heart Database from:

https://www.laurentnajman.org/heart/H_data.html

The dataset owner publishes this archive MD5:

```text
8ff8a93358a77f1f29e34ae71b4ab281
```

Place the extracted patient directories under:

```text
data/HeartDatabase/Pat01
...
data/HeartDatabase/Pat18
```

## Environment

The reported run used Python 3.12.10 with NumPy 2.4.3, SciPy 1.17.1, and Matplotlib 3.10.8.

```bash
python3.12 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

## Run

Generate the fold-0 reference artifact:

```bash
python -m src.prepare_heart_dataset \
  --dataset-root data/HeartDatabase \
  --output output/heart_prepared \
  --seed 25794129 \
  --folds 5 \
  --fold-index 0 \
  --copies 4
```

The study used the fixed split seed `25794129`, chosen a priori to match the grant identifier
`AP25794129` for traceability. The seed was not tuned against QC or downstream performance results.

Run the unit tests:

```bash
python -m unittest discover -s tests -v
```

Recompute the five-fold QC and injected-failure analysis:

```bash
python run_all_folds_qc.py
```

The pipeline writes patient splits, CSV/JSONL provenance, QC values, transform
parameters, per-item seeds, and SHA-256 checksums. Validation and test samples
are never augmented.

## Included files

- `src/heart_database.py`: dataset discovery, P7/PAM parsing, and sample loading.
- `src/augmentation.py`: paired spatial transforms, intensity perturbations, and QC gates.
- `src/prepare_heart_dataset.py`: fold-0 preparation workflow.
- `run_all_folds_qc.py`: five-fold QC and injected-failure recomputation.
- `tests/test_heart_dataset.py`: parser and preparation tests.
- `requirements.txt`: pinned core dependencies.
- `results/splits.json`: fixed patient-level folds.
- `results/manifest.csv` and `results/manifest.jsonl`: fold-0 provenance.
- `results/expert_agreement.json`: inter-observer Dice.
- `results/all_folds_qc.json`: five-fold QC and injected-failure results.

NPZ container checksums may vary when archive metadata changes; array identity and recorded
transform provenance remain the primary deterministic checks.

## Data availability and redistribution

The Heart Database is distributed by its original maintainers. Because the source dataset does not
provide a standard redistribution license for the image volumes, this repository does not mirror
the original archive or any derived MRI/label volumes. Users should obtain the source data directly
from the dataset page listed above.

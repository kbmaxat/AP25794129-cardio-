# AP25794129 Cardio Preprocessing Research Prototype

Research prototype for filtering and preprocessing cardiovascular biomedical images and evaluating their effect on downstream image analysis.

This repository supports the technical implementation of the PhD/grant project:

**AP25794129 — Development of an algorithm for filtering and preprocessing biomedical images for cardiological diagnostics**

## Important scope

This is a **research prototype**, not a certified medical diagnostic system.

The project is intended for:

- controlled preprocessing experiments;
- image-quality metric calculation;
- structure-level proxy segmentation;
- patient-level experiment tracing;
- reproducible CSV-based reporting;
- future integration with FastAPI/web interface.

Do not upload identifiable patient data, DICOM files with personal metadata, or real clinical data into GitHub.

## Core dissertation logic

```text
input image -> preprocessing mode -> optional proxy segmentation -> metrics -> patient-level aggregation -> experiment report
```

The main goal is not autonomous diagnosis. The goal is to evaluate whether preprocessing changes image-level and downstream metrics under a reproducible protocol.

## Repository structure

```text
ap25794129-cardio-preprocessing/
├── cardiac_image_system/
│   ├── core/
│   │   ├── io.py
│   │   ├── preprocessing.py
│   │   ├── segmentation.py
│   │   ├── metrics.py
│   │   ├── validation.py
│   │   └── manifest.py
│   └── experiments/
│       ├── run_preprocessing_comparison.py
│       └── run_ablation.py
├── backend/
│   └── app/
│       └── main.py
├── data/
│   ├── sample/
│   └── README.md
├── docs/
│   ├── architecture.md
│   ├── data_contract.md
│   ├── experiment_protocol.md
│   └── codex_tasks.md
├── scripts/
│   └── create_synthetic_sample.py
├── tests/
├── requirements.txt
├── pyproject.toml
└── .gitignore
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
```

Run FastAPI demo:

```bash
uvicorn backend.app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

## Create synthetic demo data

```bash
python scripts/create_synthetic_sample.py
```

## Run preprocessing comparison

```bash
python -m cardiac_image_system.experiments.run_preprocessing_comparison   --manifest data/sample/manifest.csv   --output-dir outputs/experiment_001   --modes none gaussian wavelet nlm clahe hybrid
```

## Preprocessing modes

- `none`
- `gaussian`
- `wavelet`
- `nlm`
- `clahe`
- `hybrid` = wavelet + NLM + CLAHE

## Metrics

- PSNR
- SSIM
- Dice
- IoU
- HD95
- relative area error

## Next development steps for Codex

1. Add DICOM/NIfTI loaders.
2. Add patient-level train/validation/test split generator.
3. Add U-Net locked inference experiment.
4. Add mode-specific U-Net training experiment.
5. Add bootstrap confidence intervals and Wilcoxon/Holm reporting.
6. Add GitHub Actions tests.
7. Add a minimal web UI.

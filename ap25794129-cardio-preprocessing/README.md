# AP25794129 Cardio Preprocessing Research Prototype

Research prototype for filtering and preprocessing cardiovascular biomedical images and evaluating their effect on downstream image analysis.

This repository supports the technical implementation of the PhD/grant project:

**AP25794129 — Development of an algorithm for filtering and preprocessing biomedical images for cardiological diagnostics**

## Project website

- https://maxatlab.kz/en/

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

Tested with Python 3.12. `requirements.txt` pins exact versions that are verified to install
and pass the full test suite together.

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

1. Add locked external-validation benchmarks across ACDC and CAMUS.
2. Add mode-specific U-Net training experiment with statistical testing.
3. Add bootstrap confidence intervals and Wilcoxon/Holm reporting.
4. Add GitHub Actions tests.
5. Add a minimal web UI.

## U-Net baseline experiment

Run a compact downstream segmentation baseline on a manifest:

```bash
python -m cardiac_image_system.experiments.train_unet_baseline ^
  --manifest data/manifests_local/segmentation_public_combined.csv ^
  --output-dir outputs/unet_baseline_combined_none ^
  --mode none ^
  --epochs 10 ^
  --batch-size 8
```

Useful options:

- `--dataset-filter ACDC`
- `--dataset-filter CAMUS`
- `--mode gaussian|wavelet|nlm|clahe|hybrid`
- `--max-train-samples 128 --max-val-samples 64 --max-test-samples 64` for smoke tests

Main outputs:

- `resolved_splits/*.csv`
- `history.csv`
- `test_slice_level.csv`
- `test_patient_level.csv`
- `summary.json`
- `checkpoint_best.pt`

Run all preprocessing modes:

```powershell
.\scripts\run_unet_mode_grid.ps1 -DatasetFilter ACDC -Epochs 10 -BatchSize 8
```

Each grid run is written into a unique timestamped session folder under `outputs/unet_mode_grid/`.

Run the remaining public segmentation suite after an active ACDC job finishes:

```powershell
.\scripts\run_public_segmentation_suite.ps1 -AfterProcessId 9736 -Epochs 10 -BatchSize 8
```

This follow-up script runs:

1. `CAMUS`
2. `ACDC + CAMUS` combined

## Reviewer-driven binary follow-up scripts

Run the top-3 binary modes (`none`, `wavelet`, `nlm`) across multiple seeds on CAMUS:

```powershell
.\scripts\run_unet_camus_multiseed_top3.ps1
```

Run the same multi-seed sweep on the mixed `ACDC + CAMUS` corpus:

```powershell
.\scripts\run_unet_combined_multiseed_top3.ps1
```

Run a long-schedule CAMUS follow-up with early stopping:

```powershell
.\scripts\run_unet_camus_longschedule_top3.ps1
```

Run the same long-schedule follow-up on the mixed corpus:

```powershell
.\scripts\run_unet_combined_longschedule_top3.ps1
```

Summarize a completed multi-seed session:

```bash
python scripts/summarize_unet_binary_seed_sweep.py \
  --session-root outputs/unet_binary_multiseed_camus_top3/20260702_120000
```

## Statistical comparison across preprocessing modes

`cardiac_image_system/core/stats.py` implements the paired-comparison statistics
(bootstrap confidence intervals, Holm-adjusted Wilcoxon signed-rank tests, and TOST
equivalence testing) as version-controlled, unit-tested code, rather than as numbers
transcribed by hand. Given a `test_patient_level.csv` per mode (produced by
`train_unet_baseline` / `train_unet_multiclass`), it reproduces a table3-style
mode-vs-baseline comparison:

```bash
python scripts/compare_preprocessing_modes.py \
  --run none=outputs/unet_baseline_combined_none \
  --run wavelet=outputs/unet_baseline_combined_wavelet \
  --run nlm=outputs/unet_baseline_combined_nlm \
  --baseline-mode none \
  --metric dice \
  --output-csv outputs/mode_comparison_dice.csv
```

The TOST (two one-sided tests) check reports whether a mode is statistically equivalent
to the baseline within a chosen margin — this is the appropriate test for a "preprocessing
does not help" claim, since a non-significant Wilcoxon/Holm result on its own only means
failure to reject the null, not proof of equivalence.

## Final manuscript results (RadiologyAI submission)

The finalized benchmark results extracted from `RadiologyAI_Main_Manuscript_Submission.docx` are available here:

- `docs/final_results_radiology_ai.md`
- `docs/results/table2_best_mode_summary.csv`
- `docs/results/table3_dice_vs_none_inference.csv`
- `docs/results/table4_acdc_multiclass_top3.csv`

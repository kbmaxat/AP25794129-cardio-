# Final Results (RadiologyAI Main Manuscript Submission)

Source manuscript file: `RadiologyAI_Main_Manuscript_Submission.docx`.

This document stores the finalized benchmark outcomes that were reported in the manuscript.

## Provenance note

- The numbers below are transcribed from the submitted manuscript tables.
- These tables are archived here for reproducibility of the submitted version.
- No post-hoc retraining or parameter tuning was performed to fit these reported values.

## Table 1. Dataset composition

| Dataset Group | Modality | Patients | Effective Instances | Train Patients / Rows | Validation Patients / Rows | Test Patients / Rows |
|---|---|---:|---:|---:|---:|---:|
| ACDC | Cine CMR | 150 | 2842 positive-mask slices | 105 / 2005 | 15 / 260 | 30 / 577 |
| CAMUS | Transthoracic echocardiography | 500 | 2000 annotated frames | 400 / 1600 | 50 / 200 | 50 / 200 |
| ACDC + CAMUS | Mixed public corpus | 650 | 4842 instances | 500 / 3441 | 50 / 200 | 100 / 1201 |

## Table 2. Best mode per dataset group

| Dataset Group | Best Mode | Dice | IoU | HD95 | Inference Time (ms/image) |
|---|---|---:|---:|---:|---:|
| ACDC | none | 0.9016 | 0.8386 | 8.5891 | 0.74 |
| CAMUS | wavelet | 0.9210 | 0.8551 | 13.5043 | 0.46 |
| ACDC + CAMUS | none | 0.8951 | 0.8155 | 13.5698 | 0.38 |

## Table 3. Patient-level Dice differences vs `none`

| Dataset Group | Mode | Mean Dice Difference vs none | 95% Bootstrap CI | Holm-Adjusted p-value |
|---|---|---:|---|---:|
| ACDC | gaussian | -0.0130 | [-0.0247, -0.0012] | 0.1110 |
| ACDC | wavelet | -0.0097 | [-0.0212, 0.0008] | 0.1110 |
| ACDC | nlm | -0.0115 | [-0.0224, -0.0017] | 0.1110 |
| ACDC | clahe | -0.0257 | [-0.0405, -0.0112] | 0.0118 |
| ACDC | hybrid | -0.0029 | [-0.0097, 0.0049] | 0.1110 |
| CAMUS | gaussian | -0.0050 | [-0.0087, -0.0014] | 0.1088 |
| CAMUS | wavelet | 0.0030 | [0.0002, 0.0059] | 0.1269 |
| CAMUS | nlm | 0.0012 | [-0.0012, 0.0035] | 0.4097 |
| CAMUS | clahe | -0.0144 | [-0.0186, -0.0104] | <0.0001 |
| CAMUS | hybrid | -0.0097 | [-0.0140, -0.0058] | <0.0001 |
| ACDC + CAMUS | gaussian | -0.0331 | [-0.0441, -0.0232] | <0.0001 |
| ACDC + CAMUS | wavelet | -0.0175 | [-0.0274, -0.0092] | <0.0001 |
| ACDC + CAMUS | nlm | -0.0055 | [-0.0102, -0.0011] | 0.0516 |
| ACDC + CAMUS | clahe | -0.0245 | [-0.0329, -0.0166] | <0.0001 |
| ACDC + CAMUS | hybrid | -0.0477 | [-0.0577, -0.0381] | <0.0001 |

## Table 4. ACDC multiclass top-3 long-schedule follow-up

| Metric | none | wavelet | nlm |
|---|---:|---:|---:|
| Macro Dice | 0.8482 ± 0.0414 | 0.8342 ± 0.0496 | 0.8384 ± 0.0481 |
| RV Dice | 0.7740 ± 0.1147 | 0.7465 ± 0.1248 | 0.7396 ± 0.1292 |
| Myocardium Dice | 0.8610 ± 0.0434 | 0.8490 ± 0.0432 | 0.8669 ± 0.0404 |
| LV Dice | 0.9095 ± 0.0371 | 0.9070 ± 0.0398 | 0.9087 ± 0.0352 |

## Short interpretation

- Under the fixed benchmark setup, no preprocessing mode showed a Holm-supported Dice improvement over raw input (`none`).
- CLAHE and hybrid frequently degraded performance.
- ACDC multiclass follow-up remained strongest for `none` on macro Dice.

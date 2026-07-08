# Data Contract

## Manifest columns

| Column | Required | Description |
|---|---|---|
| patient_id | yes | anonymized independent patient identifier |
| phase | yes | diastole/systole/other |
| image_path | yes | local path to image |
| mask_path | yes | local path to binary expert/proxy mask |

## Optional manifest columns

| Column | Description |
|---|---|
| source_patient_id | original dataset-local patient folder name |
| dataset | source dataset label such as `ACDC` or `CAMUS` |
| subset | official or generated split label |
| modality | imaging modality |
| view | acquisition view, for example `SAX`, `2CH`, `4CH` |
| slice_index | slice to extract from a 3D NIfTI volume |
| frame_id | original frame code or phase frame number |
| group | dataset-provided disease group or cohort label |
| has_positive_mask | convenience flag for positive-structure slices |

## Patient-level rule

The same `patient_id` must never appear in both train and test splits.

For combined multi-dataset manifests, `patient_id` must be globally unique.  
Recommended pattern:

- `ACDC_patient001`
- `CAMUS_patient0001`

## Real medical data

Real clinical files must remain outside GitHub. Use local paths and private storage only.

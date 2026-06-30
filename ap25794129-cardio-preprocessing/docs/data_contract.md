# Data Contract

## Manifest columns

| Column | Required | Description |
|---|---|---|
| patient_id | yes | anonymized independent patient identifier |
| phase | yes | diastole/systole/other |
| image_path | yes | local path to image |
| mask_path | yes | local path to binary expert/proxy mask |

## Patient-level rule

The same `patient_id` must never appear in both train and test splits.

## Real medical data

Real clinical files must remain outside GitHub. Use local paths and private storage only.

# Dataset preparation

Images are not redistributed. Download the public datasets from their official providers and create a CSV manifest with:

`path,label,patient_id,modality,dataset`

- `label`: `0` for normal and `1` for pathology/risk.
- `patient_id`: stable subject identifier; all images from a subject stay in one fold.
- `modality`: `echo` or `mri`.
- `dataset`: source name such as `CAMUS`, `ACDC`, or the future internal cohort.

For CAMUS, the manuscript uses LVEF above 55% as normal and below 45% as risk; cases from 45–55% are excluded. For ACDC, the healthy class is normal and the disease groups are mapped to pathology. Verify these mappings against the licenses and metadata supplied with the downloaded datasets.

Place local data below `data/raw/`; this directory is ignored by Git. The ACDC and CAMUS datasets are publicly available from their respective repositories. The internal clinical dataset is not publicly available because of institutional and data-protection restrictions; authorized users can connect it by appending manifest rows without changing the code.

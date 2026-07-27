# Data policy

Do not commit real medical data to GitHub.

Allowed:

- tiny synthetic images;
- fully anonymized demo images;
- toy masks;
- CSV manifests with fake/demo paths.

Forbidden:

- identifiable DICOM files;
- patient names;
- patient IDs from clinics;
- raw clinical exports;
- private hospital datasets.

For real experiments, keep data outside the repository and reference it through local manifests.

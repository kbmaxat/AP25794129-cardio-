# Codex Task List

Use small tasks. Do not ask Codex to build the entire project at once.

## Task 1 — Repository audit

Study the repository and produce a short report:
- what works;
- what is missing;
- what should be implemented next.

## Task 2 — Add tests for preprocessing

Add unit tests for:
- normalize_minmax;
- gaussian;
- wavelet;
- nlm;
- clahe;
- hybrid.

## Task 3 — Improve experiment runner

Add:
- parameter JSON logging;
- config file support;
- error logging for corrupted images;
- runtime measurement.

## Task 4 — Add statistical report

Implement:
- patient-level aggregation;
- Wilcoxon paired tests;
- Holm correction;
- bootstrap confidence intervals.

## Task 5 — Add U-Net locked inference placeholder

Create structure for:
- model loading;
- fold definition;
- locked inference;
- Dice/HD95 reporting.

## Task 6 — Add GitHub Actions

Add CI to run pytest on every pull request.

## Rule

Every Codex task must preserve the statement:
“This is a research prototype, not clinical diagnostic software.”

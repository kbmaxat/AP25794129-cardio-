import importlib.util
import json
from pathlib import Path

import pytest


SOURCE = Path(__file__).resolve().parents[2] / "research/temporal_v01/export_pilot.py"
CODE_SHA = "a" * 40
spec = importlib.util.spec_from_file_location("temporal_export", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_export_rejects_incomplete_run(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    module.write_json(run / "status.json", {"status": "failed"})
    with pytest.raises(ValueError, match="completed"):
        module.export_numeric(run, tmp_path / "public", CODE_SHA)
    assert not (tmp_path / "public").exists()


def test_export_excludes_private_text_and_paths(tmp_path):
    run, public = tmp_path / "run", tmp_path / "public"
    run.mkdir()
    marker = "PRIVATE_CONTENT_MUST_NOT_APPEAR"
    module.write_json(run / "status.json", {"status": "completed", "notes": marker})
    module.write_json(run / "protocol.json", {"dataset_root": marker, "train_patients": 32, "dev_patients": 8, "epochs": 12})
    module.write_json(run / "environment.json", {**dict.fromkeys(module.ENV_KEYS, "value"), "executable": marker})
    module.write_json(run / "source_hashes.json", {"method.py": "a" * 64, "private.md": marker})
    module.write_json(run / "patient_split.json", {"train": [marker]})
    (run / "journal.md").write_text(marker)
    fields = ("mode", "stage", "best_epoch", "dev_loss", "dev_patient_dice", "dev_n_patients", "training_seconds",
              "gpu_peak_allocated_bytes", "parameters", "preprocessor_parameters", "mean_abs_correction",
              "max_abs_correction", "initial_segmenter_sha256", "checkpoint_sha256")
    for mode in module.MODES:
        folder = run / mode
        folder.mkdir()
        module.write_json(folder / "summary.json", {**dict.fromkeys(fields, 0), "mode": mode, "notes": marker})
        module.write_csv(folder / "history.csv", [{"epoch": 1, "train_loss": 1, "dev_loss": 1, "dev_patient_dice": .5, "notes": marker}])
    module.export_numeric(run, public, CODE_SHA)
    assert all(marker not in p.read_text(encoding="utf-8") for p in public.iterdir())
    assert json.loads((public / "config.json").read_text())["dataset_root"] is None
    assert json.loads((public / "metadata.json").read_text())["github_code_commit_format_validated"]
    assert {p.suffix for p in public.iterdir()} == {".csv", ".json"}
    assert all(b"\r\n" not in p.read_bytes() for p in public.iterdir())
    manifest = module.read_json(public / "files_sha256.json")
    assert all(module.sha256(public / name) == value for name, value in manifest.items())


def test_replay_compares_all_artifacts(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    names = ["patient_split.json", "data_manifest.json", "protocol.json"]
    names += [f"{mode}/{name}" for mode in module.MODES
              for name in ("history.csv", "dev_phase_metrics.csv", "dev_patient_metrics.csv", "best.pt")]
    for folder in (first, second):
        for name in names:
            target = folder / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"identical")
    assert module.verify_replay(first, second)["all_identical"]
    (second / "temporal/best.pt").write_bytes(b"changed")
    result = module.verify_replay(first, second)
    assert not result["all_identical"]
    assert sum(not row["identical"] for row in result["checks"]) == 1


def test_export_rejects_unverifiable_commit_identifier(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    module.write_json(run / "status.json", {"status": "completed"})
    with pytest.raises(ValueError, match="40-character"):
        module.export_numeric(run, tmp_path / "public", "commit")
    assert not (tmp_path / "public").exists()

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODES = ("none", "direct_temporal", "spatial", "temporal")
ENV_KEYS = ("torch", "cuda_runtime", "numpy", "opencv", "gpu", "base_commit", "protocol_sha256")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8", newline="\n")


def csv_rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def verify_replay(first, replay):
    rows = []
    for relative in ("patient_split.json", "data_manifest.json", "protocol.json"):
        rows.append({"artifact": relative, "identical": sha256(first / relative) == sha256(replay / relative)})
    for mode in MODES:
        for name in ("history.csv", "dev_phase_metrics.csv", "dev_patient_metrics.csv", "best.pt"):
            relative = f"{mode}/{name}"
            rows.append({"artifact": relative, "identical": sha256(first / relative) == sha256(replay / relative)})
    return {"first_run": first.name, "replay_run": replay.name, "checks": rows,
            "all_identical": all(row["identical"] for row in rows)}


def export_numeric(run, destination, remote_code_commit):
    if read_json(run / "status.json")["status"] != "completed":
        raise ValueError("Only completed runs can be exported")
    destination.mkdir(parents=True, exist_ok=False)
    config = read_json(run / "protocol.json")
    config["dataset_root"] = None
    write_json(destination / "config.json", config)
    summary_fields = ("mode", "stage", "best_epoch", "dev_loss", "dev_patient_dice", "dev_n_patients",
                      "training_seconds", "gpu_peak_allocated_bytes", "parameters", "preprocessor_parameters",
                      "mean_abs_correction", "max_abs_correction", "initial_segmenter_sha256", "checkpoint_sha256")
    summaries, histories = [], []
    for mode in MODES:
        summary = read_json(run / mode / "summary.json")
        summaries.append({key: summary[key] for key in summary_fields})
        for row in csv_rows(run / mode / "history.csv"):
            histories.append({"mode": mode, **{key: row[key] for key in
                             ("epoch", "train_loss", "dev_loss", "dev_patient_dice")}})
    write_csv(destination / "summary.csv", summaries)
    write_csv(destination / "history.csv", histories)
    environment = read_json(run / "environment.json")
    source_hashes = read_json(run / "source_hashes.json")
    metadata = {
        "run_id": run.name, "stage": "technical_pilot_not_confirmatory", "dataset": "CAMUS",
        "dataset_partition_access": "official_training_only", "test_evaluation": False,
        "n_train_patients": config["train_patients"], "n_dev_patients": config["dev_patients"],
        "n_training_seeds": 1, "epochs": config["epochs"],
        "environment": {key: environment[key] for key in ENV_KEYS},
        "github_code_commit": remote_code_commit,
        "patient_split_sha256": sha256(run / "patient_split.json"),
        "code_source_sha256": {key.replace("\\", "/"): value for key, value in source_hashes.items()
                               if key.endswith(".py")},
    }
    write_json(destination / "metadata.json", metadata)
    write_json(destination / "files_sha256.json", {p.name: sha256(p) for p in sorted(destination.iterdir()) if p.is_file()})


def plot_local(run):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    output = HERE / "private" / run.name
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for mode in MODES:
        rows = csv_rows(run / mode / "history.csv")
        epochs = [int(row["epoch"]) for row in rows]
        for axis, key in zip(axes, ("dev_loss", "dev_patient_dice")):
            axis.plot(epochs, [float(row[key]) for row in rows], label=mode)
    for axis, label in zip(axes, ("Development loss", "Development patient-mean Dice")):
        axis.set(xlabel="Epoch", ylabel=label)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    fig.suptitle("CAMUS technical pilot | train 32 / dev 8 | one seed | no test evaluation", fontsize=11)
    fig.savefig(output / "learning_curves.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(3, 4, figsize=(11, 8), layout="constrained")
    for column, mode in enumerate(MODES):
        example = np.load(run / mode / "fixed_dev_example.npz")
        axes[0, column].imshow(example["processed"], cmap="gray", vmin=0, vmax=1)
        axes[0, column].set_title(mode, fontsize=11)
        correction = axes[1, column].imshow(example["correction"], cmap="RdBu_r", vmin=-0.01, vmax=0.01)
        axes[2, column].imshow(example["raw"][0], cmap="gray", vmin=0, vmax=1)
        axes[2, column].contour(example["mask"], levels=[0.5], colors=["#00bfa5"], linewidths=1)
        axes[2, column].contour(example["prediction"], levels=[0.5], colors=["#ef476f"], linewidths=1)
        for row in range(3):
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
    for row, label in enumerate(("Processed image", "Intensity change", "GT teal / prediction red")):
        axes[row, 0].set_ylabel(label, fontsize=10)
    fig.colorbar(correction, ax=list(axes[1]), shrink=0.75, label="Intensity change (shared scale)")
    fig.suptitle("Fixed first development case | ED | not selected by performance", fontsize=12)
    fig.savefig(output / "fixed_case.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--github-code-commit", required=True)
    args = parser.parse_args()
    check = verify_replay(args.run, args.replay)
    destination = HERE / "public_results" / args.run.name
    export_numeric(args.run, destination, args.github_code_commit)
    write_json(destination / "reproducibility.json", check)
    write_json(destination / "files_sha256.json", {p.name: sha256(p) for p in sorted(destination.iterdir())
                                                 if p.name != "files_sha256.json"})
    plot_local(args.run)
    print(json.dumps({"exported": str(destination), "replay_identical": check["all_identical"]}))
    if not check["all_identical"]:
        raise SystemExit("Replay mismatch: inspect reproducibility.json")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import ExperimentConfig
from .data import assign_grouped_folds, load_manifest


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(prog="joig-cardio")
    sub = command.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate-manifest")
    validate.add_argument("manifest"); validate.add_argument("--skip-file-check", action="store_true")
    train = sub.add_parser("cross-validate")
    train.add_argument("manifest"); train.add_argument("--config", default="configs/default.json"); train.add_argument("--output", default="outputs/cv")
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("manifest"); evaluate.add_argument("checkpoint"); evaluate.add_argument("--config", default="configs/default.json"); evaluate.add_argument("--output", default="outputs/external")
    return command


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if args.command == "validate-manifest":
        frame = load_manifest(args.manifest, check_files=not args.skip_file_check)
        print(json.dumps({"images": len(frame), "patients": frame.patient_id.nunique(), "datasets": sorted(frame.dataset.unique())}, indent=2))
        return 0
    config = ExperimentConfig.load(args.config)
    frame = load_manifest(args.manifest)
    if args.command == "cross-validate":
        from .engine import train_cross_validation
        frame = assign_grouped_folds(frame, config.folds, config.seed)
        train_cross_validation(frame, config, args.output)
    else:
        from .engine import evaluate_checkpoint
        metrics = evaluate_checkpoint(frame, config, args.checkpoint, args.output)
        print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
from pathlib import Path

from cardiac_image_system.experiments.run_preprocessing_comparison import run_experiment

ABLATION_MODES = ["none", "wavelet", "nlm", "clahe", "hybrid"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    run_experiment(args.manifest, args.output_dir, ABLATION_MODES)


if __name__ == "__main__":
    main()

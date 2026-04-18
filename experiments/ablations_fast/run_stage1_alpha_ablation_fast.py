"""FAST ablation: Stage 1 allocation strength."""

import argparse
import os

from common_fast import load_base_config, run_experiments


def main():
    parser = argparse.ArgumentParser(description="FAST Stage1 alpha ablation")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--alphas", type=str, default="0,0.5,1.0")
    parser.add_argument("--save-root", type=str,
                        default="./experiments/ablations_fast/results/stage1_alpha")
    args = parser.parse_args()

    base_cfg = load_base_config(args.config)
    alphas = [float(a.strip()) for a in args.alphas.split(',') if a.strip()]

    experiments = []
    for alpha in alphas:
        name = f"alpha_{alpha:g}"
        overrides = {"uncertainty_alpha": alpha}
        experiments.append({"name": name, "overrides": overrides})

    summary_path = os.path.join(args.save_root, "summary.csv")
    run_experiments(base_cfg, experiments, args.save_root, summary_path)


if __name__ == "__main__":
    main()

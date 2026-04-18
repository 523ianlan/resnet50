"""FAST ablation: Stage 2 first-order mode comparison."""

import argparse
import os

from common_fast import load_base_config, run_experiments


def main():
    parser = argparse.ArgumentParser(description="FAST Stage2 first-order ablation")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--save-root", type=str,
                        default="./experiments/ablations_fast/results/stage2_first_order")
    args = parser.parse_args()

    base_cfg = load_base_config(args.config)

    variants = [
        ("mean_abs", "mean_abs"),
        ("abs_mean", "abs_mean"),
    ]

    experiments = []
    for name, mode in variants:
        overrides = {"fisher_first_order_mode": mode}
        experiments.append({"name": name, "overrides": overrides})

    summary_path = os.path.join(args.save_root, "summary.csv")
    run_experiments(base_cfg, experiments, args.save_root, summary_path)


if __name__ == "__main__":
    main()

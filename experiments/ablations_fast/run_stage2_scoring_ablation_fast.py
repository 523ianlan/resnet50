"""FAST ablation: Stage 2 scoring weights and mode."""

import argparse
import os

from common_fast import load_base_config, run_experiments


def main():
    parser = argparse.ArgumentParser(description="FAST Stage2 scoring ablation")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--save-root", type=str,
                        default="./experiments/ablations_fast/results/stage2_scoring")
    args = parser.parse_args()

    base_cfg = load_base_config(args.config)

    variants = [
        ("fisher_default", True, 1.0, 0.5),
        ("first_only", True, 1.0, 0.0),
        ("second_only", True, 0.0, 1.0),
        ("equal_weights", True, 1.0, 1.0),
        ("magnitude_only", False, 0.0, 0.0),
    ]

    experiments = []
    for name, use_fisher, w1, w2 in variants:
        overrides = {
            "use_fisher_scores": use_fisher,
            "fisher_first_order_weight": w1,
            "fisher_second_order_weight": w2,
        }
        experiments.append({"name": name, "overrides": overrides})

    summary_path = os.path.join(args.save_root, "summary.csv")
    run_experiments(base_cfg, experiments, args.save_root, summary_path)


if __name__ == "__main__":
    main()

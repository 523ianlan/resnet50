"""FAST ablation: Stage 1 numeric stability switches."""

import argparse
import os

from common_fast import load_base_config, run_experiments


def main():
    parser = argparse.ArgumentParser(description="FAST Stage1 stability ablation")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--save-root", type=str,
                        default="./experiments/ablations_fast/results/stage1_stability")
    args = parser.parse_args()

    base_cfg = load_base_config(args.config)

    variants = [
        ("no_stab", False, 0.0, 0.0),
        ("var_floor", False, 0.0, 1e-6),
        ("log_only", True, 0.0, 0.0),
        ("clip_only", False, 1.0, 0.0),
        ("log_clip", True, 1.0, 0.0),
        ("log_clip_floor", True, 1.0, 1e-6),
    ]

    experiments = []
    for name, log_on, clip_pct, var_floor in variants:
        overrides = {
            "uncertainty_log": log_on,
            "uncertainty_clip_percentile": clip_pct,
            "uncertainty_var_floor": var_floor,
        }
        experiments.append({"name": name, "overrides": overrides})

    summary_path = os.path.join(args.save_root, "summary.csv")
    run_experiments(base_cfg, experiments, args.save_root, summary_path)


if __name__ == "__main__":
    main()

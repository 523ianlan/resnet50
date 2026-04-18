"""Table: Effect of uncertainty normalization knobs (ResNet-50)."""

import argparse
import os

from common import load_base_config, run_experiments


def main():
    parser = argparse.ArgumentParser(description="Uncertainty normalization ablation (ResNet-50)")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--save-root", type=str,
                        default="./experiments/paper_tables/results/ablation_uncertainty_norm")
    parser.add_argument("--target-compression", type=float, default=0.4)
    parser.add_argument("--clip-percentile", type=float, default=1.0)
    parser.add_argument("--var-floor", type=float, default=1e-6)
    args = parser.parse_args()

    base_cfg = load_base_config(args.config)

    base_overrides = {
        "target_compression": args.target_compression,
        "uncertainty_alpha": 1.0,
        "allocation_strategy": "binary_search",
        "stage2_score_metric": "fisher",
        "use_fisher_scores": True,
    }

    experiments = [
        {
            "name": "log_only",
            "overrides": {
                **base_overrides,
                "uncertainty_log": True,
                "uncertainty_clip_percentile": 0.0,
                "uncertainty_var_floor": 0.0,
            },
        },
        {
            "name": "clip_only",
            "overrides": {
                **base_overrides,
                "uncertainty_log": False,
                "uncertainty_clip_percentile": args.clip_percentile,
                "uncertainty_var_floor": 0.0,
            },
        },
        {
            "name": "var_floor_only",
            "overrides": {
                **base_overrides,
                "uncertainty_log": False,
                "uncertainty_clip_percentile": 0.0,
                "uncertainty_var_floor": args.var_floor,
            },
        },
        {
            "name": "all_on",
            "overrides": {
                **base_overrides,
                "uncertainty_log": True,
                "uncertainty_clip_percentile": args.clip_percentile,
                "uncertainty_var_floor": args.var_floor,
            },
        },
    ]

    summary_path = os.path.join(args.save_root, "summary.csv")
    run_experiments(base_cfg, experiments, args.save_root, summary_path)


if __name__ == "__main__":
    main()

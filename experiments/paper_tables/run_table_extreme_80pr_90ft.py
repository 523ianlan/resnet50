"""Table: Extreme test at 80% compression with 90-epoch fine-tuning (ResNet-50)."""

import argparse
import os

from common import load_base_config, run_experiments


def main():
    parser = argparse.ArgumentParser(description="Extreme 80% pruning, 90-epoch FT")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--save-root", type=str,
                        default="./experiments/paper_tables/results/extreme_80pr_90ft")
    parser.add_argument("--target-compression", type=float, default=0.8)
    parser.add_argument("--fine-tune-epochs", type=int, default=90)
    args = parser.parse_args()

    base_cfg = load_base_config(args.config)

    base_overrides = {
        "target_compression": args.target_compression,
        "fine_tune_epochs": args.fine_tune_epochs,
        "uncertainty_alpha": 1.0,
        "allocation_strategy": "binary_search",
        "stage2_score_metric": "fisher",
        "use_fisher_scores": True,
    }

    experiments = [
        {
            "name": "global_fisher",
            "overrides": {
                **base_overrides,
                "allocation_strategy": "global_fisher",
            },
        },
        {
            "name": "mc_dropout_raw_norm",
            "overrides": {
                **base_overrides,
                "uncertainty_log": False,
                "uncertainty_clip_percentile": 0.0,
                "uncertainty_var_floor": 0.0,
            },
        },
        {
            "name": "mc_dropout_full",
            "overrides": {
                **base_overrides,
                "uncertainty_log": True,
                "uncertainty_clip_percentile": float(getattr(base_cfg, "uncertainty_clip_percentile", 0.0)),
                "uncertainty_var_floor": float(getattr(base_cfg, "uncertainty_var_floor", 0.0)),
            },
        },
    ]

    summary_path = os.path.join(args.save_root, "summary.csv")
    run_experiments(base_cfg, experiments, args.save_root, summary_path, use_paper_defaults=False)


if __name__ == "__main__":
    main()

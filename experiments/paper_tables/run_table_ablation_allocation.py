"""Table: Allocation strategies under equal FLOPs budget (ResNet-50)."""

import argparse
import os

from common import load_base_config, run_experiments


def main():
    parser = argparse.ArgumentParser(description="Allocation strategy ablation (ResNet-50)")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--save-root", type=str,
                        default="./experiments/paper_tables/results/ablation_allocation")
    parser.add_argument("--target-compression", type=float, default=0.4)
    args = parser.parse_args()

    base_cfg = load_base_config(args.config)

    experiments = [
        {
            "name": "uniform",
            "overrides": {
                "target_compression": args.target_compression,
                "uncertainty_alpha": 0.0,
                "allocation_strategy": "binary_search",
                "stage2_score_metric": "fisher",
            },
        },
        {
            "name": "global_fisher",
            "overrides": {
                "target_compression": args.target_compression,
                "allocation_strategy": "global_fisher",
                "stage2_score_metric": "fisher",
            },
        },
        {
            "name": "ufalp_full",
            "overrides": {
                "target_compression": args.target_compression,
                "uncertainty_alpha": 1.0,
                "allocation_strategy": "binary_search",
                "stage2_score_metric": "fisher",
            },
        },
    ]

    summary_path = os.path.join(args.save_root, "summary.csv")
    run_experiments(base_cfg, experiments, args.save_root, summary_path)


if __name__ == "__main__":
    main()

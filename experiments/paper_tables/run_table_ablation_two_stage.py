"""Table: Ablation on two-stage design (ResNet-50)."""

import argparse
import os

from common import load_base_config, run_experiments


def main():
    parser = argparse.ArgumentParser(description="Two-stage ablation (ResNet-50)")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--save-root", type=str,
                        default="./experiments/paper_tables/results/ablation_two_stage")
    parser.add_argument("--target-compression", type=float, default=0.4)
    args = parser.parse_args()

    base_cfg = load_base_config(args.config)

    experiments = [
        # {
        #     "name": "wo_uncertainty",
        #     "overrides": {
        #         "target_compression": args.target_compression,
        #         "uncertainty_alpha": 0.0,
        #         "use_fisher_scores": True,
        #         "stage2_score_metric": "fisher",
        #     },
        # },
        {
            "name": "wo_fisher",
            "overrides": {
                "target_compression": args.target_compression,
                "uncertainty_alpha": 1.0,
                "use_fisher_scores": False,
                "stage2_score_metric": "magnitude",
            },
        },
        {
            "name": "ufalp_full",
            "overrides": {
                "target_compression": args.target_compression,
                "uncertainty_alpha": 1.0,
                "use_fisher_scores": True,
                "stage2_score_metric": "fisher",
            },
        },
    ]

    summary_path = os.path.join(args.save_root, "summary.csv")
    run_experiments(base_cfg, experiments, args.save_root, summary_path)


if __name__ == "__main__":
    main()

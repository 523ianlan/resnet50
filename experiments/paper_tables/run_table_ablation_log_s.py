"""Table: Effect of log-singular parameterization (ResNet-50)."""

import argparse
import os

from common import load_base_config, run_experiments


def main():
    parser = argparse.ArgumentParser(description="Log-s ablation (ResNet-50)")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--save-root", type=str,
                        default="./experiments/paper_tables/results/ablation_log_s")
    parser.add_argument("--target-compression", type=float, default=0.4)
    args = parser.parse_args()

    base_cfg = load_base_config(args.config)

    experiments = [
        {
            "name": "direct_sigma",
            "overrides": {
                "target_compression": args.target_compression,
                "use_log_s": False,
            },
        },
        {
            "name": "log_sigma",
            "overrides": {
                "target_compression": args.target_compression,
                "use_log_s": True,
            },
        },
    ]

    summary_path = os.path.join(args.save_root, "summary.csv")
    run_experiments(base_cfg, experiments, args.save_root, summary_path)


if __name__ == "__main__":
    main()

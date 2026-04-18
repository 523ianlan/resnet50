"""Ablation: Stage 1 metric choice for layer importance."""

import argparse
import os

from sweep_common import load_base_config, run_experiments


def main():
    parser = argparse.ArgumentParser(description="Stage1 metric ablation")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--metrics", type=str, default="mu_over_var,mu,var,inv_var")
    parser.add_argument("--save-root", type=str,
                        default="./experiments/ablations/results/stage1_metrics")
    args = parser.parse_args()

    base_cfg = load_base_config(args.config)
    metrics = [m.strip() for m in args.metrics.split(',') if m.strip()]

    experiments = []
    for metric in metrics:
        name = f"metric_{metric}"
        overrides = {"uncertainty_metric": metric}
        experiments.append({"name": name, "overrides": overrides})

    summary_path = os.path.join(args.save_root, "summary.csv")
    run_experiments(base_cfg, experiments, args.save_root, summary_path)


if __name__ == "__main__":
    main()

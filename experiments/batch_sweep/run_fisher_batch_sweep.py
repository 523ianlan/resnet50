"""Sweep fisher batch counts to study convergence of Stage 2 Fisher impact."""

import argparse
import os

from sweep_common import (
    parse_int_list,
    load_base_config,
    run_experiments,
)


def build_experiments(fisher_batches_list, calib_batches, tag_prefix):
    experiments = []
    for fb in fisher_batches_list:
        name = f"{tag_prefix}cb{calib_batches}_fb{fb}"
        overrides = {
            "calib_batches": calib_batches,
            "fisher_batches": fb,
        }
        experiments.append({"name": name, "overrides": overrides})
    return experiments


def main():
    parser = argparse.ArgumentParser(description="Fisher-batches sweep for UFALP")
    parser.add_argument("--config", type=str, default=None, help="Optional config JSON")
    parser.add_argument("--fisher-batches", type=str, default="25,50,100,200",
                        help="Comma-separated fisher_batches list")
    parser.add_argument("--calib-batches", type=int, default=10, help="Fixed calib_batches")
    parser.add_argument("--save-root", type=str,
                        default="./experiments/batch_sweep/results/fisher_batches",
                        help="Root directory for experiment outputs")
    parser.add_argument("--tag", type=str, default="", help="Optional name prefix")
    args = parser.parse_args()

    base_cfg = load_base_config(args.config)
    fisher_batches_list = parse_int_list(args.fisher_batches)
    if not fisher_batches_list:
        raise ValueError("--fisher-batches must not be empty")

    tag_prefix = f"{args.tag}_" if args.tag else ""
    experiments = build_experiments(fisher_batches_list, args.calib_batches, tag_prefix)

    summary_path = os.path.join(args.save_root, "summary.csv")
    run_experiments(base_cfg, experiments, args.save_root, summary_path)


if __name__ == "__main__":
    main()

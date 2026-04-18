"""Sweep calibration batch counts to study convergence of Stage 1 uncertainty."""

import argparse
import os

from sweep_common import (
    parse_int_list,
    load_base_config,
    run_experiments,
)


def build_experiments(calib_batches_list, fisher_batches, tag_prefix):
    experiments = []
    for cb in calib_batches_list:
        name = f"{tag_prefix}cb{cb}_fb{fisher_batches}"
        overrides = {
            "calib_batches": cb,
            "fisher_batches": fisher_batches,
        }
        experiments.append({"name": name, "overrides": overrides})
    return experiments


def main():
    parser = argparse.ArgumentParser(description="Calib-batches sweep for UFALP")
    parser.add_argument("--config", type=str, default=None, help="Optional config JSON")
    parser.add_argument("--calib-batches", type=str, default="5,10,20,50",
                        help="Comma-separated calib_batches list")
    parser.add_argument("--fisher-batches", type=int, default=100, help="Fixed fisher_batches")
    parser.add_argument("--save-root", type=str,
                        default="./experiments/batch_sweep/results/calib_batches",
                        help="Root directory for experiment outputs")
    parser.add_argument("--tag", type=str, default="", help="Optional name prefix")
    args = parser.parse_args()

    base_cfg = load_base_config(args.config)
    calib_batches_list = parse_int_list(args.calib_batches)
    if not calib_batches_list:
        raise ValueError("--calib-batches must not be empty")

    tag_prefix = f"{args.tag}_" if args.tag else ""
    experiments = build_experiments(calib_batches_list, args.fisher_batches, tag_prefix)

    summary_path = os.path.join(args.save_root, "summary.csv")
    run_experiments(base_cfg, experiments, args.save_root, summary_path)


if __name__ == "__main__":
    main()

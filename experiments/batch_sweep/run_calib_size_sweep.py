"""Sweep calibration set sizes (D_cal) to study Stage 1 convergence."""

import argparse
import os

from sweep_common import (
    parse_int_list,
    load_base_config,
    run_experiments,
)


def build_experiments(calib_samples_list, fisher_batches, tag_prefix, use_val_transform, exclude_from_train):
    experiments = []
    for cs in calib_samples_list:
        name = f"{tag_prefix}cs{cs}_fb{fisher_batches}"
        overrides = {
            "calib_samples": cs,
            "calib_split_ratio": 0.0,
            "calib_batches": 0,
            "calib_use_val_transform": use_val_transform,
            "calib_exclude_from_train": exclude_from_train,
            "fisher_batches": fisher_batches,
        }
        experiments.append({"name": name, "overrides": overrides})
    return experiments


def main():
    parser = argparse.ArgumentParser(description="Calibration set size sweep for UFALP")
    parser.add_argument("--config", type=str, default=None, help="Optional config JSON")
    parser.add_argument("--calib-samples", type=str, default="1000,2000,5000,10000",
                        help="Comma-separated calib_samples list")
    parser.add_argument("--fisher-batches", type=int, default=100, help="Fixed fisher_batches")
    parser.add_argument("--use-val-transform", action="store_true",
                        help="Use validation transform for D_cal")
    parser.add_argument("--exclude-from-train", action="store_true",
                        help="Exclude D_cal from training set")
    parser.add_argument("--save-root", type=str,
                        default="./experiments/batch_sweep/results/calib_sizes",
                        help="Root directory for experiment outputs")
    parser.add_argument("--tag", type=str, default="", help="Optional name prefix")
    args = parser.parse_args()

    base_cfg = load_base_config(args.config)
    calib_samples_list = parse_int_list(args.calib_samples)
    if not calib_samples_list:
        raise ValueError("--calib-samples must not be empty")

    tag_prefix = f"{args.tag}_" if args.tag else ""
    experiments = build_experiments(
        calib_samples_list,
        args.fisher_batches,
        tag_prefix,
        args.use_val_transform,
        args.exclude_from_train,
    )

    summary_path = os.path.join(args.save_root, "summary.csv")
    run_experiments(base_cfg, experiments, args.save_root, summary_path)


if __name__ == "__main__":
    main()

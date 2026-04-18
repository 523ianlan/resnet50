"""Common helpers for UFALP batch/convergence sweeps."""

import os
import sys
import csv
import copy
import re
from typing import Dict, Any, List, Optional

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from configs.config import PruningConfig
from main import main_r50


def parse_int_list(value: str) -> List[int]:
    if not value:
        return []
    return [int(v.strip()) for v in value.split(',') if v.strip()]


def parse_float_list(value: str) -> List[float]:
    if not value:
        return []
    return [float(v.strip()) for v in value.split(',') if v.strip()]


def load_base_config(path: Optional[str]) -> PruningConfig:
    if path:
        return PruningConfig.load(path)
    return PruningConfig()


def parse_report(report_path: str) -> Dict[str, Any]:
    if not os.path.isfile(report_path):
        return {}

    metrics: Dict[str, Any] = {"report_path": report_path}
    patterns = {
        "final_top1": r"Final Model Top-1:\s*([0-9.]+)%",
        "final_accuracy_change": r"Final Accuracy Change:\s*([+-]?[0-9.]+)%",
        "achieved_compression": r"Achieved Compression:\s*([0-9.]+)%",
        "target_compression": r"Target Compression:\s*([0-9.]+)%",
    }

    with open(report_path, "r", encoding="utf-8") as f:
        text = f.read()

    for key, pattern in patterns.items():
        m = re.search(pattern, text)
        if m:
            try:
                metrics[key] = float(m.group(1))
            except ValueError:
                pass

    return metrics


def append_summary(summary_path: str, rows: List[Dict[str, Any]]):
    if not rows:
        return

    os.makedirs(os.path.dirname(summary_path), exist_ok=True)

    # build header as union of keys
    header: List[str] = []
    for row in rows:
        for k in row.keys():
            if k not in header:
                header.append(k)

    write_header = not os.path.isfile(summary_path)

    with open(summary_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_experiments(
    base_config: PruningConfig,
    experiments: List[Dict[str, Any]],
    save_root: str,
    summary_path: str,
) -> None:
    rows: List[Dict[str, Any]] = []

    for exp in experiments:
        cfg = copy.deepcopy(base_config)
        name = exp["name"]
        overrides = exp.get("overrides", {})

        for key, value in overrides.items():
            setattr(cfg, key, value)

        cfg.save_dir = os.path.join(save_root, name)
        cfg.custom_tag = ""
        cfg._compression_percentage = int(cfg.target_compression * 100)

        print("=" * 80)
        print(f"Running experiment: {name}")
        print("Overrides:")
        for k, v in overrides.items():
            print(f"  {k} = {v}")
        print("=" * 80)

        main_r50(cfg)

        exp_dir = cfg.get_experiment_dir()
        report_path = os.path.join(exp_dir, "complete_pruning_report.txt")
        metrics = parse_report(report_path)

        row = {"name": name, **overrides, **metrics}
        rows.append(row)

    append_summary(summary_path, rows)

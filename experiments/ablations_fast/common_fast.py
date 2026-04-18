"""Common helpers for fast ablation experiments."""

import os
import sys
import csv
import copy
import json
import re
from typing import Dict, Any, List, Optional

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from configs.config import PruningConfig
from main import main_r50


def load_base_config(path: Optional[str]) -> PruningConfig:
    if path:
        return PruningConfig.load(path)
    return PruningConfig()


def apply_fast_defaults(cfg: PruningConfig) -> None:
    """Fast settings for quick sanity runs."""
    cfg.fine_tune_epochs = 1
    cfg.train_max_batches = 200
    # Use full validation to avoid inflated accuracy from small subsets.
    cfg.eval_max_batches = 0
    cfg.calib_batches = 5
    cfg.fisher_batches = 20


def parse_report(report_path: str) -> Dict[str, Any]:
    if not os.path.isfile(report_path):
        return {}

    metrics: Dict[str, Any] = {"report_path": report_path}
    patterns = {
        "final_top1": r"Final Model Top-1:\s*([0-9.]+)%",
        "final_top5": r"Final Model Top-5:\s*([0-9.]+)%",
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


def write_overrides(exp_dir: str, name: str, overrides: Dict[str, Any]) -> None:
    os.makedirs(exp_dir, exist_ok=True)
    path = os.path.join(exp_dir, "ablation_overrides.json")
    payload = {"name": name, "overrides": overrides, "mode": "fast"}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def append_summary(summary_path: str, rows: List[Dict[str, Any]]):
    if not rows:
        return

    os.makedirs(os.path.dirname(summary_path), exist_ok=True)

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
        apply_fast_defaults(cfg)

        name = exp["name"]
        overrides = exp.get("overrides", {})

        for key, value in overrides.items():
            setattr(cfg, key, value)

        cfg.save_dir = os.path.join(save_root, name)
        cfg.custom_tag = ""
        cfg._compression_percentage = int(cfg.target_compression * 100)

        print("=" * 80)
        print(f"Running FAST experiment: {name}")
        print("Overrides:")
        for k, v in overrides.items():
            print(f"  {k} = {v}")
        print("FAST defaults: finetune=1, train_max_batches=200, eval_max_batches=50, fisher_batches=20")
        print("=" * 80)

        main_r50(cfg)

        exp_dir = cfg.get_experiment_dir()
        write_overrides(exp_dir, name, overrides)
        report_path = os.path.join(exp_dir, "complete_pruning_report.txt")
        metrics = parse_report(report_path)

        row = {"name": name, **overrides, **metrics}
        rows.append(row)

    append_summary(summary_path, rows)

"""Stage-2 score overlap analysis (Taylor vs Fisher vs Hessian) for ResNet-50."""

import argparse
import os
import json
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from common import load_base_config
from data.build import get_resnet_data_loaders_with_calib
from models.resnet_setup import setup_resnet_model
from models.custom_layers import SimpleSVDConv
from models.utils import collect_resnet_conv_layers, get_resnet_parent_and_name
from pruning.stage2_fisher import compute_fisher_components_stage2_r50


def spearmanr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    return float(np.corrcoef(rx, ry)[0, 1])


def topk_overlap(a: np.ndarray, b: np.ndarray, k: int) -> float:
    if k <= 0:
        return float("nan")
    idx_a = np.argpartition(a, -k)[-k:]
    idx_b = np.argpartition(b, -k)[-k:]
    inter = np.intersect1d(idx_a, idx_b)
    return float(len(inter) / k)


def main():
    parser = argparse.ArgumentParser(description="Stage-2 score overlap analysis")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--max-layers", type=int, default=10)
    parser.add_argument("--max-components", type=int, default=0)
    parser.add_argument("--keep-ratio", type=float, default=None)
    parser.add_argument("--fisher-batches", type=int, default=None)
    parser.add_argument("--save-root", type=str,
                        default="./experiments/paper_tables/results/score_overlap")
    parser.add_argument("--device", type=str, default=None,
                        help="Override device (e.g., cpu or cuda).")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Override num_workers for data loading.")
    args = parser.parse_args()

    cfg = load_base_config(args.config)
    if args.fisher_batches is not None:
        cfg.fisher_batches = int(args.fisher_batches)
    if args.device:
        cfg.device = args.device
    if args.num_workers is not None:
        cfg.num_workers = int(args.num_workers)
    if str(cfg.device).lower() == "cpu":
        cfg.num_workers = 0
        cfg.persistent_workers = False
        cfg.pin_memory = False

    keep_ratio = args.keep_ratio
    if keep_ratio is None:
        keep_ratio = max(0.01, min(0.99, 1.0 - float(cfg.target_compression)))

    device = cfg.device
    model, _ = setup_resnet_model(cfg)
    model.eval()

    train_loader, val_loader, calib_loader = get_resnet_data_loaders_with_calib(cfg)
    if calib_loader is None:
        calib_loader = train_loader

    conv_paths = collect_resnet_conv_layers(model)
    svd_layers = {}
    for path in conv_paths:
        parent, layer_name = get_resnet_parent_and_name(model, path)
        if parent is None:
            continue
        original_conv = None
        if hasattr(parent, layer_name):
            original_conv = getattr(parent, layer_name)
        elif hasattr(parent, "_modules") and layer_name in parent._modules:
            original_conv = parent._modules[layer_name]
        if original_conv is None or not isinstance(original_conv, nn.Conv2d):
            continue
        svd_conv = SimpleSVDConv(original_conv, path, config=cfg).to(cfg.device)
        if hasattr(parent, layer_name):
            setattr(parent, layer_name, svd_conv)
        elif hasattr(parent, "_modules") and layer_name in parent._modules:
            parent._modules[layer_name] = svd_conv
        svd_layers[path] = svd_conv

    fisher_components = compute_fisher_components_stage2_r50(
        model, svd_layers, train_loader, config=cfg, device=device
    )

    layers = list(svd_layers.keys())[: args.max_layers]
    report: Dict[str, Dict] = {
        "meta": {
            "keep_ratio": keep_ratio,
            "max_layers": args.max_layers,
            "max_components": args.max_components,
            "fisher_batches": cfg.fisher_batches,
        },
        "layers": {},
        "averages": {},
    }

    overlaps_tf: List[float] = []
    overlaps_fh: List[float] = []
    overlaps_th: List[float] = []
    rho_tf: List[float] = []
    rho_fh: List[float] = []
    rho_th: List[float] = []
    corr_first_fisher: List[float] = []

    for lname in layers:
        layer = svd_layers[lname]
        comp = fisher_components.get(lname)
        if comp is None:
            continue

        sigma = layer.get_sigma().detach().cpu().numpy()
        first_order = comp["first_order"]
        fisher_diag = comp["fisher_diag"]

        max_len = len(sigma)
        if args.max_components and args.max_components > 0:
            max_len = min(max_len, args.max_components)

        sigma = sigma[:max_len]
        first_order = first_order[:max_len]
        fisher_diag = fisher_diag[:max_len]

        taylor = first_order
        hessian = fisher_diag
        fisher = first_order + 0.5 * fisher_diag

        k = max(1, int(round(max_len * keep_ratio)))

        layer_stats = {
            "rank": int(max_len),
            "k": int(k),
            "overlap_taylor_fisher": topk_overlap(taylor, fisher, k),
            "overlap_fisher_hessian": topk_overlap(fisher, hessian, k),
            "overlap_taylor_hessian": topk_overlap(taylor, hessian, k),
            "rho_taylor_fisher": spearmanr(taylor, fisher),
            "rho_fisher_hessian": spearmanr(fisher, hessian),
            "rho_taylor_hessian": spearmanr(taylor, hessian),
            "rho_first_fisherdiag": spearmanr(first_order, fisher_diag),
        }

        overlaps_tf.append(layer_stats["overlap_taylor_fisher"])
        overlaps_fh.append(layer_stats["overlap_fisher_hessian"])
        overlaps_th.append(layer_stats["overlap_taylor_hessian"])
        rho_tf.append(layer_stats["rho_taylor_fisher"])
        rho_fh.append(layer_stats["rho_fisher_hessian"])
        rho_th.append(layer_stats["rho_taylor_hessian"])
        corr_first_fisher.append(layer_stats["rho_first_fisherdiag"])

        report["layers"][lname] = layer_stats

    def _avg(vals: List[float]) -> float:
        vals = [v for v in vals if not np.isnan(v)]
        return float(np.mean(vals)) if vals else float("nan")

    report["averages"] = {
        "overlap_taylor_fisher": _avg(overlaps_tf),
        "overlap_fisher_hessian": _avg(overlaps_fh),
        "overlap_taylor_hessian": _avg(overlaps_th),
        "rho_taylor_fisher": _avg(rho_tf),
        "rho_fisher_hessian": _avg(rho_fh),
        "rho_taylor_hessian": _avg(rho_th),
        "rho_first_fisherdiag": _avg(corr_first_fisher),
    }

    os.makedirs(args.save_root, exist_ok=True)
    report_path = os.path.join(args.save_root, "score_overlap_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    text_path = os.path.join(args.save_root, "score_overlap_report.txt")
    with open(text_path, "w", encoding="utf-8") as f:
        f.write("Stage-2 Score Overlap (averages)\n")
        for k, v in report["averages"].items():
            f.write(f"{k}: {v:.4f}\n")

    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()

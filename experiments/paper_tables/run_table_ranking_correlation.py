"""Table: Ranking correlation and flat spectrum analysis (ResNet-50)."""

import argparse
import os
import json
import math
from typing import Dict, List, Tuple

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
        return float('nan')
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    return np.corrcoef(rx, ry)[0, 1]


def kendalltau(x: np.ndarray, y: np.ndarray) -> float:
    n = len(x)
    if n < 2:
        return float('nan')
    num = 0
    den = n * (n - 1) / 2
    for i in range(n):
        for j in range(i + 1, n):
            num += math.copysign(1.0, (x[i] - x[j]) * (y[i] - y[j])) if (x[i] - x[j]) * (y[i] - y[j]) != 0 else 0.0
    return num / den


def compute_loss(model: nn.Module, loader, device, max_batches: int) -> float:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            if max_batches > 0 and i >= max_batches:
                break
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            bs = images.size(0)
            total_loss += loss.item() * bs
            total += bs
    return total_loss / max(total, 1)


def oracle_scores_for_layer(model, layer: SimpleSVDConv, loader, device, max_batches: int, max_components: int) -> np.ndarray:
    base_loss = compute_loss(model, loader, device, max_batches)
    scores = []

    param = layer.get_score_param()
    eps = layer.config.svd_epsilon
    max_k = min(max_components, layer.full_rank)

    for i in range(max_k):
        with torch.no_grad():
            saved = param[i].item()
            if layer.use_log_s:
                param[i] = math.log(eps)
            else:
                param[i] = 0.0
        loss_i = compute_loss(model, loader, device, max_batches)
        with torch.no_grad():
            param[i] = saved
        scores.append(loss_i - base_loss)

    return np.array(scores, dtype=np.float64)


def main():
    parser = argparse.ArgumentParser(description="Ranking correlation analysis")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--max-layers", type=int, default=10)
    parser.add_argument("--max-components", type=int, default=16)
    parser.add_argument("--oracle-batches", type=int, default=1)
    parser.add_argument("--flat-threshold", type=float, default=10.0)
    parser.add_argument("--save-root", type=str,
                        default="./experiments/paper_tables/results/ranking_correlation")
    args = parser.parse_args()

    cfg = load_base_config(args.config)
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
        elif hasattr(parent, '_modules') and layer_name in parent._modules:
            original_conv = parent._modules[layer_name]
        if original_conv is None or not isinstance(original_conv, nn.Conv2d):
            continue
        svd_conv = SimpleSVDConv(original_conv, path, config=cfg).to(cfg.device)
        if hasattr(parent, layer_name):
            setattr(parent, layer_name, svd_conv)
        elif hasattr(parent, '_modules') and layer_name in parent._modules:
            parent._modules[layer_name] = svd_conv
        svd_layers[path] = svd_conv

    fisher_components = compute_fisher_components_stage2_r50(
        model, svd_layers, train_loader, config=cfg, device=device
    )

    layers = list(svd_layers.keys())[: args.max_layers]
    report = {
        "layers": {},
        "averages": {},
        "flat_threshold": args.flat_threshold,
    }

    score_names = ["magnitude", "energy", "taylor", "hessian", "fisher"]
    per_score_rho = {k: [] for k in score_names}
    per_score_tau = {k: [] for k in score_names}
    per_score_rho_flat = {k: [] for k in score_names}
    per_score_tau_flat = {k: [] for k in score_names}
    per_score_rho_decay = {k: [] for k in score_names}
    per_score_tau_decay = {k: [] for k in score_names}

    for lname in layers:
        layer = svd_layers[lname]
        comp = fisher_components.get(lname)
        if comp is None:
            continue

        oracle = oracle_scores_for_layer(
            model, layer, calib_loader, device, args.oracle_batches, args.max_components
        )
        if len(oracle) < 2:
            continue

        sigma = layer.get_sigma().detach().cpu().numpy()[: len(oracle)]
        first_order = comp["first_order"][: len(oracle)]
        fisher_diag = comp["fisher_diag"][: len(oracle)]

        scores = {
            "magnitude": np.abs(sigma),
            "energy": sigma ** 2,
            "taylor": first_order,
            "hessian": fisher_diag,
            "fisher": first_order + 0.5 * fisher_diag,
        }

        ratio = float(sigma[0] / max(sigma[-1], cfg.svd_epsilon))
        is_flat = ratio < args.flat_threshold

        layer_stats = {"ratio": ratio, "flat": is_flat}
        for key, s in scores.items():
            rho = spearmanr(s, oracle)
            tau = kendalltau(s, oracle)
            layer_stats[f"{key}_rho"] = rho
            layer_stats[f"{key}_tau"] = tau
            per_score_rho[key].append(rho)
            per_score_tau[key].append(tau)
            if is_flat:
                per_score_rho_flat[key].append(rho)
                per_score_tau_flat[key].append(tau)
            else:
                per_score_rho_decay[key].append(rho)
                per_score_tau_decay[key].append(tau)

        report["layers"][lname] = layer_stats

    def _avg(vals: List[float]) -> float:
        vals = [v for v in vals if not np.isnan(v)]
        return float(np.mean(vals)) if vals else float('nan')

    report["averages"]["overall"] = {
        k: {"rho": _avg(per_score_rho[k]), "tau": _avg(per_score_tau[k])}
        for k in score_names
    }
    report["averages"]["flat"] = {
        k: {"rho": _avg(per_score_rho_flat[k]), "tau": _avg(per_score_tau_flat[k])}
        for k in score_names
    }
    report["averages"]["decaying"] = {
        k: {"rho": _avg(per_score_rho_decay[k]), "tau": _avg(per_score_tau_decay[k])}
        for k in score_names
    }

    os.makedirs(args.save_root, exist_ok=True)
    report_path = os.path.join(args.save_root, "ranking_correlation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Write a simple text report
    text_path = os.path.join(args.save_root, "ranking_correlation_report.txt")
    with open(text_path, "w", encoding="utf-8") as f:
        f.write("Ranking Correlation (overall)\n")
        for k in score_names:
            v = report["averages"]["overall"][k]
            f.write(f"{k}: rho={v['rho']:.4f}, tau={v['tau']:.4f}\n")
        f.write("\nDecaying spectrum\n")
        for k in score_names:
            v = report["averages"]["decaying"][k]
            f.write(f"{k}: rho={v['rho']:.4f}, tau={v['tau']:.4f}\n")
        f.write("\nFlat spectrum\n")
        for k in score_names:
            v = report["averages"]["flat"][k]
            f.write(f"{k}: rho={v['rho']:.4f}, tau={v['tau']:.4f}\n")

    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()

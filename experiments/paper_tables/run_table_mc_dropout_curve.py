"""Layer-wise MC Dropout uncertainty curve experiment (ResNet-50).

This script computes U_l(r) over a dropout grid r and MC sample count K:
    U_l(r) = mean_c [ Var_t(||f_c||_2) / (E_t(||f_c||_2) + eps) ]

Outputs are saved under experiments/paper_tables/results by default.
"""

import argparse
import json
import math
import os
import random
import sys
import time
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from configs.config import PruningConfig
from data.build import get_resnet_data_loaders_with_calib
from main import build_custom_tag
from models.custom_layers import SimpleSVDConv
from models.resnet_setup import setup_resnet_model
from models.utils import collect_resnet_conv_layers, get_resnet_parent_and_name

try:
    import matplotlib.pyplot as plt

    HAS_MPL = True
except Exception:
    HAS_MPL = False


def parse_float_grid(text: str) -> List[float]:
    vals = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(float(part))
    return vals


def parse_int_grid(text: str) -> List[int]:
    vals = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(int(part))
    return vals


def sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return float(obj)
    return obj


def set_seed(seed: int, deterministic: bool = True) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)


def safe_name(path: str) -> str:
    return path.replace(".", "_").replace("/", "_")


def replace_module(parent: nn.Module, name: str, new_module: nn.Module) -> bool:
    if hasattr(parent, name):
        setattr(parent, name, new_module)
        return True
    if hasattr(parent, "_modules") and name in parent._modules:
        parent._modules[name] = new_module
        return True
    if isinstance(parent, nn.Sequential) and name.isdigit():
        idx = int(name)
        if idx < len(parent):
            parent[idx] = new_module
            return True
    return False


def set_dropout_active_only(model: nn.Module) -> None:
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            m.train()
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()


def select_layers(layer_paths: List[str], mode: str, max_layers: int) -> List[str]:
    if mode == "early":
        chosen = [p for p in layer_paths if p.startswith("conv1") or p.startswith("layer1") or p.startswith("layer2")]
    elif mode == "deep":
        chosen = [p for p in layer_paths if p.startswith("layer3") or p.startswith("layer4")]
    else:
        chosen = layer_paths
    if max_layers > 0:
        chosen = chosen[:max_layers]
    return chosen


def probe_uncertainty_for_layer(
    model: nn.Module,
    layer_path: str,
    layer_module: nn.Module,
    loader,
    device: torch.device,
    dropout_p: float,
    mc_samples: int,
    max_batches: int,
    epsilon: float,
) -> float:
    parent, layer_name = get_resnet_parent_and_name(model, layer_path)
    if parent is None:
        return float("nan")

    wrapper = nn.Sequential(layer_module, nn.Dropout2d(p=dropout_p)).to(device)
    if not replace_module(parent, layer_name, wrapper):
        return float("nan")

    collected: List[torch.Tensor] = []

    def hook(_module, _inputs, output):
        with torch.no_grad():
            flat = output.reshape(output.size(0), output.size(1), -1)
            norms = torch.norm(flat, p=2, dim=2).mean(dim=0)
            collected.append(norms.detach().cpu())

    handle = wrapper.register_forward_hook(hook)
    model.eval()
    set_dropout_active_only(model)

    batch_scores: List[float] = []
    try:
        with torch.no_grad():
            for bi, (images, _labels) in enumerate(loader):
                if max_batches > 0 and bi >= max_batches:
                    break
                images = images.to(device, non_blocking=True)
                collected.clear()
                for _ in range(mc_samples):
                    _ = model(images)
                if not collected:
                    continue
                stack = torch.stack(collected, dim=0).float()
                mu = torch.mean(stack, dim=0)
                var = torch.var(stack, dim=0, unbiased=False)
                score = torch.mean(var / (mu + epsilon)).item()
                batch_scores.append(float(score))
    finally:
        handle.remove()
        replace_module(parent, layer_name, layer_module)
        model.eval()

    if not batch_scores:
        return float("nan")
    return float(np.mean(batch_scores))


def plot_layer_curves(
    save_dir: str,
    layer_name: str,
    dropout_grid: Sequence[float],
    curves_by_k: Dict[str, Sequence[float]],
) -> None:
    if not HAS_MPL:
        return
    fig, ax = plt.subplots(figsize=(7, 4.8))
    for k, vals in curves_by_k.items():
        ax.plot(dropout_grid, vals, marker="o", label=f"K={k}")
    ax.set_title(f"U_l(r): {layer_name}")
    ax.set_xlabel("Dropout ratio r")
    ax.set_ylabel("Uncertainty U_l(r)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f"curve_{safe_name(layer_name)}.png"), dpi=180)
    plt.close(fig)


def plot_layer_heatmap(
    save_dir: str,
    layer_names: List[str],
    dropout_grid: List[float],
    values: np.ndarray,
    title: str,
    filename: str,
) -> None:
    if not HAS_MPL or values.size == 0:
        return
    fig_h = max(4, 0.25 * len(layer_names))
    fig, ax = plt.subplots(figsize=(7.6, fig_h))
    im = ax.imshow(values, aspect="auto", cmap="viridis")
    ax.set_title(title)
    ax.set_xticks(np.arange(len(dropout_grid)))
    ax.set_xticklabels([f"{r:.2f}" for r in dropout_grid], rotation=45)
    ax.set_yticks(np.arange(len(layer_names)))
    ax.set_yticklabels(layer_names, fontsize=7)
    ax.set_xlabel("Dropout ratio r")
    ax.set_ylabel("Layer")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, filename), dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="MC Dropout curve only")
    parser.add_argument("--config", type=str, default=None, help="Path to baseline config.json")
    parser.add_argument(
        "--save-root",
        type=str,
        default="./experiments/paper_tables/results/mc_dropout_curve",
    )
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"])
    parser.add_argument("--max-layers", type=int, default=53, help="0 means all")
    parser.add_argument("--layer-mode", type=str, default="all", choices=["all", "early", "deep"])
    parser.add_argument("--dropout-grid", type=str, default="0.05,0.1,0.15,0.2")
    parser.add_argument("--mc-samples-grid", type=str, default="10,20,30")
    parser.add_argument("--probe-batches", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--allow-prefetch-to-gpu", action="store_true", default=False)
    args = parser.parse_args()

    cfg = PruningConfig.load(args.config) if args.config else PruningConfig()
    if args.device is not None:
        cfg.device = torch.device(args.device)
    cfg.num_workers = int(args.num_workers)
    cfg.persistent_workers = bool(cfg.num_workers > 0)
    cfg.prefetch_to_gpu = bool(args.allow_prefetch_to_gpu)
    cfg.calib_batches = 0
    cfg.custom_tag = ""
    cfg._compression_percentage = int(cfg.target_compression * 100)

    dropout_grid = parse_float_grid(args.dropout_grid)
    mc_samples_grid = parse_int_grid(args.mc_samples_grid)
    set_seed(args.seed, deterministic=bool(getattr(cfg, "deterministic", True)))

    exp_time = time.strftime("%Y%m%d_%H%M%S")
    base_name = f"mc_dropout_curve_{exp_time}"
    if args.tag:
        base_name = f"{base_name}_{args.tag}"
    save_dir = os.path.join(args.save_root, base_name)
    os.makedirs(save_dir, exist_ok=True)
    vis_dir = os.path.join(save_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    cfg_json = cfg.to_dict()
    cfg_json["save_dir"] = save_dir
    with open(os.path.join(save_dir, "config_used.json"), "w", encoding="utf-8") as f:
        json.dump(cfg_json, f, indent=2, ensure_ascii=False)

    print("=" * 88)
    print("MC DROPOUT CURVE EXPERIMENT")
    print(f"Save dir: {save_dir}")
    print(f"Device: {cfg.device}")
    print(f"Layer mode: {args.layer_mode}, max_layers={args.max_layers}")
    print(f"Dropout grid: {dropout_grid}")
    print(f"MC samples grid: {mc_samples_grid}")
    print("=" * 88)

    model, _ = setup_resnet_model(cfg, pretrained=True)
    model.eval()
    train_loader, _val_loader, calib_loader = get_resnet_data_loaders_with_calib(cfg)
    if calib_loader is None:
        calib_loader = train_loader

    conv_paths = collect_resnet_conv_layers(model)
    svd_layers: Dict[str, SimpleSVDConv] = {}
    for path in conv_paths:
        parent, layer_name = get_resnet_parent_and_name(model, path)
        if parent is None:
            continue
        target = None
        if hasattr(parent, layer_name):
            target = getattr(parent, layer_name)
        elif hasattr(parent, "_modules") and layer_name in parent._modules:
            target = parent._modules[layer_name]
        if not isinstance(target, nn.Conv2d):
            continue
        svd = SimpleSVDConv(target, path, config=cfg).to(cfg.device)
        replace_module(parent, layer_name, svd)
        svd_layers[path] = svd

    chosen_layers = select_layers(list(svd_layers.keys()), mode=args.layer_mode, max_layers=args.max_layers)
    print(f"Total SVD layers: {len(svd_layers)} | Selected: {len(chosen_layers)}")

    layers_report: Dict[str, Dict] = {}
    csv_rows: List[Dict] = []

    # For dominance analysis
    dominance: Dict[str, Dict[str, float]] = {str(k): {} for k in mc_samples_grid}

    for i, layer_name in enumerate(chosen_layers):
        layer = svd_layers[layer_name]
        print(f"[{i+1}/{len(chosen_layers)}] {layer_name}")

        by_k: Dict[str, Dict[str, float]] = {}
        curves_for_plot: Dict[str, List[float]] = {}
        for k in mc_samples_grid:
            curve = {}
            curve_vals = []
            for r in dropout_grid:
                u = probe_uncertainty_for_layer(
                    model=model,
                    layer_path=layer_name,
                    layer_module=layer,
                    loader=calib_loader,
                    device=cfg.device,
                    dropout_p=float(r),
                    mc_samples=int(k),
                    max_batches=int(args.probe_batches),
                    epsilon=float(cfg.uncertainty_epsilon),
                )
                curve[f"{r:.4f}"] = float(u)
                curve_vals.append(float(u))
                csv_rows.append(
                    {
                        "layer": layer_name,
                        "mc_samples": int(k),
                        "dropout_ratio": float(r),
                        "uncertainty": float(u),
                    }
                )
            by_k[str(k)] = curve
            curves_for_plot[str(k)] = curve_vals
        layers_report[layer_name] = {
            "full_rank": int(layer.full_rank),
            "uncertainty": by_k,
        }
        plot_layer_curves(vis_dir, layer_name, dropout_grid, curves_for_plot)

    # Dominance statistics: max layer share of uncertainty sum
    for k in mc_samples_grid:
        k_str = str(k)
        for r in dropout_grid:
            key = f"{r:.4f}"
            vals = []
            names = []
            for lname in chosen_layers:
                v = float(layers_report[lname]["uncertainty"][k_str][key])
                if np.isnan(v):
                    continue
                vals.append(v)
                names.append(lname)
            if not vals:
                dominance[k_str][key] = float("nan")
                continue
            arr = np.array(vals, dtype=np.float64)
            total = float(np.sum(arr))
            if total <= 1e-12:
                dominance[k_str][key] = float("nan")
                continue
            share = arr / total
            max_idx = int(np.argmax(share))
            dominance[k_str][key] = float(share[max_idx])

    # Heatmap values (use largest K as default view)
    if chosen_layers and dropout_grid:
        k_show = str(max(mc_samples_grid))
        mat = np.array(
            [[float(layers_report[l]["uncertainty"][k_show][f"{r:.4f}"]) for r in dropout_grid] for l in chosen_layers],
            dtype=np.float64,
        )
        plot_layer_heatmap(
            vis_dir,
            chosen_layers,
            dropout_grid,
            mat,
            title=f"MC Dropout U_l(r) heatmap (K={k_show})",
            filename=f"heatmap_uncertainty_k{k_show}.png",
        )

    report = {
        "meta": {
            "timestamp": exp_time,
            "save_dir": save_dir,
            "config_signature": build_custom_tag(cfg),
            "device": str(cfg.device),
            "seed": int(args.seed),
            "layer_mode": args.layer_mode,
            "selected_layers": len(chosen_layers),
            "dropout_grid": dropout_grid,
            "mc_samples_grid": mc_samples_grid,
            "probe_batches": int(args.probe_batches),
            "num_workers": int(cfg.num_workers),
            "prefetch_to_gpu": bool(cfg.prefetch_to_gpu),
            "has_matplotlib": HAS_MPL,
        },
        "dominance_max_share": dominance,
        "layers": layers_report,
    }

    json_path = os.path.join(save_dir, "mc_dropout_curve_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(sanitize_for_json(report), f, indent=2, ensure_ascii=False)

    csv_path = os.path.join(save_dir, "mc_dropout_curve_points.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("layer,mc_samples,dropout_ratio,uncertainty\n")
        for row in csv_rows:
            f.write(
                f"{row['layer']},{row['mc_samples']},{row['dropout_ratio']},{row['uncertainty']}\n"
            )

    txt_path = os.path.join(save_dir, "analysis_summary.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("MC Dropout Curve Summary\n")
        f.write("=" * 80 + "\n")
        f.write(f"Save dir: {save_dir}\n")
        f.write(f"Selected layers: {len(chosen_layers)}\n")
        f.write(f"Dropout grid: {dropout_grid}\n")
        f.write(f"MC samples grid: {mc_samples_grid}\n")
        f.write("Dominance max-share (higher means one-layer dominance):\n")
        for k in mc_samples_grid:
            k_str = str(k)
            f.write(f"  K={k_str}: {dominance[k_str]}\n")
        f.write(f"JSON report: {json_path}\n")
        f.write(f"CSV points: {csv_path}\n")

    print("=" * 88)
    print("MC Dropout curve experiment completed.")
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")
    print(f"TXT:  {txt_path}")
    print("=" * 88)


if __name__ == "__main__":
    main()


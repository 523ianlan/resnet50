"""MC-Dropout proxy analysis for layer-wise pruning distortion (ResNet-50).

Protocol:
1) Stochastic probing: U_l(r) from MC-Dropout at layer l
2) Random pruning baseline: D_rand_l(rho)
3) Ranking pruning: D_rank_l(rho) for selected ranking methods
4) Derived metrics: gain, normalized gain, proxy correlation, proxy-real gap,
   integrated efficiency, intrinsic robustness
"""

import argparse
import contextlib
import json
import math
import os
import random
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from configs.config import PruningConfig
from data.build import get_resnet_data_loaders_with_calib
from main import build_custom_tag
from models.custom_layers import SimpleSVDConv
from models.resnet_setup import setup_resnet_model
from models.utils import collect_resnet_conv_layers, get_resnet_parent_and_name
from pruning.stage2_fisher import compute_fisher_components_stage2_r50

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


def rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    return ranks


def pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    x = x - np.mean(x)
    y = y - np.mean(y)
    den = np.linalg.norm(x) * np.linalg.norm(y)
    if den < 1e-12:
        return float("nan")
    return float(np.dot(x, y) / den)


def spearmanr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    rx = rankdata(x)
    ry = rankdata(y)
    return pearsonr(rx, ry)


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


@contextlib.contextmanager
def masked_singular_components(layer: SimpleSVDConv, keep_indices: torch.Tensor):
    param = layer.get_score_param()
    saved = param.detach().clone()
    keep_indices = keep_indices.to(param.device, dtype=torch.long)
    with torch.no_grad():
        if layer.use_log_s:
            floor = math.log(layer.config.svd_epsilon)
            param.fill_(floor)
            param[keep_indices] = saved[keep_indices]
        else:
            param.zero_()
            param[keep_indices] = saved[keep_indices]
    try:
        yield
    finally:
        with torch.no_grad():
            param.copy_(saved)


def evaluate_loss_and_acc(
    model: nn.Module,
    loader,
    device: torch.device,
    max_batches: int,
) -> Tuple[float, float]:
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            if max_batches > 0 and i >= max_batches:
                break
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if labels.dim() > 1 and labels.size(1) > 1:
                labels = torch.argmax(labels, dim=1)
            outputs = model(images)
            loss = criterion(outputs, labels)
            bs = images.size(0)
            total_loss += loss.item() * bs
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += bs
    if total == 0:
        return float("nan"), float("nan")
    return total_loss / total, 100.0 * total_correct / total


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


def keep_count_for_ratio(layer: SimpleSVDConv, prune_ratio: float, min_rank: int) -> int:
    full = int(layer.full_rank)
    keep = int(round(full * (1.0 - prune_ratio)))
    keep = max(min_rank, keep)
    keep = min(full, max(1, keep))
    return keep


def score_by_method(
    layer: SimpleSVDConv,
    method: str,
    fisher_components: Optional[Dict[str, Dict[str, np.ndarray]]],
    layer_name: str,
) -> Optional[np.ndarray]:
    sigma = layer.get_sigma().detach().cpu().numpy()
    if method in {"svd", "magnitude"}:
        return np.abs(sigma)
    if method == "energy":
        return sigma ** 2
    if fisher_components is None:
        return None
    comp = fisher_components.get(layer_name)
    if comp is None:
        return None
    first = comp["first_order"][: len(sigma)]
    hess = comp["fisher_diag"][: len(sigma)]
    if method == "taylor":
        return first
    if method == "hessian":
        return hess
    if method == "fisher":
        return first + 0.5 * hess
    return None


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


def plot_layer_curves(
    save_dir: str,
    layer_name: str,
    dropout_grid: Sequence[float],
    uncertainty_curve: Sequence[float],
    prune_grid: Sequence[float],
    random_curve: Sequence[float],
    rank_curves: Dict[str, Sequence[float]],
) -> None:
    if not HAS_MPL:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(dropout_grid, uncertainty_curve, marker="o", label="U_l (MC Dropout)")
    ax.plot(prune_grid, random_curve, marker="s", label="D_rand (delta loss)")
    for method, vals in rank_curves.items():
        ax.plot(prune_grid, vals, marker="^", label=f"D_rank:{method}")
    ax.set_title(layer_name)
    ax.set_xlabel("ratio")
    ax.set_ylabel("score / distortion")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f"curve_{safe_name(layer_name)}.png"), dpi=180)
    plt.close(fig)


def plot_gain_heatmap(save_dir: str, gain_matrix: np.ndarray, layer_names: List[str], prune_grid: List[float], method: str) -> None:
    if not HAS_MPL or gain_matrix.size == 0:
        return
    fig_h = max(4, 0.25 * len(layer_names))
    fig, ax = plt.subplots(figsize=(8, fig_h))
    im = ax.imshow(gain_matrix, aspect="auto", cmap="coolwarm")
    ax.set_title(f"Ranking Gain Heatmap ({method})")
    ax.set_xticks(np.arange(len(prune_grid)))
    ax.set_xticklabels([f"{r:.2f}" for r in prune_grid], rotation=45)
    ax.set_yticks(np.arange(len(layer_names)))
    ax.set_yticklabels(layer_names, fontsize=7)
    ax.set_xlabel("Prune ratio")
    ax.set_ylabel("Layer")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f"heatmap_gain_{method}.png"), dpi=180)
    plt.close(fig)


def plot_scatter_proxy(save_dir: str, u_vals: np.ndarray, d_vals: np.ndarray, title: str, name: str) -> None:
    if not HAS_MPL or len(u_vals) == 0:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(u_vals, d_vals, s=18, alpha=0.75)
    ax.set_title(title)
    ax.set_xlabel("U_l(r)")
    ax.set_ylabel("D_rand_l(r) (delta loss)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, name), dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="MC Dropout proxy analysis")
    parser.add_argument("--config", type=str, default=None, help="Path to baseline config.json")
    parser.add_argument(
        "--save-root",
        type=str,
        default="./experiments/paper_tables/results/mc_dropout_proxy_analysis",
    )
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"])
    parser.add_argument("--max-layers", type=int, default=12, help="0 means all")
    parser.add_argument("--layer-mode", type=str, default="all", choices=["all", "early", "deep"])
    parser.add_argument("--dropout-grid", type=str, default="0.05,0.1,0.15,0.2")
    parser.add_argument("--prune-grid", type=str, default="0.05,0.1,0.15,0.2")
    parser.add_argument("--mc-samples-grid", type=str, default="10,20")
    parser.add_argument("--analysis-mc-samples", type=int, default=20)
    parser.add_argument("--random-trials", type=int, default=8)
    parser.add_argument("--probe-batches", type=int, default=2)
    parser.add_argument("--eval-batches", type=int, default=2)
    parser.add_argument("--fisher-batches", type=int, default=50)
    parser.add_argument("--rank-methods", type=str, default="fisher,svd,taylor")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-prefetch-to-gpu", action="store_true", default=True)
    parser.add_argument("--allow-prefetch-to-gpu", action="store_true", default=False)
    args = parser.parse_args()

    cfg = PruningConfig.load(args.config) if args.config else PruningConfig()
    if args.device is not None:
        cfg.device = torch.device(args.device)
    if args.allow_prefetch_to_gpu:
        cfg.prefetch_to_gpu = True
    else:
        cfg.prefetch_to_gpu = False
    cfg.fisher_batches = int(args.fisher_batches)
    cfg.calib_batches = 0
    cfg.eval_max_batches = 0
    cfg.train_max_batches = 0
    cfg.num_workers = int(args.num_workers)
    cfg.persistent_workers = bool(cfg.num_workers > 0)
    cfg.custom_tag = ""
    cfg._compression_percentage = int(cfg.target_compression * 100)

    dropout_grid = parse_float_grid(args.dropout_grid)
    prune_grid = parse_float_grid(args.prune_grid)
    mc_samples_grid = parse_int_grid(args.mc_samples_grid)
    rank_methods = [m.strip() for m in args.rank_methods.split(",") if m.strip()]
    analysis_mc = int(args.analysis_mc_samples)
    if analysis_mc not in mc_samples_grid:
        mc_samples_grid.append(analysis_mc)
        mc_samples_grid = sorted(set(mc_samples_grid))

    set_seed(args.seed, deterministic=bool(getattr(cfg, "deterministic", True)))

    exp_time = time.strftime("%Y%m%d_%H%M%S")
    base_name = f"mc_dropout_proxy_{exp_time}"
    if args.tag:
        base_name = f"{base_name}_{args.tag}"
    save_dir = os.path.join(args.save_root, base_name)
    os.makedirs(save_dir, exist_ok=True)
    vis_dir = os.path.join(save_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    cfg_json = cfg.to_dict()
    cfg_json["save_dir"] = save_dir
    with open(os.path.join(save_dir, "config_used.json"), "w", encoding="utf-8") as f:
        json.dump(cfg_json, f, indent=2)

    print("=" * 90)
    print("MC DROPOUT PROXY ANALYSIS (Layer-wise)")
    print(f"Save dir: {save_dir}")
    print(f"Device: {cfg.device}")
    print(f"Dropout grid: {dropout_grid}")
    print(f"Prune grid: {prune_grid}")
    print(f"MC samples grid: {mc_samples_grid}")
    print(f"Rank methods: {rank_methods}")
    print("=" * 90)

    model, _ = setup_resnet_model(cfg, pretrained=True)
    model.eval()
    train_loader, val_loader, calib_loader = get_resnet_data_loaders_with_calib(cfg)
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
    print(f"Total SVD layers: {len(svd_layers)} | Selected for analysis: {len(chosen_layers)}")

    base_loss, base_acc = evaluate_loss_and_acc(model, val_loader, cfg.device, args.eval_batches)
    print(f"Baseline eval (limited batches={args.eval_batches}): loss={base_loss:.6f}, top1={base_acc:.2f}%")

    need_fisher = any(m in {"fisher", "taylor", "hessian"} for m in rank_methods)
    fisher_components = None
    if need_fisher:
        print("Computing Fisher components once for ranking methods...")
        fisher_components = compute_fisher_components_stage2_r50(
            model, svd_layers, train_loader, config=cfg, device=cfg.device
        )

    rng = np.random.default_rng(args.seed)
    per_layer = {}
    summary_rows = []

    scatter_u = []
    scatter_d = []
    mean_u_by_layer = []
    mean_d_by_layer = []

    gains_by_method = {m: [] for m in rank_methods}
    layer_order = []

    for li, layer_name in enumerate(chosen_layers):
        layer = svd_layers[layer_name]
        layer_order.append(layer_name)
        print(f"[{li+1}/{len(chosen_layers)}] Layer: {layer_name}")

        # 1) MC-dropout probing U_l(r)
        uncertainty_by_k: Dict[str, Dict[str, float]] = {}
        for k in mc_samples_grid:
            u_curve = {}
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
                u_curve[f"{r:.4f}"] = float(u)
            uncertainty_by_k[str(k)] = u_curve

        # 2) Random pruning D_rand_l(rho)
        random_by_ratio = {}
        random_curve_loss = []
        for rho in prune_grid:
            keep_num = keep_count_for_ratio(layer, rho, cfg.min_rank)
            trial_delta_loss = []
            trial_delta_acc = []
            for _ in range(args.random_trials):
                keep_idx = rng.choice(layer.full_rank, size=keep_num, replace=False)
                keep_idx = torch.tensor(np.sort(keep_idx), dtype=torch.long, device=cfg.device)
                with masked_singular_components(layer, keep_idx):
                    loss, acc = evaluate_loss_and_acc(model, val_loader, cfg.device, args.eval_batches)
                trial_delta_loss.append(float(loss - base_loss))
                trial_delta_acc.append(float(base_acc - acc))

            random_by_ratio[f"{rho:.4f}"] = {
                "keep_num": int(keep_num),
                "effective_prune_ratio": float(1.0 - keep_num / layer.full_rank),
                "delta_loss_mean": float(np.mean(trial_delta_loss)),
                "delta_loss_std": float(np.std(trial_delta_loss)),
                "delta_acc_mean": float(np.mean(trial_delta_acc)),
                "delta_acc_std": float(np.std(trial_delta_acc)),
            }
            random_curve_loss.append(float(np.mean(trial_delta_loss)))

        # 3) Ranking-based pruning D_rank_l(rho)
        rank_by_method = {}
        rank_curves_loss = {}
        gain_curves = {}
        norm_gain_curves = {}

        for method in rank_methods:
            scores = score_by_method(layer, method, fisher_components, layer_name)
            if scores is None:
                continue
            method_out = {}
            rank_curve_loss = []
            gain_curve = []
            norm_gain_curve = []
            for ridx, rho in enumerate(prune_grid):
                keep_num = keep_count_for_ratio(layer, rho, cfg.min_rank)
                top_idx = np.argsort(-scores)[:keep_num]
                keep_idx = torch.tensor(np.sort(top_idx), dtype=torch.long, device=cfg.device)
                with masked_singular_components(layer, keep_idx):
                    loss, acc = evaluate_loss_and_acc(model, val_loader, cfg.device, args.eval_batches)
                d_loss = float(loss - base_loss)
                d_acc = float(base_acc - acc)
                rand_mean = float(random_curve_loss[ridx])
                gain = rand_mean - d_loss
                norm_gain = gain / (rand_mean + 1e-12) if abs(rand_mean) > 1e-12 else float("nan")
                method_out[f"{rho:.4f}"] = {
                    "keep_num": int(keep_num),
                    "effective_prune_ratio": float(1.0 - keep_num / layer.full_rank),
                    "delta_loss": d_loss,
                    "delta_acc": d_acc,
                    "gain_vs_random": gain,
                    "normalized_gain": norm_gain,
                }
                rank_curve_loss.append(d_loss)
                gain_curve.append(gain)
                norm_gain_curve.append(norm_gain)

            rank_by_method[method] = method_out
            rank_curves_loss[method] = rank_curve_loss
            gain_curves[method] = gain_curve
            norm_gain_curves[method] = norm_gain_curve

        # 4) Proxy metrics
        analysis_u = uncertainty_by_k.get(str(analysis_mc), {})
        u_aligned = []
        d_rand_aligned = []
        for rho in prune_grid:
            key = f"{rho:.4f}"
            if key in analysis_u and key in random_by_ratio:
                u_aligned.append(float(analysis_u[key]))
                d_rand_aligned.append(float(random_by_ratio[key]["delta_loss_mean"]))
        if len(u_aligned) >= 2:
            u_np = np.array(u_aligned, dtype=np.float64)
            d_np = np.array(d_rand_aligned, dtype=np.float64)
            corr_p = pearsonr(u_np, d_np)
            corr_s = spearmanr(u_np, d_np)
        else:
            corr_p = float("nan")
            corr_s = float("nan")

        proxy_gap = {}
        integrated_eff = {}
        for method, curve in rank_curves_loss.items():
            gaps = {}
            for ridx, rho in enumerate(prune_grid):
                key = f"{rho:.4f}"
                u = float(analysis_u.get(key, float("nan")))
                gaps[key] = u - float(curve[ridx])
            proxy_gap[method] = gaps
            integrated_eff[method] = float(
                np.trapezoid(np.array(gain_curves[method], dtype=np.float64), np.array(prune_grid, dtype=np.float64))
            )

        intrinsic_robustness = float(
            np.trapezoid(np.array(random_curve_loss, dtype=np.float64), np.array(prune_grid, dtype=np.float64))
        )

        per_layer[layer_name] = {
            "full_rank": int(layer.full_rank),
            "uncertainty": uncertainty_by_k,
            "random_distortion": random_by_ratio,
            "rank_distortion": rank_by_method,
            "proxy_correlation": {
                "analysis_mc_samples": int(analysis_mc),
                "pearson_u_vs_drand": corr_p,
                "spearman_u_vs_drand": corr_s,
            },
            "proxy_real_gap": proxy_gap,
            "integrated_efficiency": integrated_eff,
            "intrinsic_robustness": intrinsic_robustness,
        }

        # Plot per-layer curve with analysis_mc uncertainty
        uncertainty_curve = [float(analysis_u.get(f"{r:.4f}", float("nan"))) for r in dropout_grid]
        plot_layer_curves(
            vis_dir,
            layer_name,
            dropout_grid,
            uncertainty_curve,
            prune_grid,
            random_curve_loss,
            rank_curves_loss,
        )

        # Aggregates for summary
        mean_u = float(np.nanmean(uncertainty_curve)) if uncertainty_curve else float("nan")
        mean_d = float(np.mean(random_curve_loss)) if random_curve_loss else float("nan")
        mean_u_by_layer.append(mean_u)
        mean_d_by_layer.append(mean_d)

        for ridx, rho in enumerate(prune_grid):
            key = f"{rho:.4f}"
            u_v = float(analysis_u.get(key, float("nan")))
            d_v = float(random_by_ratio[key]["delta_loss_mean"])
            if not (np.isnan(u_v) or np.isnan(d_v)):
                scatter_u.append(u_v)
                scatter_d.append(d_v)

        for method in rank_methods:
            if method in gain_curves:
                gains_by_method[method].append(gain_curves[method])

        row = {
            "layer": layer_name,
            "full_rank": int(layer.full_rank),
            "proxy_corr_spearman": corr_s,
            "proxy_corr_pearson": corr_p,
            "intrinsic_robustness": intrinsic_robustness,
        }
        for method, val in integrated_eff.items():
            row[f"integrated_eff_{method}"] = val
        summary_rows.append(row)

    # Cross-layer summary correlations
    if len(mean_u_by_layer) >= 2:
        layer_rank_corr = spearmanr(np.array(mean_u_by_layer), np.array(mean_d_by_layer))
    else:
        layer_rank_corr = float("nan")

    # Visualizations: heatmaps and scatter
    for method in rank_methods:
        mats = gains_by_method.get(method, [])
        if mats:
            gain_mat = np.array(mats, dtype=np.float64)
            plot_gain_heatmap(vis_dir, gain_mat, layer_order, prune_grid, method)
    if scatter_u:
        plot_scatter_proxy(
            vis_dir,
            np.array(scatter_u, dtype=np.float64),
            np.array(scatter_d, dtype=np.float64),
            title=f"Proxy Scatter (K={analysis_mc})",
            name="scatter_proxy_vs_random.png",
        )
    if len(mean_u_by_layer) >= 2:
        plot_scatter_proxy(
            vis_dir,
            np.array(mean_u_by_layer, dtype=np.float64),
            np.array(mean_d_by_layer, dtype=np.float64),
            title="Layer Mean: U vs D_rand",
            name="scatter_layer_mean_u_vs_drand.png",
        )

    report = {
        "meta": {
            "timestamp": exp_time,
            "save_dir": save_dir,
            "config_signature": build_custom_tag(cfg),
            "device": str(cfg.device),
            "seed": args.seed,
            "dropout_grid": dropout_grid,
            "prune_grid": prune_grid,
            "mc_samples_grid": mc_samples_grid,
            "analysis_mc_samples": analysis_mc,
            "random_trials": args.random_trials,
            "probe_batches": args.probe_batches,
            "eval_batches": args.eval_batches,
            "rank_methods": rank_methods,
            "selected_layers": len(chosen_layers),
            "has_matplotlib": HAS_MPL,
        },
        "baseline": {
            "loss": float(base_loss),
            "top1": float(base_acc),
        },
        "cross_layer": {
            "spearman_layer_mean_u_vs_drand": float(layer_rank_corr),
        },
        "layers": per_layer,
    }

    json_path = os.path.join(save_dir, "mc_dropout_proxy_report.json")
    report_json = sanitize_for_json(report)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_json, f, indent=2, ensure_ascii=False)

    csv_path = os.path.join(save_dir, "layer_summary.csv")
    if summary_rows:
        cols = []
        for r in summary_rows:
            for k in r.keys():
                if k not in cols:
                    cols.append(k)
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(",".join(cols) + "\n")
            for r in summary_rows:
                f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")

    txt_path = os.path.join(save_dir, "analysis_summary.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("MC-Dropout Proxy Analysis Summary\n")
        f.write("=" * 80 + "\n")
        f.write(f"Save dir: {save_dir}\n")
        f.write(f"Selected layers: {len(chosen_layers)}\n")
        f.write(f"Baseline loss: {base_loss:.6f}\n")
        f.write(f"Baseline top1: {base_acc:.4f}\n")
        f.write(f"Cross-layer Spearman(mean U, mean D_rand): {layer_rank_corr:.6f}\n")
        f.write(f"JSON report: {json_path}\n")
        f.write(f"CSV summary: {csv_path}\n")

    print("=" * 90)
    print("MC-Dropout proxy analysis completed.")
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")
    print(f"TXT:  {txt_path}")
    print("=" * 90)


if __name__ == "__main__":
    main()

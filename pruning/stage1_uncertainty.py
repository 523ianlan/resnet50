"""Stage 1: Uncertainty-guided inter-layer allocation for ResNet-50."""

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from typing import Dict, Optional, Tuple
from torch.utils.data import DataLoader

from models.utils import get_resnet_parent_and_name


class ResNetActivationMonitor:
    """Collect per-layer activation norms during forward passes."""

    def __init__(self, model: nn.Module, layers_to_monitor: Dict[str, nn.Module]):
        self.hooks = []
        self.activations = {name: [] for name in layers_to_monitor}
        self.device = next(model.parameters()).device
        for name, layer in layers_to_monitor.items():
            hook = layer.register_forward_hook(self._get_hook(name))
            self.hooks.append(hook)

    def _get_hook(self, name: str):
        def hook(module, input, output):
            with torch.no_grad():
                # output: [batch, channels, height, width]
                # Use reshape/flatten to support channels_last or non-contiguous tensors.
                flat = output.reshape(output.size(0), output.size(1), -1)
                l2_norm = torch.norm(flat, p=2, dim=2)
                self.activations[name].append(l2_norm.cpu())
            return output
        return hook

    def reset_activations(self):
        self.activations = {name: [] for name in self.activations}

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()


def _set_dropout_active_only(model: nn.Module):
    """Enable dropout layers while keeping BatchNorm frozen."""
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            m.train()
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()


def _replace_module(parent: nn.Module, name: str, new_module: nn.Module) -> bool:
    if hasattr(parent, name):
        setattr(parent, name, new_module)
        return True
    if hasattr(parent, '_modules') and name in parent._modules:
        parent._modules[name] = new_module
        return True
    if isinstance(parent, nn.Sequential) and name.isdigit():
        idx = int(name)
        if idx < len(parent):
            parent[idx] = new_module
            return True
    return False


def _wrap_layers_with_dropout(
    model: nn.Module,
    svd_layers: Dict[str, nn.Module],
    dropout_p: float
) -> Tuple[Dict[str, nn.Module], Dict[str, nn.Module]]:
    """Wrap SVD layers with Dropout2d for MC Dropout sampling."""
    wrapped_layers: Dict[str, nn.Module] = {}
    original_layers: Dict[str, nn.Module] = {}

    for name, layer in svd_layers.items():
        parent, layer_name = get_resnet_parent_and_name(model, name)
        if parent is None or layer_name is None:
            continue
        original = layer
        dropout = nn.Dropout2d(p=dropout_p) if getattr(layer, "layer_kind", "conv") == "conv" else nn.Dropout(p=dropout_p)
        wrapper = nn.Sequential(original, dropout)
        wrapper = wrapper.to(next(original.parameters()).device)
        if _replace_module(parent, layer_name, wrapper):
            wrapped_layers[name] = wrapper
            original_layers[name] = original

    return wrapped_layers, original_layers


def _restore_wrapped_layers(model: nn.Module, original_layers: Dict[str, nn.Module]):
    for name, original in original_layers.items():
        parent, layer_name = get_resnet_parent_and_name(model, name)
        if parent is None or layer_name is None:
            continue
        _replace_module(parent, layer_name, original)


def compute_uncertainty_stage1_r50(
    model: nn.Module,
    svd_layers: Dict[str, nn.Module],
    cal_loader: DataLoader,
    config=None,
    device=None
) -> Dict[str, float]:
    """Estimate layer-wise uncertainty via MC Dropout."""
    if config is None:
        from configs.config import PruningConfig
        config = PruningConfig()
    if device is None:
        device = config.device

    print(f"[Stage 1] Estimating Layer-wise Uncertainty via MC Dropout "
        f"({config.mc_samples} passes, p={getattr(config, 'mc_dropout_p', 'na')}, "
        f"calib_batches={getattr(config, 'calib_batches', 'na')})..."
    )
    print(f"[Stage 1] uncertainty_metric={getattr(config, 'uncertainty_metric', 'mu_over_var')}")

    mc_p = float(getattr(config, 'mc_dropout_p', 0.0))
    if mc_p <= 0:
        print("Warning: mc_dropout_p <= 0, MC Dropout stochasticity may be insufficient.")

    wrapped_layers, original_layers = _wrap_layers_with_dropout(model, svd_layers, mc_p)
    # Monitor post-dropout outputs to reflect MC Dropout uncertainty.
    monitor_layers = wrapped_layers if wrapped_layers else svd_layers
    monitor = ResNetActivationMonitor(model, monitor_layers)

    _set_dropout_active_only(model)

    layer_importance_sum = {name: 0.0 for name in svd_layers}
    layer_importance_count = {name: 0 for name in svd_layers}

    calib_batches = getattr(config, 'calib_batches', 0)
    max_batches = calib_batches if calib_batches and calib_batches > 0 else None

    for batch_idx, (images, _) in enumerate(tqdm(cal_loader, desc="Stage 1: Calibration Batches")):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images = images.to(device)
        monitor.reset_activations()

        with torch.no_grad():
            for _ in tqdm(range(config.mc_samples), desc="Stage 1: MC Sampling", leave=False):
                _ = model(images)

        for name, norm_list in monitor.activations.items():
            if not norm_list:
                continue

            norms_stack = torch.stack(norm_list, dim=0).float()
            mu_c = torch.mean(norms_stack, dim=0)
            var_c = torch.var(norms_stack, dim=0, unbiased=False)
            var_floor = float(getattr(config, 'uncertainty_var_floor', 0.0))
            if var_floor > 0:
                var_c = torch.clamp(var_c, min=var_floor)

            metric = getattr(config, "uncertainty_metric", "mu_over_var")
            if metric == "mu":
                importance = torch.mean(mu_c).item()
            elif metric == "var":
                importance = torch.mean(var_c).item()
            elif metric == "inv_var":
                importance = torch.mean(1.0 / (var_c + config.uncertainty_epsilon)).item()
            else:
                ratio_c = mu_c / (var_c + config.uncertainty_epsilon)
                importance = torch.mean(ratio_c).item()

            layer_importance_sum[name] += importance
            layer_importance_count[name] += 1

    layer_importance = {}
    for name, total in layer_importance_sum.items():
        count = layer_importance_count.get(name, 0)
        if count > 0:
            layer_importance[name] = total / count

    monitor.remove_hooks()
    _restore_wrapped_layers(model, original_layers)

    if not layer_importance:
        print("Warning: no layer importance computed.")
        return {}

    raw_vals = np.array(list(layer_importance.values()), dtype=np.float64)
    print(
        f"Stage 1 raw importance stats: min={raw_vals.min():.6f}, "
        f"max={raw_vals.max():.6f}, mean={raw_vals.mean():.6f}, std={raw_vals.std():.6f}"
    )

    names = list(layer_importance.keys())
    scores = np.array([layer_importance[n] for n in names], dtype=np.float64)

    if getattr(config, "uncertainty_log", False):
        scores = np.log1p(scores)

    clip_pct = float(getattr(config, "uncertainty_clip_percentile", 0.0))
    if clip_pct > 0:
        low = np.percentile(scores, clip_pct)
        high = np.percentile(scores, 100.0 - clip_pct)
        if high > low:
            scores = np.clip(scores, low, high)

    min_s, max_s = scores.min(), scores.max()
    normalized_importance = {}

    for name, score in zip(names, scores):
        if max_s - min_s > 1e-8:
            norm_score = (score - min_s) / (max_s - min_s)
        else:
            norm_score = 0.5
        normalized_importance[name] = norm_score

    print("Stage 1 Complete. Layer importance calculated.")
    return normalized_importance

"""Stage 2: Fisher-aware intra-layer component scoring for ResNet-50."""

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from typing import Dict
from torch.utils.data import DataLoader

from utils.task_utils import build_criterion, prepare_loss_inputs


def compute_fisher_impact_stage2_r50(
    model: nn.Module,
    svd_layers: Dict[str, nn.Module],
    data_loader: DataLoader,
    config=None,
    device=None
) -> Dict[str, np.ndarray]:
    """Compute Fisher-aware impact scores for singular components."""
    if config is None:
        from configs.config import PruningConfig
        config = PruningConfig()
    if device is None:
        device = config.device

    print(
        f"\n[Stage 2] Computing Fisher-Aware Impact Scores "
        f"(using {config.fisher_batches} batches)..."
    )

    model.eval()
    model.zero_grad()

    accum_grad = {
        name: torch.zeros(layer.full_rank, device=device)
        for name, layer in svd_layers.items()
    }
    accum_abs_grad = {
        name: torch.zeros(layer.full_rank, device=device)
        for name, layer in svd_layers.items()
    }

    criterion = build_criterion(config)
    total_batches = 0

    for layer in svd_layers.values():
        layer.reset_fisher_accum()

    for param in model.parameters():
        param.requires_grad = False
    for layer in svd_layers.values():
        score_param = layer.get_score_param()
        score_param.requires_grad = True

    fisher_batches = int(getattr(config, "fisher_batches", 0))
    if fisher_batches <= 0:
        print("Warning: fisher_batches <= 0, using all available batches.")
        max_batches = len(data_loader)
    else:
        max_batches = min(fisher_batches, len(data_loader))

    for i, (images, labels) in enumerate(tqdm(data_loader, desc="Stage 2: Fisher Tracing")):
        if i >= max_batches:
            break

        images = images.to(device)
        labels = labels.to(device)

        if isinstance(labels, tuple):
            labels_a, labels_b, lam = labels
            labels_a = labels_a.to(device)
            labels_b = labels_b.to(device)

            model.zero_grad()
            outputs = model(images)
            outputs_a, labels_a = prepare_loss_inputs(outputs, labels_a, config)
            outputs_b, labels_b = prepare_loss_inputs(outputs, labels_b, config)
            loss_a = criterion(outputs_a, labels_a)
            loss_b = criterion(outputs_b, labels_b)
            loss = lam * loss_a + (1 - lam) * loss_b
        else:
            model.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

        target_params = [layer.get_score_param() for layer in svd_layers.values()]
        grads = torch.autograd.grad(loss, target_params, create_graph=False)

        with torch.no_grad():
            for idx, (name, layer) in enumerate(svd_layers.items()):
                g_i = grads[idx]
                if g_i is None:
                    continue
                accum_grad[name] += g_i.detach()
                accum_abs_grad[name] += torch.abs(g_i.detach())
                layer.update_fisher_accum(g_i.detach())

        total_batches += 1

    final_impact_scores = {}
    print("\nCalculating final impact scores...")

    for name, layer in svd_layers.items():
        if total_batches > 0:
            first_order_mode = getattr(config, "fisher_first_order_mode", "mean_abs")
            if first_order_mode == "abs_mean":
                mean_grad = accum_grad[name] / total_batches
                first_order = torch.abs(mean_grad)
            else:
                first_order = accum_abs_grad[name] / total_batches
            fisher_diag = layer.get_fisher_diagonal()
            metric = getattr(config, "stage2_score_metric", "fisher")
            if metric == "taylor":
                impact = first_order
            elif metric == "hessian":
                impact = fisher_diag
            else:
                w1 = float(getattr(config, "fisher_first_order_weight", 1.0))
                w2 = float(getattr(config, "fisher_second_order_weight", 0.5))
                impact = w1 * first_order + w2 * fisher_diag
            final_impact_scores[name] = impact.detach().cpu().numpy()

    for param in model.parameters():
        param.requires_grad = True

    print(f"Stage 2 Complete. Computed impact scores for {len(final_impact_scores)} layers.")
    return final_impact_scores


def compute_fisher_components_stage2_r50(
    model: nn.Module,
    svd_layers: Dict[str, nn.Module],
    data_loader: DataLoader,
    config=None,
    device=None
) -> Dict[str, Dict[str, np.ndarray]]:
    """Compute first-order and Fisher diagonal components for analysis."""
    if config is None:
        from configs.config import PruningConfig
        config = PruningConfig()
    if device is None:
        device = config.device

    print(
        f"\n[Stage 2] Computing Fisher components "
        f"(using {config.fisher_batches} batches)..."
    )

    model.eval()
    model.zero_grad()

    accum_grad = {
        name: torch.zeros(layer.full_rank, device=device)
        for name, layer in svd_layers.items()
    }
    accum_abs_grad = {
        name: torch.zeros(layer.full_rank, device=device)
        for name, layer in svd_layers.items()
    }

    criterion = build_criterion(config)
    total_batches = 0

    for layer in svd_layers.values():
        layer.reset_fisher_accum()

    for param in model.parameters():
        param.requires_grad = False
    for layer in svd_layers.values():
        score_param = layer.get_score_param()
        score_param.requires_grad = True

    fisher_batches = int(getattr(config, "fisher_batches", 0))
    if fisher_batches <= 0:
        print("Warning: fisher_batches <= 0, using all available batches.")
        max_batches = len(data_loader)
    else:
        max_batches = min(fisher_batches, len(data_loader))

    for i, (images, labels) in enumerate(tqdm(data_loader, desc="Stage 2: Fisher Tracing")):
        if i >= max_batches:
            break

        images = images.to(device)
        labels = labels.to(device)

        if isinstance(labels, tuple):
            labels_a, labels_b, lam = labels
            labels_a = labels_a.to(device)
            labels_b = labels_b.to(device)

            model.zero_grad()
            outputs = model(images)
            outputs_a, labels_a = prepare_loss_inputs(outputs, labels_a, config)
            outputs_b, labels_b = prepare_loss_inputs(outputs, labels_b, config)
            loss_a = criterion(outputs_a, labels_a)
            loss_b = criterion(outputs_b, labels_b)
            loss = lam * loss_a + (1 - lam) * loss_b
        else:
            model.zero_grad()
            outputs = model(images)
            outputs, labels = prepare_loss_inputs(outputs, labels, config)
            outputs, labels = prepare_loss_inputs(outputs, labels, config)
            loss = criterion(outputs, labels)

        target_params = [layer.get_score_param() for layer in svd_layers.values()]
        grads = torch.autograd.grad(loss, target_params, create_graph=False)

        with torch.no_grad():
            for idx, (name, layer) in enumerate(svd_layers.items()):
                g_i = grads[idx]
                if g_i is None:
                    continue
                accum_grad[name] += g_i.detach()
                accum_abs_grad[name] += torch.abs(g_i.detach())
                layer.update_fisher_accum(g_i.detach())

        total_batches += 1

    components: Dict[str, Dict[str, np.ndarray]] = {}
    print("\nCalculating Fisher components...")

    for name, layer in svd_layers.items():
        if total_batches > 0:
            first_order_mode = getattr(config, "fisher_first_order_mode", "mean_abs")
            if first_order_mode == "abs_mean":
                mean_grad = accum_grad[name] / total_batches
                first_order = torch.abs(mean_grad)
            else:
                first_order = accum_abs_grad[name] / total_batches
            fisher_diag = layer.get_fisher_diagonal()
            components[name] = {
                "first_order": first_order.detach().cpu().numpy(),
                "fisher_diag": fisher_diag.detach().cpu().numpy(),
            }

    for param in model.parameters():
        param.requires_grad = True

    print(f"Stage 2 Complete. Computed Fisher components for {len(components)} layers.")
    return components

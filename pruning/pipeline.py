"""Stage-6 pruning orchestration for two-stage allocation workflows."""

import os
from typing import Any, Dict

import numpy as np
import torch.nn as nn

from pruning.allocation import allocate_pruning_binary_search_r50
from pruning.stage1_uncertainty import compute_uncertainty_stage1_r50
from pruning.stage2_fisher import compute_fisher_impact_stage2_r50
from utils.visualization import (
    analyze_all_layer_sensitivity_r50,
    plot_layer_budget_allocation_r50,
)


def _compute_layer_importance_from_scores(scores_dict: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Collapse per-component scores into normalized per-layer importance."""
    raw: Dict[str, float] = {}
    for layer_name, scores in scores_dict.items():
        if scores is None or len(scores) == 0:
            continue
        raw[layer_name] = float(np.mean(scores))

    if not raw:
        return {}

    values = np.array(list(raw.values()), dtype=np.float64)
    min_value, max_value = values.min(), values.max()

    normalized: Dict[str, float] = {}
    for layer_name, value in raw.items():
        if max_value - min_value > 1e-8:
            normalized[layer_name] = (value - min_value) / (max_value - min_value)
        else:
            normalized[layer_name] = 0.5
    return normalized


def _compute_energy_scores(svd_layers: Dict[str, nn.Module]) -> Dict[str, np.ndarray]:
    """Use squared singular values as a lightweight component score."""
    return {
        name: (layer.get_sigma().detach().cpu().numpy() ** 2)
        for name, layer in svd_layers.items()
    }


def run_two_stage_pruning_allocation_r50(
    model: nn.Module,
    svd_layers: Dict[str, nn.Module],
    train_loader,
    calib_loader,
    save_dir: str,
    orig_params: int,
    config,
) -> Dict[str, Any]:
    """
    Execute stage-6 orchestration and return the artifacts needed by later steps.

    This preserves the previous main.py behavior while moving strategy-specific
    branching out of the top-level training script.
    """
    print("\n6. Two-Stage Adaptive Budget Allocation")

    allocation_strategy = getattr(config, "allocation_strategy", "binary_search")
    stage2_metric = getattr(config, "stage2_score_metric", "fisher")

    layer_importance_stage1: Dict[str, float] = {}
    component_impact_scores: Dict[str, np.ndarray] = {}

    if allocation_strategy == "global_fisher":
        print("\n" + "=" * 80)
        print("Stage 1 (Global Fisher Allocation): using Fisher-based layer scores")
        print("=" * 80)

        # We always need Fisher for the inter-layer allocation in this mode.
        fisher_scores = compute_fisher_impact_stage2_r50(
            model, svd_layers, train_loader, config=config
        )
        layer_importance_stage1 = _compute_layer_importance_from_scores(fisher_scores)

        # Decouple the component selection metric from the allocation metric.
        if stage2_metric == "magnitude":
            component_impact_scores = {}
            print("Budget Allocation: Fisher-Mean | Component Selection: Magnitude")
        elif stage2_metric == "energy":
            component_impact_scores = _compute_energy_scores(svd_layers)
            print("Budget Allocation: Fisher-Mean | Component Selection: Energy")
        else:
            component_impact_scores = fisher_scores
            print("Budget Allocation: Fisher-Mean | Component Selection: Fisher")
    else:
        print("\n" + "=" * 80)
        print("Stage 1: Inter-layer Budget Allocation (Uncertainty Estimation)")
        print("=" * 80)

        layer_importance_stage1 = compute_uncertainty_stage1_r50(
            model, svd_layers, calib_loader, config=config
        )

        print("\n" + "=" * 80)
        print("Stage 2: Intra-layer Component Selection (Fisher-Aware Scoring)")
        print("=" * 80)

        if stage2_metric == "magnitude":
            component_impact_scores = {}
            print("Stage 2 metric: magnitude (no Fisher scores).")
        elif stage2_metric == "energy":
            component_impact_scores = _compute_energy_scores(svd_layers)
            print("Stage 2 metric: energy (sigma^2).")
        else:
            if bool(getattr(config, "use_fisher_scores", True)):
                component_impact_scores = compute_fisher_impact_stage2_r50(
                    model, svd_layers, train_loader, config=config
                )
            else:
                print("Stage 2 Fisher scoring disabled; using magnitude-based ranking.")

    vis_dir = os.path.join(save_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    print("\n--- Sensitivity Analysis Visualization ---")
    analyze_all_layer_sensitivity_r50(svd_layers, component_impact_scores, vis_dir)

    print("\n" + "=" * 80)
    print(f"Allocation: Binary Search for Target Compression ({config.target_compression * 100:.1f}%)")
    print("=" * 80)

    keep_ratios = allocate_pruning_binary_search_r50(
        layer_importance_stage1,
        svd_layers,
        config=config,
        total_model_params=orig_params,
    )

    print("\n--- Visualize Budget Allocation ---")
    type_keep_ratios = plot_layer_budget_allocation_r50(
        svd_layers,
        keep_ratios,
        layer_importance_stage1,
        vis_dir,
    )

    return {
        "layer_importance_stage1": layer_importance_stage1,
        "component_impact_scores": component_impact_scores,
        "keep_ratios": keep_ratios,
        "type_keep_ratios": type_keep_ratios,
    }

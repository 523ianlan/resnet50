"""Budget Allocation - 對應 ViT 的 allocation.py"""

import torch
import numpy as np
import torch.nn as nn
from typing import Dict, Tuple
from configs.config import PruningConfig


def allocate_pruning_binary_search_r50(
    layer_importance_stage1: Dict[str, float],
    svd_layers: Dict[str, nn.Module],
    config: PruningConfig = None,
    total_model_params: int = None
) -> Dict[str, float]:
    # Stage 1 uncertainty-guided allocation with binary search.
    if config is None:
        config = PruningConfig()

    # 1. Parameter budget setup
    total_prunable_params = 0
    min_prunable_params = 0
    for layer in svd_layers.values():
        total_prunable_params += layer.get_original_param_count()

        rank_min = min(config.min_rank, layer.full_rank)
        min_prunable_params += layer.get_pruned_param_count(rank_min)

    fixed_params = 0
    if total_model_params is not None:
        fixed_params = max(total_model_params - total_prunable_params, 0)
        target_total_params = total_model_params * (1 - config.target_compression)
        target_params = target_total_params - fixed_params
        if target_params < min_prunable_params:
            print("Warning: target budget below minimum prunable params; clamping.")
            target_params = min_prunable_params
    else:
        target_params = total_prunable_params * (1 - config.target_compression)

    # 2. Binary search for base scale
    low, high = config.binary_search_low, config.binary_search_high
    best_scale = 0.0
    final_keep_ratios = {}
    best_error = float('inf')

    if total_model_params is not None:
        print("")
        print(f"Binary search for target prunable params: {target_params/1e6:.2f}M (total target: {target_total_params/1e6:.2f}M)")
    else:
        print("")
        print(f"Binary search for target params: {target_params/1e6:.2f}M")

    for iteration in range(config.binary_search_iterations):
        mid_scale = (low + high) / 2.0
        current_total_params = 0
        temp_keep_ratios = {}

        for name, layer in svd_layers.items():
            importance = layer_importance_stage1.get(name, 0.5)

            pruning_ratio = mid_scale * (1.0 - config.uncertainty_alpha * importance)
            pruning_ratio = np.clip(
                pruning_ratio,
                config.pruning_clip_low,
                config.pruning_clip_high
            )

            keep_ratio = 1.0 - pruning_ratio
            rank = layer.get_keep_count(keep_ratio)
            keep_ratio = rank / layer.full_rank
            layer_params = layer.get_pruned_param_count(rank)

            current_total_params += layer_params
            temp_keep_ratios[name] = keep_ratio

        denom = max(target_params, 1e-12)
        current_error = abs(current_total_params - target_params) / denom

        if current_error < best_error:
            best_error = current_error
            best_scale = mid_scale
            final_keep_ratios = temp_keep_ratios.copy()

        print(f"  Iter {iteration+1}: scale={mid_scale:.4f}, "
              f"params={current_total_params/1e6:.2f}M, "
              f"error={current_error:.4f}")

        if current_error < config.binary_search_tolerance:
            print(f"  Converged early at iteration {iteration+1}")
            break

        if current_total_params > target_params:
            low = mid_scale
        else:
            high = mid_scale

    print("")
    print(f"Optimal Base Scale found: {best_scale:.4f} (error: {best_error:.4f})")

    # Summary table
    print("")
    print(f"{'Layer Name':<35} {'Stability':<12} {'Keep Ratio':<12} {'Rank':<10}")
    print("-" * 70)

    final_params = 0
    for name in svd_layers:
        imp = layer_importance_stage1.get(name, 0)
        kr = final_keep_ratios[name]
        rank = svd_layers[name].get_keep_count(kr)
        kr = rank / svd_layers[name].full_rank

        full_rank = svd_layers[name].full_rank
        print(f"{name.split('.')[-1]:<35} {imp:.4f}      {kr:.2%}      {rank}/{full_rank}")

        final_params += svd_layers[name].get_pruned_param_count(rank)

    if total_model_params is not None:
        final_total_params = final_params + fixed_params
        actual_compression = 1 - (final_total_params / total_model_params)
        print("-" * 70)
        print(f"Target Compression: {config.target_compression:.2%}")
        print(f"Actual Compression (total): {actual_compression:.2%}")
        print(f"Total Params: {total_model_params/1e6:.2f}M -> {final_total_params/1e6:.2f}M")
        print(f"Prunable Params: {total_prunable_params/1e6:.2f}M -> {final_params/1e6:.2f}M")
    else:
        actual_compression = 1 - (final_params / total_prunable_params)
        print("-" * 70)
        print(f"Target Compression: {config.target_compression:.2%}")
        print(f"Actual Compression: {actual_compression:.2%}")
        print(f"Total Params: {total_prunable_params/1e6:.2f}M -> {final_params/1e6:.2f}M")

    return final_keep_ratios

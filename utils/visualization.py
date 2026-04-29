"""Visualization tools - Corresponds to ViT visualization.py"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple


def plot_training_history(history: Dict, save_dir: str):
    """Plot training history"""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss curve
    axes[0, 0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0, 0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Top-1 accuracy
    axes[0, 1].plot(history['train_top1'], label='Train Top-1', linewidth=2)
    axes[0, 1].plot(history['val_top1'], label='Val Top-1', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Top-1 Accuracy (%)')
    axes[0, 1].set_title('Top-1 Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Top-5 accuracy
    axes[1, 0].plot(history['train_top5'], label='Train Top-5', linewidth=2)
    axes[1, 0].plot(history['val_top5'], label='Val Top-5', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Top-5 Accuracy (%)')
    axes[1, 0].set_title('Top-5 Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Learning rate curve
    axes[1, 1].plot(history['learning_rate'], label='Learning Rate', color='purple', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'fine_tune_history.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Training history plot saved to: {save_path}")


def plot_singular_value_sensitivity_r50(
    svd_layer: nn.Module,
    impact_scores: np.ndarray,
    layer_name: str,
    save_dir: str
) -> Tuple[float, List]:
    """
    Plot comparison between Singular Values and Fisher Impact Score
    
    Returns:
        (correlation, []) keep the same return format as ViT version
    """
    # Get singular values
    singular_values = svd_layer.get_sigma().cpu().detach().numpy()
    
    # Ensure consistent length
    min_len = min(len(singular_values), len(impact_scores))
    singular_values = singular_values[:min_len]
    impact_scores = impact_scores[:min_len]
    
    # Create plots
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # 1. Singular value distribution
    axes[0].bar(range(min_len), singular_values, alpha=0.7, color='skyblue')
    axes[0].set_xlabel('Singular Value Index')
    axes[0].set_ylabel('Magnitude')
    axes[0].set_title(f'{layer_name} - Singular Value Distribution')
    axes[0].grid(True, alpha=0.3)
    
    # 2. Fisher Impact Score distribution
    axes[1].bar(range(min_len), impact_scores, alpha=0.7, color='salmon')
    axes[1].set_xlabel('Singular Value Index')
    axes[1].set_ylabel('Fisher Impact Score')
    axes[1].set_title(f'{layer_name} - Fisher Impact Score Distribution')
    axes[1].grid(True, alpha=0.3)
    
    # 3. Comparison scatter plot
    axes[2].scatter(singular_values, impact_scores, alpha=0.6, s=30)
    
    # Calculate correlation coefficient
    correlation = np.corrcoef(singular_values, impact_scores)[0, 1]
    
    # Add trend line
    z = np.polyfit(singular_values, impact_scores, 1)
    p = np.poly1d(z)
    axes[2].plot(np.sort(singular_values), p(np.sort(singular_values)),
                "r--", alpha=0.8, label=f'Trend line (r={correlation:.3f})')
    
    axes[2].set_xlabel('Singular Value Magnitude')
    axes[2].set_ylabel('Fisher Impact Score')
    axes[2].set_title(f'{layer_name} - Magnitude vs Impact Score')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    os.makedirs(save_dir, exist_ok=True)
    safe_name = layer_name.replace('.', '_').replace('/', '_')
    save_path = os.path.join(save_dir, f'singular_sensitivity_{safe_name}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return correlation, []


def plot_layer_budget_allocation_r50(
    svd_layers: Dict[str, nn.Module],
    keep_ratios: Dict[str, float],
    layer_importance: Dict[str, float],
    save_dir: str
) -> Dict[str, List[float]]:
    """
    Plot layer-wise budget allocation
    
    Returns:
        Dictionary of keep ratios for each layer type
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Prepare Data
    layer_names = list(svd_layers.keys())
    keep_values = [keep_ratios[name] for name in layer_names]
    importance_values = [layer_importance.get(name, 0.5) for name in layer_names]
    
    # Distinguish conv bottlenecks, standard conv, and linear layers
    is_bottleneck = [getattr(svd_layers[name], "is_bottleneck", False) for name in layer_names]
    layer_kinds = [getattr(svd_layers[name], "layer_kind", "conv") for name in layer_names]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Keep ratio distribution
    axes[0, 0].hist(keep_values, bins=20, alpha=0.7, color='skyblue')
    axes[0, 0].set_xlabel('Keep Ratio')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Keep Ratio Distribution')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Keep ratio vs Importance
    colors = [
        'red' if bottleneck else ('green' if kind == 'linear' else 'blue')
        for bottleneck, kind in zip(is_bottleneck, layer_kinds)
    ]
    axes[0, 1].scatter(importance_values, keep_values, c=colors, alpha=0.6)
    axes[0, 1].set_xlabel('Importance Score (Stage 1)')
    axes[0, 1].set_ylabel('Keep Ratio')
    axes[0, 1].set_title('Importance vs Keep Ratio (Red=Bottleneck, Green=Linear)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Layer type statistics
    bottleneck_keep = [keep for keep, b in zip(keep_values, is_bottleneck) if b]
    standard_keep = [keep for keep, b, kind in zip(keep_values, is_bottleneck, layer_kinds) if not b and kind != 'linear']
    linear_keep = [keep for keep, kind in zip(keep_values, layer_kinds) if kind == 'linear']

    boxplot_values = []
    boxplot_labels = []
    if bottleneck_keep:
        boxplot_values.append(bottleneck_keep)
        boxplot_labels.append('Bottleneck')
    if standard_keep:
        boxplot_values.append(standard_keep)
        boxplot_labels.append('Standard')
    if linear_keep:
        boxplot_values.append(linear_keep)
        boxplot_labels.append('Linear')
    if not boxplot_values:
        boxplot_values = [keep_values]
        boxplot_labels = ['All']

    axes[1, 0].boxplot(boxplot_values, labels=boxplot_labels)
    axes[1, 0].set_ylabel('Keep Ratio')
    axes[1, 0].set_title('Keep Ratio by Layer Type')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Sorted keep ratios
    sorted_indices = np.argsort(keep_values)
    sorted_keep = [keep_values[i] for i in sorted_indices]
    sorted_names = [layer_names[i].split('.')[-1] for i in sorted_indices]
    
    axes[1, 1].barh(range(len(sorted_keep)), sorted_keep)
    axes[1, 1].set_yticks(range(len(sorted_keep)))
    axes[1, 1].set_yticklabels(sorted_names, fontsize=8)
    axes[1, 1].set_xlabel('Keep Ratio')
    axes[1, 1].set_title('Layers Sorted by Keep Ratio')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'layer_budget_allocation.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # Collect keep ratios for each layer type
    type_keep_ratios = {
        'bottleneck': bottleneck_keep,
        'standard': standard_keep,
        'linear': linear_keep,
        'all': keep_values
    }
    
    return type_keep_ratios


def analyze_all_layer_sensitivity_r50(
    svd_layers: Dict[str, nn.Module],
    component_impact_scores: Dict[str, np.ndarray],
    save_dir: str
) -> Dict[str, float]:
    """
    Analyze singular value sensitivity for all layers
    
    Returns:
        Dictionary of correlation coefficients for each layer
    """
    print("\n" + "="*80)
    print("ANALYZING SINGULAR VALUE SENSITIVITY")
    print("="*80)
    
    correlations = {}
    
    for name, layer in svd_layers.items():
        if name not in component_impact_scores:
            continue
        
        print(f"Analyzing {name}...")
        corr, _ = plot_singular_value_sensitivity_r50(
            layer, component_impact_scores[name], name, save_dir
        )
        correlations[name] = corr
    
    if correlations:
        avg_correlation = np.mean(list(correlations.values()))
        print(f"\nAverage correlation: {avg_correlation:.3f}")
    
    return correlations

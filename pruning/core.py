"""Pruning core execution for convolutional and linear layers."""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional
from configs.config import PruningConfig
from models.custom_layers import LowRankLinear, SimpleSVDConv, SimpleSVDLinear
from models.utils import get_resnet_parent_and_name


def replace_prunable_layer(
    model: nn.Module,
    layer_path: str,
    svd_layer,
    keep_ratio: float,
    impact_scores: Optional[np.ndarray] = None,
    config: PruningConfig = None
) -> Optional[int]:
    """
    Replace a Conv2d or Linear layer with a low-rank decomposition.
    
    Args:
        model: Target model
        layer_path: Layer path
        svd_layer: Original SVD layer
        keep_ratio: Keep ratio
        impact_scores: Fisher impact scores
        config: Configuration
    
    Returns:
        Pruned rank, return None if failed
    """
    if config is None:
        config = PruningConfig()
    
    parent, layer_name = get_resnet_parent_and_name(model, layer_path)
    
    if parent is None:
        print(f" Warning: Cannot find parent for {layer_path}")
        return None
    
    U_selected, S_selected, Vh_selected = svd_layer.get_uvs_for_ratio(
        keep_ratio, impact_scores
    )
    rank = U_selected.shape[1]

    if isinstance(svd_layer, SimpleSVDConv):
        first = nn.Conv2d(
            svd_layer.cin, rank,
            kernel_size=(svd_layer.kh, svd_layer.kw),
            stride=svd_layer.stride,
            padding=svd_layer.padding,
            bias=False
        ).to(config.device)
        second = nn.Conv2d(
            rank, svd_layer.cout,
            kernel_size=1,
            bias=(svd_layer.bias is not None)
        ).to(config.device)

        with torch.no_grad():
            left = (torch.diag(S_selected) @ Vh_selected).view(
                rank, svd_layer.cin, svd_layer.kh, svd_layer.kw
            )
            right = U_selected.view(svd_layer.cout, rank, 1, 1)
            first.weight.copy_(left)
            second.weight.copy_(right)
            if svd_layer.bias is not None:
                second.bias.copy_(svd_layer.bias.to(config.device))

        replacement = nn.Sequential(first, second)
    elif isinstance(svd_layer, SimpleSVDLinear):
        first = nn.Linear(svd_layer.in_features, rank, bias=False).to(config.device)
        second = nn.Linear(rank, svd_layer.out_features, bias=(svd_layer.bias is not None)).to(config.device)

        with torch.no_grad():
            left = torch.diag(S_selected) @ Vh_selected
            right = U_selected
            first.weight.copy_(left)
            second.weight.copy_(right)
            if svd_layer.bias is not None:
                second.bias.copy_(svd_layer.bias.to(config.device))

        replacement = nn.Sequential(first, second)
    else:
        print(f" Warning: Unsupported SVD layer type for {layer_path}")
        return None

    if hasattr(parent, layer_name):
        setattr(parent, layer_name, replacement)
    elif hasattr(parent, '_modules') and layer_name in parent._modules:
        parent._modules[layer_name] = replacement
    elif isinstance(parent, nn.Sequential) and layer_name.isdigit():
        parent[int(layer_name)] = replacement

    return rank


def replace_conv_layer(
    model: nn.Module,
    layer_path: str,
    svd_layer: SimpleSVDConv,
    keep_ratio: float,
    impact_scores: Optional[np.ndarray] = None,
    config: PruningConfig = None
) -> Optional[int]:
    return replace_prunable_layer(
        model, layer_path, svd_layer, keep_ratio, impact_scores=impact_scores, config=config
    )


def replace_resnet_conv_layer(
    model: nn.Module,
    layer_path: str,
    svd_layer: SimpleSVDConv,
    keep_ratio: float,
    impact_scores: Optional[np.ndarray] = None,
    config: PruningConfig = None
) -> Optional[int]:
    """
    ResNet specific convolutional layer replacement (corresponds to ViT)
    Effectively an alias for replace_conv_layer
    """
    return replace_prunable_layer(
        model, layer_path, svd_layer, keep_ratio,
        impact_scores=impact_scores, config=config
    )

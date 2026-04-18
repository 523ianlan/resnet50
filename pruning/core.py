"""Pruning core execution - Corresponds to ViT core.py"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Tuple
from configs.config import PruningConfig
from models.custom_layers import LowRankConv, SimpleSVDConv
from models.utils import get_resnet_parent_and_name


def replace_conv_layer(
    model: nn.Module,
    layer_path: str,
    svd_layer: SimpleSVDConv,
    keep_ratio: float,
    impact_scores: Optional[np.ndarray] = None,
    config: PruningConfig = None
) -> Optional[int]:
    """
    Replace convolutional layer with low-rank decomposition format
    
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
    
    # Calculate target rank
    
    # Get pruned weights
    U_selected, S_selected, Vh_selected = svd_layer.get_uvs_for_ratio(
        keep_ratio, impact_scores
    )
    rank = U_selected.shape[1]

    conv1 = nn.Conv2d(
        svd_layer.cin, rank,
        kernel_size=(svd_layer.kh, svd_layer.kw),
        stride=svd_layer.stride,
        padding=svd_layer.padding,
        bias=False
    ).to(config.device)
    
    # Second convolution: 1x1 conv, output original channels
    conv2 = nn.Conv2d(
        rank, svd_layer.cout,
        kernel_size=1,
        bias=(svd_layer.bias is not None)
    ).to(config.device)
    
    # Copy weights
    with torch.no_grad():
        # A = diag(S_selected) @ Vh_selected: [rank, cin, kh, kw]
        A = (torch.diag(S_selected) @ Vh_selected).view(rank, svd_layer.cin, svd_layer.kh, svd_layer.kw)
        conv1.weight.copy_(A)
        
        # B = U_selected: [cout, rank, 1, 1]
        B = U_selected.view(svd_layer.cout, rank, 1, 1)
        conv2.weight.copy_(B)
        
        if svd_layer.bias is not None:
            conv2.bias.copy_(svd_layer.bias.to(config.device))
    
    # Replace with Sequential
    seq = nn.Sequential(conv1, conv2)
    
    if hasattr(parent, layer_name):
        setattr(parent, layer_name, seq)
    elif hasattr(parent, '_modules') and layer_name in parent._modules:
        parent._modules[layer_name] = seq
    elif isinstance(parent, nn.Sequential) and layer_name.isdigit():
        parent[int(layer_name)] = seq
    
    return rank


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
    return replace_conv_layer(
        model, layer_path, svd_layer, keep_ratio,
        impact_scores=impact_scores, config=config
    )

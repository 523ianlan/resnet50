"""ResNet model工具函數 - 對應 ViT 的 utils.py"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, List, Any


def get_resnet_parent_and_name(
    model: nn.Module,
    layer_path: str
) -> Tuple[Optional[nn.Module], Optional[str]]:
    """
    Get parent module and layer name based on layer path
    
    Args:
        model: 模型
        layer_path: Layer path，e.g. 'layer1.0.conv1'
    
    Returns:
        (parent_module, layer_name) or (None, None) if not found
    """
    parts = layer_path.split('.')
    parent = model
    
    # Traverse up to the second-to-last layer
    for part in parts[:-1]:
        if part.isdigit():
            part = int(part)
        
        if isinstance(parent, nn.Sequential):
            if part < len(parent):
                parent = parent[part]
            else:
                return None, None
        elif hasattr(parent, part):
            parent = getattr(parent, part)
        elif hasattr(parent, '_modules') and part in parent._modules:
            parent = parent._modules[part]
        else:
            # Try integer index
            if isinstance(part, int) and hasattr(parent, '_modules'):
                modules = list(parent._modules.values())
                if part < len(modules):
                    parent = modules[part]
                else:
                    return None, None
    
    layer_name = parts[-1]
    return parent, layer_name


def collect_resnet_conv_layers(model: nn.Module) -> List[str]:
    """
    Collect paths of all convolutional layers in ResNet
    
    Args:
        model: ResNet model
    
    Returns:
        List of convolutional layer paths
    """
    conv_paths = []
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_paths.append(name)
    
    return conv_paths


def collect_prunable_layers(
    model: nn.Module,
    include_conv: bool = True,
    include_linear: bool = True,
) -> List[str]:
    """Collect prunable Conv2d and/or Linear layer paths."""
    layer_paths: List[str] = []

    for name, module in model.named_modules():
        if include_conv and isinstance(module, nn.Conv2d):
            layer_paths.append(name)
        elif include_linear and isinstance(module, nn.Linear):
            layer_paths.append(name)

    return layer_paths


def get_resnet_layer_info(model: nn.Module) -> Dict[str, Dict[str, Any]]:
    """
    Get detailed information for each layer in ResNet (for analysis)
    
    Returns:
        Dictionary of layer info, containing:
        - type: Layer type
        - in_channels: Input channels
        - out_channels: Output channels
        - kernel_size: Kernel size
        - is_bottleneck: Whether it's a bottleneck layer
        - block_idx: Parent block index
        - layer_idx: Parent layer index
    """
    layer_info = {}
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            info = {
                'type': 'conv2d',
                'in_channels': module.in_channels,
                'out_channels': module.out_channels,
                'kernel_size': module.kernel_size,
                'stride': module.stride,
                'padding': module.padding,
                'has_bias': module.bias is not None
            }
            
            # Determine if it's a bottleneck layer
            is_bottleneck = (
                module.kernel_size == (1, 1) and 
                'downsample' not in name and
                module.in_channels != module.out_channels
            )
            info['is_bottleneck'] = is_bottleneck
            
            # Parse position information
            if name.startswith('layer'):
                parts = name.split('.')
                info['layer_idx'] = int(parts[0].replace('layer', ''))
                if len(parts) > 1 and parts[1].isdigit():
                    info['block_idx'] = int(parts[1])
            
            layer_info[name] = info
    
    return layer_info

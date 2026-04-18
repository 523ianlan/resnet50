"""ResNet Model setup - Corresponds to ViT vit_setup.py"""

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from typing import Tuple, Optional, Dict, Any

def setup_resnet_model(
    config,
    pretrained: bool = True,
    weights_version: str = "IMAGENET1K_V1"
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Load and setup ResNet-50 model
    
    Args:
        config: Pruning config
        pretrained: Whether to use pretrained weights
        weights_version: Pretrained weights version
    
    Returns:
        model: ResNet-50 model
        data_config: Data config (ViT compatible format)
    """
    
    if pretrained:
        if weights_version == "IMAGENET1K_V1":
            weights = ResNet50_Weights.IMAGENET1K_V1
        else:
            weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
        print(f"Loaded pretrained ResNet-50 with {weights_version}")
    else:
        model = resnet50(weights=None)
        print("Loaded ResNet-50 without pretraining")
    
    model = model.to(config.device)
    model.eval()
    
    # Construct ViT compatible data config
    data_config = {
        'input_size': 224,
        'interpolation': 'bilinear',
        'crop_pct': 0.875,
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    }
    
    return model, data_config


def get_resnet_model(config) -> nn.Module:
    """Simplified model loading function"""
    model, _ = setup_resnet_model(config)
    return model
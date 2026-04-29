"""Evaluation metrics - Corresponds to ViT metrics.py"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
from torch.utils.data import DataLoader
from fvcore.nn import FlopCountAnalysis

from configs.config import PruningConfig

def evaluate_with_topk_r50(
    model: nn.Module,
    loader: DataLoader,
    config: PruningConfig = None,
    device: Optional[torch.device] = None
) -> Tuple[float, float]:
    """
    Evaluate model, return top-1 and top-5 accuracy (using Float64 high precision)
    """
    if config is None:
        config = PruningConfig()
    if device is None:
        device = config.device
    
    model.eval()
    top1_correct = 0
    top5_correct = 0
    total = 0

    device = device if device is not None else next(model.parameters()).device
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            if config and hasattr(config, 'eval_max_batches') and config.eval_max_batches > 0 and i >= config.eval_max_batches:
                break

            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            
            # Top-1
            _, top1_pred = torch.max(outputs, 1)
            top1_correct += (top1_pred == labels).sum().item()
            
            topk = min(5, outputs.size(1))
            _, top5_pred = outputs.topk(topk, 1, True, True)
            top5_correct += (top5_pred == labels.view(-1, 1)).any(dim=1).sum().item()
            
            total += labels.size(0)
    
    return 100.0 * top1_correct / total, 100.0 * top5_correct / total


def compute_flops_resnet(
    model: nn.Module,
    input_size: Tuple[int, int, int, int] = (1, 3, 224, 224)
) -> Optional[float]:
    """
    Calculate model FLOPs
    """
    model.eval()
    device = next(model.parameters()).device
    dummy_input = torch.randn(input_size).to(device)
    
    if hasattr(model, 'to') and hasattr(dummy_input, 'to'):
        # Ensure dummy input follows channel format
        if next(model.parameters()).is_contiguous(memory_format=torch.channels_last):
            dummy_input = dummy_input.to(memory_format=torch.channels_last)

    try:
        flops = FlopCountAnalysis(model, dummy_input)
        return flops.total()
    except ImportError:
        print("fvcore not installed. Install with: pip install fvcore")
        return None
    except Exception as e:
        print(f"Error computing FLOPs: {e}")
        return None

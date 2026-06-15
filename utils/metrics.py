"""Evaluation metrics - Corresponds to ViT metrics.py"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
from torch.utils.data import DataLoader
from fvcore.nn import FlopCountAnalysis

from configs.config import PruningConfig
from utils.task_utils import finalize_metric_totals, init_metric_totals, is_regression_task, update_metric_totals

def evaluate_with_topk_r50(
    model: nn.Module,
    loader: DataLoader,
    config: PruningConfig = None,
    device: Optional[torch.device] = None
) -> Tuple[float, float]:
    """
    Evaluate model, return top-1 and top-5 accuracy (using Float64 high precision)
    """
    if config is not None and is_regression_task(config):
        raise ValueError("evaluate_with_topk_r50 is classification-only. Use evaluate_model for regression.")
    if config is None:
        config = PruningConfig()
    if device is None:
        device = config.device
    
    metrics = evaluate_model(model, loader, config=config, device=device)
    return float(metrics["top1"]), float(metrics["top5"])


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    config: PruningConfig = None,
    device: Optional[torch.device] = None
) -> Dict[str, float]:
    """Evaluate model and return task-aware metrics."""
    if config is None:
        config = PruningConfig()
    if device is None:
        device = config.device

    model.eval()
    totals = init_metric_totals(config)
    device = device if device is not None else next(model.parameters()).device

    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            if (
                config
                and hasattr(config, 'eval_max_batches')
                and config.eval_max_batches > 0
                and i >= config.eval_max_batches
            ):
                break

            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            update_metric_totals(totals, outputs, labels, config)

    return finalize_metric_totals(totals, config)


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

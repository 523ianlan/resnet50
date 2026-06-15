"""Task-aware helpers shared across training, evaluation, and reporting."""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


CLASSIFICATION = "classification"
REGRESSION = "regression"


def get_task_type(config) -> str:
    return str(getattr(config, "task_type", CLASSIFICATION)).strip().lower()


def is_regression_task(config) -> bool:
    return get_task_type(config) == REGRESSION


def get_output_dim(config) -> int:
    if is_regression_task(config):
        target_columns = getattr(config, "target_columns", None)
        if isinstance(target_columns, (list, tuple)) and len(target_columns) > 0:
            return len(target_columns)
        return int(getattr(config, "target_dim", 1))
    return int(getattr(config, "num_classes", 1000))


def get_loss_name(config) -> str:
    explicit = getattr(config, "loss_name", None)
    if explicit:
        return str(explicit).strip().lower()
    return "mse" if is_regression_task(config) else "cross_entropy"


def get_primary_metric_name(config) -> str:
    explicit = getattr(config, "primary_metric", None)
    if explicit:
        return str(explicit).strip().lower()
    return "rmse" if is_regression_task(config) else "top1"


def get_secondary_metric_name(config) -> Optional[str]:
    explicit = getattr(config, "secondary_metric", None)
    if explicit:
        return str(explicit).strip().lower()
    return "mae" if is_regression_task(config) else "top5"


def is_higher_better(config, metric_name: Optional[str] = None) -> bool:
    if metric_name is None or metric_name == get_primary_metric_name(config):
        if hasattr(config, "higher_is_better"):
            return bool(getattr(config, "higher_is_better"))
        return not is_regression_task(config)

    metric_name = metric_name.lower()
    return metric_name in {"top1", "top5", "accuracy"}


def get_metric_display_name(metric_name: Optional[str]) -> str:
    if metric_name is None:
        return "Metric"

    display_map = {
        "top1": "Top-1 Accuracy (%)",
        "top5": "Top-5 Accuracy (%)",
        "mse": "MSE",
        "rmse": "RMSE",
        "mae": "MAE",
        "loss": "Loss",
    }
    return display_map.get(metric_name.lower(), metric_name.upper())


def build_criterion(config) -> nn.Module:
    loss_name = get_loss_name(config)

    if is_regression_task(config):
        if loss_name == "mse":
            return nn.MSELoss()
        if loss_name == "l1":
            return nn.L1Loss()
        if loss_name == "smooth_l1":
            return nn.SmoothL1Loss()
        raise ValueError(f"Unsupported regression loss: {loss_name}")

    if loss_name != "cross_entropy":
        raise ValueError(
            f"Unsupported classification loss: {loss_name}. "
            "Use 'cross_entropy' for classification."
        )

    if bool(getattr(config, "use_label_smoothing", False)):
        return nn.CrossEntropyLoss(label_smoothing=float(getattr(config, "label_smoothing", 0.0)))
    return nn.CrossEntropyLoss()


def prepare_loss_inputs(
    outputs: torch.Tensor,
    labels: torch.Tensor,
    config,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not is_regression_task(config):
        return outputs, labels.long()

    outputs = outputs.float()
    labels = labels.to(device=outputs.device, dtype=outputs.dtype)
    target_dim = max(1, int(getattr(config, "target_dim", 1)))

    if outputs.ndim == 1:
        outputs = outputs.unsqueeze(1)
    if labels.ndim == 0:
        labels = labels.unsqueeze(0)
    if labels.ndim == 1:
        if target_dim == 1:
            labels = labels.unsqueeze(1)
        else:
            labels = labels.view(-1, target_dim)
    elif labels.ndim > 2:
        labels = labels.view(labels.size(0), -1)

    if outputs.shape != labels.shape:
        if outputs.numel() != labels.numel():
            raise ValueError(
                f"Regression output/target shape mismatch: {tuple(outputs.shape)} vs {tuple(labels.shape)}"
            )
        labels = labels.view_as(outputs)

    return outputs, labels


def init_metric_totals(config) -> Dict[str, float]:
    if is_regression_task(config):
        return {
            "sum_sq": 0.0,
            "sum_abs": 0.0,
            "count": 0.0,
        }
    return {
        "top1_correct": 0.0,
        "top5_correct": 0.0,
        "count": 0.0,
    }


def update_metric_totals(
    totals: Dict[str, float],
    outputs: torch.Tensor,
    labels: torch.Tensor,
    config,
) -> None:
    if is_regression_task(config):
        outputs, labels = prepare_loss_inputs(outputs.detach(), labels.detach(), config)
        diff = outputs - labels
        totals["sum_sq"] += float(diff.pow(2).sum().item())
        totals["sum_abs"] += float(diff.abs().sum().item())
        totals["count"] += float(labels.numel())
        return

    labels = labels.detach()
    outputs = outputs.detach()
    _, top1_pred = torch.max(outputs, 1)
    totals["top1_correct"] += float((top1_pred == labels).sum().item())

    topk = min(5, outputs.size(1))
    _, top5_pred = outputs.topk(topk, 1, True, True)
    totals["top5_correct"] += float((top5_pred == labels.view(-1, 1)).any(dim=1).sum().item())
    totals["count"] += float(labels.size(0))


def finalize_metric_totals(totals: Dict[str, float], config) -> Dict[str, float]:
    if is_regression_task(config):
        denom = max(totals["count"], 1.0)
        mse = totals["sum_sq"] / denom
        return {
            "mse": mse,
            "rmse": math.sqrt(mse),
            "mae": totals["sum_abs"] / denom,
        }

    denom = max(totals["count"], 1.0)
    return {
        "top1": 100.0 * totals["top1_correct"] / denom,
        "top5": 100.0 * totals["top5_correct"] / denom,
    }


def get_primary_metric_value(metrics: Dict[str, float], config) -> float:
    metric_name = get_primary_metric_name(config)
    if metric_name not in metrics:
        raise KeyError(f"Primary metric '{metric_name}' missing from metrics: {metrics.keys()}")
    return float(metrics[metric_name])


def get_secondary_metric_value(metrics: Dict[str, float], config) -> Optional[float]:
    metric_name = get_secondary_metric_name(config)
    if metric_name is None:
        return None
    value = metrics.get(metric_name)
    return None if value is None else float(value)


def is_better_metric(candidate: float, best: Optional[float], config) -> bool:
    if best is None:
        return True
    if is_higher_better(config):
        return candidate > best
    return candidate < best

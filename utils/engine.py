"""Training utilities for the ResNet-50 pruning pipeline."""

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Any
from torch.utils.data import DataLoader
import copy
import os
import time

from configs.config import PruningConfig
from utils.metrics import evaluate_model
from utils.task_utils import (
    build_criterion,
    finalize_metric_totals,
    get_metric_display_name,
    get_primary_metric_name,
    get_primary_metric_value,
    get_secondary_metric_name,
    get_secondary_metric_value,
    init_metric_totals,
    is_better_metric,
    prepare_loss_inputs,
    update_metric_totals,
)

@torch.no_grad()
def recalibrate_bn(model, loader, device, num_batches=100):
    """
    Recalibrate BatchNorm statistics by running a few batches through the model.
    Note: Standard ImageNet recalibration usually needs ~1000 batches for stability,
    but we use a smaller number here for speed.
    """
    print(f"\nRecalibrating BN with {num_batches} batches...")
    model.train()
    # Reset stats
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.reset_running_stats()
            m.momentum = None # Use cumulative average
    
    with torch.no_grad():
        for i, (images, _) in enumerate(tqdm(loader, desc="BN Recalibration", leave=False)):
            if i >= num_batches:
                break
            images = images.to(device, non_blocking=True)
            model(images)
            
    # Restore momentum if needed (PyTorch default is 0.1)
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.momentum = 0.1
            
    print("BN Recalibration Done.")

class EarlyStopping:
    """Simple early stopping helper."""

    def __init__(self, config: PruningConfig, verbose: bool = True):
        self.patience = config.early_stopping_patience
        self.min_delta = config.early_stopping_min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_state_dict = None

    def __call__(self, val_loss: float, model: nn.Module):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_state_dict = copy.deepcopy(model.state_dict())
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if self.verbose and val_loss < self.best_loss:
                print(
                    f"Validation loss improved from {self.best_loss:.6f} to {val_loss:.6f}"
                )
            self.best_loss = val_loss
            self.best_state_dict = copy.deepcopy(model.state_dict())
            self.counter = 0


def get_layer_wise_optimizer_params_resnet(
    model: nn.Module,
    config: PruningConfig,
    base_lr: Optional[float] = None,
    weight_decay: Optional[float] = None,
) -> List[Dict]:
    """Build parameter groups with optional layer-wise LR decay."""

    if not hasattr(config, "layer_decay") or config.layer_decay >= 1.0:
        print("Using single parameter group (no layer decay)")
        return [
            {
                "params": [p for p in model.parameters() if p.requires_grad],
                "lr": base_lr if base_lr is not None else config.base_lr,
                "weight_decay": weight_decay if weight_decay is not None else config.weight_decay,
            }
        ]

    print(f"Using layer-wise decay with rate: {config.layer_decay}")

    param_groups: List[Dict] = []
    base_lr = base_lr if base_lr is not None else config.base_lr
    weight_decay = weight_decay if weight_decay is not None else config.weight_decay
    layer_decay = config.layer_decay

    layer_groups = {
        "conv1": 0,
        "layer1": 1,
        "layer2": 2,
        "layer3": 3,
        "layer4": 4,
        "fc": 5,
    }
    num_layers = max(layer_groups.values()) + 1

    layer_params: Dict[int, List[torch.nn.Parameter]] = {i: [] for i in range(num_layers)}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        layer_idx = num_layers - 1
        for layer_name, idx in layer_groups.items():
            if layer_name in name:
                layer_idx = idx
                break

        layer_params[layer_idx].append(param)

    for layer_idx in range(num_layers):
        if not layer_params[layer_idx]:
            continue

        decay_rate = layer_decay ** (num_layers - 1 - layer_idx)
        lr = base_lr * max(decay_rate, 0.1)

        param_groups.append(
            {
                "params": layer_params[layer_idx],
                "lr": lr,
                "weight_decay": weight_decay,
            }
        )
        print(f"  Layer {layer_idx}: {len(layer_params[layer_idx])} params, lr={lr:.2e}")

    return param_groups


def train_one_epoch_r50(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: PruningConfig,
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler] = None,
    mixup_fn: Optional[Any] = None,
) -> Tuple[float, Dict[str, float]]:
    """Train one epoch."""

    model.train()
    train_loss = 0.0
    totals = init_metric_totals(config)
    total = 0

    log_interval = getattr(config, "log_interval", 50)
    channels_last = bool(getattr(config, "channels_last", False)) and device.type == "cuda"
    profile_data = bool(getattr(config, "profile_data_time", False))
    profile_interval = int(getattr(config, "profile_interval", 50))
    last_end = time.perf_counter()
    primary_metric_name = get_primary_metric_name(config)

    pbar = tqdm(loader, desc=f"Epoch {epoch} Training", leave=False)

    for batch_idx, (images, labels) in enumerate(pbar):
        if profile_data:
            data_time = time.perf_counter() - last_end
        # ALWAYS move to device unless we are sure it's already there
        if images.device != device:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
        if channels_last:
            images = images.to(memory_format=torch.channels_last)

        # Do not use GPU MixUp here, restore original structure
        optimizer.zero_grad()
        compute_start = time.perf_counter()
        
        if scaler:
            with torch.amp.autocast(device_type=device.type, enabled=True):
                outputs = model(images)
                outputs, labels_for_loss = prepare_loss_inputs(outputs, labels, config)
                loss = criterion(outputs, labels_for_loss)

            scaler.scale(loss).backward()
            if getattr(config, "use_gradient_clip", False):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            outputs, labels_for_loss = prepare_loss_inputs(outputs, labels, config)
            loss = criterion(outputs, labels_for_loss)
            loss.backward()
            if getattr(config, "use_gradient_clip", False):
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad_norm)
            optimizer.step()

        do_profile = profile_data and (
            (batch_idx + 1) % profile_interval == 0 or (batch_idx + 1) % log_interval == 0
        )
        if do_profile and device.type == "cuda":
            torch.cuda.synchronize()
        if profile_data:
            compute_time = time.perf_counter() - compute_start

        # Metric accumulation (original method)
        bs = images.size(0)
        train_loss += loss.item() * bs
        update_metric_totals(totals, outputs, labels, config)
        
        total += bs
        
        if (batch_idx + 1) % log_interval == 0:
            running_metrics = finalize_metric_totals(totals, config)
            postfix = {
                "loss": f"{loss.item():.4f}",
                primary_metric_name: f"{running_metrics[primary_metric_name]:.4f}",
            }
            if profile_data:
                total_time = data_time + compute_time
                imgs_s = bs / total_time if total_time > 0 else 0.0
                postfix.update(
                    {
                        "data_ms": f"{data_time*1000:.1f}",
                        "compute_ms": f"{compute_time*1000:.1f}",
                        "img/s": f"{imgs_s:.0f}",
                    }
                )
            pbar.set_postfix(postfix)
        last_end = time.perf_counter()

    return train_loss / total, finalize_metric_totals(totals, config)


def validate_one_epoch_r50(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    config: PruningConfig,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    """Validate one epoch."""

    model.eval()
    val_loss = torch.tensor(0.0, device=device, dtype=torch.float64)
    totals = init_metric_totals(config)
    total_samples = 0

    channels_last = bool(getattr(config, "channels_last", False)) and device.type == "cuda"

    with torch.no_grad():
        for i, (images, labels) in enumerate(
            tqdm(
                loader,
                desc="Validation",
                leave=False,
                mininterval=5.0,
                total=len(loader)
            )
        ):
            # Robust device transfer
            if images.device != device:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
            
            if channels_last:
                images = images.to(memory_format=torch.channels_last)

            outputs = model(images)
            outputs, labels_for_loss = prepare_loss_inputs(outputs, labels, config)
            loss = criterion(outputs, labels_for_loss)

            bs = images.size(0)
            total_samples += bs
            val_loss += loss.to(torch.float64) * bs
            update_metric_totals(totals, outputs, labels, config)

    final_loss = (val_loss / total_samples).item()
    return final_loss, finalize_metric_totals(totals, config)


class FixedLRScheduler:
    """Freeze LR after a given epoch."""

    def __init__(self, optimizer, base_scheduler, freeze_epoch=85, freeze_lr=None):
        self.optimizer = optimizer
        self.base_scheduler = base_scheduler
        self.freeze_epoch = freeze_epoch
        self.freeze_lr = freeze_lr
        self.fixed_lr = None

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.base_scheduler.last_epoch + 1

        if epoch < self.freeze_epoch:
            self.base_scheduler.step()
        else:
            if self.fixed_lr is None:
                self.fixed_lr = self.optimizer.param_groups[0]["lr"]
                print(f"\n  Freezing learning rate at {self.fixed_lr:.2e} from epoch {epoch}")

            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.fixed_lr


def fine_tune_resnet_improved(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: Optional[DataLoader] = None,
    config: PruningConfig = None,
    save_dir: str = None,
    start_epoch: int = 1,
    initial_history: Optional[Dict] = None,
) -> Tuple[Dict[str, float], Dict]:
    """Fine-tune ResNet-50."""

    if config is None:
        config = PruningConfig()
    if save_dir is None:
        save_dir = config.save_dir

    print("\n" + "=" * 80)
    print("OPT-FINE-TUNING FOR RESNET-50 (DECREASED SYNC)")
    print("=" * 80)

    device = config.device
    model = model.to(device)
    if device.type == "cuda" and bool(getattr(config, "channels_last", False)):
        model = model.to(memory_format=torch.channels_last)

    criterion = build_criterion(config)

    ft_lr = getattr(config, "fine_tune_lr", None)
    if ft_lr is None:
        ft_lr = config.base_lr
    ft_wd = getattr(config, "fine_tune_weight_decay", None)
    if ft_wd is None:
        ft_wd = config.weight_decay

    param_groups = get_layer_wise_optimizer_params_resnet(
        model, config, base_lr=ft_lr, weight_decay=ft_wd
    )

    if config.optimizer["name"] == "adamw":
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=ft_lr,
            weight_decay=ft_wd,
            betas=tuple(config.optimizer["betas"]),
        )
    else:
        optimizer = torch.optim.SGD(
            param_groups,
            lr=ft_lr,
            momentum=0.9,
            weight_decay=ft_wd,
        )

    if config.scheduler["name"] == "cosine":
        base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.fine_tune_epochs,
            eta_min=ft_lr * config.scheduler["cosine"]["eta_min_ratio"],
        )
    else:
        base_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.scheduler["step"]["step_size"],
            gamma=config.scheduler["step"]["gamma"],
        )

    freeze_epoch = getattr(config, "freeze_epoch", 85)
    freeze_lr = getattr(config, "freeze_lr", None)
    scheduler = FixedLRScheduler(optimizer, base_scheduler, freeze_epoch, freeze_lr)

    if config.warmup_epochs > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=config.warmup_init_lr / config.base_lr,
            end_factor=1.0,
            total_iters=config.warmup_epochs,
        )
    else:
        warmup_scheduler = None

    scaler = (
        torch.amp.GradScaler("cuda")
        if (config.mixed_precision and config.device.type == "cuda")
        else None
    )

    early_stopping = EarlyStopping(config, verbose=True)

    primary_metric_name = get_primary_metric_name(config)
    secondary_metric_name = get_secondary_metric_name(config)
    best_val_primary = None
    best_epoch = 0
    best_state_dict = None
    if initial_history is not None:
        history = initial_history
    else:
        history = {
            "train_loss": [],
            "val_loss": [],
            "train_primary_metric": [],
            "val_primary_metric": [],
            "train_secondary_metric": [],
            "val_secondary_metric": [],
            "learning_rate": [],
            "primary_metric_name": primary_metric_name,
            "secondary_metric_name": secondary_metric_name,
        }
    history.setdefault("train_loss", [])
    history.setdefault("val_loss", [])
    history.setdefault("train_primary_metric", [])
    history.setdefault("val_primary_metric", [])
    history.setdefault("train_secondary_metric", [])
    history.setdefault("val_secondary_metric", [])
    history.setdefault("learning_rate", [])
    history["primary_metric_name"] = primary_metric_name
    history["secondary_metric_name"] = secondary_metric_name

    os.makedirs(save_dir, exist_ok=True)
    print(f"\nModel checkpoints will be saved to: {save_dir}")

    final_info_path = os.path.join(save_dir, "final_results.txt")

    print(f"\nStarting fine-tuning for {config.fine_tune_epochs} epochs")
    print(f"Base learning rate: {ft_lr}")
    print(f"Layer decay: {config.layer_decay}")
    print(f"Warmup epochs: {config.warmup_epochs}")
    print(f"Mixed precision: {config.mixed_precision}")
    print(f"Primary metric: {get_metric_display_name(primary_metric_name)}")
    
    # Optional BN Recalibration
    # recalibrate_bn(model, train_loader, device, 100)

    # If resuming, step the scheduler to the correct epoch
    if start_epoch > 1:
        print(f"Resuming from epoch {start_epoch}, advancing scheduler...")
        for e in range(1, start_epoch):
            if warmup_scheduler and e <= config.warmup_epochs:
                warmup_scheduler.step()
            if e > config.warmup_epochs:
                scheduler.step(e)

    for epoch in range(start_epoch, config.fine_tune_epochs + 1):
        if warmup_scheduler and epoch <= config.warmup_epochs:
            warmup_scheduler.step()

        train_loss, train_metrics = train_one_epoch_r50(
            model,
            train_loader,
            criterion,
            optimizer,
            epoch,
            config,
            device,
            scaler,
        )

        val_loss, val_metrics = validate_one_epoch_r50(
            model, val_loader, criterion, config, device
        )

        if epoch > config.warmup_epochs:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        train_primary = get_primary_metric_value(train_metrics, config)
        val_primary = get_primary_metric_value(val_metrics, config)
        train_secondary = get_secondary_metric_value(train_metrics, config)
        val_secondary = get_secondary_metric_value(val_metrics, config)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_primary_metric"].append(train_primary)
        history["val_primary_metric"].append(val_primary)
        history["train_secondary_metric"].append(train_secondary)
        history["val_secondary_metric"].append(val_secondary)
        history["learning_rate"].append(current_lr)
        if primary_metric_name == "top1":
            history.setdefault("train_top1", []).append(train_primary)
            history.setdefault("val_top1", []).append(val_primary)
        if secondary_metric_name == "top5":
            history.setdefault("train_top5", []).append(train_secondary if train_secondary is not None else 0.0)
            history.setdefault("val_top5", []).append(val_secondary if val_secondary is not None else 0.0)

        train_summary = f"Loss={train_loss:.4f}, {primary_metric_name}={train_primary:.4f}"
        val_summary = f"Loss={val_loss:.4f}, {primary_metric_name}={val_primary:.4f}"
        if train_secondary is not None and secondary_metric_name is not None:
            train_summary += f", {secondary_metric_name}={train_secondary:.4f}"
        if val_secondary is not None and secondary_metric_name is not None:
            val_summary += f", {secondary_metric_name}={val_secondary:.4f}"
        print(
            f"Epoch {epoch:3d}/{config.fine_tune_epochs} | "
            f"Train: {train_summary} | "
            f"Val: {val_summary} | "
            f"LR={current_lr:.2e}"
        )
        torch.save(model.state_dict(), os.path.join(save_dir, f"epoch{epoch}.pth"))

        if is_better_metric(val_primary, best_val_primary, config):
            best_val_primary = val_primary
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())
            torch.save(best_state_dict, os.path.join(save_dir, "best_model.pth"))
            print(f"NEW BEST MODEL, Epoch {epoch} ({primary_metric_name}: {best_val_primary:.4f})")

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        print(
            f"\nLoaded best model from epoch {best_epoch} with validation {primary_metric_name}: {best_val_primary:.4f}"
        )

    eval_loader = test_loader if test_loader is not None else val_loader
    test_metrics = evaluate_model(model, eval_loader, config=config, device=device)

    print("\nFine-tuning completed.")
    if best_val_primary is not None:
        print(f"Best validation {primary_metric_name}: {best_val_primary:.4f} (epoch {best_epoch})")

    with open(final_info_path, "w") as f:
        if best_val_primary is not None:
            f.write(f"Best {primary_metric_name}: {best_val_primary:.6f} at epoch {best_epoch}\n")
        for metric_name, metric_value in test_metrics.items():
            f.write(f"Final Test {metric_name}: {metric_value:.6f}\n")

    return test_metrics, history

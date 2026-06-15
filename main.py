"""ResNet-50 Pruning Main Script - Corresponds to ViT main.py"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse
import random
import copy
import torch
import torch.nn as nn
import numpy as np
import time
import json
from typing import Any, Dict, Optional

from configs.config import PruningConfig
from data.build import get_resnet_data_loaders_with_calib
from models.resnet_setup import get_model_family, setup_builtin_model, setup_resnet_model
from models.custom_layers import SimpleSVDConv, SimpleSVDLinear
from models.utils import collect_prunable_layers, get_resnet_parent_and_name
from pruning.core import replace_prunable_layer
from pruning.pipeline import run_two_stage_pruning_allocation_r50
from utils.engine import fine_tune_resnet_improved
from utils.metrics import compute_flops_resnet, evaluate_model
from utils.visualization import (
    plot_training_history,
)
from utils.helpers import ModelStructureManager, generate_report, batch_clean_experiment
from utils.task_utils import (
    get_metric_display_name,
    get_output_dim,
    get_primary_metric_name,
    get_primary_metric_value,
    get_secondary_metric_name,
    get_secondary_metric_value,
    get_task_type,
    is_regression_task,
)


def _format_tag_value(value) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def build_custom_tag(config: PruningConfig) -> str:
    parts = [
        f"mc{_format_tag_value(getattr(config, 'mc_samples', 'na'))}",
        f"mdp{_format_tag_value(getattr(config, 'mc_dropout_p', 'na'))}",
        f"cb{_format_tag_value(getattr(config, 'calib_batches', 'na'))}",
        f"fb{_format_tag_value(getattr(config, 'fisher_batches', 'na'))}",
        f"tm{_format_tag_value(getattr(config, 'train_max_batches', 'na'))}",
        f"em{_format_tag_value(getattr(config, 'eval_max_batches', 'na'))}",
        f"low{_format_tag_value(getattr(config, 'pruning_clip_low', 'na'))}",
        f"high{_format_tag_value(getattr(config, 'pruning_clip_high', 'na'))}",
        f"lr{_format_tag_value(getattr(config, 'base_lr', 'na'))}",
        f"ld{_format_tag_value(getattr(config, 'layer_decay', 'na'))}",
        f"bs{_format_tag_value(getattr(config, 'batch_size', 'na'))}",
        f"nw{_format_tag_value(getattr(config, 'num_workers', 'na'))}",
        f"mp{int(getattr(config, 'mixed_precision', False))}",
        f"seed{_format_tag_value(getattr(config, 'seed', 'na'))}",
    ]
    return "_".join(parts)


def build_pruning_config(
    pruning_ratio: float,
    fine_tune_epochs: int,
    base_config: Optional[PruningConfig] = None,
    **overrides: Any,
) -> PruningConfig:
    """
    Build a pruning config for programmatic use.

    Args:
        pruning_ratio: Target pruning/compression ratio in [0, 1).
        fine_tune_epochs: Number of fine-tuning epochs.
        base_config: Optional existing config to reuse.
        **overrides: Any extra config fields to override.

    Returns:
        A configured PruningConfig instance.
    """
    if not 0.0 <= pruning_ratio < 1.0:
        raise ValueError("pruning_ratio must be in the range [0.0, 1.0).")
    if fine_tune_epochs < 0:
        raise ValueError("fine_tune_epochs must be >= 0.")

    config = base_config if base_config is not None else PruningConfig()
    config.target_compression = pruning_ratio
    config.fine_tune_epochs = fine_tune_epochs

    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise AttributeError(f"Unknown config field: {key}")

    _configure_task_defaults(config)
    config._compression_percentage = int(config.target_compression * 100)
    return config


def _detect_model_family(model: nn.Module) -> str:
    has_conv = any(isinstance(module, nn.Conv2d) for module in model.modules())
    has_linear = any(isinstance(module, nn.Linear) for module in model.modules())

    if has_conv:
        return "cnn"
    if has_linear:
        return "mlp"
    raise ValueError("The provided model has no prunable Conv2d or Linear layers.")


def _prepare_runtime(
    config: PruningConfig,
    model: Optional[nn.Module] = None,
    train_loader=None,
    val_loader=None,
    calib_loader=None,
    model_family: Optional[str] = None,
):
    if model is None:
        built_in_model_name = getattr(config, "model_name", "resnet50")
        print(
            "\n1. No custom nn.Module was provided; "
            f"loading built-in model from config.model_name: {built_in_model_name}"
        )
        model, data_config, family = setup_builtin_model(
            config,
            model_name=built_in_model_name,
            pretrained=bool(getattr(config, "pretrained", True)),
        )
        print("\n2. Preparing data loaders for the built-in model")
        train_loader, val_loader, calib_loader = get_resnet_data_loaders_with_calib(config)
        if calib_loader is None:
            print("Calibration loader not specified; using train_loader as D_cal.")
            calib_loader = train_loader
        else:
            print(f"Calibration loader ready: {len(calib_loader)} batches")
        model_name = built_in_model_name
        model_builder = lambda: setup_builtin_model(
            config,
            model_name=built_in_model_name,
            pretrained=bool(getattr(config, "pretrained", True)),
        )[0]
        return model, train_loader, val_loader, calib_loader, family, model_name, model_builder

    if train_loader is None or val_loader is None:
        raise ValueError(
            "When passing a custom nn.Module, train_loader and val_loader must also be provided."
        )

    family = model_family or _detect_model_family(model)
    model_name = model.__class__.__name__
    print(f"\n1. Using caller-provided custom nn.Module: {model_name}")
    print("2. Using caller-provided data loaders")
    if calib_loader is None:
        calib_loader = train_loader
        print("Calibration loader not provided; using train_loader as D_cal.")
    base_model = copy.deepcopy(model).to(config.device)
    working_model = copy.deepcopy(base_model).to(config.device)
    model_builder = lambda: copy.deepcopy(base_model).to(config.device)
    return working_model, train_loader, val_loader, calib_loader, family, model_name, model_builder


def _build_svd_layers(model: nn.Module, config: PruningConfig, model_family: str):
    include_conv = model_family == "cnn"
    include_linear = True
    layer_paths = collect_prunable_layers(model, include_conv=include_conv, include_linear=include_linear)
    svd_layers = {}

    for path in layer_paths:
        parent, layer_name = get_resnet_parent_and_name(model, path)
        if parent is None or layer_name is None:
            continue

        original_layer = getattr(parent, layer_name, None)
        if original_layer is None and hasattr(parent, '_modules') and layer_name in parent._modules:
            original_layer = parent._modules[layer_name]

        if isinstance(original_layer, nn.Conv2d):
            svd_layer = SimpleSVDConv(original_layer, path, config=config).to(config.device)
        elif isinstance(original_layer, nn.Linear):
            svd_layer = SimpleSVDLinear(original_layer, path, config=config).to(config.device)
        else:
            continue

        if hasattr(parent, layer_name):
            setattr(parent, layer_name, svd_layer)
        elif hasattr(parent, '_modules') and layer_name in parent._modules:
            parent._modules[layer_name] = svd_layer
        svd_layers[path] = svd_layer

    return layer_paths, svd_layers


def _safe_compute_flops(model: nn.Module, config: PruningConfig):
    try:
        return compute_flops_resnet(model)
    except Exception:
        return None


def _build_pruning_detail(svd_layer, keep_ratio: float, pruned_rank: int, used_fisher_selection: bool):
    detail = {
        'layer_kind': getattr(svd_layer, 'layer_kind', 'conv'),
        'is_bottleneck': getattr(svd_layer, 'is_bottleneck', False),
        'keep_ratio': keep_ratio,
        'pruned_rank': pruned_rank,
        'original_rank': svd_layer.full_rank,
        'recon_error': svd_layer.get_reconstruction_error(keep_ratio),
        'has_bias': svd_layer.bias is not None,
        'used_fisher_selection': used_fisher_selection,
    }
    if getattr(svd_layer, 'layer_kind', 'conv') == 'conv':
        detail.update({
            'original_cin': svd_layer.cin,
            'original_cout': svd_layer.cout,
            'kernel_size': (svd_layer.kh, svd_layer.kw),
        })
    else:
        detail.update({
            'original_in_features': svd_layer.in_features,
            'original_out_features': svd_layer.out_features,
        })
    return detail


def _parse_int_list(value: Optional[str]):
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _parse_str_list(value: Optional[str]):
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def _configure_task_defaults(config: PruningConfig):
    task_type = get_task_type(config)
    target_columns = getattr(config, "target_columns", None)
    if isinstance(target_columns, str):
        target_columns = [part.strip() for part in target_columns.split(",") if part.strip()]
        config.target_columns = target_columns
    if isinstance(target_columns, list) and target_columns:
        config.target_dim = len(target_columns)
    if task_type == "regression":
        if getattr(config, "dataset_type", "imagefolder") == "imagefolder":
            config.dataset_type = "csv_regression"
        if getattr(config, "loss_name", None) in {None, "", "cross_entropy"}:
            config.loss_name = "mse"
        if getattr(config, "primary_metric", None) in {None, "", "top1"}:
            config.primary_metric = "rmse"
        if getattr(config, "secondary_metric", None) in {None, "", "top5"}:
            config.secondary_metric = "mae"
        config.higher_is_better = False
    else:
        if getattr(config, "dataset_type", None) in {None, "", "csv_regression"}:
            config.dataset_type = "imagefolder"
        if getattr(config, "loss_name", None) in {None, "", "mse"}:
            config.loss_name = "cross_entropy"
        if getattr(config, "primary_metric", None) in {None, "", "rmse"}:
            config.primary_metric = "top1"
        if getattr(config, "secondary_metric", None) in {None, "", "mae"}:
            config.secondary_metric = "top5"
        config.higher_is_better = True


def _format_metric_value(metric_name: Optional[str], metric_value: Optional[float]) -> str:
    if metric_name is None or metric_value is None:
        return "-"
    label = get_metric_display_name(metric_name)
    return f"{label}={metric_value:.4f}"


def _print_metric_summary(title: str, metrics: Dict[str, float], config: PruningConfig, baseline_metrics: Optional[Dict[str, float]] = None):
    primary_metric_name = get_primary_metric_name(config)
    secondary_metric_name = get_secondary_metric_name(config)
    primary_value = get_primary_metric_value(metrics, config)
    secondary_value = get_secondary_metric_value(metrics, config)

    print(f"\n{title}:")
    print(f"  {_format_metric_value(primary_metric_name, primary_value)}")
    if secondary_value is not None and secondary_metric_name is not None:
        print(f"  {_format_metric_value(secondary_metric_name, secondary_value)}")

    if baseline_metrics is not None:
        baseline_primary = get_primary_metric_value(baseline_metrics, config)
        print(f"  Primary Change: {primary_value - baseline_primary:+.4f}")


def _flatten_metric_result(prefix: str, metrics: Dict[str, float]) -> Dict[str, float]:
    return {f"{prefix}_{name}": value for name, value in metrics.items()}


def main_r50(
    config: Optional[PruningConfig] = None,
    model: Optional[nn.Module] = None,
    train_loader=None,
    val_loader=None,
    calib_loader=None,
    model_family: Optional[str] = None,
):
    """
    Main pruning flow for built-in or caller-provided models.

    Args:
        config: Pruning config.
        model: Optional custom nn.Module. If omitted, a built-in model is loaded from config.model_name.
        train_loader: Training loader for a custom model.
        val_loader: Validation loader for a custom model.
        calib_loader: Optional calibration loader. Defaults to train_loader for custom models.
        model_family: Optional explicit family for a custom model, such as cnn or mlp.
    """
    if config is None:
        config = PruningConfig()
    
    print("=" * 80)
    print("SECOND-ORDER IMPACT PRUNING")
    print("=" * 80)
    
    # Create save directory
    save_dir = config.get_experiment_dir()
    before_finetune_path = config.get_model_path("before_finetune")
    finetuned_path = config.get_model_path(f"{config.fine_tune_epochs}ft")
    
    # Save configuration
    config_path = os.path.join(save_dir, "config.json")
    config.save(config_path)
    print(f"Config saved to: {config_path}")
    
    # Set random seed
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    if bool(getattr(config, "deterministic", False)):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)

    # GPU performance knobs
    if config.device.type == "cuda":
        if bool(getattr(config, "deterministic", False)):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        else:
            torch.backends.cudnn.benchmark = bool(getattr(config, "cudnn_benchmark", True))
        deterministic = bool(getattr(config, "deterministic", False))
        enable_tf32 = bool(getattr(config, "enable_tf32", True)) and not deterministic
        torch.backends.cuda.matmul.allow_tf32 = enable_tf32
        torch.backends.cudnn.allow_tf32 = enable_tf32
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    
    # ========== 1-2. Load Model / Prepare Data ==========
    model, train_loader, val_loader, calib_loader, resolved_family, model_name, model_builder = _prepare_runtime(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        calib_loader=calib_loader,
        model_family=model_family,
    )
    model.eval()
    model_name_lower = model_name.lower()
    if model_name_lower == "resnet18":
        config.model_prefix = "r18"
    elif model_name_lower == "resnet34":
        config.model_prefix = "r34"
    elif model_name_lower == "resnet50":
        config.model_prefix = "r50"
    else:
        config.model_prefix = resolved_family
    config.experiment_name = f"{model_name}_SecondOrderPruning_{int(config.target_compression*100)}pr"
    print(f"Model: {model_name} | Family: {resolved_family}")
    print(f"Task: {get_task_type(config)} | Output dim: {get_output_dim(config)}")
    
    # ========== 3. Evaluate Original Model ==========
    print("\n3. Evaluating original model")
    orig_metrics = evaluate_model(model, val_loader, config=config)
    orig_params = sum(p.numel() for p in model.parameters())
    orig_flops = _safe_compute_flops(model, config)
    
    print(f"\nOriginal Model Results:")
    _print_metric_summary("Original Metrics", orig_metrics, config)
    print(f"  Parameters: {orig_params/1e6:.2f}M")
    if orig_flops:
        print(f"  FLOPs: {orig_flops/1e9:.2f}G")

    if config.target_compression <= 0.0:
        print("\nTarget compression is 0. Skipping pruning and fine-tuning to preserve the true baseline.")

        config_dict = config.to_dict()
        config_dict.update({'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')})

        ModelStructureManager.save_complete_model(
            model, {}, {}, {},
            before_finetune_path,
            orig_metrics,
            orig_metrics,
            test_metrics=orig_metrics,
            config=config_dict
        )

        generate_report(
            save_dir=save_dir,
            config=config,
            model=model,
            svd_layers={},
            layer_importance_stage1={},
            component_impact_scores={},
            keep_ratios={},
            pruning_details={},
            orig_params=orig_params,
            pruned_params=orig_params,
            orig_metrics=orig_metrics,
            pruned_metrics=orig_metrics,
            final_metrics=orig_metrics,
            orig_flops=orig_flops,
            pruned_flops=orig_flops,
            history=None
        )

        result = {
            "final_model": model,
            "pruned_model": model,
            "pruning_details": {},
            "keep_ratios": {},
            "layer_importance_stage1": {},
            "component_impact_scores": {},
            "history": None,
            "metrics": {
                **_flatten_metric_result("orig", orig_metrics),
                **_flatten_metric_result("pruned", orig_metrics),
                **_flatten_metric_result("final", orig_metrics),
                "orig_params": orig_params,
                "pruned_params": orig_params,
                "orig_flops": orig_flops,
                "pruned_flops": orig_flops,
            },
            "paths": {
                "save_dir": save_dir,
                "before_finetune_path": before_finetune_path,
                "finetuned_path": before_finetune_path,
            },
            "config": config,
        }
        return result
    
    # ========== 4-5. Collect Layers / Replace with SVD Layers ==========
    print("\n4. Collecting prunable layers")
    layer_paths, svd_layers = _build_svd_layers(model, config, resolved_family)
    print(f"Found {len(layer_paths)} prunable layers")
    print("\n5. Replacing prunable layers with SVD layers")
    print(f"Created {len(svd_layers)} SVD layers")
    
    # ========== 6. Two-Stage Pruning Allocation ==========
    allocation_outputs = run_two_stage_pruning_allocation_r50(
        model=model,
        svd_layers=svd_layers,
        train_loader=train_loader,
        calib_loader=calib_loader,
        save_dir=save_dir,
        orig_params=orig_params,
        config=config,
    )
    layer_importance_stage1 = allocation_outputs["layer_importance_stage1"]
    component_impact_scores = allocation_outputs["component_impact_scores"]
    keep_ratios = allocation_outputs["keep_ratios"]
    type_keep_ratios = allocation_outputs["type_keep_ratios"]
    
    # ========== 7. Execute Pruning ==========
    print("\n7. Executing pruning")
    
    # Reload original model for pruning
    pruned_model = model_builder()
    pruned_model.eval()
    
    pruning_details = {}
    pruned_count = 0
    
    for name, svd_layer in svd_layers.items():
        if name not in keep_ratios:
            continue
        
        keep_ratio = keep_ratios[name]
        layer_impact_scores = component_impact_scores.get(name, None)
        
        pruned_rank = replace_prunable_layer(
            pruned_model, name, svd_layer, keep_ratio,
            impact_scores=layer_impact_scores,
            config=config
        )
        
        if pruned_rank is not None:
            pruned_count += 1
            pruning_details[name] = _build_pruning_detail(
                svd_layer, keep_ratio, pruned_rank, used_fisher_selection=layer_impact_scores is not None
            )
    
    print(f"\nSuccessfully pruned {pruned_count}/{len(keep_ratios)} layers")
    
    # ========== 8. Evaluate Pruned Model ==========
    print("\n8. Evaluating pruned model (before fine-tuning)")
    pruned_metrics = evaluate_model(pruned_model, val_loader, config=config)
    pruned_params = sum(p.numel() for p in pruned_model.parameters())
    
    # ========== 9. Save Model (Before FT) ==========
    print("\n9. Saving pruned model (Before FT)")
    config_dict = config.to_dict()
    config_dict.update({'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')})
    
    ModelStructureManager.save_complete_model(
        pruned_model, pruning_details, keep_ratios, layer_importance_stage1,
        before_finetune_path,
        orig_metrics,
        pruned_metrics,
        config=config_dict
    )
    
    # ========== Results Summary ==========
    print("\n" + "="*80)
    print("RESULTS SUMMARY (BEFORE FINE-TUNING)")
    print("="*80)
    
    param_reduction = 100 * (1 - pruned_params / orig_params)
    
    print(f"\nOriginal Model:")
    print(f"  Parameters: {orig_params/1e6:.2f}M")
    if orig_flops:
        print(f"  FLOPs: {orig_flops/1e9:.2f}G")
    _print_metric_summary("Original Metrics", orig_metrics, config)
    
    print(f"\nPruned Model (Before Fine-tuning):")
    print(f"  Parameters: {pruned_params/1e6:.2f}M ({param_reduction:.1f}% reduction)")
    pruned_flops = _safe_compute_flops(pruned_model, config)
    if pruned_flops and orig_flops:
        flops_reduction = 100 * (1 - pruned_flops / orig_flops)
        print(f"  FLOPs: {pruned_flops/1e9:.2f}G ({flops_reduction:.1f}% reduction)")
    
    _print_metric_summary("Pruned Metrics", pruned_metrics, config, baseline_metrics=orig_metrics)
    
    # Layer statistics
    print(f"\n10. Layer Statistics:")
    for layer_type, ratios in type_keep_ratios.items():
        avg_keep = np.mean(ratios) if ratios else 0
        print(f"  {layer_type}: {len(ratios)} layers, avg keep ratio: {avg_keep:.2%}")
    
    avg_keep = np.mean(list(keep_ratios.values()))
    print(f"\n  Avg keep ratio (all layers): {avg_keep:.2%}")
    
    # ========== 10. Fine-tuning ==========
    print("\n10. Starting fine-tuning")
    
    final_model = pruned_model
    final_metrics = dict(pruned_metrics)
    fine_tune_history = None
    ft_dir = os.path.join(save_dir, 'finetunemodel')
    
    if config.fine_tune_epochs > 0:
        final_metrics, fine_tune_history = fine_tune_resnet_improved(
            model=pruned_model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=None,
            config=config,
            save_dir=ft_dir
        )
        plot_training_history(fine_tune_history, ft_dir)
        
        # Save fine-tuned model
        ModelStructureManager.save_complete_model(
            final_model, pruning_details, keep_ratios, layer_importance_stage1,
            finetuned_path,
            orig_metrics,
            final_metrics,
            test_metrics=final_metrics,
            config=config_dict
        )
        _print_metric_summary("Fine-tuned Metrics", final_metrics, config, baseline_metrics=orig_metrics)
    
    # ========== 11. Generate Report ==========
    print("\n11. Generating final report")
    generate_report(
    save_dir=save_dir,
    config=config,
    model=pruned_model,
    svd_layers=svd_layers,
    layer_importance_stage1=layer_importance_stage1,
    component_impact_scores=component_impact_scores,
    keep_ratios=keep_ratios,
    pruning_details=pruning_details,
    orig_params=orig_params,
    pruned_params=pruned_params,  # Can pass, but function recalculates
    orig_metrics=orig_metrics,
    pruned_metrics=pruned_metrics,
    final_metrics=final_metrics,
    orig_flops=orig_flops,
    pruned_flops=pruned_flops,
    history=fine_tune_history  # Ensure history is passed
    )
    # ========== 12. Cleanup ==========
    print("\n12. Auto-cleaning pruned models to reduce file size")
    print('save_dir cleaning:')
    batch_clean_experiment(save_dir, overwrite=True)
    if os.path.isdir(ft_dir):
        print('ft_dir cleaning:')
        batch_clean_experiment(ft_dir, overwrite=True)
    else:
        print('ft_dir cleaning: skipped (directory not found)')
    result = {
        "final_model": final_model,
        "pruned_model": pruned_model,
        "pruning_details": pruning_details,
        "keep_ratios": keep_ratios,
        "layer_importance_stage1": layer_importance_stage1,
        "component_impact_scores": component_impact_scores,
        "history": fine_tune_history,
        "metrics": {
            **_flatten_metric_result("orig", orig_metrics),
            **_flatten_metric_result("pruned", pruned_metrics),
            **_flatten_metric_result("final", final_metrics),
            "orig_params": orig_params,
            "pruned_params": pruned_params,
            "orig_flops": orig_flops,
            "pruned_flops": pruned_flops,
        },
        "paths": {
            "save_dir": save_dir,
            "before_finetune_path": before_finetune_path,
            "finetuned_path": finetuned_path,
        },
        "config": config,
    }
    return result


def prune_and_finetune_model(
    pruning_ratio: float,
    fine_tune_epochs: int,
    model_type: Optional[str] = None,
    model: Optional[nn.Module] = None,
    train_loader=None,
    val_loader=None,
    calib_loader=None,
    config: Optional[PruningConfig] = None,
    **config_overrides: Any,
) -> Dict[str, Any]:
    """
    Unified entry point for the pruning pipeline.

    Args:
        model_type: Optional model family name such as cnn, resnet50, mlp.
        pruning_ratio: Target pruning/compression ratio.
        fine_tune_epochs: Number of fine-tuning epochs after pruning.
        model: Optional custom nn.Module instance. If omitted, the built-in model named by
            config.model_name is loaded. The default config model_name is resnet50.
        train_loader: Required when model is provided.
        val_loader: Required when model is provided.
        calib_loader: Optional calibration loader. Defaults to train_loader.
        config: Optional base config.
        **config_overrides: Extra config fields such as save_dir, device, batch_size.

    Returns:
        A result dict containing the final model and experiment metadata.
    """
    normalized_model_type = model_type.strip().lower() if model_type is not None else None
    runtime_config = build_pruning_config(
        pruning_ratio=pruning_ratio,
        fine_tune_epochs=fine_tune_epochs,
        base_config=config,
        **config_overrides,
    )

    if model is not None:
        detected_family = normalized_model_type or _detect_model_family(model)
        if detected_family not in {"cnn", "resnet", "resnet50", "mlp"}:
            raise ValueError(f"Unsupported custom model family: {detected_family}")
        family = "cnn" if detected_family in {"cnn", "resnet", "resnet50"} else "mlp"
        return main_r50(
            runtime_config,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            calib_loader=calib_loader,
            model_family=family,
        )

    if normalized_model_type in {None, "cnn", "resnet", "resnet50", "mlp"}:
        return main_r50(runtime_config)

    raise ValueError(
        f"Unsupported model_type: {model_type}. "
        "Use one of: 'cnn', 'resnet', 'resnet50', 'mlp'."
    )


def main():
    """Main entry point - parameter parsing and execution"""
    import multiprocessing
    multiprocessing.freeze_support()
    
    parser = argparse.ArgumentParser(
        description='ResNet-50 Second-Order Impact Pruning - Complete Parameter Control',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # ==================== Basic Settings ====================
    parser.add_argument('--experiment-name', type=str, default=None,
                        help='Experiment name')
    parser.add_argument('--model-name', type=str, default=None,
                        help='Built-in model name: resnet18, resnet34, resnet50, simple_cnn, mlp_small, mlp_medium')
    parser.add_argument('--pretrained', action='store_true', default=None,
                        help='Use pretrained weights when available')
    parser.add_argument('--no-pretrained', action='store_false', default=None,
                        dest='pretrained',
                        help='Disable pretrained weights')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default=None,
                        help='Device to use')
    parser.add_argument('--deterministic', action='store_true', default=None,
                        help='Enable deterministic execution')
    parser.add_argument('--no-deterministic', action='store_false', default=None,
                        dest='deterministic',
                        help='Disable deterministic execution')
    parser.add_argument('--cudnn-benchmark', action='store_true', default=None,
                        help='Enable cuDNN benchmark (non-deterministic)')
    parser.add_argument('--no-cudnn-benchmark', action='store_false', default=None,
                        dest='cudnn_benchmark',
                        help='Disable cuDNN benchmark')
    
    # ==================== Data Settings ====================
    parser.add_argument('--train-root', type=str, default=None,
                        help='Training data root path')
    parser.add_argument('--val-root', type=str, default=None,
                        help='Validation data root path')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size')
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Number of data loading workers')
    parser.add_argument('--prefetch-factor', type=int, default=None,
                        help='DataLoader prefetch factor (effective when num_workers > 0)')
    parser.add_argument('--prefetch-to-gpu', action='store_true', default=None,
                        help='Enable CUDA prefetch for input batches')
    parser.add_argument('--no-prefetch-to-gpu', action='store_false', default=None,
                        dest='prefetch_to_gpu',
                        help='Disable CUDA prefetch for input batches')
    parser.add_argument('--channels-last', action='store_true', default=None,
                        help='Use channels_last memory format on CUDA')
    parser.add_argument('--no-channels-last', action='store_false', default=None,
                        dest='channels_last',
                        help='Disable channels_last memory format')
    parser.add_argument('--profile-data-time', action='store_true', default=None,
                        help='Enable data/compute time profiling')
    parser.add_argument('--profile-interval', type=int, default=None,
                        help='Profiling interval in batches')
    parser.add_argument('--shuffle-val', action='store_true', default=None,
                        help='Shuffle validation loader')
    parser.add_argument('--no-shuffle-val', action='store_false', default=None,
                        dest='shuffle_val',
                        help='Disable validation loader shuffling')
    parser.add_argument('--random-eval-subset', action='store_true', default=None,
                        help='Use a random validation subset when eval-max-batches > 0')
    parser.add_argument('--no-random-eval-subset', action='store_false', default=None,
                        dest='random_eval_subset',
                        help='Disable random validation subset sampling')
    parser.add_argument('--input-channels', type=int, default=None,
                        help='Model input channels for built-in toy models')
    parser.add_argument('--input-height', type=int, default=None,
                        help='Model input height for built-in toy models')
    parser.add_argument('--input-width', type=int, default=None,
                        help='Model input width for built-in toy models')
    parser.add_argument('--num-classes', type=int, default=None,
                        help='Number of output classes for built-in toy models')
    parser.add_argument('--task-type', type=str, choices=['classification', 'regression'], default=None,
                        help='Problem type')
    parser.add_argument('--dataset-type', type=str, choices=['imagefolder', 'csv_regression'], default=None,
                        help='Dataset loader type')
    parser.add_argument('--loss-name', type=str, choices=['cross_entropy', 'mse', 'l1', 'smooth_l1'], default=None,
                        help='Loss function name')
    parser.add_argument('--target-dim', type=int, default=None,
                        help='Regression target dimension')
    parser.add_argument('--train-csv', type=str, default=None,
                        help='Training CSV path for regression mode')
    parser.add_argument('--val-csv', type=str, default=None,
                        help='Validation CSV path for regression mode')
    parser.add_argument('--image-column', type=str, default=None,
                        help='CSV column that stores image paths')
    parser.add_argument('--target-column', type=str, default=None,
                        help='Single CSV target column for regression mode')
    parser.add_argument('--target-columns', type=str, default=None,
                        help='Comma-separated CSV target columns for regression mode')
    parser.add_argument('--csv-delimiter', type=str, default=None,
                        help='CSV delimiter for regression annotations')
    parser.add_argument('--mlp-hidden-dims', type=str, default=None,
                        help='Comma-separated hidden dims for mlp_medium, e.g. 1024,512')
    parser.add_argument('--cnn-channels', type=str, default=None,
                        help='Comma-separated channel sizes for simple_cnn, e.g. 32,64,128')
    
    # ==================== Pruning Core Parameters ====================
    parser.add_argument('--target-compression', type=float, default=None,
                        help='Target compression ratio (0.0-1.0)')
    parser.add_argument('--min-rank', type=int, default=None,
                        help='Minimum rank to keep')
    parser.add_argument('--use-log-s', action='store_true', default=None,
                        help='Use log-singular parameterization')
    parser.add_argument('--no-log-s', action='store_false', default=None,
                        dest='use_log_s',
                        help='Use direct singular values (no log)')
    
    # ==================== Stage 1 Parameters ====================
    parser.add_argument('--mc-samples', type=int, default=None,
                        help='Number of MC Dropout samples')
    parser.add_argument('--mc-dropout-p', type=float, default=None,
                        help='MC Dropout probability (Stage 1)')
    parser.add_argument('--calib-batches', type=int, default=None,
                        help='Number of calibration batches (0 = all)')
    parser.add_argument('--calib-split-ratio', type=float, default=None,
                        help='Fraction of training data used for D_cal (0 = disabled)')
    parser.add_argument('--calib-samples', type=int, default=None,
                        help='Number of samples for D_cal (overrides split ratio)')
    parser.add_argument('--calib-seed', type=int, default=None,
                        help='Random seed for D_cal sampling')
    parser.add_argument('--calib-exclude-from-train', action='store_true', default=None,
                        help='Exclude D_cal samples from training set')
    parser.add_argument('--calib-use-val-transform', action='store_true', default=None,
                        help='Use validation transform for D_cal')
    parser.add_argument('--uncertainty-alpha', type=float, default=None,
                        help='Uncertainty alpha factor')
    parser.add_argument('--uncertainty-log', action='store_true', default=None,
                        help='Enable log1p stabilization for Stage 1 scores')
    parser.add_argument('--no-uncertainty-log', action='store_false', default=None,
                        dest='uncertainty_log',
                        help='Disable log1p stabilization for Stage 1 scores')
    parser.add_argument('--uncertainty-clip-percentile', type=float, default=None,
                        help='Percentile clip for Stage 1 scores (0 = disabled)')
    parser.add_argument('--uncertainty-var-floor', type=float, default=None,
                        help='Variance floor for Stage 1 scores (0 = disabled)')
    parser.add_argument('--uncertainty-metric', type=str, default=None,
                        help='Stage 1 metric: mu_over_var, mu, var, inv_var')
    
    # ==================== Stage 2 Parameters ====================
    parser.add_argument('--fisher-batches', type=int, default=None,
                        help='Number of batches for Fisher calculation')
    parser.add_argument('--fisher-first-order-weight', type=float, default=None,
                        help='Weight for first-order term in Fisher impact')
    parser.add_argument('--fisher-second-order-weight', type=float, default=None,
                        help='Weight for second-order term in Fisher impact')
    parser.add_argument('--use-fisher-scores', action='store_true', default=None,
                        help='Enable Fisher-based ranking in Stage 2')
    parser.add_argument('--no-fisher-scores', action='store_false', default=None,
                        dest='use_fisher_scores',
                        help='Disable Fisher ranking (use magnitude only)')
    parser.add_argument('--allocation-strategy', type=str, default=None,
                        help='Allocation strategy: binary_search, global_fisher, uniform')
    parser.add_argument('--stage2-score-metric', type=str, default=None,
                        help='Stage 2 metric: fisher, taylor, hessian, energy, magnitude')
    parser.add_argument('--pruning-clip-low', type=float, default=None,
                        help='Minimum pruning ratio clip (default: 0.05)')
    parser.add_argument('--pruning-clip-high', type=float, default=None,
                        help='Maximum pruning ratio clip (default: 0.95)')
    parser.add_argument('--binary-search-iterations', type=int, default=None,
                        help='Number of binary-search iterations for global budget allocation')
    parser.add_argument('--binary-search-low', type=float, default=None,
                        help='Lower bound of binary-search scale')
    parser.add_argument('--binary-search-high', type=float, default=None,
                        help='Upper bound of binary-search scale')
    parser.add_argument('--binary-search-tolerance', type=float, default=None,
                        help='Convergence tolerance of binary-search allocation')
    
    # ==================== Fine-tuning Parameters ====================
    parser.add_argument('--fine-tune-epochs', type=int, default=None,
                        help='Number of fine-tuning epochs')
    parser.add_argument('--base-lr', type=float, default=None,
                        help='Base learning rate')
    parser.add_argument('--fine-tune-lr', type=float, default=None,
                        help='Fine-tuning learning rate (overrides base-lr for FT)')
    parser.add_argument('--fine-tune-weight-decay', type=float, default=None,
                        help='Fine-tuning weight decay')
    parser.add_argument('--freeze-epoch', type=int, default=None,
                    help='Epoch to freeze learning rate (default: 85)')
    parser.add_argument('--freeze-lr', type=float, default=None,
                    help='Learning rate to freeze at')
    parser.add_argument('--min-lr-ratio', type=float, default=None,
                    help='Minimum learning rate ratio')
    parser.add_argument('--layer-decay', type=float, default=None,
                        help='Layer-wise decay rate')
    parser.add_argument('--warmup-epochs', type=int, default=None,
                        help='Number of warmup epochs')
    parser.add_argument('--loss-type', type=str, choices=['ce', 'mse'], default=None,
                        help='Loss function to use during fine-tuning (ce for CrossEntropy, mse for Square Error)')
    parser.add_argument('--mixed-precision', action='store_true', default=None,
                        help='Enable AMP mixed precision')
    parser.add_argument('--eval-max-batches', type=int, default=None,
                        help='Limit evaluation batches (0 = full)')
    parser.add_argument('--train-max-batches', type=int, default=None,
                        help='Limit training batches per epoch (0 = full)')
    
    # ==================== Training Augmentation ====================
    parser.add_argument('--use-mixup', action='store_true', default=None,
                        help='Use MixUp augmentation')
    parser.add_argument('--mixup-alpha', type=float, default=None,
                        help='MixUp alpha parameter')
    
    # ==================== Logging & Storage ====================
    parser.add_argument('--save-dir', type=str, default=None,
                        help='Directory to save results')
    parser.add_argument('--wandb', action='store_true', default=None,
                        help='Use Weights & Biases logging')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config JSON file')
    
    args = parser.parse_args()
    
    # ==================== Initialize Config ====================
    if args.config:
        print(f"Loading base configuration from file: {args.config}")
        config = PruningConfig.load(args.config)
    else:
        config = PruningConfig()
        print("Using PruningConfig defaults as the base configuration")
    
    # ==================== CLI Parameter Override ====================
    for key, value in vars(args).items():
        if value is not None and hasattr(config, key):
            setattr(config, key, value)

    parsed_mlp_hidden_dims = _parse_int_list(args.mlp_hidden_dims)
    if parsed_mlp_hidden_dims is not None:
        config.mlp_hidden_dims = parsed_mlp_hidden_dims

    parsed_cnn_channels = _parse_int_list(args.cnn_channels)
    if parsed_cnn_channels is not None:
        config.cnn_channels = parsed_cnn_channels

    parsed_target_columns = _parse_str_list(args.target_columns)
    if parsed_target_columns is not None:
        config.target_columns = parsed_target_columns

    _configure_task_defaults(config)
     
    # Special handling for boolean flags
    if args.use_mixup is not None:
        if 'augmentation' not in config.__dict__:
            config.augmentation = {'train': {}}
        config.augmentation['train']['mixup_alpha'] = args.mixup_alpha if args.mixup_alpha else 0.2
    
    if args.wandb is not None:
        if 'logging' not in config.__dict__:
            config.logging = {}
        config.logging['wandb'] = args.wandb
    
    # Update compression percentage
    config._compression_percentage = int(config.target_compression * 100)

    if not getattr(config, 'custom_tag', ''):
        config.custom_tag = ""
    
    # ==================== Show Final Settings ====================
    print("\n" + "="*60)
    print("Final Configuration:")
    print("="*60)
    for key, value in config.to_dict().items():
        if not key.startswith('_'):
            print(f"{key:30}: {value}")
    print("="*60 + "\n")
    
    prune_and_finetune_model(
        model_type=get_model_family(getattr(config, "model_name", "resnet50")),
        pruning_ratio=config.target_compression,
        fine_tune_epochs=config.fine_tune_epochs,
        config=config,
    )


if __name__ == '__main__':
    main()

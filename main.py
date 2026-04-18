"""ResNet-50 Pruning Main Script - Corresponds to ViT main.py"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse
import random
import torch
import torch.nn as nn
import numpy as np
import time
import json
from typing import Optional

from configs.config import PruningConfig
from data.build import get_resnet_data_loaders_with_calib
from models.resnet_setup import setup_resnet_model
from models.custom_layers import SimpleSVDConv
from models.utils import collect_resnet_conv_layers, get_resnet_parent_and_name
from pruning.stage1_uncertainty import compute_uncertainty_stage1_r50
from pruning.stage2_fisher import compute_fisher_impact_stage2_r50
from pruning.allocation import allocate_pruning_binary_search_r50
from pruning.core import replace_resnet_conv_layer
from utils.engine import fine_tune_resnet_improved
from utils.metrics import evaluate_with_topk_r50, compute_flops_resnet
from utils.visualization import (
    plot_training_history,
    plot_layer_budget_allocation_r50,
    analyze_all_layer_sensitivity_r50
)
from utils.helpers import ModelStructureManager, generate_report, batch_clean_experiment


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


def main_r50(config: Optional[PruningConfig] = None):
    """
    Main pruning flow for ResNet-50
    
    Args:
        config: Pruning config
    """
    if config is None:
        config = PruningConfig()
    
    # Update experiment name
    config.experiment_name = f"ResNet50_SecondOrderPruning_{int(config.target_compression*100)}pr"
    
    print("=" * 80)
    print("RESNET-50 SECOND-ORDER IMPACT PRUNING")
    print("Model: ResNet-50 trained on ImageNet-1k (~76.1% top-1)")
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
    
    # ========== 1. Load Model ==========
    print("\n1. Loading ResNet-50 model")
    model, data_config = setup_resnet_model(config)
    model.eval()
    
    # ========== 2. Prepare Data ==========
    print("\n2. Preparing data")
    train_loader, val_loader, calib_loader = get_resnet_data_loaders_with_calib(config)
    if calib_loader is None:
        print("Calibration loader not specified; using train_loader as D_cal.")
        calib_loader = train_loader
    else:
        print(f"Calibration loader ready: {len(calib_loader)} batches")
    
    # ========== 3. Evaluate Original Model ==========
    print("\n3. Evaluating original ResNet-50 model")
    orig_top1, orig_top5 = evaluate_with_topk_r50(model, val_loader, config=config)
    orig_params = sum(p.numel() for p in model.parameters())
    orig_flops = compute_flops_resnet(model)
    
    print(f"\nResNet-50 Results:")
    print(f"  Top-1 Accuracy: {orig_top1:.2f}%")
    print(f"  Top-5 Accuracy: {orig_top5:.2f}%")
    print(f"  Parameters: {orig_params/1e6:.2f}M")
    if orig_flops:
        print(f"  FLOPs: {orig_flops/1e9:.2f}G")
    
    # ========== 4. Collect Conv Layers ==========
    print("\n4. Collecting convolutional layers for pruning")
    conv_paths = collect_resnet_conv_layers(model)
    print(f"Found {len(conv_paths)} convolutional layers")
    
    # ========== 5. Replace with SVD Layers ==========
    print("\n5. Replacing convolutional layers with SVD layers")
    svd_layers = {}
    
    for path in conv_paths:
        parent, layer_name = get_resnet_parent_and_name(model, path)
        if parent is None:
            continue
        
        # Get original conv layer
        original_conv = None
        if hasattr(parent, layer_name):
            original_conv = getattr(parent, layer_name)
        elif hasattr(parent, '_modules') and layer_name in parent._modules:
            original_conv = parent._modules[layer_name]
        
        if original_conv is None or not isinstance(original_conv, nn.Conv2d):
            continue
        
        # Create SVD layer
        svd_conv = SimpleSVDConv(original_conv, path, config=config).to(config.device)
        
        # Replace layer in model
        if hasattr(parent, layer_name):
            setattr(parent, layer_name, svd_conv)
        elif hasattr(parent, '_modules') and layer_name in parent._modules:
            parent._modules[layer_name] = svd_conv
        
        svd_layers[path] = svd_conv
    
    print(f"Created {len(svd_layers)} SVD layers")
    
    # ========== 6. Two-Stage Pruning Allocation ==========
    print("\n6. Two-Stage Adaptive Budget Allocation")

    allocation_strategy = getattr(config, "allocation_strategy", "binary_search")
    stage2_metric = getattr(config, "stage2_score_metric", "fisher")

    def _compute_layer_importance_from_scores(scores_dict):
        raw = {}
        for lname, scores in scores_dict.items():
            if scores is None or len(scores) == 0:
                continue
            raw[lname] = float(np.mean(scores))
        if not raw:
            return {}
        vals = np.array(list(raw.values()), dtype=np.float64)
        min_v, max_v = vals.min(), vals.max()
        norm = {}
        for k, v in raw.items():
            if max_v - min_v > 1e-8:
                norm[k] = (v - min_v) / (max_v - min_v)
            else:
                norm[k] = 0.5
        return norm

    layer_importance_stage1 = {}
    component_impact_scores = {}

    if allocation_strategy == "global_fisher":
        print("\n" + "="*80)
        print("Stage 1 (Global Fisher Allocation): using Fisher-based layer scores")
        print("="*80)
        
        # We always need Fisher for the inter-layer allocation in this mode
        fisher_scores = compute_fisher_impact_stage2_r50(
            model, svd_layers, train_loader, config=config
        )
        layer_importance_stage1 = _compute_layer_importance_from_scores(fisher_scores)
        
        # Decouple Selection metric for Stage 2
        if stage2_metric == "magnitude":
            component_impact_scores = {}
            print("Budget Allocation: Fisher-Mean | Component Selection: Magnitude")
        elif stage2_metric == "energy":
            component_impact_scores = {
                name: (layer.get_sigma().detach().cpu().numpy() ** 2)
                for name, layer in svd_layers.items()
            }
            print("Budget Allocation: Fisher-Mean | Component Selection: Energy")
        else:
            component_impact_scores = fisher_scores
            print("Budget Allocation: Fisher-Mean | Component Selection: Fisher")
    else:
        # --- Stage 1: Inter-layer Uncertainty ---
        print("\n" + "="*80)
        print("Stage 1: Inter-layer Budget Allocation (Uncertainty Estimation)")
        print("="*80)

        layer_importance_stage1 = compute_uncertainty_stage1_r50(
            model, svd_layers, calib_loader, config=config
        )

        # --- Stage 2: Intra-layer Component Selection ---
        print("\n" + "="*80)
        print("Stage 2: Intra-layer Component Selection (Fisher-Aware Scoring)")
        print("="*80)

        if stage2_metric == "magnitude":
            component_impact_scores = {}
            print("Stage 2 metric: magnitude (no Fisher scores).")
        elif stage2_metric == "energy":
            component_impact_scores = {
                name: (layer.get_sigma().detach().cpu().numpy() ** 2)
                for name, layer in svd_layers.items()
            }
            print("Stage 2 metric: energy (sigma^2).")
        else:
            if bool(getattr(config, "use_fisher_scores", True)):
                component_impact_scores = compute_fisher_impact_stage2_r50(
                    model, svd_layers, train_loader, config=config
                )
            else:
                print("Stage 2 Fisher scoring disabled; using magnitude-based ranking.")

    # --- ????閬箏? ---
    vis_dir = os.path.join(save_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    analyze_all_layer_sensitivity_r50(svd_layers, component_impact_scores, vis_dir)
    
    # --- Global Allocation: Binary Search ---
    print("\n" + "="*80)
    print(f"Allocation: Binary Search for Target Compression ({config.target_compression*100:.1f}%)")
    print("="*80)
    
    keep_ratios = allocate_pruning_binary_search_r50(
        layer_importance_stage1, svd_layers, config=config, total_model_params=orig_params
    )
    
    # 蝜芾ˊ撅丁udget Allocation??
    type_keep_ratios = plot_layer_budget_allocation_r50(
        svd_layers, keep_ratios, layer_importance_stage1, vis_dir
    )
    
    # ========== 7. Execute Pruning ==========
    print("\n7. Executing pruning on ResNet-50")
    
    # Reload original model for pruning
    pruned_model, _ = setup_resnet_model(config)
    pruned_model.eval()
    
    pruning_details = {}
    pruned_count = 0
    
    for name, svd_layer in svd_layers.items():
        if name not in keep_ratios:
            continue
        
        keep_ratio = keep_ratios[name]
        layer_impact_scores = component_impact_scores.get(name, None)
        
        # Execute replacement: Conv2d -> LowRankConv
        pruned_rank = replace_resnet_conv_layer(
            pruned_model, name, svd_layer, keep_ratio,
            impact_scores=layer_impact_scores,
            config=config
        )
        
        if pruned_rank is not None:
            pruned_count += 1
            recon_error = svd_layer.get_reconstruction_error(keep_ratio)
            pruning_details[name] = {
                'is_bottleneck': svd_layer.is_bottleneck,
                'keep_ratio': keep_ratio,
                'pruned_rank': pruned_rank,
                'original_rank': svd_layer.full_rank,
                'recon_error': recon_error,
                'original_cin': svd_layer.cin,
                'original_cout': svd_layer.cout,
                'kernel_size': (svd_layer.kh, svd_layer.kw),
                'has_bias': svd_layer.bias is not None,
                'used_fisher_selection': layer_impact_scores is not None
            }
    
    print(f"\nSuccessfully pruned {pruned_count}/{len(keep_ratios)} layers")
    
    # ========== 8. Evaluate Pruned Model ==========
    print("\n8. Evaluating pruned ResNet-50 model (before fine-tuning)")
    pruned_top1, pruned_top5 = evaluate_with_topk_r50(pruned_model, val_loader, config=config)
    pruned_params = sum(p.numel() for p in pruned_model.parameters())
    
    # ========== 9. Save Model (Before FT) ==========
    print("\n9. Saving pruned ResNet-50 model (Before FT)")
    config_dict = config.to_dict()
    config_dict.update({'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')})
    
    ModelStructureManager.save_complete_model(
        pruned_model, pruning_details, keep_ratios, layer_importance_stage1,
        before_finetune_path,
        {'top1': orig_top1, 'top5': orig_top5},
        {'top1': pruned_top1, 'top5': pruned_top5},
        config=config_dict
    )
    
    # ========== Results Summary ==========
    print("\n" + "="*80)
    print("RESNET-50 RESULTS SUMMARY (BEFORE FINE-TUNING)")
    print("="*80)
    
    param_reduction = 100 * (1 - pruned_params / orig_params)
    top1_change = pruned_top1 - orig_top1
    top5_change = pruned_top5 - orig_top5
    
    print(f"\nOriginal ResNet-50 Model:")
    print(f"  Parameters: {orig_params/1e6:.2f}M")
    if orig_flops:
        print(f"  FLOPs: {orig_flops/1e9:.2f}G")
    print(f"  Top-1 Accuracy: {orig_top1:.2f}%")
    print(f"  Top-5 Accuracy: {orig_top5:.2f}%")
    
    print(f"\nPruned ResNet-50 Model (Before Fine-tuning):")
    print(f"  Parameters: {pruned_params/1e6:.2f}M ({param_reduction:.1f}% reduction)")
    pruned_flops = compute_flops_resnet(pruned_model)
    if pruned_flops and orig_flops:
        flops_reduction = 100 * (1 - pruned_flops / orig_flops)
        print(f"  FLOPs: {pruned_flops/1e9:.2f}G ({flops_reduction:.1f}% reduction)")
    
    print(f"  Top-1 Accuracy: {pruned_top1:.2f}% ({top1_change:+.2f}%)")
    print(f"  Top-5 Accuracy: {pruned_top5:.2f}% ({top5_change:+.2f}%)")
    
    # Layer statistics
    print(f"\n10. Layer Statistics:")
    for layer_type, ratios in type_keep_ratios.items():
        avg_keep = np.mean(ratios) if ratios else 0
        print(f"  {layer_type}: {len(ratios)} layers, avg keep ratio: {avg_keep:.2%}")
    
    avg_keep = np.mean(list(keep_ratios.values()))
    print(f"\n  Avg keep ratio (all layers): {avg_keep:.2%}")
    
    # ========== 10. Fine-tuning ==========
    print("\n10. Starting fine-tuning for ResNet-50")
    
    final_model = pruned_model
    final_top1, final_top5 = pruned_top1, pruned_top5
    fine_tune_history = None
    ft_dir = os.path.join(save_dir, 'finetunemodel')
    
    if config.fine_tune_epochs > 0:
        final_top1, final_top5, fine_tune_history = fine_tune_resnet_improved(
            model=pruned_model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=None,
            config=config,
            save_dir=ft_dir
        )
        
        # Save fine-tuned model
        ModelStructureManager.save_complete_model(
            final_model, pruning_details, keep_ratios, layer_importance_stage1,
            finetuned_path,
            {'top1': orig_top1, 'top5': orig_top5},
            {'top1': final_top1, 'top5': final_top5},
            test_accuracy={'top1': final_top1, 'top5': final_top5},
            config=config_dict
        )
    
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
    orig_top1=orig_top1, orig_top5=orig_top5,
    pruned_top1=pruned_top1, pruned_top5=pruned_top5,
    final_top1=final_top1, final_top5=final_top5,
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
    return pruned_model, pruning_details


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
        print(f"Loading configuration from: {args.config}")
        config = PruningConfig.load(args.config)
    else:
        config = PruningConfig()
        print("Using default configuration")
    
    # ==================== CLI Parameter Override ====================
    for key, value in vars(args).items():
        if value is not None and hasattr(config, key):
            setattr(config, key, value)
    
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
    
    # ==================== Execute Main Program ====================
    main_r50(config)


if __name__ == '__main__':
    main()


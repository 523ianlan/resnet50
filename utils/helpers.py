"""Helper tools - Corresponds to ViT helpers.py"""

import os
import json
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional
import gc
import time


class ModelStructureManager:
    """Model structure manager - Shared with ViT version"""
    
    @staticmethod
    def analyze_model_structure(model: nn.Module) -> Dict:
        """Analyze model structure"""
        structure_info = {}
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                structure_info[name] = {
                    'type': 'conv2d',
                    'in_channels': module.in_channels,
                    'out_channels': module.out_channels,
                    'kernel_size': module.kernel_size,
                    'stride': module.stride,
                    'padding': module.padding,
                    'bias': module.bias is not None,
                    'is_original': True
                }
            elif isinstance(module, nn.Linear):
                structure_info[name] = {
                    'type': 'linear',
                    'in_features': module.in_features,
                    'out_features': module.out_features,
                    'bias': module.bias is not None,
                    'is_original': True
                }
            elif isinstance(module, nn.Sequential) and len(module) == 2:
                if isinstance(module[0], nn.Conv2d) and isinstance(module[1], nn.Conv2d):
                    conv1, conv2 = module[0], module[1]
                    structure_info[name] = {
                        'type': 'decomposed_conv',
                        'conv1': {
                            'in_channels': conv1.in_channels,
                            'out_channels': conv1.out_channels,
                            'kernel_size': conv1.kernel_size,
                            'stride': conv1.stride,
                            'padding': conv1.padding,
                            'bias': conv1.bias is not None
                        },
                        'conv2': {
                            'in_channels': conv2.in_channels,
                            'out_channels': conv2.out_channels,
                            'kernel_size': conv2.kernel_size,
                            'stride': conv2.stride,
                            'padding': conv2.padding,
                            'bias': conv2.bias is not None
                        },
                        'is_original': False,
                        'decomposed': True
                    }
                elif isinstance(module[0], nn.Linear) and isinstance(module[1], nn.Linear):
                    linear1, linear2 = module[0], module[1]
                    structure_info[name] = {
                        'type': 'decomposed_linear',
                        'linear1': {
                            'in_features': linear1.in_features,
                            'out_features': linear1.out_features,
                            'bias': linear1.bias is not None
                        },
                        'linear2': {
                            'in_features': linear2.in_features,
                            'out_features': linear2.out_features,
                            'bias': linear2.bias is not None
                        },
                        'is_original': False,
                        'decomposed': True
                    }
        
        return structure_info
    
    @staticmethod
    def save_complete_model(
        pruned_model: nn.Module,
        pruning_details: Dict,
        keep_ratios: Dict,
        layer_importance_stage1: Dict,
        save_path: str,
        original_accuracy: Dict,
        pruned_accuracy: Dict,
        test_accuracy: Dict = None,
        config: Dict = None
    ):
        """Save complete pruned model information"""
        print(f"\nSaving complete model information to {save_path}")
        
        model_structure = ModelStructureManager.analyze_model_structure(pruned_model)
        
        saved_data = {
            'model_state_dict': pruned_model.state_dict(),
            'model_structure': model_structure,
            'pruning_details': pruning_details,
            'keep_ratios': keep_ratios,
            'layer_importance_stage1': layer_importance_stage1,
            'original_accuracy': original_accuracy,
            'pruned_accuracy': pruned_accuracy,
            'config': config if config else {}
        }
        
        if test_accuracy is not None:
            saved_data['test_accuracy'] = test_accuracy
        
        torch.save(saved_data, save_path)
        
                # Save readable config file
        config_path = save_path.replace('.pth', '_config.json')
        with open(config_path, 'w') as f:
            json_data = {
                'model_structure': {k: str(v) for k, v in model_structure.items()},
                'pruning_summary': {
                    'total_layers_pruned': len(pruning_details),
                    'bottleneck_layers': sum(1 for d in pruning_details.values() if d.get('is_bottleneck', False)),
                    'standard_layers': len(pruning_details) - sum(1 for d in pruning_details.values() if d.get('is_bottleneck', False)),
                    'avg_keep_ratio': float(np.mean(list(keep_ratios.values())) if keep_ratios else 0),
                    'compression_percentage': int(config.get('target_compression', 0) * 100) if config else 0
                },
                'accuracy_info': {
                    'original': original_accuracy,
                    'pruned': pruned_accuracy,
                    'test': test_accuracy if test_accuracy else {}
                }
            }
            json.dump(json_data, f, indent=2, default=str)
        
        print(f"Complete model saved to: {save_path}")
        print(f"Config file saved to: {config_path}")
        
        return saved_data

def generate_report(
    save_dir: str,
    config,
    model: nn.Module,
    svd_layers: Dict,
    layer_importance_stage1: Dict,
    component_impact_scores: Dict,
    keep_ratios: Dict,
    pruning_details: Dict,
    orig_params: int,
    pruned_params: int,  # This parameter might not be used, recalculated from pruning_details
    orig_top1: float,
    orig_top5: float,
    pruned_top1: float,
    pruned_top5: float,
    final_top1: float,
    final_top5: float,
    orig_flops: float = None,
    pruned_flops: float = None,
    history: Dict = None
) -> str:
    """
    Generate complete pruning report
    
    Args:
        save_dir: Save directory
        config: Configuration object
        model: Model (used to get structure info)
        svd_layers: SVD layers dictionary
        layer_importance_stage1: Stage 1 importance scores
        component_impact_scores: Stage 2 impact scores
        keep_ratios: Keep ratio per layer
        pruning_details: Pruning details (includes pruned rank per layer etc)
        orig_params: Original parameter count
        pruned_params: Pruned parameter count
        orig_top1: Original Top-1 accuracy
        orig_top5: Original Top-5 accuracy
        pruned_top1: Pruned Top-1 accuracy
        pruned_top5: Pruned Top-5 accuracy
        final_top1: Fine-tuned Top-1 accuracy
        final_top5: Fine-tuned Top-5 accuracy
        orig_flops: Original FLOPs
        pruned_flops: Pruned FLOPs
        history: Training history (includes loss and accuracy per epoch)
    
    Returns:
        Report file path
    """
    report_lines = []
    
    # ========== Recalculate pruned params from pruning_details ==========
    calculated_pruned_params = 0
    layer_param_details = []
    
    for name, detail in pruning_details.items():
        if 'pruned_rank' in detail:
            rank = detail['pruned_rank']
            
            # Calculate params based on layer type
            if 'original_cin' in detail and 'original_cout' in detail:
                cin = detail['original_cin']
                cout = detail['original_cout']
                kh, kw = detail.get('kernel_size', (3, 3))
                
                # Params after low-rank decomposition: conv1 + conv2
                conv1_params = cin * kh * kw * rank
                conv2_params = cout * rank
                layer_params = conv1_params + conv2_params
                
                # Add bias (if any)
                if detail.get('has_bias', False):
                    layer_params += cout
                
                calculated_pruned_params += layer_params
                layer_param_details.append({
                    'name': name,
                    'rank': rank,
                    'params': layer_params,
                    'orig_params': cin * kh * kw * cout + (cout if detail.get('has_bias', False) else 0)
                })
            elif 'original_in_features' in detail and 'original_out_features' in detail:
                in_features = detail['original_in_features']
                out_features = detail['original_out_features']
                layer_params = (in_features * rank) + (out_features * rank)
                if detail.get('has_bias', False):
                    layer_params += out_features

                calculated_pruned_params += layer_params
                layer_param_details.append({
                    'name': name,
                    'rank': rank,
                    'params': layer_params,
                    'orig_params': in_features * out_features + (out_features if detail.get('has_bias', False) else 0)
                })
    
    # Fix: Trust the comprehensive `pruned_params` computed accurately via `sum(p.numel())`,
    # rather than just the partial convolution parameter sum `calculated_pruned_params`.
    final_pruned_params = pruned_params
    print(f"Using accurately tracked real pruned params (Full Network): {final_pruned_params/1e6:.2f}M")
    if calculated_pruned_params > 0:
        print(f"  (Info: SVD reduced convolutional parameters sub-total: {calculated_pruned_params/1e6:.2f}M)")
    
    # ========== Title ==========
    report_lines.append("=" * 100)
    report_lines.append("RESNET-50 SECOND-ORDER IMPACT PRUNING COMPLETE REPORT")
    report_lines.append("=" * 100)
    report_lines.append(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Model: ResNet-50")
    report_lines.append("")
    
    # ========== Config Summary ==========
    report_lines.append("-" * 100)
    report_lines.append("PRUNING CONFIGURATION")
    report_lines.append("-" * 100)
    report_lines.append(f"Target Compression: {config.target_compression:.1%}")
    report_lines.append(f"Min Rank: {config.min_rank}")
    report_lines.append(f"MC Samples (Stage 1): {config.mc_samples}")
    report_lines.append(f"MC Dropout p (Stage 1): {getattr(config, 'mc_dropout_p', 'na')}")
    report_lines.append(f"Calibration Batches (Stage 1): {getattr(config, 'calib_batches', 'na')}")
    report_lines.append(f"Fisher Batches (Stage 2): {config.fisher_batches}")
    report_lines.append(f"Train Max Batches (FT): {getattr(config, 'train_max_batches', 'na')}")
    report_lines.append(f"Eval Max Batches: {getattr(config, 'eval_max_batches', 'na')}")
    report_lines.append(f"Uncertainty Alpha: {config.uncertainty_alpha}")
    report_lines.append(f"Pruning Clip Range: [{config.pruning_clip_low:.1%}, {config.pruning_clip_high:.1%}]")
    report_lines.append("")
    
    # ========== Stage 1: Inter-layer Uncertainty ==========
    report_lines.append("=" * 100)
    report_lines.append("STAGE 1: INTER-LAYER UNCERTAINTY ESTIMATION")
    report_lines.append("=" * 100)
    report_lines.append(f"{'Layer Name':<40} {'Type':<15} {'Stability':<12}")
    report_lines.append("-" * 100)
    
    sorted_importance = sorted(layer_importance_stage1.items(), key=lambda x: x[1], reverse=True)
    for name, score in sorted_importance:
        layer_obj = svd_layers.get(name)
        if layer_obj is not None and getattr(layer_obj, "layer_kind", "conv") == "linear":
            layer_type = "Linear"
        else:
            layer_type = "Bottleneck" if (layer_obj is not None and layer_obj.is_bottleneck) else "Standard"
        report_lines.append(f"{name:<40} {layer_type:<15} {score:.6f}")
    
    report_lines.append("")
    
    # ========== Stage 2: Fisher Impact ==========
    report_lines.append("=" * 100)
    report_lines.append("STAGE 2: INTRA-LAYER FISHER IMPACT ANALYSIS")
    report_lines.append("=" * 100)
    report_lines.append(f"{'Layer Name':<40} {'Type':<15} {'Correlation':<12}")
    report_lines.append("-" * 100)
    
    correlations = []
    for name, layer in svd_layers.items():
        if name in component_impact_scores:
            impact_scores = component_impact_scores[name]
            singular_values = layer.get_sigma().cpu().detach().numpy()
            min_len = min(len(singular_values), len(impact_scores))
            if min_len > 1:
                corr = np.corrcoef(singular_values[:min_len], impact_scores[:min_len])[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)
                    layer_type = "Linear" if getattr(layer, "layer_kind", "conv") == "linear" else ("Bottleneck" if layer.is_bottleneck else "Standard")
                    report_lines.append(f"{name:<40} {layer_type:<15} {corr:.4f}")
    
    report_lines.append("")
    if correlations:
        report_lines.append(f"Average Correlation: {np.mean(correlations):.4f}")
        report_lines.append(f"Min Correlation: {np.min(correlations):.4f}")
        report_lines.append(f"Max Correlation: {np.max(correlations):.4f}")
    report_lines.append("")
    
    # ========== Budget Allocation ==========
    report_lines.append("=" * 100)
    report_lines.append("LAYER-WISE PRUNING ALLOCATION")
    report_lines.append("=" * 100)
    report_lines.append(f"{'Layer Name':<40} {'Type':<15} {'Stability':<10} {'Keep%':<8} {'Rank':<12} {'Params(K)':<10}")
    report_lines.append("-" * 100)
    
    total_orig_layer_params = 0
    total_pruned_layer_params = 0
    
    for name, layer in svd_layers.items():
        if name not in keep_ratios:
            continue
        
        keep_ratio = keep_ratios[name]
        stability = layer_importance_stage1.get(name, 0)
        layer_type = "Linear" if getattr(layer, "layer_kind", "conv") == "linear" else ("Bottleneck" if layer.is_bottleneck else "Standard")

        rank = layer.get_keep_count(keep_ratio)
        orig_layer_params = layer.get_original_param_count()
        pruned_layer_params = layer.get_pruned_param_count(rank)
        
        total_orig_layer_params += orig_layer_params
        total_pruned_layer_params += pruned_layer_params
        
        report_lines.append(
            f"{name:<40} {layer_type:<15} {stability:.4f}    "
            f"{keep_ratio:6.1%} {rank:3d}/{layer.full_rank:<6} {pruned_layer_params/1000:>8.2f}"
        )
    
    report_lines.append("-" * 100)
    report_lines.append(f"{'TOTAL':<40} {'':<15} {'':<10} {'':<8} {'':<12} {total_pruned_layer_params/1000:>8.2f}")
    report_lines.append("")
    
    # ========== Pruning Details ==========
    report_lines.append("=" * 100)
    report_lines.append("PRUNING DETAILS")
    report_lines.append("=" * 100)
    report_lines.append(f"{'Layer Name':<40} {'Type':<15} {'Keep%':<8} {'Rank':<12} {'Recon Error':<12}")
    report_lines.append("-" * 100)
    
    for name, detail in pruning_details.items():
        detail_kind = detail.get('layer_kind', 'conv')
        layer_type = "Linear" if detail_kind == "linear" else ("Bottleneck" if detail.get('is_bottleneck', False) else "Standard")
        keep_ratio = detail.get('keep_ratio', 0)
        pruned_rank = detail.get('pruned_rank', 0)
        original_rank = detail.get('original_rank', 0)
        recon_error = detail.get('recon_error', 0)
        
        report_lines.append(
            f"{name:<40} {layer_type:<15} {keep_ratio:6.1%} "
            f"{pruned_rank:3d}/{original_rank:<6} {recon_error:11.4f}"
        )
    
    report_lines.append("")
    # ========== FINE-TUNING LOG ==========
    if history is not None and len(history.get('train_loss', [])) > 0:
        report_lines.append("=" * 100)
        report_lines.append("FINE-TUNING LOG")
        report_lines.append("=" * 100)
        total_epochs = config.fine_tune_epochs if hasattr(config, 'fine_tune_epochs') else len(history['train_loss'])
        best_val_top1 = float("-inf")
        best_val_loss = None
        min_delta = getattr(config, 'early_stopping_min_delta', 0)
        num_epochs = len(history['train_loss'])
        for i in range(num_epochs):
            epoch = i + 1
            train_loss = history['train_loss'][i]
            train_top1 = history['train_top1'][i]
            train_top5 = history['train_top5'][i] if 'train_top5' in history and i < len(history['train_top5']) else 0
            val_loss = history['val_loss'][i] if i < len(history['val_loss']) else 0
            val_top1 = history['val_top1'][i] if i < len(history['val_top1']) else 0
            val_top5 = history['val_top5'][i] if 'val_top5' in history and i < len(history['val_top5']) else 0
            lr = history['learning_rate'][i] if i < len(history['learning_rate']) else 0

            report_lines.append(
                f"Epoch {epoch:>3d}/{total_epochs} | Train: Loss={train_loss:.4f}, Top1={train_top1:.2f}%,  Top5={train_top5:.2f}%| "
                f"Val: Loss={val_loss:.4f}, Top1={val_top1:.2f}%, Top5={val_top5:.2f}% | LR={lr:.2e}"
            )

            if val_top1 > best_val_top1:
                best_val_top1 = val_top1
                report_lines.append(f"  New best model saved (Top1: {best_val_top1:.2f}%)")

            if best_val_loss is None:
                best_val_loss = val_loss
            elif val_loss < best_val_loss - min_delta:
                report_lines.append(f"Validation loss improved from {best_val_loss:.6f} to {val_loss:.6f}")
                best_val_loss = val_loss

        report_lines.append("")

    # ========== ACCURACY RESULTS ==========
    report_lines.append("=" * 100)
    report_lines.append("ACCURACY RESULTS")
    report_lines.append("=" * 100)
    report_lines.append(f"{'Model State':<25} {'Top-1 Acc':<12} {'Top-5 Acc':<12} {'Change':<12}")
    report_lines.append("-" * 100)
    report_lines.append(f"{'Original':<25} {orig_top1:>10.2f}%    {orig_top5:>10.2f}%    {'-':<12}")
    report_lines.append(f"{'Pruned (No FT)':<25} {pruned_top1:>10.2f}%    {pruned_top5:>10.2f}%    "
                        f"{pruned_top1 - orig_top1:>+10.2f}%")
    report_lines.append(f"{'Final (After FT)':<25} {final_top1:>10.2f}%    {final_top5:>10.2f}%    "
                        f"{final_top1 - orig_top1:>+10.2f}%")
    report_lines.append("")
    report_lines.append("=" * 100)
    report_lines.append("RESOURCE SUMMARY")
    report_lines.append("=" * 100)
    
    param_reduction = 100 * (1 - final_pruned_params / orig_params) if orig_params > 0 else 0
    
    report_lines.append(f"{'Metric':<20} {'Original':<15} {'Pruned':<15} {'Reduction':<15}")
    report_lines.append("-" * 100)
    report_lines.append(f"{'Parameters':<20} {orig_params/1e6:>8.2f}M       {final_pruned_params/1e6:>8.2f}M       {param_reduction:>8.1f}%")
    
    if orig_flops is not None and pruned_flops is not None:
        flops_reduction = 100 * (1 - pruned_flops / orig_flops)
        report_lines.append(f"{'FLOPs':<20} {orig_flops/1e9:>8.2f}G       {pruned_flops/1e9:>8.2f}G       {flops_reduction:>8.1f}%")
    
    report_lines.append("")
    
    # ========== CONCLUSION ==========
    report_lines.append("=" * 100)
    report_lines.append("CONCLUSION")
    report_lines.append("=" * 100)
    
    final_change = final_top1 - orig_top1
    target_comp = config.target_compression * 100
    actual_comp = 100 * (1 - final_pruned_params / orig_params) if orig_params > 0 else 0
    
    report_lines.append(f"Target Compression: {target_comp:.1f}%")
    report_lines.append(f"Achieved Compression: {actual_comp:.1f}%")
    report_lines.append(f"Final Accuracy Change: {final_change:+.2f}%")
    report_lines.append(f"Final Model Top-1: {final_top1:.2f}%")
    report_lines.append("")
    report_lines.append("=" * 100)
    
    # Write to file
    report_path = os.path.join(save_dir, 'complete_pruning_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"\nComplete report generated and saved to: {report_path}")
    return report_path


def clean_pruned_model(model_path: str, overwrite: bool = True) -> str:
    """
    Clean pruned model file, remove original weights to save space
    
    Args:
        model_path: Model file path
        overwrite: Whether to overwrite original file
    
    Returns:
        Cleaned model path
    """
    print(f"Cleaning pruned model: {model_path}")
    
    # Load Model
    data = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Remove original weights
    if 'model_state_dict' in data:
        state_dict = data['model_state_dict']
        keys_to_remove = [k for k in state_dict.keys() if 'original_weight' in k]
        for k in keys_to_remove:
            del state_dict[k]
        print(f"  Removed {len(keys_to_remove)} original_weight entries")
    
    # Save cleaned model
    if overwrite:
        save_path = model_path
    else:
        save_path = model_path.replace('.pth', '_cleaned.pth')
    
    torch.save(data, save_path)
    print(f"  Cleaned model saved to: {save_path}")
    
    return save_path


def batch_clean_experiment(exp_dir: str, overwrite: bool = True):
    """
    Batch clean all models in experiment directory
    
    Args:
        exp_dir: Experiment directory
        overwrite: Whether to overwrite original file
    """
    print(f"\nBatch cleaning experiment directory: {exp_dir}")
    
    for filename in os.listdir(exp_dir):
        if filename.endswith('.pth') and not filename.endswith('_cleaned.pth'):
            file_path = os.path.join(exp_dir, filename)
            try:
                clean_pruned_model(file_path, overwrite=overwrite)
            except Exception as e:
                print(f"  Error cleaning {filename}: {e}")
    
    print("Batch cleaning completed.")


# Provide aliases to maintain compatibility with ViT version
collect_timm_vit_linear_layers = None  # Not used in ResNet
get_vit_parent_and_name = None  # Not used in ResNet

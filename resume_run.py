import os
import torch
import torch.nn as nn
import json
import numpy as np
from configs.config import PruningConfig
from models.resnet_setup import setup_resnet_model
from models.utils import get_resnet_parent_and_name
from utils.engine import fine_tune_resnet_improved
from utils.helpers import ModelStructureManager, generate_report, batch_clean_experiment
from utils.metrics import compute_flops_resnet

class MockLayer:
    """Mock SimpleSVDConv for report generation"""
    def __init__(self, detail, name):
        self.is_bottleneck = detail.get('is_bottleneck', False)
        self.full_rank = detail.get('original_rank', 0)
        self.cin = detail.get('original_cin', 0)
        self.cout = detail.get('original_cout', 0)
        self.kh, self.kw = detail.get('kernel_size', (3, 3))
        self.bias = torch.zeros(self.cout) if detail.get('has_bias', False) else None
        self.name = name
    
    def get_sigma(self):
        # We don't have the real sigma, return dummy
        return torch.ones(self.full_rank)

def reconstruct_pruned_model(model, model_structure):
    """
    Reconstruct the pruned model structure from the saved structure info.
    """
    for layer_path, info in model_structure.items():
        if info['type'] == 'decomposed_conv':
            parent, layer_name = get_resnet_parent_and_name(model, layer_path)
            if parent is None:
                print(f"Warning: Cannot find parent for {layer_path}")
                continue
            
            c1_info = info['conv1']
            c2_info = info['conv2']
            
            conv1 = nn.Conv2d(
                c1_info['in_channels'], c1_info['out_channels'],
                kernel_size=c1_info['kernel_size'],
                stride=c1_info['stride'],
                padding=c1_info['padding'],
                bias=c1_info['bias']
            )
            
            conv2 = nn.Conv2d(
                c2_info['in_channels'], c2_info['out_channels'],
                kernel_size=c2_info['kernel_size'],
                stride=c2_info['stride'],
                padding=c2_info['padding'],
                bias=c2_info['bias']
            )
            
            seq = nn.Sequential(conv1, conv2)
            
            if hasattr(parent, layer_name):
                setattr(parent, layer_name, seq)
            elif hasattr(parent, '_modules') and layer_name in parent._modules:
                parent._modules[layer_name] = seq
            else:
                print(f"Error: Could not replace layer {layer_path}")

def resume():
    # 1. Configuration and Paths
    exp_dir = r"results\FISHER mean layer importance result\comp60_ft90_low0.2_high0.8_result_20260413_194543\r50_60pr_90ft"
    save_dir = r"results\FISHER mean layer importance result\comp60_ft90_low0.2_high0.8_result_20260413_194543\r50_60pr_90ft"
    before_ft_path = os.path.join(exp_dir, "r50_60pr_before_finetune.pth")
    ckpt_path = os.path.join(exp_dir, "finetunemodel", "epoch86.pth")
    
    if not os.path.exists(before_ft_path):
        print(f"Error: {before_ft_path} not found.")
        return

    # Load initial backup to get structure and accuracies
    checkpoint = torch.load(before_ft_path, map_location='cpu', weights_only=False)
    config_dict = checkpoint['config']
    model_structure = checkpoint['model_structure']
    pruning_details = checkpoint['pruning_details']
    keep_ratios = checkpoint['keep_ratios']
    layer_importance_stage1 = checkpoint['layer_importance_stage1']
    orig_acc = checkpoint['original_accuracy']
    pruned_acc = checkpoint['pruned_accuracy']
    
    # Load PruningConfig
    config = PruningConfig()
    for k, v in config_dict.items():
        if hasattr(config, k):
            if k == 'device' and isinstance(v, str):
                setattr(config, k, torch.device(v))
            else:
                setattr(config, k, v)
    
    # Override some paths if needed
    config.save_dir = save_dir
    
    # 2. Setup Base Model
    print("Loading base ResNet-50 model...")
    model, _ = setup_resnet_model(config, pretrained=False) # We will load weights later
    
    # 3. Reconstruct Pruned Structure
    print("Reconstructing pruned model structure...")
    reconstruct_pruned_model(model, model_structure)
    
    # 4. Load Weights from Epoch 86
    print(f"Loading weights from {ckpt_path}...")
    if not os.path.exists(ckpt_path):
        print(f"Error: {ckpt_path} not found.")
        return
    
    weights = torch.load(ckpt_path, map_location=config.device, weights_only=False)
    model.load_state_dict(weights)
    model.to(config.device)
    
    # 5. Prepare Data Loaders
    from data.build import get_resnet_data_loaders_with_calib
    print("Preparing data loaders...")
    train_loader, val_loader, _ = get_resnet_data_loaders_with_calib(config)
    
    # 6. Setup Initial History (Try to load if it exists, otherwise start fresh)
    # Since we don't have a history file, we can't easily recover 1-86.
    # We will start with empty history or just show 87-90.
    initial_history = {
        "train_loss": [], "train_top1": [], "train_top5": [],
        "val_loss": [], "val_top1": [], "val_top5": [],
        "learning_rate": [],
    }
    
    # 7. Resume Fine-Tuning
    print(f"Resuming fine-tuning from epoch 87 to {config.fine_tune_epochs}...")
    ft_dir = os.path.join(save_dir, 'finetunemodel')
    
    final_top1, final_top5, fine_tune_history = fine_tune_resnet_improved(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        save_dir=ft_dir,
        start_epoch=87,
        initial_history=initial_history
    )
    
    # 8. Save Final Model
    finetuned_path = os.path.join(save_dir, f"r50_60pr_{config.fine_tune_epochs}ft.pth")
    ModelStructureManager.save_complete_model(
        model, pruning_details, keep_ratios, layer_importance_stage1,
        finetuned_path,
        orig_acc,
        pruned_acc,
        test_accuracy={'top1': final_top1, 'top5': final_top5},
        config=config_dict
    )
    
    # 9. Generate Report
    print("Generating report...")
    # Note: If history was lost due to crash, use recover_validation_history.py
    # Reconstruct original model briefly to get original params/flops
    orig_model_base, _ = setup_resnet_model(config, pretrained=False)
    orig_params = sum(p.numel() for p in orig_model_base.parameters())
    orig_flops = compute_flops_resnet(orig_model_base)
    
    pruned_params = sum(p.numel() for p in model.parameters())
    pruned_flops = compute_flops_resnet(model)
    
    # Create mock svd_layers for report
    svd_layers = {}
    for name, detail in pruning_details.items():
        svd_layers[name] = MockLayer(detail, name)

    generate_report(
        save_dir=save_dir,
        config=config,
        model=model,
        svd_layers=svd_layers,
        layer_importance_stage1=layer_importance_stage1,
        component_impact_scores={},
        keep_ratios=keep_ratios,
        pruning_details=pruning_details,
        orig_params=orig_params,
        pruned_params=pruned_params,
        orig_top1=orig_acc['top1'], orig_top5=orig_acc['top5'],
        pruned_top1=pruned_acc['top1'], pruned_top5=pruned_acc['top5'],
        final_top1=final_top1, final_top5=final_top5,
        orig_flops=orig_flops,
        pruned_flops=pruned_flops,
        history=fine_tune_history
    )
    
    # 10. Cleanup
    print("Cleaning up...")
    batch_clean_experiment(save_dir, overwrite=True)
    if os.path.isdir(ft_dir):
        batch_clean_experiment(ft_dir, overwrite=True)

    print("Resume completed successfully.")

if __name__ == "__main__":
    resume()

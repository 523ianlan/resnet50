import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kendalltau, spearmanr
from tqdm import tqdm
from typing import Dict, List
import time

from configs.config import PruningConfig
from data.build import get_optimized_data_loaders_r50
from models.resnet_setup import setup_resnet_model
from models.custom_layers import SimpleSVDConv
from models.utils import collect_resnet_conv_layers, get_resnet_parent_and_name

NUM_CLASSES = 1000
DEBUG_VALIDATE_LABELS = True

def _collect_batches(data_loader, num_batches: int):
    """Collect a small fixed set of batches for repeated evaluation."""
    if num_batches <= 0:
        return []
    batches = []
    for i, (images, labels) in enumerate(data_loader):
        if i >= num_batches:
            break
        batches.append((images, labels))
    return batches


def _compute_avg_loss_on_batches(
    model: nn.Module,
    batches: List,
    criterion: nn.Module,
    device: torch.device
) -> float:
    """Compute average loss across cached batches (no grad)."""
    if not batches:
        return 0.0
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for images, labels in batches:
            images = images.to(device)
            if isinstance(labels, tuple):
                la, lb, lam = labels
                la = la.to(device)
                lb = lb.to(device)
                outputs = model(images)
                loss = lam * criterion(outputs, la) + (1 - lam) * criterion(outputs, lb)
            else:
                labels = labels.to(device)
                if labels.dtype != torch.long:
                    labels = labels.long()
                if DEBUG_VALIDATE_LABELS:
                    min_label = int(labels.min().item())
                    max_label = int(labels.max().item())
                    if min_label < 0 or max_label >= NUM_CLASSES:
                        print(f"[GT] Invalid labels detected: min={min_label}, max={max_label}")
                        continue
                outputs = model(images)
                loss = criterion(outputs, labels)
            total_loss += float(loss.item())
            count += 1
    return total_loss / max(count, 1)

def compute_impact_scores_with_timing(g_s: torch.Tensor, sigma: torch.Tensor, F_full: torch.Tensor):
    """
    Compute different importance scores, and detailed timing
    
    Args:
        g_s: Gradient vector
        sigma: Singular value vector
        F_full: Fisher Information Matrix
    
    Returns:
        scores_dict: Scores for each method
        timing_dict: Computation time for each method
    """
    scores = {}
    timing = {}
    
    # Take absolute value of expectation |E[g]|
    abs_g = torch.abs(g_s)
    
    # 1. Hessian Full (Slowest, O(n^2))
    start = time.perf_counter()
    H_full = F_full.clone()
    diag_H_form = torch.diag(sigma.unsqueeze(0) * H_full * sigma.unsqueeze(1))
    scores['Hessian Full'] = abs_g * sigma + 0.5 * diag_H_form
    timing['Hessian Full'] = time.perf_counter() - start
    
    # 2. Hessian Diag (Fast, O(n))
    start = time.perf_counter()
    scores['Hessian Diag'] = abs_g * sigma + 0.5 * H_full.diag() * (sigma ** 2)
    timing['Hessian Diag'] = time.perf_counter() - start
    
    # 3. Fisher Full (Slow, O(n^2))
    start = time.perf_counter()
    diag_F_form = torch.diag(sigma.unsqueeze(0) * F_full * sigma.unsqueeze(1))
    scores['Fisher Full'] = abs_g * sigma + 0.5 * diag_F_form
    timing['Fisher Full'] = time.perf_counter() - start
    
    # 4. Fisher Diag (Fast, O(n))
    start = time.perf_counter()
    scores['Fisher Diag'] = abs_g + 0.5 * F_full.diag()
    timing['Fisher Diag'] = time.perf_counter() - start
    
    # 5. Taylor 1st-order (Very fast, O(n))
    start = time.perf_counter()
    scores['Taylor 1st-order'] = abs_g.clone()
    timing['Taylor 1st-order'] = time.perf_counter() - start
    
    # 6. Gradient Magnitude (Very fast, O(n))
    start = time.perf_counter()
    scores['Gradient Magnitude'] = abs_g / (sigma + 1e-9)
    timing['Gradient Magnitude'] = time.perf_counter() - start
    
    # 7. Magnitude (Sigma) (Very fast, O(n))
    start = time.perf_counter()
    scores['Magnitude (Sigma)'] = sigma.clone()
    timing['Magnitude (Sigma)'] = time.perf_counter() - start
    
    # 8. Energy (Sigma^2) (Very fast, O(n))
    start = time.perf_counter()
    scores['Energy (Sigma^2)'] = sigma ** 2
    timing['Energy (Sigma^2)'] = time.perf_counter() - start
    
    # Convert to numpy array
    scores_numpy = {k: v.detach().cpu().numpy() for k, v in scores.items()}
    
    return scores_numpy, timing


def compute_true_loss_increase_scores_with_timing(
    model: nn.Module,
    svd_layers: Dict[str, nn.Module],
    loss_batches: List,
    criterion: nn.Module,
    device: torch.device,
    svd_epsilon: float = 1e-12
) -> Dict[str, np.ndarray]:
    """
    Ground truth baseline with detailed timing.
    """
    if not loss_batches:
        return {}

    print("\n[GT] Computing true loss-increase scores (one-component removal)...")
    
    # Calculate baseline loss
    baseline_start = time.perf_counter()
    baseline_loss = _compute_avg_loss_on_batches(model, loss_batches, criterion, device)
    baseline_time = time.perf_counter() - baseline_start
    print(f"[GT] Baseline loss: {baseline_loss:.6f} (computed in {baseline_time:.2f}s)")

    min_log_s = float(np.log(svd_epsilon))
    true_loss_scores = {}
    
    total_components = 0
    total_time = 0
    
    for name, layer in svd_layers.items():
        r = layer.full_rank
        total_components += r
        scores = torch.zeros(r, device=device)
        orig_log_s = layer.log_s.detach().clone()
        
        layer_start = time.perf_counter()
        print(f"[GT] Processing {name} (rank={r})...")
        
        component_times = []
        
        for i in range(r):
            with torch.no_grad():
                layer.log_s.copy_(orig_log_s)
                layer.log_s[i] = min_log_s

            comp_start = time.perf_counter()
            loss_i = _compute_avg_loss_on_batches(model, loss_batches, criterion, device)
            comp_time = time.perf_counter() - comp_start
            component_times.append(comp_time)
            
            scores[i] = loss_i - baseline_loss
            
            # Print progress every 10 singular values or at the last one
            if (i + 1) % 10 == 0 or i == r - 1:
                avg_time = sum(component_times) / len(component_times)
                remaining = (r - i - 1) * avg_time
                print(f"  [{i+1:3d}/{r:3d}] avg={avg_time:.3f}s/comp, "
                      f"remaining={remaining:.1f}s, last={comp_time:.3f}s")

        with torch.no_grad():
            layer.log_s.copy_(orig_log_s)
        
        layer_time = time.perf_counter() - layer_start
        total_time += layer_time
        
        true_loss_scores[name] = scores.detach().cpu().numpy()
        print(f"[GT] Completed {name}: {r} components in {layer_time:.2f}s "
              f"(avg={layer_time/r:.3f}s/component)")
    
    print(f"\n[GT] TOTAL: {total_components} components processed in {total_time:.2f}s")
    print(f"[GT] Average per component: {total_time/total_components:.3f}s")
    
    return true_loss_scores


def run_experiment_detailed_timing(
    config: PruningConfig,
    target_batches_list: List[int],
    output_dir: str,
    loss_eval_batches: int = 1,
    target_layers: List[str] = None
):
    """
    Full timing record experiment version, including method computation time
    """
    os.makedirs(output_dir, exist_ok=True)
    device = config.device
    
    # Initialize total timing records
    total_experiment_start = time.perf_counter()
    
    print("\n" + "="*80)
    print("STARTING DETAILED TIMING EXPERIMENT")
    print("="*80)
    
    # ==================== Phase 1: Model Loading and SVD Transform ====================
    stage_start = time.perf_counter()
    print("\n[Stage 1] Loading ResNet-50 and applying SVD...")
    
    model, _ = setup_resnet_model(config)
    model.eval()
    
    train_loader, _ = get_optimized_data_loaders_r50(config)
    criterion = nn.CrossEntropyLoss()
    
    conv_paths = collect_resnet_conv_layers(model)
    svd_layers = {}
    
    # Process only target_layers if specified
    for path in conv_paths:
        if target_layers and path not in target_layers:
            continue
            
        parent, layer_name = get_resnet_parent_and_name(model, path)
        if parent is None: continue
        original_conv = getattr(parent, layer_name, None)
        if original_conv is None and hasattr(parent, '_modules'):
            original_conv = parent._modules.get(layer_name)
        if not isinstance(original_conv, nn.Conv2d): continue
        
        svd_conv = SimpleSVDConv(original_conv, path, config=config).to(device)
        if hasattr(parent, layer_name): setattr(parent, layer_name, svd_conv)
        elif hasattr(parent, '_modules'): parent._modules[layer_name] = svd_conv
        svd_layers[path] = svd_conv
    
    stage1_time = time.perf_counter() - stage_start
    print(f"[Stage 1] Completed: {len(svd_layers)} SVD layers created in {stage1_time:.2f}s")
    
    # Set gradients
    for param in model.parameters(): param.requires_grad = False
    for layer in svd_layers.values(): layer.log_s.requires_grad = True
    
    # ==================== Phase 2: Ground Truth Calculation ====================
    stage_start = time.perf_counter()
    print("\n[Stage 2] Computing Ground Truth (True Loss Increase)...")
    print("-" * 60)
    
    loss_batches = _collect_batches(train_loader, loss_eval_batches)
    
    # Record Ground Truth computation time in detail
    gt_start = time.perf_counter()
    true_loss_scores = compute_true_loss_increase_scores_with_timing(
        model, svd_layers, loss_batches, criterion, device, svd_epsilon=config.svd_epsilon
    )
    gt_elapsed = time.perf_counter() - gt_start
    
    stage2_time = time.perf_counter() - stage_start
    print(f"\n[Stage 2] Ground Truth completed in {stage2_time:.2f}s")
    
    # ==================== Phase 3: Fisher/Hessian Collection and Correlation ====================
    print("\n[Stage 3] Fisher/Hessian Information Collection and Correlation Analysis")
    print("-" * 60)
    
    all_results = []
    timing_records = []
    
    # Record timing for each batch count in detail
    fisher_collection_times = {}
    correlation_times = {}
    method_computation_times = {}  # Added: record computation time for each method
    
    for num_batches in target_batches_list:
        print(f"\n--- Processing {num_batches} batches ---")
        
        # 3.1 Fisher Information Collection
        fisher_start = time.perf_counter()
        print(f"  [3.1] Collecting Fisher information with {num_batches} batches...")
        
        model.zero_grad()
        accum_g = {name: torch.zeros(layer.full_rank, device=device) for name, layer in svd_layers.items()}
        accum_F_full = {name: torch.zeros((layer.full_rank, layer.full_rank), device=device) for name, layer in svd_layers.items()}
        
        max_batches = min(num_batches, len(train_loader))
        count = 0
        batch_times = []
        
        # Use tqdm to show progress
        with tqdm(total=max_batches, desc=f"    Processing batches", unit="batch") as pbar:
            for i, (images, labels) in enumerate(train_loader):
                if i >= max_batches:
                    break
                
                batch_start = time.perf_counter()
                
                images = images.to(device)
                if isinstance(labels, tuple):
                    la, lb, lam = labels
                    loss = lam * criterion(model(images), la.to(device)) + (1-lam) * criterion(model(images), lb.to(device))
                else:
                    labels = labels.to(device)
                    if labels.dtype != torch.long:
                        labels = labels.long()
                    if DEBUG_VALIDATE_LABELS:
                        min_label = int(labels.min().item())
                        max_label = int(labels.max().item())
                        if min_label < 0 or max_label >= NUM_CLASSES:
                            print(f"      [Warning] Invalid labels at batch {i}: min={min_label}, max={max_label}")
                            continue
                    loss = criterion(model(images), labels)
                    
                model.zero_grad()
                target_params = [layer.log_s for layer in svd_layers.values()]
                grads = torch.autograd.grad(loss, target_params, create_graph=False)
                
                with torch.no_grad():
                    for idx, (name, layer) in enumerate(svd_layers.items()):
                        g_i = grads[idx]
                        if g_i is not None:
                            accum_g[name] += torch.abs(g_i)
                            accum_F_full[name] += torch.outer(g_i, g_i)
                
                count += 1
                batch_time = time.perf_counter() - batch_start
                batch_times.append(batch_time)
                
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({
                    'batch_time': f'{batch_time:.2f}s',
                    'avg_time': f'{sum(batch_times)/len(batch_times):.2f}s'
                })
        
        fisher_time = time.perf_counter() - fisher_start
        fisher_collection_times[num_batches] = {
            'total_time': fisher_time,
            'batches_processed': count,
            'avg_batch_time': sum(batch_times) / len(batch_times) if batch_times else 0,
            'total_batch_time': sum(batch_times)
        }
        
        print(f"  [3.1] Fisher collection completed in {fisher_time:.2f}s")
        print(f"       - Processed {count} batches")
        print(f"       - Average batch time: {fisher_collection_times[num_batches]['avg_batch_time']:.3f}s")
        
        # 3.2 Score calculation and correlation analysis
        corr_start = time.perf_counter()
        print(f"  [3.2] Computing scores and correlations...")
        
        base_method = 'Hessian Full'
        vis_layers = list(svd_layers.keys())[:min(5, len(svd_layers))]
        
        # Record computation time for each method
        method_compute_times = {
            'Hessian Full': [],
            'Hessian Diag': [],
            'Fisher Full': [],
            'Fisher Diag': [],
            'Taylor 1st-order': [],
            'Gradient Magnitude': [],
            'Magnitude (Sigma)': [],
            'Energy (Sigma^2)': []
        }
        
        layer_corr_times = []
        
        for name, layer in svd_layers.items():
            layer_start_time = time.perf_counter()
            
            avg_g = accum_g[name] / count
            F_full = accum_F_full[name] / count
            sigma = layer.get_sigma()
            
            # Compute scores for all methods, recording time for each
            scores_dict, method_timing = compute_impact_scores_with_timing(avg_g, sigma, F_full)
            
            # Record computation time for each method
            for method, mtime in method_timing.items():
                method_compute_times[method].append(mtime)
            
            # Add Ground Truth (if exists)
            if true_loss_scores:
                scores_dict['True Loss Increase'] = true_loss_scores[name]
            
            base_score = scores_dict[base_method]
            
            r = layer.full_rank
            top_k = max(1, int(r * 0.3))
            base_topk = set(np.argsort(base_score)[::-1][:top_k])
            
            # Calculate correlation for each method
            for method, sc in scores_dict.items():
                t_start = time.perf_counter()
                
                tau, _ = kendalltau(sc, base_score)
                sp, _ = spearmanr(sc, base_score)
                sc_topk = set(np.argsort(sc)[::-1][:top_k])
                overlap = len(sc_topk & base_topk) / top_k * 100
                
                t_elapsed = time.perf_counter() - t_start
                
                if np.isnan(tau):
                    tau = 1.0
                if np.isnan(sp):
                    sp = 1.0
                
                all_results.append({
                    'Batches': num_batches,
                    'Layer': name,
                    'Method': method,
                    'Kendall_Tau': tau,
                    'Spearman': sp,
                    'Top30_Overlap_Pct': overlap
                })
                
                timing_records.append({
                    'Batches': num_batches,
                    'Layer': name,
                    'Method': method,
                    'Correlation_Time_Sec': t_elapsed
                })
            
            layer_corr_time = time.perf_counter() - layer_start_time
            layer_corr_times.append(layer_corr_time)
            
            # Print detailed time info for the layer
            print(f"    {name}: rank={r}, score_compute={sum(method_timing.values()):.4f}s, "
                  f"correlation={layer_corr_time:.4f}s")
            
            # Visualize partial layers
            if name in vis_layers:
                layer_vis_dir = os.path.join(output_dir, f'batches_{num_batches}')
                os.makedirs(layer_vis_dir, exist_ok=True)
                
                num_methods = len(scores_dict)
                cols = 2
                rows = (num_methods + 1) // 2
                fig, axes = plt.subplots(rows, cols, figsize=(16, 4*rows))
                axes = axes.flatten()
                
                for idx, (mthd, sc) in enumerate(scores_dict.items()):
                    ax = axes[idx]
                    norm_sc = sc / (np.max(sc) + 1e-12)
                    ax.bar(range(r), norm_sc, color='skyblue')
                    ax.set_title(f"{mthd} (Normalized)")
                    ax.set_xlabel('Singular Value Index')
                    ax.set_ylabel('Impact Score')
                
                for j in range(idx+1, len(axes)):
                    axes[j].axis('off')
                    
                plt.suptitle(f'Impact Scores Comparison - {name} (Batches={num_batches})', fontsize=16)
                plt.tight_layout()
                plt.savefig(os.path.join(layer_vis_dir, f'comparison_{name.replace(".", "_")}.png'))
                plt.close(fig)
        
        corr_time = time.perf_counter() - corr_start
        correlation_times[num_batches] = {
            'total_time': corr_time,
            'avg_layer_time': sum(layer_corr_times) / len(layer_corr_times) if layer_corr_times else 0,
            'layers_processed': len(layer_corr_times)
        }
        
        # Aggregate method computation times
        method_computation_times[num_batches] = {
            method: sum(times) for method, times in method_compute_times.items() if times
        }
        
        print(f"  [3.2] Score computation + correlation completed in {corr_time:.2f}s")
        print(f"       - Layers processed: {len(layer_corr_times)}")
        print(f"       - Average time per layer: {correlation_times[num_batches]['avg_layer_time']:.3f}s")
        
        # Print method computation time summary
        print(f"  [3.2] Method computation time breakdown:")
        for method, total_time in sorted(method_computation_times[num_batches].items(), key=lambda x: x[1], reverse=True):
            avg_time = total_time / len(svd_layers)
            print(f"       - {method:25s}: {total_time:.4f}s total, {avg_time:.4f}s avg/layer")
    
    # ==================== Phase 4: Save Results and Generate Report ====================
    stage4_start = time.perf_counter()
    print("\n[Stage 4] Saving results and generating reports...")
    
    # Save main results
    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(output_dir, 'fisher_methods_comparison.csv'), index=False)
    
    # Save correlation computation times
    if timing_records:
        timing_df = pd.DataFrame(timing_records)
        timing_summary = timing_df.groupby(['Batches', 'Method'])['Correlation_Time_Sec'].sum().reset_index()
        timing_summary.to_csv(os.path.join(output_dir, 'correlation_timing.csv'), index=False)
    
    # Save method computation times
    method_time_records = []
    for num_batches, methods in method_computation_times.items():
        for method, mtime in methods.items():
            method_time_records.append({
                'Batches': num_batches,
                'Method': method,
                'Total_Computation_Time_Sec': mtime,
                'Avg_Computation_Time_Per_Layer_Sec': mtime / len(svd_layers)
            })
    
    method_time_df = pd.DataFrame(method_time_records)
    method_time_df.to_csv(os.path.join(output_dir, 'method_computation_times.csv'), index=False)
    
    # ==================== Generate Complete Time Report ====================
    total_experiment_time = time.perf_counter() - total_experiment_start
    stage4_time = time.perf_counter() - stage4_start
    
    report_path = os.path.join(output_dir, 'COMPLETE_TIMING_REPORT.txt')
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("COMPLETE TIMING ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")
        
        # Experiment Configuration
        f.write("EXPERIMENT CONFIGURATION:\n")
        f.write("-"*40 + "\n")
        f.write(f"Target layers: {len(svd_layers)} layers\n")
        for name in svd_layers.keys():
            f.write(f"  - {name} (rank={svd_layers[name].full_rank})\n")
        f.write(f"Batch sizes tested: {target_batches_list}\n")
        f.write(f"Ground truth evaluation batches: {loss_eval_batches}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Batch size for training: {config.batch_size}\n\n")
        
        # Phase 1: Model Loading
        f.write("STAGE 1: MODEL LOADING AND SVD CONVERSION\n")
        f.write("-"*40 + "\n")
        f.write(f"Time: {stage1_time:.2f} seconds\n")
        f.write(f"SVD layers created: {len(svd_layers)}\n\n")
        
        # Phase 2: Ground Truth
        f.write("STAGE 2: GROUND TRUTH (TRUE LOSS INCREASE) COMPUTATION\n")
        f.write("-"*40 + "\n")
        f.write(f"Total time: {stage2_time:.2f} seconds\n")
        f.write(f"Components processed: {sum(layer.full_rank for layer in svd_layers.values())}\n")
        f.write(f"Batches per component: {loss_eval_batches}\n")
        f.write(f"Average time per component: {stage2_time / sum(layer.full_rank for layer in svd_layers.values()):.3f} seconds\n\n")
        
        # Phase 3: Fisher/Hessian Collection and Method Computation
        f.write("STAGE 3: FISHER/HESSIAN COLLECTION AND METHOD COMPUTATION\n")
        f.write("-"*40 + "\n")
        
        for num_batches in target_batches_list:
            f.write(f"\n--- {num_batches} Batches ---\n")
            
            # Fisher Collection Time
            fisher_info = fisher_collection_times[num_batches]
            f.write(f"  Fisher Collection:\n")
            f.write(f"    Total time: {fisher_info['total_time']:.2f} seconds\n")
            f.write(f"    Batches processed: {fisher_info['batches_processed']}\n")
            f.write(f"    Average batch time: {fisher_info['avg_batch_time']:.3f} seconds\n")
            f.write(f"    Total forward+backward time: {fisher_info['total_batch_time']:.2f} seconds\n\n")
            
            # Method Computation Time (Added!)
            f.write(f"  Method Computation Time (score calculation):\n")
            f.write(f"  {'Method':30s} {'Total Time':12s} {'Avg/Layer':12s} {'vs Fastest':12s}\n")
            f.write(f"  { '-'*70 }\n")
            
            method_times = method_computation_times[num_batches]
            fastest_time = min(method_times.values())
            
            for method, total_time in sorted(method_times.items(), key=lambda x: x[1], reverse=True):
                avg_time = total_time / len(svd_layers)
                ratio = total_time / fastest_time
                f.write(f"  {method:30s} {total_time:8.4f}s   {avg_time:8.4f}s   {ratio:8.1f}x\n")
            
            f.write(f"\n  Correlation Computation Time:\n")
            corr_info = correlation_times[num_batches]
            f.write(f"    Total time: {corr_info['total_time']:.2f} seconds\n")
            f.write(f"    Average time per layer: {corr_info['avg_layer_time']:.3f} seconds\n")
            
            # Detailed Correlation Method Time
            if timing_records:
                method_corr_times = {}
                for record in timing_records:
                    if record['Batches'] == num_batches:
                        method_corr_times[record['Method']] = method_corr_times.get(record['Method'], 0) + record['Correlation_Time_Sec']
                
                if method_corr_times:
                    f.write(f"\n  Correlation Method Breakdown:\n")
                    for method, ctime in sorted(method_corr_times.items(), key=lambda x: x[1], reverse=True):
                        f.write(f"    {method:30s}: {ctime:.4f} seconds\n")
        
        f.write("\n")
        
        # Phase 4: Save Results
        f.write("STAGE 4: RESULT SAVING\n")
        f.write("-"*40 + "\n")
        f.write(f"Time: {stage4_time:.2f} seconds\n\n")
        
        # Total Time
        f.write("="*40 + "\n")
        f.write("TOTAL EXPERIMENT TIME\n")
        f.write("="*40 + "\n")
        f.write(f"Overall experiment time: {total_experiment_time:.2f} seconds ({total_experiment_time/60:.2f} minutes)\n\n")
        
        # Time Allocation Percentage
        f.write("TIME BREAKDOWN BY STAGE:\n")
        f.write("-"*40 + "\n")
        total_fisher_time = sum(info['total_time'] for info in fisher_collection_times.values())
        total_method_compute_time = sum(sum(method_times.values()) for method_times in method_computation_times.values())
        total_corr_time = sum(info['total_time'] for info in correlation_times.values())
        
        stage_times = {
            'Stage 1 (Model Setup)': stage1_time,
            'Stage 2 (Ground Truth)': stage2_time,
            'Stage 3 (Fisher Collection)': total_fisher_time,
            'Stage 3 (Method Computation)': total_method_compute_time,
            'Stage 3 (Correlation)': total_corr_time,
            'Stage 4 (Saving Results)': stage4_time
        }
        
        for stage, stime in stage_times.items():
            percentage = (stime / total_experiment_time) * 100
            f.write(f"{stage:35s}: {stime:7.2f}s ({percentage:5.1f}%)\n")
        
        # Efficiency Analysis
        f.write("\nEFFICIENCY ANALYSIS:\n")
        f.write("-"*40 + "\n")
        total_components = sum(layer.full_rank for layer in svd_layers.values())
        f.write(f"Ground Truth efficiency: {stage2_time / total_components:.3f} seconds per component\n")
        
        if fisher_collection_times:
            avg_batch_time = sum(info['avg_batch_time'] for info in fisher_collection_times.values()) / len(fisher_collection_times)
            f.write(f"Average Fisher collection batch time: {avg_batch_time:.3f} seconds\n")
        
        # Method Computation Efficiency
        f.write(f"\nMethod Computation Efficiency (per layer):\n")
        for num_batches in target_batches_list:
            f.write(f"\n  {num_batches} batches:\n")
            method_times = method_computation_times[num_batches]
            fastest_time = min(method_times.values())
            for method, total_time in sorted(method_times.items(), key=lambda x: x[1], reverse=True):
                avg_time = total_time / len(svd_layers)
                ratio = total_time / fastest_time
                f.write(f"    {method:30s}: {avg_time:.4f}s/layer ({ratio:.1f}x slower than fastest)\n")
        
        # Optimization Suggestions
        f.write("\nOPTIMIZATION SUGGESTIONS:\n")
        f.write("-"*40 + "\n")
        if stage2_time / total_components > 0.5:
            f.write("• Ground Truth is time-consuming. Consider reducing loss_eval_batches or using fewer layers for validation.\n")
        
        # Check for abnormal method computation times
        for num_batches in target_batches_list:
            method_times = method_computation_times[num_batches]
            hessian_full_time = method_times.get('Hessian Full', 0)
            hessian_diag_time = method_times.get('Hessian Diag', 0)
            if hessian_full_time > 0 and hessian_diag_time > 0:
                ratio = hessian_full_time / hessian_diag_time
                if ratio > 10:
                    f.write(f"• Hessian Full is {ratio:.1f}x slower than Hessian Diag. Consider using diagonal approximation for faster computation.\n")
        
        if total_corr_time > 10:
            f.write("• Correlation computation is taking significant time. Consider reducing the number of layers for detailed analysis.\n")
        
        if total_method_compute_time > 0:
            f.write("• Method computation time is now properly recorded. Use this data to compare method efficiency.\n")
    
    print(f"\n[Stage 4] Results saved to '{output_dir}'")
    print(f"[Stage 4] Complete timing report saved to '{report_path}'")
    
    # Print brief time summary to console
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETED - TIME SUMMARY")
    print("="*80)
    print(f"Total experiment time: {total_experiment_time:.2f}s ({total_experiment_time/60:.2f} minutes)")
    print(f"  - Ground Truth: {stage2_time:.2f}s ({stage2_time/total_experiment_time*100:.1f}%)")
    print(f"  - Fisher Collection: {total_fisher_time:.2f}s ({total_fisher_time/total_experiment_time*100:.1f}%)")
    print(f"  - Method Computation: {total_method_compute_time:.2f}s ({total_method_compute_time/total_experiment_time*100:.1f}%)")
    print(f"  - Correlation: {total_corr_time:.2f}s ({total_corr_time/total_experiment_time*100:.1f}%)")
    print("="*80)
    
    # Print method computation time comparison
    print("\nMETHOD COMPUTATION TIME COMPARISON (per layer average):")
    print("-"*60)
    for num_batches in target_batches_list:
        print(f"\n{num_batches} batches:")
        method_times = method_computation_times[num_batches]
        fastest_time = min(method_times.values())
        for method, total_time in sorted(method_times.items(), key=lambda x: x[1], reverse=True):
            avg_time = total_time / len(svd_layers)
            ratio = total_time / fastest_time
            print(f"  {method:25s}: {avg_time:.4f}s/layer ({ratio:.1f}x)")


if __name__ == '__main__':
    config = PruningConfig()
    
    # Add more representative layers for statistical significance
    target_layers = [
        # Stage 1
        'layer1.0.conv1', 'layer1.0.conv2', 'layer1.0.conv3',
        'layer1.1.conv1', 'layer1.1.conv2', 'layer1.1.conv3',
        
        # Stage 2
        'layer2.0.conv1', 'layer2.0.conv2', 'layer2.0.conv3',
        'layer2.1.conv1', 'layer2.1.conv2', 'layer2.1.conv3',
        
        # Stage 3 (Partial)
        'layer3.0.conv1', 'layer3.0.conv2',
        
        # Stage 4 (Partial)
        'layer4.0.conv1', 'layer4.0.conv2',
    ]
    
    target_batches = [10, 50, 100]
    output_directory = './fisher_comparison_results_detailed'
    config.batch_size = 256
    
    loss_eval_batches = 2  # Reduce Ground Truth time
    
    run_experiment_detailed_timing(
        config, target_batches, output_directory, 
        loss_eval_batches=loss_eval_batches,
        target_layers=target_layers
    )
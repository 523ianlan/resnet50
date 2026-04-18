# fix_report_lr.py
import os
import math

def get_correct_lr(epoch, total_epochs, base_lr, min_lr_ratio, freeze_epoch, freeze_lr):
    if epoch >= freeze_epoch:
        return freeze_lr if freeze_lr is not None else 5e-7 # based on sh file
    
    # Official PyTorch CosineAnnealingLR formula
    eta_min = base_lr * min_lr_ratio
    t_cur = epoch - 1 # starts at 0 for epoch 1
    t_max = total_epochs
    
    lr = eta_min + 0.5 * (base_lr - eta_min) * (1 + math.cos(math.pi * t_cur / t_max))
    return lr

def fix_report(report_path):
    with open(report_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    new_lines = []
    base_lr = 1e-4
    total_epochs = 90
    min_lr_ratio = 0.005
    freeze_epoch = 88
    freeze_lr = 5e-7
    
    in_log = False
    for line in lines:
        if "FINE-TUNING LOG" in line:
            in_log = True
        elif in_log and "====================" in line and line.strip() == "====================================================================================================":
             # This might be the end of the log or the start of a separator. 
             # Let's check the context.
             pass
        
        if in_log and "Epoch" in line and "/" in line and "LR=" in line:
            try:
                # Extract epoch
                parts = line.split("|")
                epoch_part = parts[0].strip() # "Epoch   1/90"
                epoch = int(epoch_part.split()[1].split("/")[0])
                
                correct_lr = get_correct_lr(epoch, total_epochs, base_lr, min_lr_ratio, freeze_epoch, freeze_lr)
                
                # Replace LR=... at the end
                lr_start = line.rfind("LR=")
                if lr_start != -1:
                    new_line = line[:lr_start] + f"LR={correct_lr:.2e}\n"
                    new_lines.append(new_line)
                    continue
            except Exception as e:
                print(f"Error processing line: {line.strip()} - {e}")
        
        new_lines.append(line)
        if "ACCURACY RESULTS" in line:
            in_log = False

    with open(report_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    print(f"Fixed LR in {report_path}")

if __name__ == '__main__':
    report_path = r"results\FISHER mean layer importance result\comp60_ft90_low0.2_high0.8_result_20260413_194543\r50_60pr_90ft\complete_pruning_report.txt"
    fix_report(report_path)

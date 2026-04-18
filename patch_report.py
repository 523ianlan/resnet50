# patch_report.py
import os
import json

def patch_report(report_path, history_json_path):
    with open(report_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    with open(history_json_path, 'r') as f:
        history = json.load(f)
    
    # Generate new log lines
    new_log_lines = []
    new_log_lines.append("====================================================================================================\n")
    new_log_lines.append("FINE-TUNING LOG (RECOVERED VALIDATION DATA)\n")
    new_log_lines.append("====================================================================================================\n")
    
    for h in history:
        # We use N/A for Train metrics as they were lost
        line = f"Epoch {h['epoch']:3d}/90 | Train: Loss=N/A, Top1=N/A, Top5=N/A | Val: Loss={h['val_loss']:.4f}, Top1={h['val_top1']:.2f}%, Top5={h['val_top5']:.2f}% | LR={h['lr']:.2e}\n"
        new_log_lines.append(line)
    new_log_lines.append("\n")

    # Find the section to replace
    start_idx = -1
    end_idx = -1
    for i, line in enumerate(lines):
        if "FINE-TUNING LOG" in line:
            start_idx = i - 1
            # Look for the next section start
            for j in range(i + 1, len(lines)):
                if "====================" in lines[j] and j > i + 5:
                    end_idx = j
                    break
            break
    
    if start_idx != -1 and end_idx != -1:
        updated_lines = lines[:start_idx] + new_log_lines + lines[end_idx:]
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.writelines(updated_lines)
        print(f"Successfully patched {report_path}")
    else:
        print("Could not find FINE-TUNING LOG section in report.")

if __name__ == '__main__':
    report_file = r"results\FISHER mean layer importance result\comp60_ft90_low0.2_high0.8_result_20260413_194543\r50_60pr_90ft\complete_pruning_report.txt"
    history_file = "full_history_val_only.json"
    if os.path.exists(history_file):
        patch_report(report_file, history_file)
    else:
        print("History JSON not found.")

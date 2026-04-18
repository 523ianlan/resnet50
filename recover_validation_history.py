# recover_validation_history.py
import os
import json
import torch
import torch.nn as nn
from torchvision.models import resnet50
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import argparse
import math

# Use the same logic as your standalone r50_val.py
def get_resnet_parent_and_name(model, layer_path):
    parts = layer_path.split('.')
    parent = model
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
    return parent, parts[-1]

def rebuild_pruned_resnet(model, structure_info, device):
    replaced_count = 0
    import ast
    for layer_path, info in structure_info.items():
        if isinstance(info, str):
            try: info = ast.literal_eval(info)
            except: continue
        if not isinstance(info, dict): continue
        if info.get('type') == 'decomposed_conv' or info.get('decomposed', False):
            parent, layer_name = get_resnet_parent_and_name(model, layer_path)
            c1_info, c2_info = info['conv1'], info['conv2']
            conv1 = nn.Conv2d(c1_info['in_channels'], c1_info['out_channels'], kernel_size=c1_info['kernel_size'],
                              stride=c1_info.get('stride', (1,1)), padding=c1_info.get('padding', (0,0)), bias=c1_info.get('bias', False)).to(device)
            conv2 = nn.Conv2d(c2_info['in_channels'], c2_info['out_channels'], kernel_size=c2_info['kernel_size'],
                              stride=c2_info.get('stride', (1,1)), padding=c2_info.get('padding', (0,0)), bias=c2_info.get('bias', False)).to(device)
            seq = nn.Sequential(conv1, conv2)
            setattr(parent, layer_name, seq)
            replaced_count += 1
    return model

def evaluate(model, loader, device):
    model.eval()
    top1_correct, top5_correct, total_loss, total = 0, 0, 0.0, 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        with torch.amp.autocast("cuda"):
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item() * labels.size(0)
                _, top1_pred = torch.max(outputs, 1)
                top1_correct += (top1_pred == labels).sum().item()
                _, top5_pred = outputs.topk(5, 1, True, True)
                top5_correct += (top5_pred == labels.view(-1, 1)).any(dim=1).sum().item()
                total += labels.size(0)
    return total_loss / total, 100.0 * top1_correct / total, 100.0 * top5_correct / total

def get_lr_at_epoch(epoch, total_epochs, base_lr, freeze_epoch, freeze_lr, min_lr_ratio):
    if epoch >= freeze_epoch:
        return freeze_lr if freeze_lr is not None else base_lr * 0.01 # logic for fixed lr
    # Simplified Cosine logic
    return 0.5 * base_lr * (1 + math.cos(math.pi * (epoch-1) / total_epochs))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-dir', type=str, required=True, help='Path to r50_60pr_90ft folder')
    parser.add_argument('--val-root', type=str, required=True, help='Path to ImageNet validation')
    parser.add_argument('--batch-size', type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load config and structure
    config_path = os.path.join(args.exp_dir, 'r50_60pr_before_finetune_config.json')
    with open(config_path, 'r') as f:
        config_data = json.load(f)
        structure_info = config_data['model_structure']
    
    # Initialize Model
    model = resnet50(weights=None)
    model = rebuild_pruned_resnet(model, structure_info, device)
    model.to(device)

    # Data
    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_dataset = ImageFolder(args.val_root, transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    history = []
    checkpoint_dir = os.path.join(args.exp_dir, 'finetunemodel')
    
    # Params for LR
    base_lr = 1e-4
    freeze_epoch = 88
    total_epochs = 90

    for epoch in range(1, total_epochs + 1):
        ckpt_path = os.path.join(checkpoint_dir, f'epoch{epoch}.pth')
        if not os.path.exists(ckpt_path):
            print(f"Skipping epoch {epoch}, checkpoint not found.")
            continue
        
        print(f"Propcessing Epoch {epoch}/90...")
        state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
        if 'model_state_dict' in state_dict: state_dict = state_dict['model_state_dict']
        model.load_state_dict(state_dict, strict=False)
        
        val_loss, val_top1, val_top5 = evaluate(model, val_loader, device)
        lr = get_lr_at_epoch(epoch, total_epochs, base_lr, freeze_epoch, None, 0.005)
        
        entry = {
            "epoch": epoch,
            "val_loss": val_loss,
            "val_top1": val_top1,
            "val_top5": val_top5,
            "lr": lr
        }
        history.append(entry)
        
        # Intermediate save
        with open('recovered_history_temp.json', 'w') as f:
            json.dump(history, f)

    print("\nRecovery Complete. Summary:")
    for h in history:
        print(f"Epoch {h['epoch']:2d}: Val Top1={h['val_top1']:.2f}%, LR={h['lr']:.2e}")

    with open('full_history_val_only.json', 'w') as f:
        json.dump(history, f)

if __name__ == '__main__':
    main()

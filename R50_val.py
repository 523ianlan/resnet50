# Validation ResNet-50 (pruned / fine-tuned)
#
# Example:
# python R50_val.py \
#     --val-root "D:\ImageNet_Organized\validation" \
#     --model-path "D:\UFALP\resnet50\scripts\results\comp10_ft1_low0.0_high0.3_result_20260330_170356\r50_10pr_1ft\finetunemodel\best_model.pth" \
#     --structure-pth "D:\UFALP\resnet50\scripts\results\comp10_ft1_low0.0_high0.3_result_20260330_170356\r50_10pr_1ft\r50_10pr_1ft.pth"
#
# Or directly use the full artifact (.pth with model_structure):
# python R50_val.py \
#     --val-root "D:\ImageNet_Organized\validation" \
#     --model-path "D:\UFALP\resnet50\scripts\results\comp70_ft90_low0.1_high0.9_result_20260330_234641\r50_70pr_90ft\finetunemodel\best_model.pth"

# python R50_val.py `
#   --val-root "D:\ImageNet_Organized\validation" `
#   --model-path "D:\UFALP\resnet50\scripts\results\comp70_ft90_low0.1_high0.9_result_20260330_234641\r50_70pr_90ft\finetunemodel\best_model.pth"


import os
import sys
import json
import argparse
import contextlib
from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn
from torchvision.models import resnet50
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure local imports work when running from elsewhere
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.transforms import get_resnet_val_transform
from models.utils import get_resnet_parent_and_name

try:
    from fvcore.nn import FlopCountAnalysis
    HAS_FVCORE = True
except Exception:
    HAS_FVCORE = False


def _to_tuple(value, default: Tuple[int, int]) -> Tuple[int, int]:
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        return tuple(int(x) for x in value)
    return (int(value), int(value))


def load_json_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    keys = list(state_dict.keys())
    if keys and all(k.startswith("module.") for k in keys):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def load_model_artifact(path: str) -> Tuple[Dict[str, torch.Tensor], Optional[Dict], Dict]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = None
    structure_info = None
    meta: Dict = {}

    if isinstance(obj, dict):
        if "model_state_dict" in obj:
            state_dict = obj["model_state_dict"]
        elif "state_dict" in obj:
            state_dict = obj["state_dict"]
        elif all(isinstance(v, torch.Tensor) for v in obj.values()):
            state_dict = obj

        if "model_structure" in obj:
            structure_info = obj["model_structure"]

        for key in ["original_accuracy", "pruned_accuracy", "test_accuracy", "config"]:
            if key in obj:
                meta[key] = obj[key]
    else:
        state_dict = obj

    if state_dict is None:
        raise ValueError(f"No state_dict found in {path}")

    state_dict = strip_module_prefix(state_dict)
    return state_dict, structure_info, meta


def load_structure_from_path(path: Optional[str]) -> Optional[Dict]:
    if not path:
        return None

    if path.lower().endswith(".pth"):
        obj = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(obj, dict) and "model_structure" in obj:
            return obj["model_structure"]
        return None

    # Assume JSON
    data = load_json_config(path)
    if isinstance(data, dict) and "model_structure" in data:
        return data["model_structure"]
    return None


def _has_model_structure(path: str) -> bool:
    if not path.lower().endswith(".pth"):
        return False
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
        return isinstance(obj, dict) and "model_structure" in obj
    except Exception:
        return False


def auto_find_structure_path(model_path: str, config: Optional[Dict]) -> Optional[str]:
    search_dirs: List[str] = []
    model_dir = os.path.abspath(os.path.dirname(model_path))

    # Search near the model path first
    cur = model_dir
    for _ in range(3):
        if cur not in search_dirs:
            search_dirs.append(cur)
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent

    # Search config save_dir if provided
    if config and isinstance(config, dict):
        save_dir = config.get("save_dir")
        if save_dir:
            save_dir = os.path.abspath(os.path.join(ROOT, save_dir))
            if save_dir not in search_dirs:
                search_dirs.append(save_dir)

    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            if not name.lower().endswith(".pth"):
                continue
            cand = os.path.abspath(os.path.join(d, name))
            if os.path.abspath(model_path) == cand:
                continue
            if _has_model_structure(cand):
                return cand
    return None


def _make_conv_from_info(info: Dict, default_stride=(1, 1), default_padding=(0, 0), default_bias=False) -> nn.Conv2d:
    kernel_size = _to_tuple(info.get("kernel_size"), (1, 1))
    stride = _to_tuple(info.get("stride"), default_stride)
    padding = _to_tuple(info.get("padding"), default_padding)
    bias = bool(info.get("bias", default_bias))
    return nn.Conv2d(
        int(info["in_channels"]),
        int(info["out_channels"]),
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=bias,
    )


def rebuild_pruned_resnet_from_structure(model: nn.Module, structure_info: Dict) -> int:
    replaced = 0
    for layer_path, info in structure_info.items():
        if not isinstance(info, dict):
            continue
        if not info.get("decomposed", False):
            continue
        if info.get("type") != "decomposed_conv":
            continue

        parent, layer_name = get_resnet_parent_and_name(model, layer_path)
        if parent is None:
            continue

        conv1 = _make_conv_from_info(info["conv1"], default_bias=False)
        conv2 = _make_conv_from_info(info["conv2"], default_bias=False)
        seq = nn.Sequential(conv1, conv2)

        if hasattr(parent, layer_name):
            setattr(parent, layer_name, seq)
        elif hasattr(parent, "_modules") and layer_name in parent._modules:
            parent._modules[layer_name] = seq
        elif isinstance(parent, nn.Sequential) and layer_name.isdigit():
            parent[int(layer_name)] = seq

        replaced += 1
    return replaced


def rebuild_pruned_resnet_from_state_dict(model: nn.Module, state_dict: Dict[str, torch.Tensor]) -> int:
    replaced = 0
    modules = list(model.named_modules())

    for name, module in modules:
        if not isinstance(module, nn.Conv2d):
            continue

        key0 = f"{name}.0.weight"
        key1 = f"{name}.1.weight"
        if key0 not in state_dict or key1 not in state_dict:
            continue

        w1 = state_dict[key0]
        w2 = state_dict[key1]

        conv1_info = {
            "in_channels": w1.shape[1],
            "out_channels": w1.shape[0],
            "kernel_size": (w1.shape[2], w1.shape[3]),
            "stride": module.stride,
            "padding": module.padding,
            "bias": False,
        }
        conv2_info = {
            "in_channels": w2.shape[1],
            "out_channels": w2.shape[0],
            "kernel_size": (w2.shape[2], w2.shape[3]),
            "stride": (1, 1),
            "padding": (0, 0),
            "bias": f"{name}.1.bias" in state_dict,
        }

        parent, layer_name = get_resnet_parent_and_name(model, name)
        if parent is None:
            continue

        conv1 = _make_conv_from_info(conv1_info, default_bias=False)
        conv2 = _make_conv_from_info(conv2_info, default_bias=False)
        seq = nn.Sequential(conv1, conv2)

        if hasattr(parent, layer_name):
            setattr(parent, layer_name, seq)
        elif hasattr(parent, "_modules") and layer_name in parent._modules:
            parent._modules[layer_name] = seq
        elif isinstance(parent, nn.Sequential) and layer_name.isdigit():
            parent[int(layer_name)] = seq

        replaced += 1

    return replaced


def compute_flops_hooks(model: nn.Module, device: torch.device, channels_last: bool) -> Optional[float]:
    model.eval()
    dummy = torch.randn(1, 3, 224, 224, device=device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
        dummy = dummy.to(memory_format=torch.channels_last)

    macs = 0
    handles = []

    def conv_hook(module: nn.Conv2d, inputs, output):
        nonlocal macs
        x = inputs[0]
        out = output
        out_elements = out.numel()
        kernel_ops = module.kernel_size[0] * module.kernel_size[1] * (module.in_channels / module.groups)
        macs += out_elements * kernel_ops

    def linear_hook(module: nn.Linear, inputs, output):
        nonlocal macs
        x = inputs[0]
        batch = x.shape[0] if x.dim() > 1 else 1
        macs += batch * module.in_features * module.out_features

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            handles.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            handles.append(m.register_forward_hook(linear_hook))

    try:
        with torch.no_grad():
            model(dummy)
    finally:
        for h in handles:
            h.remove()

    flops = 2.0 * macs
    return float(flops)


def compute_flops_fvcore(model: nn.Module, device: torch.device, channels_last: bool) -> Optional[float]:
    if not HAS_FVCORE:
        return None
    model.eval()
    dummy = torch.randn(1, 3, 224, 224, device=device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
        dummy = dummy.to(memory_format=torch.channels_last)
    try:
        flops = FlopCountAnalysis(model, dummy).total()
        return float(flops)
    except Exception as exc:
        print(f"FLOPs calculation failed: {exc}")
        return None


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
    eval_max_batches: int,
    channels_last: bool,
) -> Tuple[float, float]:
    model.eval()
    top1_correct = 0
    top5_correct = 0
    total = 0

    use_amp = amp and device.type == "cuda"
    autocast_ctx = torch.cuda.amp.autocast if use_amp else contextlib.nullcontext

    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(loader, desc="Testing Model")):
            if eval_max_batches > 0 and i >= eval_max_batches:
                break

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if channels_last:
                images = images.to(memory_format=torch.channels_last)

            if labels.dim() > 1 and labels.size(1) == 1000:
                labels = torch.argmax(labels, dim=1)

            with autocast_ctx():
                outputs = model(images)

            _, top1_pred = torch.max(outputs, 1)
            top1_correct += (top1_pred == labels).sum().item()

            _, top5_pred = outputs.topk(5, 1, True, True)
            top5_correct += (top5_pred == labels.view(-1, 1)).any(dim=1).sum().item()

            total += labels.size(0)

    top1_acc = 100.0 * top1_correct / total if total > 0 else 0.0
    top5_acc = 100.0 * top5_correct / total if total > 0 else 0.0
    return top1_acc, top5_acc


def main():
    parser = argparse.ArgumentParser(description="Validate pruned ResNet-50 model")
    parser.add_argument("--model-path", type=str, required=True, help="Path to pruned/finetuned .pth")
    parser.add_argument("--val-root", type=str, default=None, help="Path to ImageNet validation folder")
    parser.add_argument("--structure-pth", type=str, default=None, help="Optional .pth with model_structure")
    parser.add_argument("--structure-config", type=str, default=None, help="Optional JSON with model_structure")
    parser.add_argument("--config", type=str, default=None, help="Optional config.json from results")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--eval-max-batches", type=int, default=0)
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu")
    parser.add_argument("--no-amp", action="store_true", help="Disable AMP")
    parser.add_argument("--no-flops", action="store_true", help="Skip FLOPs calculation")
    parser.add_argument("--flops-method", type=str, default="hooks", choices=["hooks", "fvcore"],
                        help="FLOPs method: hooks (conv/linear only) or fvcore")
    parser.add_argument("--no-compare-original", action="store_true", help="Skip original model comparison")
    args = parser.parse_args()

    config = load_json_config(args.config) if args.config else None

    val_root = args.val_root
    if not val_root and config and "val_root" in config:
        val_root = config["val_root"]

    if not val_root:
        raise ValueError("--val-root is required (or provide --config with val_root)")

    batch_size = args.batch_size
    if batch_size is None and config and "batch_size" in config:
        batch_size = int(config["batch_size"])
    if batch_size is None:
        batch_size = 128

    num_workers = args.num_workers
    if num_workers is None and config and "num_workers" in config:
        num_workers = int(config["num_workers"])
    if num_workers is None:
        num_workers = 8

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    # Load weights and optional structure
    state_dict, structure_info, meta = load_model_artifact(args.model_path)

    if structure_info is None and args.structure_pth:
        structure_info = load_structure_from_path(args.structure_pth)

    if structure_info is None and args.structure_config:
        structure_info = load_structure_from_path(args.structure_config)

    auto_structure_path = None
    if structure_info is None:
        auto_structure_path = auto_find_structure_path(args.model_path, config)
        if auto_structure_path:
            structure_info = load_structure_from_path(auto_structure_path)

    # Build base model
    model = resnet50(weights=None)

    # Rebuild pruned architecture
    replaced_count = 0
    rebuild_source = "none"
    if structure_info and isinstance(structure_info, dict):
        replaced_count = rebuild_pruned_resnet_from_structure(model, structure_info)
        if replaced_count > 0:
            rebuild_source = "structure_info"

    if replaced_count == 0:
        replaced_count = rebuild_pruned_resnet_from_state_dict(model, state_dict)
        if replaced_count > 0:
            rebuild_source = "state_dict"

    # Load weights
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    # Move to device after loading weights
    model = model.to(device)

    channels_last = False
    if config and isinstance(config, dict):
        channels_last = bool(config.get("channels_last", False))
    if channels_last:
        model = model.to(memory_format=torch.channels_last)

    # Build validation transform
    if config and isinstance(config, dict):
        val_aug = config.get("augmentation", {}).get("val", {})
    else:
        val_aug = {"resize": 256, "center_crop": 224}

    val_transform = get_resnet_val_transform(val_aug)
    val_dataset = ImageFolder(val_root, val_transform)

    pin_memory = (device.type == "cuda")
    if config and "pin_memory" in config:
        pin_memory = bool(config["pin_memory"])

    persistent_workers = num_workers > 0
    if config and "persistent_workers" in config:
        persistent_workers = bool(config["persistent_workers"])
    if num_workers == 0:
        persistent_workers = False

    prefetch_factor = 2
    if config and "prefetch_factor" in config:
        prefetch_factor = int(config["prefetch_factor"])

    loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor

    val_loader = DataLoader(val_dataset, **loader_kwargs)

    # Evaluate
    print("\nRunning Validation...")
    top1, top5 = evaluate_model(
        model,
        val_loader,
        device,
        amp=not args.no_amp,
        eval_max_batches=args.eval_max_batches,
        channels_last=channels_last,
    )

    # Params / FLOPs
    params = sum(p.numel() for p in model.parameters())
    compute_flops_flag = (not args.no_flops)
    flops = None
    flops_method = "none"
    if compute_flops_flag:
        if args.flops_method == "fvcore":
            flops = compute_flops_fvcore(model, device, channels_last)
            flops_method = "fvcore"
        else:
            flops = compute_flops_hooks(model, device, channels_last)
            flops_method = "hooks"

    # Original model comparison
    orig_params = None
    orig_flops = None
    if not args.no_compare_original:
        base_model = resnet50(weights=None).to(device)
        if channels_last:
            base_model = base_model.to(memory_format=torch.channels_last)
        orig_params = sum(p.numel() for p in base_model.parameters())
        if compute_flops_flag:
            if args.flops_method == "fvcore":
                orig_flops = compute_flops_fvcore(base_model, device, channels_last)
            else:
                orig_flops = compute_flops_hooks(base_model, device, channels_last)

    # Accuracy deltas if available
    orig_acc = meta.get("original_accuracy") if isinstance(meta, dict) else None
    orig_top1 = orig_acc.get("top1") if isinstance(orig_acc, dict) else None
    orig_top5 = orig_acc.get("top5") if isinstance(orig_acc, dict) else None

    # Report
    print("\n" + "=" * 60)
    print("FINAL MODEL EVALUATION REPORT")
    print("=" * 60)
    print(f"Model Path        : {args.model_path}")
    print(f"Val Root          : {val_root}")
    print(f"Total Images      : {len(val_dataset)}")
    print(f"Batch Size        : {batch_size}")
    print(f"Num Workers       : {num_workers}")
    print(f"Channels Last     : {channels_last}")
    print(f"Rebuilt Layers    : {replaced_count} (source: {rebuild_source})")
    if auto_structure_path:
        print(f"Structure PTH     : {auto_structure_path}")

    if missing or unexpected:
        print("\nState Dict Warnings:")
        if missing:
            print(f"  Missing keys    : {len(missing)}")
            print(f"  Sample missing  : {missing[:8]}")
        if unexpected:
            print(f"  Unexpected keys : {len(unexpected)}")
            print(f"  Sample unexpected: {unexpected[:8]}")

    params_m = params / 1e6
    print(f"Parameters        : {params_m:.2f} M")
    if orig_params:
        reduction = 100.0 * (1.0 - params / orig_params)
        print(f"Param Reduction   : {reduction:.2f}% vs base({orig_params / 1e6:.2f} M)")

    if flops is not None:
        print(f"FLOPs             : {flops / 1e9:.2f} G (method: {flops_method})")
        if orig_flops:
            flops_red = 100.0 * (1.0 - flops / orig_flops)
            print(f"FLOPs Reduction   : {flops_red:.2f}% vs base({orig_flops / 1e9:.2f} G)")
    else:
        print("FLOPs             : N/A (disabled)")
    if flops_method == "hooks":
        print("FLOPs Definition  : conv/linear MACs x2 (excludes pool/add/BN/ReLU)")
    elif flops_method == "fvcore":
        print("FLOPs Definition  : fvcore FlopCountAnalysis (may skip unsupported ops)")

    if orig_top1 is not None:
        print(f"Top-1 Acc         : {top1:.2f}% ({top1 - orig_top1:+.2f}%)")
    else:
        print(f"Top-1 Acc         : {top1:.2f}%")

    if orig_top5 is not None:
        print(f"Top-5 Acc         : {top5:.2f}% ({top5 - orig_top5:+.2f}%)")
    else:
        print(f"Top-5 Acc         : {top5:.2f}%")

    print("=" * 60)


if __name__ == "__main__":
    main()

"""Built-in model registry and setup helpers."""

import torch
import torch.nn as nn
from torchvision.models import (
    resnet18,
    resnet34,
    resnet50,
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
)
from typing import Tuple, Dict, Any, List

from utils.task_utils import get_output_dim


class FlattenMLP(nn.Module):
    """Simple MLP that flattens image inputs internally."""

    def __init__(
        self,
        input_channels: int,
        input_height: int,
        input_width: int,
        hidden_dims: List[int],
        num_classes: int,
    ):
        super().__init__()
        input_dim = input_channels * input_height * input_width
        dims = [input_dim] + list(hidden_dims) + [num_classes]
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, 1)
        return self.net(x)


class SimpleCNN(nn.Module):
    """A lightweight CNN for end-to-end pruning smoke tests."""

    def __init__(
        self,
        input_channels: int,
        input_height: int,
        input_width: int,
        channels: List[int],
        num_classes: int,
    ):
        super().__init__()
        conv_layers = []
        in_ch = input_channels
        for out_ch in channels:
            conv_layers.extend(
                [
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                ]
            )
            in_ch = out_ch
        self.features = nn.Sequential(*conv_layers)

        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, input_height, input_width)
            feature_dim = self.features(dummy).flatten(1).shape[1]

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, max(256, channels[-1])),
            nn.ReLU(inplace=True),
            nn.Linear(max(256, channels[-1]), num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def _resnet_weights_and_builder(model_name: str, pretrained: bool):
    if model_name == "resnet18":
        return resnet18, (ResNet18_Weights.DEFAULT if pretrained else None)
    if model_name == "resnet34":
        return resnet34, (ResNet34_Weights.DEFAULT if pretrained else None)
    return resnet50, (ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)


def get_model_family(model_name: str) -> str:
    normalized = model_name.strip().lower()
    if normalized.startswith("resnet") or normalized == "simple_cnn":
        return "cnn"
    if normalized.startswith("mlp"):
        return "mlp"
    raise ValueError(f"Unsupported built-in model name: {model_name}")


def setup_builtin_model(
    config,
    model_name: str = "resnet50",
    pretrained: bool = True,
) -> Tuple[nn.Module, Dict[str, Any], str]:
    """Load one of the built-in models and return model + metadata."""
    normalized = model_name.strip().lower()
    family = get_model_family(normalized)
    output_dim = get_output_dim(config)

    if normalized in {"resnet18", "resnet34", "resnet50"}:
        builder, weights = _resnet_weights_and_builder(normalized, pretrained)
        model = builder(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, output_dim)
        print(
            f"Loaded {normalized} "
            f"{'with pretrained weights' if pretrained else 'without pretraining'} "
            f"(output_dim={output_dim})"
        )
    elif normalized == "simple_cnn":
        model = SimpleCNN(
            input_channels=int(getattr(config, "input_channels", 3)),
            input_height=int(getattr(config, "input_height", 224)),
            input_width=int(getattr(config, "input_width", 224)),
            channels=list(getattr(config, "cnn_channels", [32, 64, 128])),
            num_classes=output_dim,
        )
        print("Loaded built-in simple_cnn")
    elif normalized == "mlp_small":
        model = FlattenMLP(
            input_channels=int(getattr(config, "input_channels", 3)),
            input_height=int(getattr(config, "input_height", 224)),
            input_width=int(getattr(config, "input_width", 224)),
            hidden_dims=[512, 256],
            num_classes=output_dim,
        )
        print("Loaded built-in mlp_small")
    elif normalized == "mlp_medium":
        model = FlattenMLP(
            input_channels=int(getattr(config, "input_channels", 3)),
            input_height=int(getattr(config, "input_height", 224)),
            input_width=int(getattr(config, "input_width", 224)),
            hidden_dims=list(getattr(config, "mlp_hidden_dims", [1024, 512])),
            num_classes=output_dim,
        )
        print("Loaded built-in mlp_medium")
    else:
        raise ValueError(f"Unsupported built-in model name: {model_name}")

    model = model.to(config.device)
    model.eval()

    data_config = {
        "input_size": (
            int(getattr(config, "input_channels", 3)),
            int(getattr(config, "input_height", 224)),
            int(getattr(config, "input_width", 224)),
        ),
        "interpolation": "bilinear",
        "crop_pct": 0.875,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "model_name": normalized,
        "model_family": family,
    }
    return model, data_config, family


def setup_resnet_model(
    config,
    pretrained: bool = True,
    weights_version: str = "IMAGENET1K_V1"
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Backward-compatible wrapper.

    It now respects config.model_name and config.pretrained, but keeps the old
    function name so existing code keeps working.
    """
    model_name = getattr(config, "model_name", "resnet50")
    use_pretrained = getattr(config, "pretrained", pretrained)
    model, data_config, _ = setup_builtin_model(
        config=config,
        model_name=model_name,
        pretrained=use_pretrained,
    )
    return model, data_config


def get_resnet_model(config) -> nn.Module:
    """Simplified model loading function."""
    model, _ = setup_resnet_model(config)
    return model

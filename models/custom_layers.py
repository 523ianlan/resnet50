"""Prunable custom layers for convolutional and linear modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class LowRankConv(nn.Module):
    """
    Low-rank convolutional layer - composed of two conv layers
    Corresponds to ViT LowRankLinear
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int],
        rank: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
        device=None,
        dtype=None
    ):
        super().__init__()
        
        # First conv: spatial conv, reduce to rank
        self.conv1 = nn.Conv2d(
            in_channels, rank,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )
        
        # 第二個卷積：1x1 卷積，expand to out_channels
        self.conv2 = nn.Conv2d(
            rank, out_channels,
            kernel_size=1,
            bias=bias
        )
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rank = rank
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        return x
    
    @property
    def effective_weight(self):
        """Get combined effective weights (for analysis)"""
        # conv1.weight: [rank, in_channels, kh, kw]
        # conv2.weight: [out_channels, rank, 1, 1]
        w1 = self.conv1.weight  # [rank, in_channels, kh, kw]
        w2 = self.conv2.weight  # [out_channels, rank, 1, 1]
        
        # Reshape for matrix multiplication
        w1_flat = w1.view(self.rank, -1)  # [rank, in_channels*kh*kw]
        w2_flat = w2.view(self.out_channels, self.rank)  # [out_channels, rank]
        
        # Combine
        w_combined_flat = w2_flat @ w1_flat  # [out_channels, in_channels*kh*kw]
        
        # Reshape back to conv weight shape
        kh, kw = self.kernel_size
        w_combined = w_combined_flat.view(
            self.out_channels, self.in_channels, kh, kw
        )
        
        return w_combined


class LowRankLinear(nn.Module):
    """Low-rank linear layer composed of two Linear layers."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        bias: bool = True,
    ):
        super().__init__()
        self.linear1 = nn.Linear(in_features, rank, bias=False)
        self.linear2 = nn.Linear(rank, out_features, bias=bias)
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.linear1(x))


class _BaseSVDLayer(nn.Module):
    """Shared utilities for SVD-based prunable layers."""

    layer_kind = "generic"
    is_bottleneck = False

    def reset_fisher_accum(self):
        self.fisher_accum.zero_()
        self.fisher_samples = 0

    def update_fisher_accum(self, gradients: torch.Tensor):
        with torch.no_grad():
            self.fisher_accum += gradients ** 2
            self.fisher_samples += 1

    def get_fisher_diagonal(self) -> torch.Tensor:
        if self.fisher_samples > 0:
            return self.fisher_accum / self.fisher_samples
        return torch.zeros_like(self.fisher_accum)

    def get_sigma(self) -> torch.Tensor:
        if self.use_log_s:
            return torch.exp(self.log_s)
        return torch.clamp(self.sigma, min=self.config.svd_epsilon)

    def get_score_param(self) -> torch.Tensor:
        return self.log_s if self.use_log_s else self.sigma

    def get_keep_count(self, keep_ratio: float) -> int:
        keep_num = max(self.config.min_rank, int(self.full_rank * keep_ratio))
        return min(keep_num, self.full_rank)

    def get_uvs_for_ratio(
        self,
        keep_ratio: float,
        impact_scores: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        keep_num = self.get_keep_count(keep_ratio)

        if impact_scores is not None:
            if isinstance(impact_scores, np.ndarray):
                impact_scores = torch.from_numpy(impact_scores).to(self.get_sigma().device)

            min_len = min(len(impact_scores), self.full_rank)
            if min_len < self.full_rank:
                impact_scores = impact_scores[:min_len]

            _, top_indices = torch.topk(impact_scores, min(keep_num, min_len))
            top_indices, _ = torch.sort(top_indices)
        else:
            sigma = self.get_sigma()
            _, top_indices = torch.topk(sigma, keep_num)
            top_indices, _ = torch.sort(top_indices)

        return self.U[:, top_indices], self.get_sigma()[top_indices], self.Vh[top_indices, :]

    def get_reconstruction_error(self, keep_ratio: float) -> float:
        reconstructed = self.get_weight_for_ratio(keep_ratio)
        error = torch.norm(reconstructed - self.original_weight) / torch.norm(self.original_weight)
        return float(error.item())


class SimpleSVDConv(_BaseSVDLayer):
    """
    Learnable SVD Conv layer - Corresponds to ViT SimpleSVDLinear
    
    Decompose conv weights into U, S, V and use log-singular variables
    """
    
    def __init__(
        self,
        original_conv: nn.Conv2d,
        layer_name: str,
        config=None
    ):
        super().__init__()
        if config is None:
            from configs.config import PruningConfig
            config = PruningConfig()
        
        # Get original conv layer info
        w = original_conv.weight.data
        self.cout, self.cin, self.kh, self.kw = w.shape
        
        self.layer_name = layer_name
        self.is_bottleneck = (self.kh == 1 and self.kw == 1) and ('downsample' not in layer_name)
        
        # Flatten weights and perform SVD
        W_flat = w.view(self.cout, -1)  # [cout, cin*kh*kw]
        U, S, Vh = torch.linalg.svd(W_flat, full_matrices=False)
        
        # Save original info
        self.stride = original_conv.stride
        self.padding = original_conv.padding
        self.bias = original_conv.bias
        
        # Learnable singular values (log or direct)
        self.full_rank = len(S)
        self.use_log_s = bool(getattr(config, "use_log_s", True))
        if self.use_log_s:
            self.log_s = nn.Parameter(torch.log(S + config.svd_epsilon))
            self.sigma = None
        else:
            self.sigma = nn.Parameter(S.clone())
            self.log_s = None
        
        # Fixed singular vectors (as reference coordinate system)
        self.register_buffer('U', U[:, :self.full_rank])
        self.register_buffer('Vh', Vh[:self.full_rank, :])
        self.register_buffer('original_weight', w.clone())
        
        # Fisher information accumulator
        self.register_buffer('fisher_accum', torch.zeros(self.full_rank))
        self.fisher_samples = 0

        self.config = config
        self.layer_kind = "conv"
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: reconstruct weights using current singular values"""
        S = self.get_sigma()
        W_flat = self.U @ torch.diag(S) @ self.Vh
        W = W_flat.view(self.cout, self.cin, self.kh, self.kw)
        return F.conv2d(x, W, self.bias, self.stride, self.padding)
    
    def get_uvs_for_ratio(
        self,
        keep_ratio: float,
        impact_scores: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''Return selected U, S, Vh for the given keep_ratio and impact_scores.'''
        return _BaseSVDLayer.get_uvs_for_ratio(self, keep_ratio, impact_scores)

    def get_weight_for_ratio(
        self,
        keep_ratio: float,
        impact_scores: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        '''Reconstruct weight using selected singular components.'''
        U_selected, S_selected, Vh_selected = self.get_uvs_for_ratio(
            keep_ratio, impact_scores
        )
        W_flat = U_selected @ torch.diag(S_selected) @ Vh_selected
        return W_flat.view(self.cout, self.cin, self.kh, self.kw)

    def get_reconstruction_error(self, keep_ratio: float) -> float:
        return _BaseSVDLayer.get_reconstruction_error(self, keep_ratio)

    def get_original_param_count(self) -> int:
        params = self.cout * self.cin * self.kh * self.kw
        if self.bias is not None:
            params += self.cout
        return params

    def get_pruned_param_count(self, rank: int) -> int:
        params = (self.cin * self.kh * self.kw * rank) + (self.cout * rank)
        if self.bias is not None:
            params += self.cout
        return params


class SimpleSVDLinear(_BaseSVDLayer):
    """Learnable SVD Linear layer for FC/MLP pruning."""

    def __init__(
        self,
        original_linear: nn.Linear,
        layer_name: str,
        config=None
    ):
        super().__init__()
        if config is None:
            from configs.config import PruningConfig
            config = PruningConfig()

        w = original_linear.weight.data
        self.out_features, self.in_features = w.shape
        self.layer_name = layer_name
        self.is_bottleneck = False
        self.bias = original_linear.bias

        U, S, Vh = torch.linalg.svd(w, full_matrices=False)
        self.full_rank = len(S)
        self.use_log_s = bool(getattr(config, "use_log_s", True))
        if self.use_log_s:
            self.log_s = nn.Parameter(torch.log(S + config.svd_epsilon))
            self.sigma = None
        else:
            self.sigma = nn.Parameter(S.clone())
            self.log_s = None

        self.register_buffer('U', U[:, :self.full_rank])
        self.register_buffer('Vh', Vh[:self.full_rank, :])
        self.register_buffer('original_weight', w.clone())
        self.register_buffer('fisher_accum', torch.zeros(self.full_rank))
        self.fisher_samples = 0

        self.config = config
        self.layer_kind = "linear"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sigma = self.get_sigma()
        weight = self.U @ torch.diag(sigma) @ self.Vh
        return F.linear(x, weight, self.bias)

    def get_weight_for_ratio(
        self,
        keep_ratio: float,
        impact_scores: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        U_selected, S_selected, Vh_selected = self.get_uvs_for_ratio(
            keep_ratio, impact_scores
        )
        return U_selected @ torch.diag(S_selected) @ Vh_selected

    def get_original_param_count(self) -> int:
        params = self.out_features * self.in_features
        if self.bias is not None:
            params += self.out_features
        return params

    def get_pruned_param_count(self, rank: int) -> int:
        params = (self.in_features * rank) + (self.out_features * rank)
        if self.bias is not None:
            params += self.out_features
        return params

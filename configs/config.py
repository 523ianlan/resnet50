"""Configuration management class - shares structure with ViT version"""

import os
import json
import torch
import numpy as np
from typing import Any, Dict

class PruningConfig:
    """Unified pruning configuration class (compatible with ViT version, adds ResNet parameters)"""
    
    def __init__(self):
        # ========== Basic Settings ==========
        self.experiment_name = "ResNet50_SecondOrderPruning"
        self.model_prefix = "r50"
        self.model_name = "resnet50"
        self.pretrained = True
        self.seed = 42
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # ========== Data Paths ==========
        self.train_root = r"D:\ImageNet_Organized\train"
        self.val_root = r"D:\ImageNet_Organized\validation"
        
        # ========== Data Processing ==========
        self.batch_size = 256
        self.num_workers = 12
        self.pin_memory = True
        self.persistent_workers = True
        self.prefetch_factor = 4
        self.prefetch_to_gpu = True
        self.channels_last = False
        self.enable_tf32 = True
        self.profile_data_time = False
        self.profile_interval = 50
        self.shuffle_val = False
        self.random_eval_subset = True

        # ========== Model Shape / Toy Model Settings ==========
        self.input_channels = 3
        self.input_height = 224
        self.input_width = 224
        self.task_type = "classification"
        self.dataset_type = "imagefolder"
        self.loss_name = "cross_entropy"
        self.num_classes = 1000
        self.target_dim = 1
        self.primary_metric = "top1"
        self.secondary_metric = "top5"
        self.higher_is_better = True
        self.mlp_hidden_dims = [1024, 512]
        self.cnn_channels = [32, 64, 128]
        self.train_csv = None
        self.val_csv = None
        self.image_column = "image_path"
        self.target_column = "target"
        self.target_columns = None
        self.csv_delimiter = ","
        
        # ========== Data Augmentation (ResNet specific) ==========
        self.augmentation = {
            'train': {
                'random_resized_crop': 224,
                'random_horizontal_flip': True,
                'color_jitter': [0.2, 0.2, 0.2, 0.1],
                'auto_augment': None,           # Optional 'rand-m9-mstd0.5'
                'random_erasing': 0.0,
                'mixup_alpha': 0.0,              # MixUp parameter
                'cutmix_alpha': 0.0,              # CutMix parameter
                'mixup_prob': 0.5
            },
            'val': {
                'resize': 256,
                'center_crop': 224
            }
        }
        
        # ========== Core Pruning Parameters ==========
        self.target_compression = 0.8
        self.min_rank = 16
        
        # ========== SVD Parameters ==========
        self.svd_epsilon = 1e-12
        
        # ========== Stage 1: Inter-layer Uncertainty Scoring ==========
        self.mc_samples = 20
        self.mc_dropout_p = 0.1          # MC Dropout probability (Stage 1)
        self.calib_batches = 5           # Calibration batches (0 = use all computation from calibration loader)
        # Calibration set (D_cal) control
        self.calib_split_ratio = 0.0     # 0.0 = disabled, use train_loader
        self.calib_samples = 0           # >0 overrides calib_split_ratio
        self.calib_seed = self.seed
        self.calib_exclude_from_train = False
        self.calib_use_val_transform = False
        self.uncertainty_epsilon = 1e-8
        self.uncertainty_alpha = 1.0
        # Robust normalization for Stage 1 scores
        self.uncertainty_log = True
        self.uncertainty_clip_percentile = 1.0
        self.uncertainty_var_floor = 1e-6
        self.uncertainty_metric = "mu_over_var"  # mu_over_var, mu, var, inv_var
        
        # ========== Stage 2: Fisher Information Calculation ==========
        self.fisher_batches = 100
        self.fisher_approximation = 'diagonal'
        self.accumulate_grad_sign = True
        self.fisher_first_order_mode = 'mean_abs'
        self.fisher_first_order_weight = 1.0
        self.fisher_second_order_weight = 0.5
        self.use_fisher_scores = True
        self.stage2_score_metric = "fisher"  # fisher, taylor, hessian, energy, magnitude

        # ========== Parameterization ==========
        self.use_log_s = True
        
        # ========== Budget Allocation Parameters ==========
        self.allocation_strategy = 'binary_search'
        self.pruning_clip_low = 0.05
        self.pruning_clip_high = 0.95
        
        # ========== Binary Search Parameters ==========
        self.binary_search_iterations = 50
        self.binary_search_low = 0.0
        self.binary_search_high = 5.0
        self.binary_search_tolerance = 0.001
        
        # ========== Fine-tuning Parameters (ResNet optimized) ==========
        self.fine_tune_epochs = 90
        self.base_lr = 1e-4
        self.fine_tune_lr = 1e-4           # Backward compatibility
        self.weight_decay = 5e-4
        self.fine_tune_weight_decay = 5e-4  # Backward compatibility
        self.freeze_epoch = 85               # Epoch to start freezing learning rate
        self.freeze_lr = None                 # Learning rate when frozen (None means use the current learning rate)
        
        # Advanced fine-tuning parameters
        self.layer_decay = 1.0               # Layer-wise decay (1.0 = off)
        self.warmup_epochs = 0                # Warmup epochs
        self.warmup_init_lr = 1e-6
        self.min_lr_ratio = 0.01
        self.decay_end_ratio = 0.75
        
        # Optimizer settings
        self.optimizer = {
            'name': 'adamw',                  # Supported: 'adamw' or 'sgd'
            'betas': [0.9, 0.999],
            'eps': 1e-8
        }
        
        # Learning rate scheduler
        self.scheduler = {
            'name': 'cosine',                  # Supported: 'cosine' or 'step'
            'cosine': {
                'eta_min_ratio': 0.01
            },
            'step': {
                'step_size': 30,
                'gamma': 0.1
            }
        }
        
        # Regularization
        self.use_label_smoothing = True
        self.label_smoothing = 0.1
        self.use_gradient_clip = True
        self.clip_grad_norm = 1.0
        self.stochastic_depth_prob = 0.0       # Stochastic Depth
        self.loss_type = 'ce'                  # 'ce' or 'mse'
        
        # Mixed precision training
        self.mixed_precision = False
        self.gradient_accumulation_steps = 1
        # ========== Reproducibility ==========
        self.deterministic = True
        self.cudnn_benchmark = False
        
        # ========== Early Stopping Parameters ==========
        self.early_stopping_patience = 15
        self.early_stopping_min_delta = 0.001
        
        # ========== Evaluation Parameters ==========
        self.eval_max_batches = 0
        self.train_max_batches = 0
        self.topk = (1, 5)
        self.test_interval = 1
        self.save_best = True
        self.log_interval = 200                 # Print training progress every N batches
        
        # ========== Logging & Storage ==========
        self.save_dir = "./results"
        self.logging = {
            'level': 'INFO',
            'wandb': False,
            'tensorboard': True,
            'save_checkpoints': True,
            'checkpoint_interval': 5
        }
        
        # ========== Visualization Parameters ==========
        self.visualization = {
            'plot_curves': True,
            'plot_sensitivity': True,
            'plot_allocation': True,
            'save_format': 'png',
            'dpi': 150
        }
        
        # Derived experiment naming fields
        self._compression_percentage = int(self.target_compression * 100)
        self.custom_tag = ''
    
    @property
    def compression_percentage(self):
        return self._compression_percentage
    
    def to_dict(self) -> Dict[str, Any]:
        data = {}
        for k, v in self.__dict__.items():
            if not callable(v) and not k.startswith('_'):
                if k == 'device':
                    data[k] = str(v)
                else:
                    data[k] = v
        return data
    
    def save(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
    
    @classmethod
    def load(cls, path: str):
        config = cls()
        with open(path, 'r', encoding='utf-8-sig') as f:
            data = json.load(f)
        for k, v in data.items():
            if hasattr(config, k):
                if k == 'device' and isinstance(v, str):
                    setattr(config, k, torch.device(v))
                else:
                    setattr(config, k, v)
        config._compression_percentage = int(config.target_compression * 100)
        return config
    
    def get_experiment_dir(self) -> str:
        exp_dir = os.path.join(
            self.save_dir,
            f'{self.model_prefix}_{self.compression_percentage}pr_{self.fine_tune_epochs}ft'
        )
        if self.custom_tag:
            exp_dir += f'_{self.custom_tag}'
        os.makedirs(exp_dir, exist_ok=True)
        return exp_dir
    
    def get_model_path(self, suffix: str = "") -> str:
        exp_dir = self.get_experiment_dir()
        base_name = f'{self.model_prefix}_{self.compression_percentage}pr'
        if suffix:
            return os.path.join(exp_dir, f'{base_name}_{suffix}.pth')
        return os.path.join(exp_dir, f'{base_name}.pth')


# test 123(commit test 1 amend)

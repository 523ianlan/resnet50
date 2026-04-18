"""ResNet 資料轉換函式 - 對應 ViT 的 transforms.py"""

import torchvision.transforms as transforms
from typing import Dict, List, Optional, Tuple
import numpy as np

# ImageNet standard normalization parameters
IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]

def get_resnet_train_transform(
    config: Dict,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None
) -> transforms.Compose:
    """
    Get ResNet training data augmentation
    
    Args:
        config: Config dictionary (config.augmentation.train)
        mean: Normalization mean, default uses ImageNet standard
        std: Normalization std, default uses ImageNet standard
    """
    if mean is None:
        mean = IMAGENET_DEFAULT_MEAN
    if std is None:
        std = IMAGENET_DEFAULT_STD
    
    transform_list = []
    
    # 1. Random crop
    if 'random_resized_crop' in config:
        transform_list.append(
            transforms.RandomResizedCrop(config['random_resized_crop'])
        )
    else:
        transform_list.append(transforms.RandomResizedCrop(224))
    
    # 2. Random horizontal flip
    if config.get('random_horizontal_flip', True):
        transform_list.append(transforms.RandomHorizontalFlip())
    
    # 3. Color jitter
    color_jitter = config.get('color_jitter')
    if color_jitter:
        if isinstance(color_jitter, (list, tuple)):
            transform_list.append(
                transforms.ColorJitter(
                    brightness=color_jitter[0],
                    contrast=color_jitter[1],
                    saturation=color_jitter[2],
                    hue=color_jitter[3] if len(color_jitter) > 3 else 0
                )
            )
        elif isinstance(color_jitter, (int, float)):
            transform_list.append(transforms.ColorJitter(color_jitter))
    
    # 4. Convert to Tensor
    transform_list.append(transforms.ToTensor())
    
    # 5. Normalize
    transform_list.append(transforms.Normalize(mean=mean, std=std))
    
    # 6. Random Erasing (optional)
    if config.get('random_erasing', 0) > 0:
        transform_list.append(
            transforms.RandomErasing(
                p=config['random_erasing'],
                scale=(0.02, 0.33),
                ratio=(0.3, 3.3),
                value=0
            )
        )
    
    return transforms.Compose(transform_list)


def get_resnet_val_transform(
    config: Dict,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None
) -> transforms.Compose:
    """
    Get ResNet validation data transform
    
    Args:
        config: Config dictionary (config.augmentation.val)
        mean: Normalization mean, default uses ImageNet standard
        std: Normalization std, default uses ImageNet standard
    """
    if mean is None:
        mean = IMAGENET_DEFAULT_MEAN
    if std is None:
        std = IMAGENET_DEFAULT_STD
    
    transform_list = []
    
    # 1. Resize
    resize_size = config.get('resize', 256)
    transform_list.append(transforms.Resize(resize_size))
    
    # 2. Center Crop
    crop_size = config.get('center_crop', 224)
    transform_list.append(transforms.CenterCrop(crop_size))
    
    # 3. Convert to Tensor
    transform_list.append(transforms.ToTensor())
    
    # 4. Normalize
    transform_list.append(transforms.Normalize(mean=mean, std=std))
    
    return transforms.Compose(transform_list)

# # test 123(commit test 2)
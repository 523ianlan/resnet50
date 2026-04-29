"""ResNet Data Loader builder - Corresponds to ViT build.py"""

import torch
import random
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import ImageFolder
import numpy as np
from typing import Tuple, Optional, Dict, Any

from .transforms import get_resnet_train_transform, get_resnet_val_transform
from configs.config import PruningConfig


def _seed_worker(worker_id: int):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


class FastBatchCollator:
    """
    Ultra-fast batch collator - Only stacks, no calculation
    Leave data augmentation (MixUp/CutMix) to GPU
    """
    def __call__(self, batch):
        images = torch.stack([item[0] for item in batch])
        labels = torch.tensor([item[1] for item in batch])
        return images, labels


class CUDAPrefetcher:
    """
    Prefetch batches to GPU on a separate CUDA stream to overlap H2D copy with compute.
    """

    def __init__(self, loader: DataLoader, device: torch.device, channels_last: bool = False):
        self.loader = loader
        self.device = device
        self.channels_last = bool(channels_last) and device.type == "cuda"
        self.stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None
        self._iter = None
        self._next_batch = None

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        self._iter = iter(self.loader)
        self._preload()
        return self

    def _move_to_device(self, obj):
        if torch.is_tensor(obj):
            return obj.to(self.device, non_blocking=True)
        if isinstance(obj, tuple):
            return tuple(self._move_to_device(x) for x in obj)
        return obj

    def _preload(self):
        try:
            batch = next(self._iter)
        except StopIteration:
            self._next_batch = None
            return

        if self.stream is None:
            self._next_batch = batch
            return

        with torch.cuda.stream(self.stream):
            images, labels = batch
            images = images.to(self.device, non_blocking=True)
            if self.channels_last:
                images = images.to(memory_format=torch.channels_last)
            labels = self._move_to_device(labels)
            self._next_batch = (images, labels)

    def __next__(self):
        if self._next_batch is None:
            raise StopIteration
        if self.stream is not None:
            torch.cuda.current_stream(self.device).wait_stream(self.stream)
        batch = self._next_batch
        self._preload()
        return batch


def _loader_common_kwargs(config: PruningConfig) -> Dict[str, Any]:
    kwargs = {
        "num_workers": config.num_workers,
        "pin_memory": config.pin_memory,
        "persistent_workers": bool(config.persistent_workers) and config.num_workers > 0,
        "worker_init_fn": _seed_worker,
    }
    if config.num_workers > 0:
        kwargs["prefetch_factor"] = getattr(config, "prefetch_factor", 2)
    return kwargs


def _build_calibration_subset(
    config: PruningConfig,
    train_dataset: Dataset
) -> Optional[Subset]:
    calib_samples = int(getattr(config, "calib_samples", 0))
    calib_ratio = float(getattr(config, "calib_split_ratio", 0.0))
    if calib_samples <= 0 and calib_ratio <= 0:
        return None

    total = len(train_dataset)
    if calib_samples <= 0:
        calib_samples = max(1, int(total * calib_ratio))
    calib_samples = min(calib_samples, total)

    g = torch.Generator()
    calib_seed = int(getattr(config, "calib_seed", getattr(config, "seed", 42)))
    g.manual_seed(calib_seed)
    perm = torch.randperm(total, generator=g).tolist()
    calib_indices = perm[:calib_samples]
    return Subset(train_dataset, calib_indices)


def _maybe_build_eval_subset(
    config: PruningConfig,
    val_dataset: Dataset
) -> Tuple[Dataset, bool]:
    """
    When eval_max_batches is small, evaluate on a random subset instead of the
    first few class-sorted batches. This keeps quick smoke tests meaningful.
    """
    eval_max_batches = int(getattr(config, "eval_max_batches", 0))
    batch_size = int(getattr(config, "batch_size", 1))
    use_random_subset = bool(getattr(config, "random_eval_subset", True))

    if eval_max_batches <= 0 or not use_random_subset:
        return val_dataset, bool(getattr(config, "shuffle_val", False))

    subset_size = min(len(val_dataset), max(1, eval_max_batches * batch_size))
    g = torch.Generator()
    g.manual_seed(int(getattr(config, "seed", 42)) + 999)
    perm = torch.randperm(len(val_dataset), generator=g).tolist()
    subset = Subset(val_dataset, perm[:subset_size])
    print(
        f"Using random validation subset for quick eval: "
        f"{subset_size} samples (~{eval_max_batches} batches)"
    )
    return subset, True


def get_resnet_data_loaders_with_calib(
    config: PruningConfig,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Build train/val loaders and an optional calibration loader (D_cal).
    """
    train_transform = get_resnet_train_transform(config.augmentation['train'])
    val_transform = get_resnet_val_transform(config.augmentation['val'])

    train_dataset = ImageFolder(
        root=config.train_root,
        transform=train_transform
    )
    val_dataset = ImageFolder(
        root=config.val_root,
        transform=val_transform
    )

    # Optional calibration dataset (subset of train)
    calib_dataset = None
    if bool(getattr(config, "calib_use_val_transform", False)):
        calib_base = ImageFolder(
            root=config.train_root,
            transform=val_transform
        )
        calib_dataset = _build_calibration_subset(config, calib_base)
    else:
        calib_dataset = _build_calibration_subset(config, train_dataset)

    # Optionally exclude calibration samples from training
    if calib_dataset is not None and bool(getattr(config, "calib_exclude_from_train", False)):
        calib_indices = set(calib_dataset.indices)
        train_indices = [i for i in range(len(train_dataset)) if i not in calib_indices]
        train_dataset = Subset(train_dataset, train_indices)

    loader_kwargs = _loader_common_kwargs(config)
    generator = torch.Generator()
    generator.manual_seed(int(getattr(config, "seed", 42)))

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        generator=generator,
        **loader_kwargs
    )
    eval_dataset, eval_shuffle = _maybe_build_eval_subset(config, val_dataset)

    val_loader = DataLoader(
        eval_dataset,
        batch_size=config.batch_size,
        shuffle=eval_shuffle,
        generator=generator,
        **loader_kwargs
    )

    calib_loader = None
    if calib_dataset is not None:
        calib_loader = DataLoader(
            calib_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            generator=generator,
            **loader_kwargs
        )

    if getattr(config, "prefetch_to_gpu", False) and config.device.type == "cuda":
        print("Using CUDA prefetcher for data loading")
        train_loader = CUDAPrefetcher(
            train_loader, config.device, channels_last=getattr(config, "channels_last", False)
        )
        val_loader = CUDAPrefetcher(
            val_loader, config.device, channels_last=getattr(config, "channels_last", False)
        )
        if calib_loader is not None:
            calib_loader = CUDAPrefetcher(
                calib_loader, config.device, channels_last=getattr(config, "channels_last", False)
            )

    return train_loader, val_loader, calib_loader

def get_resnet_data_loaders(
    config: PruningConfig,
    train_dataset: Optional[Dataset] = None,
    val_dataset: Optional[Dataset] = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Get ResNet training and validation data loaders
    
    Args:
        config: Pruning config (PruningConfig 對象)
        train_dataset: Custom training dataset (optional)
        val_dataset: Custom validation dataset (optional)
    
    Returns:
        (train_loader, val_loader)
    """
    
    # Add debugging info
    print(f"Debug - Train root: {config.train_root}")
    print(f"Debug - Val root: {config.val_root}")
    
    # Load from path if datasets not provided
    if train_dataset is None:
        train_transform = get_resnet_train_transform(config.augmentation['train'])
        print(f"Loading training data from: {config.train_root}")
        train_dataset = ImageFolder(
            root=config.train_root,
            transform=train_transform
        )
    
    if val_dataset is None:
        val_transform = get_resnet_val_transform(config.augmentation['val'])
        print(f"Loading validation data from: {config.val_root}")
        val_dataset = ImageFolder(
            root=config.val_root,
            transform=val_transform
        )
    # DataLoader settings
    loader_kwargs = _loader_common_kwargs(config)
    generator = torch.Generator()
    generator.manual_seed(int(getattr(config, "seed", 42)))

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        generator=generator,
        **loader_kwargs
    )
    
    # Validation DataLoader
    eval_dataset, eval_shuffle = _maybe_build_eval_subset(config, val_dataset)

    val_loader = DataLoader(
        eval_dataset,
        batch_size=config.batch_size,
        shuffle=eval_shuffle,
        generator=generator,
        **loader_kwargs
    )

    if getattr(config, "prefetch_to_gpu", False) and config.device.type == "cuda":
        print("Using CUDA prefetcher for data loading")
        train_loader = CUDAPrefetcher(
            train_loader, config.device, channels_last=getattr(config, "channels_last", False)
        )
        val_loader = CUDAPrefetcher(
            val_loader, config.device, channels_last=getattr(config, "channels_last", False)
        )
    
    return train_loader, val_loader


def get_optimized_data_loaders_r50(
    config: PruningConfig
) -> Tuple[DataLoader, DataLoader]:
    """
    Optimized ResNet DataLoader
    """
    return get_resnet_data_loaders(config)

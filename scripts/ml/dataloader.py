"""
PyTorch DataLoaders for OceanEmbed.

Creates DataLoaders from the train, validation, and test
OceanDataset instances.

Expected batch shapes:

    x:
        [B, T, 7, 101, 241]

    y:
        [B, 15, 101, 241]

    mask:
        [B, 15, 101, 241]
"""

import torch
from torch.utils.data import DataLoader

from scripts.ml.dataset import (
    create_train_dataset,
    create_val_dataset,
    create_test_dataset,
)


def create_dataloaders(
    batch_size,
    num_workers=0,
    pin_memory=False,
):
    """
    Create train, validation, and test DataLoaders.

    Parameters
    ----------
    batch_size : int
        Number of samples per batch.

    num_workers : int
        Number of DataLoader worker processes.
        Use 0 initially for reliable Windows testing.

    pin_memory : bool
        Whether to pin CPU tensors for faster GPU transfer.

    Returns
    -------
    train_loader, val_loader, test_loader
    """

    train_dataset = create_train_dataset()
    val_dataset = create_val_dataset()
    test_dataset = create_test_dataset()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
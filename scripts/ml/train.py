"""
Training script for OceanEmbed.

Train:
    2015-01-01 -> 2021-12-31

Validation:
    2022-01-01 -> 2022-12-31

Test:
    2023-01-01 -> 2024-12-31

The test set is NOT used during training.

Model input:
    [B, 7, 7, 101, 241]

Model output:
    [B, 15, 101, 241]
"""

from pathlib import Path
import time

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from scripts.ml.dataloader import create_dataloaders
from scripts.ml.losses import MaskedMSELoss
from scripts.ml.models.temporal_unet import TemporalUNet


# ============================================================
# Configuration
# ============================================================

BATCH_SIZE = 32
NUM_WORKERS = 8

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

NUM_EPOCHS = 20

USE_AMP = True

CHECKPOINT_DIR = Path("checkpoints")

BEST_CHECKPOINT_PATH = (
    CHECKPOINT_DIR / "temporal_unet_best.pt"
)

LAST_CHECKPOINT_PATH = (
    CHECKPOINT_DIR / "temporal_unet_last.pt"
)


def train_one_epoch(
    model,
    loader,
    loss_fn,
    optimizer,
    scaler,
    device,
    use_amp,
):
    """
    Run one complete training epoch.
    """

    model.train()

    total_loss = 0.0
    num_batches = 0

    for batch in loader:

        x = batch["x"].to(
            device,
            non_blocking=True,
        )

        y = batch["y"].to(
            device,
            non_blocking=True,
        )

        mask = batch["mask"].to(
            device,
            non_blocking=True,
        )

        # ----------------------------------------------------
        # Clear previous gradients.
        # ----------------------------------------------------
        optimizer.zero_grad(
            set_to_none=True
        )

        # ----------------------------------------------------
        # Forward pass + loss under AMP.
        # ----------------------------------------------------
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            prediction = model(x)
            loss = loss_fn(prediction,y,mask,)

        # ----------------------------------------------------
        # Backpropagation with gradient scaling.
        # ----------------------------------------------------
        scaler.scale(loss).backward()

        # ----------------------------------------------------
        # Update parameters.
        # ----------------------------------------------------
        scaler.step(optimizer)

        # ----------------------------------------------------
        # Update gradient scale.
        # ----------------------------------------------------
        scaler.update()

        total_loss += loss.item()
        num_batches += 1

    if num_batches == 0:
        raise RuntimeError(
            "Training DataLoader produced zero batches."
        )

    return total_loss / num_batches


@torch.no_grad()
def validate_one_epoch(
    model,
    loader,
    loss_fn,
    device,
    use_amp,
):
    """
    Run validation without gradient computation.
    """

    model.eval()

    total_loss = 0.0
    num_batches = 0

    for batch in loader:

        x = batch["x"].to(
            device,
            non_blocking=True,
        )

        y = batch["y"].to(
            device,
            non_blocking=True,
        )

        mask = batch["mask"].to(
            device,
            non_blocking=True,
        )

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            prediction = model(x)

            loss = loss_fn(prediction, y, mask,)

        total_loss += loss.item()
        num_batches += 1

    if num_batches == 0:
        raise RuntimeError(
            "Validation DataLoader produced zero batches."
        )

    return total_loss / num_batches


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    epoch,
    train_loss,
    val_loss,
):
    """
    Save a complete training checkpoint.
    """

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
        },
        path,
    )


def main():

    print("=" * 70)
    print("TRAINING OCEANEMBED TEMPORAL U-NET")
    print("=" * 70)

    # ========================================================
    # Device
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    print()
    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    # ========================================================
    # Checkpoint directory
    # ========================================================

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # DataLoaders
    # ========================================================

    print()
    print("Creating DataLoaders...")

    train_loader, val_loader, test_loader = (
        create_dataloaders(
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
        )
    )

    # --------------------------------------------------------
    # We intentionally create the test loader because it is
    # part of the dataset pipeline, but it is NOT touched
    # during training.
    # --------------------------------------------------------
    del test_loader

    # ========================================================
    # Model
    # ========================================================

    model = TemporalUNet(
        in_channels=7,
        out_channels=15,
        base_channels=32,
    ).to(device)

    num_parameters = sum(parameter.numel() for parameter in model.parameters())

    print(f"Trainable parameters: {num_parameters:,}")

    # ========================================================
    # Loss
    # ========================================================

    loss_fn = MaskedMSELoss()

    # ========================================================
    # Optimizer
    # ========================================================

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=USE_AMP and device.type == "cuda",
    )

    # ========================================================
    # Learning-rate scheduler
    #
    # If validation loss stops improving, reduce LR.
    # ========================================================

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )

    # ========================================================
    # Training
    # ========================================================

    best_val_loss = float("inf")

    print()
    print("=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    for epoch in range(1, NUM_EPOCHS + 1):

        epoch_start = time.perf_counter()
        train_start = time.perf_counter()
        
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=USE_AMP and device.type == "cuda",
        )

        train_time = time.perf_counter() - train_start
        val_start = time.perf_counter()

        val_loss = validate_one_epoch(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            device=device,
            use_amp=USE_AMP and device.type == "cuda",
        )

        val_time = time.perf_counter() - val_start
        epoch_time = time.perf_counter() - epoch_start

        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        # ----------------------------------------------------
        # Save best model.
        # ----------------------------------------------------
        if val_loss < best_val_loss:

            best_val_loss = val_loss

            save_checkpoint(
                path=BEST_CHECKPOINT_PATH,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
            )

            best_marker = "  <-- BEST"

        else:
            best_marker = ""

        # ----------------------------------------------------
        # Save latest checkpoint.
        # ----------------------------------------------------
        save_checkpoint(
            path=LAST_CHECKPOINT_PATH,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
        )

        print(
            f"Epoch {epoch:03d}/{NUM_EPOCHS:03d} | "
            f"Train: {train_loss:.6f} | "
            f"Val: {val_loss:.6f} | "
            f"LR: {current_lr:.2e} | "
            f"TrainTime: {train_time:.1f}s | "
            f"ValTime: {val_time:.1f}s | "
            f"Total: {epoch_time:.1f}s"
            f"{best_marker}"
        )

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print()
    print(
        f"Best validation loss: "
        f"{best_val_loss:.6f}"
    )

    print()
    print(
        f"Best checkpoint: "
        f"{BEST_CHECKPOINT_PATH}"
    )

    print(
        f"Last checkpoint: "
        f"{LAST_CHECKPOINT_PATH}"
    )


if __name__ == "__main__":
    main()
"""
Training script for the OceanEmbed Causal Swin Transformer.

Train:
    2015-01-01 -> 2021-12-31

Validation:
    2022-01-01 -> 2022-12-31

Test:
    2023-01-01 -> 2024-12-31

The test set is created by the existing dataloader but is NOT
used during training.

Input:
    [B, 7, 7, 101, 241]

Output:
    [B, 15, 101, 241]

Architecture:
    CNN input projection
        ->
    causal temporal attention
        ->
    hierarchical Swin Transformer
        ->
    U-Net decoder
        ->
    15-depth output head
"""

from pathlib import Path
import time

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from scripts.ml.dataloader import create_dataloaders
from scripts.ml.losses import MaskedMSELoss
from scripts.ml.models.ocean_transformer import OceanTransformer


# ============================================================
# Configuration
# ============================================================

# ------------------------------------------------------------
# Batch size / memory
# ------------------------------------------------------------

# The old U-Net used 32 directly.
#
# This Transformer is considerably more memory-intensive, so
# start with a physical batch size of 4.
BATCH_SIZE = 4

# 4 x 8 = effective batch size of 32.
GRADIENT_ACCUMULATION_STEPS = 8

NUM_WORKERS = 8

# ------------------------------------------------------------
# Optimization
# ------------------------------------------------------------

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4

NUM_EPOCHS = 20

# Clip gradients after unscaling AMP gradients.
MAX_GRAD_NORM = 1.0

# ------------------------------------------------------------
# Mixed precision
# ------------------------------------------------------------

USE_AMP = True

# ------------------------------------------------------------
# Model regularization
# ------------------------------------------------------------

DROPOUT = 0.1
STOCHASTIC_DEPTH_PROB = 0.1

# ------------------------------------------------------------
# Architecture
# ------------------------------------------------------------

TEMPORAL_LENGTH = 7

IN_CHANNELS = 7
OUT_CHANNELS = 15

FEATURE_CHANNELS = 64

TEMPORAL_HEADS = 8

SWIN_CHANNELS = (64, 128, 256)
SWIN_DEPTHS = (2, 2, 4)
SWIN_HEADS = (4, 8, 16)

WINDOW_SIZE = 7

USE_TEMPORAL_ATTENTION = True
USE_COORDINATE_FEATURES = True

# ------------------------------------------------------------
# Checkpoints
# ------------------------------------------------------------

CHECKPOINT_DIR = (
    Path("checkpoints") / "ocean_transformer"
)

BEST_CHECKPOINT_PATH = (
    CHECKPOINT_DIR / "ocean_transformer_best.pt"
)

LAST_CHECKPOINT_PATH = (
    CHECKPOINT_DIR / "ocean_transformer_last.pt"
)


# ============================================================
# Model configuration
# ============================================================

MODEL_CONFIG = {
    "in_channels": IN_CHANNELS,
    "out_channels": OUT_CHANNELS,
    "temporal_length": TEMPORAL_LENGTH,

    "feature_channels": FEATURE_CHANNELS,

    "temporal_heads": TEMPORAL_HEADS,

    "swin_channels": SWIN_CHANNELS,
    "swin_depths": SWIN_DEPTHS,
    "swin_heads": SWIN_HEADS,

    "window_size": WINDOW_SIZE,

    "dropout": DROPOUT,
    "stochastic_depth_prob": STOCHASTIC_DEPTH_PROB,

    "use_temporal_attention": USE_TEMPORAL_ATTENTION,
    "use_coordinate_features": USE_COORDINATE_FEATURES,
}


# ============================================================
# Training
# ============================================================

def train_one_epoch(
    model,
    loader,
    loss_fn,
    optimizer,
    scaler,
    device,
    use_amp,
    accumulation_steps,
):
    """
    Run one complete training epoch.

    Gradient accumulation is used so that:

        physical batch = BATCH_SIZE

        effective batch =
            BATCH_SIZE * accumulation_steps

    The optimizer is updated once every accumulation_steps
    batches.
    """

    model.train()

    total_loss = 0.0
    num_batches = 0

    optimizer.zero_grad(set_to_none=True)

    for batch_idx, batch in enumerate(loader):

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
        # Forward pass + loss
        # ----------------------------------------------------

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            prediction = model(x)

            loss = loss_fn(
                prediction,
                y,
                mask,
            )

        # ----------------------------------------------------
        # Keep the reported loss as the actual batch loss.
        # ----------------------------------------------------

        total_loss += loss.item()
        num_batches += 1

        # ----------------------------------------------------
        # Divide before backward so accumulated gradients
        # correspond to the mean over the effective batch.
        # ----------------------------------------------------

        scaled_loss = loss / accumulation_steps

        scaler.scale(scaled_loss).backward()

        # ----------------------------------------------------
        # Optimizer step
        #
        # Also handle the final partial accumulation group.
        # ----------------------------------------------------

        should_step = (
            (batch_idx + 1) % accumulation_steps == 0
            or (batch_idx + 1) == len(loader)
        )

        if should_step:

            # Unscale BEFORE gradient clipping.
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=MAX_GRAD_NORM,
            )

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(
                set_to_none=True
            )

    if num_batches == 0:
        raise RuntimeError(
            "Training DataLoader produced zero batches."
        )

    return total_loss / num_batches


# ============================================================
# Validation
# ============================================================

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

            loss = loss_fn(
                prediction,
                y,
                mask,
            )

        total_loss += loss.item()
        num_batches += 1

    if num_batches == 0:
        raise RuntimeError(
            "Validation DataLoader produced zero batches."
        )

    return total_loss / num_batches


# ============================================================
# Checkpoint
# ============================================================

def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    train_loss,
    val_loss,
):
    """
    Save everything required to resume training and
    reconstruct the model.
    """

    torch.save(
        {
            # ------------------------------------------------
            # Training state
            # ------------------------------------------------
            "epoch": epoch,

            "train_loss": train_loss,
            "val_loss": val_loss,

            # ------------------------------------------------
            # Model
            # ------------------------------------------------
            "model_state_dict": model.state_dict(),

            # ------------------------------------------------
            # Optimizer / scheduler
            # ------------------------------------------------
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),

            # ------------------------------------------------
            # AMP scaler
            # ------------------------------------------------
            "scaler_state_dict": scaler.state_dict(),

            # ------------------------------------------------
            # Architecture
            #
            # This means evaluation does not have to guess
            # which architecture configuration was trained.
            # ------------------------------------------------
            "model_config": MODEL_CONFIG,

            # ------------------------------------------------
            # Training configuration
            # ------------------------------------------------
            "batch_size": BATCH_SIZE,
            "gradient_accumulation_steps": (
                GRADIENT_ACCUMULATION_STEPS
            ),
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
        },
        path,
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("TRAINING OCEANEMBED CAUSAL SWIN TRANSFORMER")
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

    if device.type == "cuda":
        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            f"CUDA: "
            f"{torch.version.cuda}"
        )

    # ========================================================
    # Configuration summary
    # ========================================================

    print()
    print("Training configuration:")
    print(f"  Physical batch size:       {BATCH_SIZE}")
    print(
        f"  Gradient accumulation:     "
        f"{GRADIENT_ACCUMULATION_STEPS}"
    )
    print(
        f"  Effective batch size:      "
        f"{BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}"
    )
    print(f"  Learning rate:              {LEARNING_RATE}")
    print(f"  Weight decay:               {WEIGHT_DECAY}")
    print(f"  Epochs:                     {NUM_EPOCHS}")
    print(f"  AMP:                        {USE_AMP}")
    print(f"  Gradient clip:              {MAX_GRAD_NORM}")

    print()
    print("Model configuration:")
    print(f"  Temporal length:            {TEMPORAL_LENGTH}")
    print(f"  Input variables:            {IN_CHANNELS}")
    print(f"  Output depths:              {OUT_CHANNELS}")
    print(f"  Feature channels:           {FEATURE_CHANNELS}")
    print(f"  Temporal attention heads:   {TEMPORAL_HEADS}")
    print(f"  Swin channels:              {SWIN_CHANNELS}")
    print(f"  Swin depths:                {SWIN_DEPTHS}")
    print(f"  Swin heads:                 {SWIN_HEADS}")
    print(f"  Window size:                {WINDOW_SIZE}")
    print(
        f"  Temporal attention:         "
        f"{USE_TEMPORAL_ATTENTION}"
    )
    print(
        f"  Coordinate features:        "
        f"{USE_COORDINATE_FEATURES}"
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

    # The test loader is deliberately not used.
    del test_loader

    # ========================================================
    # Model
    # ========================================================

    print()
    print("Creating model...")

    model = OceanTransformer(
        **MODEL_CONFIG,
    ).to(device)

    num_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"Trainable parameters: "
        f"{num_parameters:,}"
    )

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

    # ========================================================
    # AMP scaler
    # ========================================================

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=USE_AMP and device.type == "cuda",
    )

    # ========================================================
    # Learning-rate scheduler
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

    for epoch in range(
        1,
        NUM_EPOCHS + 1,
    ):

        epoch_start = time.perf_counter()

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        train_start = time.perf_counter()

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=(
                USE_AMP
                and device.type == "cuda"
            ),
            accumulation_steps=(
                GRADIENT_ACCUMULATION_STEPS
            ),
        )

        train_time = (
            time.perf_counter()
            - train_start
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        val_start = time.perf_counter()

        val_loss = validate_one_epoch(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            device=device,
            use_amp=(
                USE_AMP
                and device.type == "cuda"
            ),
        )

        val_time = (
            time.perf_counter()
            - val_start
        )

        epoch_time = (
            time.perf_counter()
            - epoch_start
        )

        # ----------------------------------------------------
        # Learning-rate scheduler
        # ----------------------------------------------------

        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            save_checkpoint(
                path=BEST_CHECKPOINT_PATH,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
            )

            best_marker = "  <-- BEST"

        else:
            best_marker = ""

        # ----------------------------------------------------
        # Save latest model
        # ----------------------------------------------------

        save_checkpoint(
            path=LAST_CHECKPOINT_PATH,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
        )

        # ----------------------------------------------------
        # Logging
        # ----------------------------------------------------

        print()
        print(
            f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
            f"Train: {train_loss:.6f} | "
            f"Val: {val_loss:.6f} | "
            f"LR: {current_lr:.2e} | "
            f"Train: {train_time:.1f}s | "
            f"Val: {val_time:.1f}s | "
            f"Total: {epoch_time:.1f}s"
            f"{best_marker}"
        )

    # ========================================================
    # Finished
    # ========================================================

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
        f"Best checkpoint:\n"
        f"  {BEST_CHECKPOINT_PATH}"
    )

    print(
        f"Last checkpoint:\n"
        f"  {LAST_CHECKPOINT_PATH}"
    )


if __name__ == "__main__":
    main()
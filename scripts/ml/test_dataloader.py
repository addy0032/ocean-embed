import torch

from scripts.ml.dataloader import create_dataloaders


def main():

    print("=" * 70)
    print("TESTING OCEAN DATALOADERS")
    print("=" * 70)

    # --------------------------------------------------------
    # Initial smoke-test configuration.
    # --------------------------------------------------------
    batch_size = 2

    train_loader, val_loader, test_loader = create_dataloaders(
        batch_size=batch_size,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    print()
    print("DataLoader sizes:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    print(f"  Test batches:  {len(test_loader)}")

    # --------------------------------------------------------
    # Read one training batch.
    # --------------------------------------------------------
    print()
    print("Reading first training batch...")

    batch = next(iter(train_loader))

    x = batch["x"]
    y = batch["y"]
    mask = batch["mask"]

    # --------------------------------------------------------
    # Shapes.
    # --------------------------------------------------------
    print()
    print("Batch shapes:")
    print(f"  x:    {x.shape}")
    print(f"  y:    {y.shape}")
    print(f"  mask: {mask.shape}")

    # --------------------------------------------------------
    # Expected shapes.
    # --------------------------------------------------------
    expected_x_shape = (
        batch_size,
        7,
        7,
        101,
        241,
    )

    expected_y_shape = (
        batch_size,
        15,
        101,
        241,
    )

    expected_mask_shape = (
        batch_size,
        15,
        101,
        241,
    )

    if x.shape != expected_x_shape:
        raise ValueError(
            f"Unexpected x shape.\n"
            f"Expected: {expected_x_shape}\n"
            f"Found:    {tuple(x.shape)}"
        )

    if y.shape != expected_y_shape:
        raise ValueError(
            f"Unexpected y shape.\n"
            f"Expected: {expected_y_shape}\n"
            f"Found:    {tuple(y.shape)}"
        )

    if mask.shape != expected_mask_shape:
        raise ValueError(
            f"Unexpected mask shape.\n"
            f"Expected: {expected_mask_shape}\n"
            f"Found:    {tuple(mask.shape)}"
        )

    # --------------------------------------------------------
    # Dtypes.
    # --------------------------------------------------------
    print()
    print("Dtypes:")
    print(f"  x:    {x.dtype}")
    print(f"  y:    {y.dtype}")
    print(f"  mask: {mask.dtype}")

    if x.dtype != torch.float32:
        raise TypeError(f"Expected x to be float32, got {x.dtype}")

    if y.dtype != torch.float32:
        raise TypeError(f"Expected y to be float32, got {y.dtype}")

    if mask.dtype != torch.bool:
        raise TypeError(f"Expected mask to be bool, got {mask.dtype}")

    # --------------------------------------------------------
    # NaN / Inf checks.
    # --------------------------------------------------------
    x_nan = torch.isnan(x).sum().item()
    y_nan = torch.isnan(y).sum().item()

    x_inf = torch.isinf(x).sum().item()
    y_inf = torch.isinf(y).sum().item()

    print()
    print("NaN / Inf check:")
    print(f"  x NaNs: {x_nan}")
    print(f"  y NaNs: {y_nan}")
    print(f"  x Infs: {x_inf}")
    print(f"  y Infs: {y_inf}")

    if x_nan > 0:
        raise ValueError("Input batch contains NaNs.")

    if y_nan > 0:
        raise ValueError("Target batch contains NaNs.")

    if x_inf > 0:
        raise ValueError("Input batch contains Inf values.")

    if y_inf > 0:
        raise ValueError("Target batch contains Inf values.")

    # --------------------------------------------------------
    # Mask statistics.
    # --------------------------------------------------------
    print()
    print("Mask:")
    print(f"  valid: {mask.sum().item()}")
    print(f"  total: {mask.numel()}")
    print(
        f"  valid percentage: "
        f"{100.0 * mask.float().mean().item():.2f}%"
    )

    # --------------------------------------------------------
    # Check that mask contains only True / False.
    # --------------------------------------------------------
    unique_mask_values = torch.unique(mask)

    if not torch.all(
        (unique_mask_values == 0) | (unique_mask_values == 1)
    ):
        raise ValueError("Mask contains values other than 0/1.")

    # --------------------------------------------------------
    # Check GPU availability.
    # --------------------------------------------------------
    print()
    print("Device:")
    print(f"  CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    print()
    print("=" * 70)
    print("DATALOADER TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
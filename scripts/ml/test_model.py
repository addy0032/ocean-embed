import torch

from scripts.ml.dataloader import create_dataloaders
from scripts.ml.models.temporal_unet import TemporalUNet


def main():

    print("=" * 70)
    print("TESTING TEMPORAL U-NET")
    print("=" * 70)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print()
    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # --------------------------------------------------------
    # Load one batch.
    #
    # Keep batch size small for the smoke test.
    # --------------------------------------------------------
    train_loader, _, _ = create_dataloaders(
        batch_size=2,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    batch = next(iter(train_loader))

    x = batch["x"]
    y = batch["y"]
    mask = batch["mask"]

    print()
    print("Input:")
    print(f"  x:    {x.shape}")
    print(f"  y:    {y.shape}")
    print(f"  mask: {mask.shape}")

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------
    model = TemporalUNet(
        in_channels=7,
        out_channels=15,
        base_channels=32,
    )

    model = model.to(device)

    x = x.to(device)
    y = y.to(device)
    mask = mask.to(device)

    # --------------------------------------------------------
    # Forward pass.
    # --------------------------------------------------------
    print()
    print("Running forward pass...")

    model.eval()

    with torch.no_grad():
        prediction = model(x)

    print()
    print("Output:")
    print(f"  prediction: {prediction.shape}")

    # --------------------------------------------------------
    # Verify output shape.
    # --------------------------------------------------------
    expected_shape = (
        x.shape[0],
        15,
        101,
        241,
    )

    if prediction.shape != expected_shape:
        raise ValueError(
            f"Unexpected prediction shape.\n"
            f"Expected: {expected_shape}\n"
            f"Found:    {tuple(prediction.shape)}"
        )

    # --------------------------------------------------------
    # Numerical checks.
    # --------------------------------------------------------
    print()
    print("Prediction:")
    print(f"  dtype: {prediction.dtype}")
    print(f"  min:   {prediction.min().item():.6f}")
    print(f"  max:   {prediction.max().item():.6f}")
    print(f"  mean:  {prediction.mean().item():.6f}")

    if torch.isnan(prediction).any():
        raise ValueError("Model output contains NaNs.")

    if torch.isinf(prediction).any():
        raise ValueError("Model output contains Inf values.")

    # --------------------------------------------------------
    # Parameter count.
    # --------------------------------------------------------
    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print()
    print("Parameters:")
    print(f"  total:     {total_parameters:,}")
    print(f"  trainable: {trainable_parameters:,}")

    print()
    print("=" * 70)
    print("MODEL TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
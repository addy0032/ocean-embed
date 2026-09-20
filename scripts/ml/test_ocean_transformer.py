import sys
from pathlib import Path

import torch


# Allow running:
# uv run scripts/ml/test_ocean_transformer.py
ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from scripts.ml.models.ocean_transformer import OceanTransformer


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 70)
    print("TESTING OCEAN TRANSFORMER")
    print("=" * 70)

    print(f"Device: {device}")

    model = OceanTransformer(
        in_channels=7,
        out_channels=15,
        temporal_length=7,

        feature_channels=64,

        temporal_heads=8,

        swin_channels=(64, 128, 256),
        swin_depths=(2, 2, 4),
        swin_heads=(4, 8, 16),

        window_size=7,

        dropout=0.1,
        stochastic_depth_prob=0.1,

        use_temporal_attention=True,
        use_coordinate_features=True,
    ).to(device)

    # ---------------------------------------------------------
    # Synthetic input with the EXACT real shape.
    # ---------------------------------------------------------

    x = torch.randn(
        2,
        7,
        7,
        101,
        241,
        device=device,
    )

    print()
    print(f"Input shape:  {tuple(x.shape)}")

    # ---------------------------------------------------------
    # Forward pass
    # ---------------------------------------------------------

    with torch.no_grad():
        y = model(x)

    print(f"Output shape: {tuple(y.shape)}")

    # ---------------------------------------------------------
    # Verify expected shape.
    # ---------------------------------------------------------

    expected = (
        2,
        15,
        101,
        241,
    )

    assert tuple(y.shape) == expected, (
        f"Wrong output shape. "
        f"Expected {expected}, got {tuple(y.shape)}"
    )

    # ---------------------------------------------------------
    # Check numerical stability.
    # ---------------------------------------------------------

    assert torch.isfinite(y).all(), (
        "Model output contains NaN or Inf."
    )

    # ---------------------------------------------------------
    # Parameter count.
    # ---------------------------------------------------------

    parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print()
    print(
        f"Trainable parameters: "
        f"{parameters:,}"
    )

    print()
    print("PASS")
    print("=" * 70)


if __name__ == "__main__":
    main()
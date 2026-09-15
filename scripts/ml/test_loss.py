import torch

from scripts.ml.losses import MaskedMSELoss


def main():

    print("=" * 70)
    print("TESTING MASKED MSE LOSS")
    print("=" * 70)

    loss_fn = MaskedMSELoss()

    # --------------------------------------------------------
    # Create a small deterministic example.
    #
    # Shape:
    # [B, depth, H, W]
    # --------------------------------------------------------
    prediction = torch.tensor(
        [
            [
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                ]
            ]
        ],
        dtype=torch.float32,
    )

    target = torch.tensor(
        [
            [
                [
                    [1.0, 0.0],
                    [5.0, 4.0],
                ]
            ]
        ],
        dtype=torch.float32,
    )

    # Valid cells:
    #
    # [ True, False ]
    # [ True, True  ]
    #
    # Errors:
    #
    # (1-1)^2 = 0
    # (2-0)^2 = ignored
    # (3-5)^2 = 4
    # (4-4)^2 = 0
    #
    # Expected:
    #
    # (0 + 4 + 0) / 3 = 4/3
    # --------------------------------------------------------
    mask = torch.tensor(
        [
            [
                [
                    [True, False],
                    [True, True],
                ]
            ]
        ],
        dtype=torch.bool,
    )

    loss = loss_fn(
        prediction,
        target,
        mask,
    )

    expected_loss = 4.0 / 3.0

    print()
    print("Basic masked-loss test:")
    print(f"  calculated: {loss.item():.6f}")
    print(f"  expected:   {expected_loss:.6f}")

    if not torch.isclose(
        loss,
        torch.tensor(expected_loss),
        atol=1e-6,
    ):
        raise ValueError("Basic masked MSE test failed.")

    # --------------------------------------------------------
    # Critical test:
    #
    # Changing an INVALID target value must NOT change the loss.
    # --------------------------------------------------------
    target_changed = target.clone()

    # This location is invalid according to the mask.
    target_changed[0, 0, 0, 1] = 100000.0

    loss_changed = loss_fn(
        prediction,
        target_changed,
        mask,
    )

    print()
    print("Invalid-cell invariance test:")
    print(f"  original loss: {loss.item():.6f}")
    print(f"  changed loss:  {loss_changed.item():.6f}")

    if not torch.isclose(
        loss,
        loss_changed,
        atol=1e-6,
    ):
        raise ValueError("Masked loss changed when an invalid target value was modified.")

    # --------------------------------------------------------
    # Test that valid cells DO affect the loss.
    # --------------------------------------------------------
    target_valid_changed = target.clone()

    # This location is valid.
    target_valid_changed[0, 0, 0, 0] = 10.0

    loss_valid_changed = loss_fn(
        prediction,
        target_valid_changed,
        mask,
    )

    print()
    print("Valid-cell sensitivity test:")
    print(f"  original loss: {loss.item():.6f}")
    print(f"  changed loss: {loss_valid_changed.item():.6f}")

    if torch.isclose(
        loss,
        loss_valid_changed,
        atol=1e-6,
    ):
        raise ValueError("Masked loss did not respond to a valid target value being changed.")

    # --------------------------------------------------------
    # Test shape validation.
    # --------------------------------------------------------
    try:
        wrong_target = torch.zeros(
            1, 2, 2, 3,
            dtype=torch.float32,
        )

        loss_fn(
            prediction,
            wrong_target,
            mask,
        )

    except ValueError:
        print()
        print("Shape validation test: passed")

    else:
        raise ValueError("Shape validation test failed.")

    # --------------------------------------------------------
    # Test mask dtype validation.
    # --------------------------------------------------------
    try:
        wrong_mask = mask.float()

        loss_fn(
            prediction,
            target,
            wrong_mask,
        )

    except TypeError:
        print("Mask dtype validation test: passed")

    else:
        raise ValueError("Mask dtype validation test failed.")

    print()
    print("=" * 70)
    print("MASKED LOSS TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
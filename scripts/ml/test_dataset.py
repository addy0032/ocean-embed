import numpy as np

from scripts.ml.dataset import (
    create_train_dataset,
    create_val_dataset,
    create_test_dataset,
)


def main():

    print("=" * 70)
    print("TESTING OCEAN DATASET")
    print("=" * 70)

    train = create_train_dataset()
    val = create_val_dataset()
    test = create_test_dataset()

    print()
    print("Dataset sizes:")
    print(f"  Train: {len(train)}")
    print(f"  Val:   {len(val)}")
    print(f"  Test:  {len(test)}")

    print()
    print("Reading first training sample...")

    sample = train[0]

    print()
    print("Sample:")
    print(f"  time: {sample['time']}")
    print(f"  x shape: {sample['x'].shape}")
    print(f"  y shape: {sample['y'].shape}")
    print(f"  mask shape: {sample['mask'].shape}")

    print()
    print("Dtypes:")
    print(f"  x:    {sample['x'].dtype}")
    print(f"  y:    {sample['y'].dtype}")
    print(f"  mask: {sample['mask'].dtype}")

    print()
    print("Input range:")
    print(f"  min: {np.nanmin(sample['x'].numpy()):.4f}")
    print(f"  max: {np.nanmax(sample['x'].numpy()):.4f}")
    
    x = sample["x"].numpy()
    print()
    print("Input ranges by variable:")
    for i, name in enumerate(train.input_names):
        values = x[:, i]

        print(
            f"  {name:10s}: "
            f"min={np.nanmin(values):8.4f}, "
            f"max={np.nanmax(values):8.4f}, "
            f"mean={np.nanmean(values):8.4f}, "
            f"std={np.nanstd(values):8.4f}"
        )

    print()
    print("Target range:")
    print(f"  min: {np.nanmin(sample['y'].numpy()):.4f}")
    print(f"  max: {np.nanmax(sample['y'].numpy()):.4f}")

    print()
    print("Mask:")
    print(f"  valid: {sample['mask'].sum().item()}")
    print(f"  total: {sample['mask'].numel()}")

    print()
    print("=" * 70)
    print("DATASET TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
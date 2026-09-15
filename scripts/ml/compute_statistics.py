"""
Compute training-only normalization statistics.

Statistics are calculated ONLY from the training period:
2015-01-01 through 2021-12-31.

Input statistics:
    mean/std for each of the 7 surface variables.

Target statistics:
    mean/std independently for each of the 15 depth levels.
"""

from pathlib import Path

import numpy as np
import xarray as xr

from config import (
    INPUT_VARIABLES,
    TRAIN_START,
    TRAIN_END,
)

X_PATH = "data/processed/X_inputs_025deg.zarr"
Y_PATH = "data/processed/y_target_025deg.zarr"

OUTPUT_PATH = Path("data/processed/ml/training_statistics.npz")


def compute_stats(da):
    """
    Compute mean and standard deviation over all dimensions
    except depth, if depth exists.

    NaNs are ignored.
    """

    if "depth" in da.dims:
        dims = [d for d in da.dims if d != "depth"]

        mean = da.mean(dim=dims, skipna=True).compute()
        std = da.std(dim=dims, skipna=True).compute()

    else:
        mean = da.mean(skipna=True).compute()
        std = da.std(skipna=True).compute()

    return mean.values, std.values


def main():

    print("=" * 70)
    print("COMPUTING TRAINING-ONLY NORMALIZATION STATISTICS")
    print("=" * 70)

    X = xr.open_zarr(X_PATH)
    Y = xr.open_zarr(Y_PATH)

    X = X.sel(time=slice(TRAIN_START, TRAIN_END))
    Y = Y.sel(time=slice(TRAIN_START, TRAIN_END))

    print()
    print("Training period:")
    print(X.time.values[0])
    print(X.time.values[-1])

    input_means = {}
    input_stds = {}

    print()
    print("INPUT STATISTICS")
    print("-" * 70)

    for name, info in INPUT_VARIABLES.items():

        print(f"Computing {name}...")

        da = X[name]

        mean, std = compute_stats(da)

        input_means[name] = float(mean)
        input_stds[name] = float(std)

        print(f"  mean = {input_means[name]:.6f}")
        print(f"  std  = {input_stds[name]:.6f}")

        if not np.isfinite(input_means[name]):
            raise ValueError(f"{name}: mean is not finite.")

        if not np.isfinite(input_stds[name]) or input_stds[name] <= 0:
            raise ValueError(f"{name}: invalid standard deviation.")

    print()
    print("TARGET STATISTICS")
    print("-" * 70)
    target_da = Y["target"]

    # Read the actual depth coordinate from the canonical target dataset.
    target_depths = target_da["depth"].values.astype(np.float32)

    if len(target_depths) != 15:
        raise ValueError(
            f"Expected 15 target depths, found {len(target_depths)}."
        )

    target_mean, target_std = compute_stats(target_da)

    print("Target means by depth:")
    for depth, mean in zip(target_depths, target_mean):
        print(f"  {depth:>6.1f} m : {mean:.6f}")

    print()
    print("Target stds by depth:")
    for depth, std in zip(target_depths, target_std):
        print(f"  {depth:>6.1f} m : {std:.6f}")

    if not np.all(np.isfinite(target_mean)):
        raise ValueError("Target contains non-finite means.")

    if not np.all(np.isfinite(target_std)):
        raise ValueError("Target contains non-finite standard deviations.")

    if np.any(target_std <= 0):
        raise ValueError("Target contains zero/negative standard deviations.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        OUTPUT_PATH,
        input_names=np.array(list(input_means.keys())),
        input_means=np.array(list(input_means.values()), dtype=np.float32),
        input_stds=np.array(list(input_stds.values()), dtype=np.float32),
        target_depths=target_depths,
        target_means=target_mean.astype(np.float32),
        target_stds=target_std.astype(np.float32),
    )

    print()
    print("=" * 70)
    print(f"Saved statistics to:")
    print(f"  {OUTPUT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
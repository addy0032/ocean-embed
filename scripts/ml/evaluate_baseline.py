import os

import numpy as np
import pandas as pd
import xarray as xr


Y_PATH = "data/processed/y_target_025deg.zarr"

OUTPUT_DIR = "evaluation"

TRAIN_START = "2015-01-01"
TRAIN_END = "2021-12-31"

TEST_START = "2023-01-01"
TEST_END = "2024-12-31"


# ============================================================
# Metrics
# ============================================================

def rmse(pred, target):
    return np.sqrt(np.mean((pred - target) ** 2))


def mae(pred, target):
    return np.mean(np.abs(pred - target))


def bias(pred, target):
    return np.mean(pred - target)


def correlation(pred, target):
    if np.std(pred) == 0 or np.std(target) == 0:
        return np.nan

    return np.corrcoef(pred, target)[0, 1]


def r2(pred, target):
    ss_res = np.sum((target - pred) ** 2)
    ss_tot = np.sum(
        (target - np.mean(target)) ** 2
    )

    if ss_tot == 0:
        return np.nan

    return 1.0 - ss_res / ss_tot


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("GLORYS CLIMATOLOGY BASELINE EVALUATION")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print("\nLoading GLORYS target data...")

    ds = xr.open_zarr(Y_PATH)

    target = ds["target"]
    depth_mask = ds["depth_valid_mask"]

    print("Target dimensions:", target.dims)
    print("Depth mask dimensions:", depth_mask.dims)

    # --------------------------------------------------------
    # Training climatology
    # --------------------------------------------------------

    print(
        f"\nCalculating climatology from "
        f"{TRAIN_START} → {TRAIN_END}..."
    )

    train = target.sel(
        time=slice(TRAIN_START, TRAIN_END)
    )

    climatology = train.mean(
        dim="time",
        skipna=True,
    )

    print(
        "Climatology shape:",
        climatology.shape,
    )

    # Convert once
    climatology_values = (
        climatology.values.astype(np.float32)
    )

    # --------------------------------------------------------
    # Test data
    # --------------------------------------------------------

    print(
        f"\nLoading test data "
        f"{TEST_START} → {TEST_END}..."
    )

    test = target.sel(
        time=slice(TEST_START, TEST_END)
    )

    target_values = (
        test.values.astype(np.float32)
    )

    # Static validity mask:
    #
    # depth × latitude × longitude
    #
    mask_values = (
        depth_mask.values.astype(bool)
    )

    depths = ds["depth"].values.astype(np.float32)

    print(
        "Test target shape:",
        target_values.shape,
    )

    print(
        "Static mask shape:",
        mask_values.shape,
    )

    # --------------------------------------------------------
    # Depth-wise metrics
    # --------------------------------------------------------

    print("\nCalculating baseline metrics...")

    results = []

    for d, depth in enumerate(depths):

        # Climatology at this depth:
        #
        # [latitude, longitude]
        baseline = climatology_values[d]

        # Test target:
        #
        # [time, latitude, longitude]
        truth = target_values[:, d]

        # Broadcast climatology across time
        prediction = np.broadcast_to(
            baseline,
            truth.shape,
        )

        # Static ocean/depth validity mask
        valid = (
            mask_values[d][None, :, :]
            & np.isfinite(truth)
            & np.isfinite(prediction)
        )

        pred_valid = prediction[valid]
        truth_valid = truth[valid]

        results.append(
            {
                "Depth_m": float(depth),
                "RMSE_C": rmse(
                    pred_valid,
                    truth_valid,
                ),
                "MAE_C": mae(
                    pred_valid,
                    truth_valid,
                ),
                "Bias_C": bias(
                    pred_valid,
                    truth_valid,
                ),
                "Correlation": correlation(
                    pred_valid,
                    truth_valid,
                ),
                "R2": r2(
                    pred_valid,
                    truth_valid,
                ),
                "N": len(pred_valid),
            }
        )

    results_df = pd.DataFrame(results)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary_metrics = {
        "Mean_Depth_RMSE_C":
            results_df["RMSE_C"].mean(),

        "Median_Depth_RMSE_C":
            results_df["RMSE_C"].median(),

        "Mean_Depth_MAE_C":
            results_df["MAE_C"].mean(),

        "Median_Depth_MAE_C":
            results_df["MAE_C"].median(),

        "Mean_Depth_Bias_C":
            results_df["Bias_C"].mean(),

        "Mean_Depth_Correlation":
            results_df["Correlation"].mean(),

        "Mean_Depth_R2":
            results_df["R2"].mean(),
    }

    summary_df = pd.DataFrame(
        [summary_metrics]
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    depth_output = (
        f"{OUTPUT_DIR}/baseline_metrics_by_depth.csv"
    )

    summary_output = (
        f"{OUTPUT_DIR}/baseline_metrics_summary.csv"
    )

    results_df.to_csv(
        depth_output,
        index=False,
    )

    summary_df.to_csv(
        summary_output,
        index=False,
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("BASELINE DEPTH-WISE RESULTS")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print("=" * 70)
    print("BASELINE SUMMARY")
    print("=" * 70)

    for key, value in summary_metrics.items():
        print(f"{key:30s}: {value:.4f}")

    print()
    print(f"Saved: {depth_output}")
    print(f"Saved: {summary_output}")


if __name__ == "__main__":
    main()
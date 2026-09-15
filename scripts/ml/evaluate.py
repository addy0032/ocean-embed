from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from scripts.ml.dataset import create_test_dataset
from scripts.ml.models.temporal_unet import TemporalUNet


# ============================================================
# Configuration
# ============================================================

CHECKPOINT_PATH = Path(
    "checkpoints/fully_trained_temporal_unet_best_32batch.pt"
)

STATS_PATH = Path(
    "data/processed/ml/training_statistics.npz"
)

OUTPUT_DIR = Path(
    "evaluation"
)

BATCH_SIZE = 16
NUM_WORKERS = 8


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(prediction, target, mask):
    """
    Calculate regression metrics over valid cells.
    """

    prediction = prediction[mask]
    target = target[mask]

    error = prediction - target

    mse = np.mean(error ** 2)
    rmse = np.sqrt(mse)

    mae = np.mean(np.abs(error))

    bias = np.mean(error)

    # Pearson correlation
    target_centered = target - np.mean(target)
    prediction_centered = prediction - np.mean(prediction)

    denominator = (
        np.sqrt(
            np.sum(target_centered ** 2)
            * np.sum(prediction_centered ** 2)
        )
    )

    if denominator == 0:
        correlation = np.nan
    else:
        correlation = (
            np.sum(
                target_centered
                * prediction_centered
            )
            / denominator
        )

    # Coefficient of determination
    ss_res = np.sum((target - prediction) ** 2)
    ss_tot = np.sum(
        (target - np.mean(target)) ** 2
    )

    if ss_tot == 0:
        r2 = np.nan
    else:
        r2 = 1.0 - (ss_res / ss_tot)

    return {
        "RMSE_C": rmse,
        "MAE_C": mae,
        "Bias_C": bias,
        "Correlation": correlation,
        "R2": r2,
        "N": len(target),
    }


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("OCEANEMBED MODEL EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # --------------------------------------------------------
    # Load test dataset
    # --------------------------------------------------------

    print()
    print("Loading test dataset...")

    test_dataset = create_test_dataset()

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(NUM_WORKERS > 0),
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
    )

    # --------------------------------------------------------
    # Load normalization statistics
    # --------------------------------------------------------

    stats = np.load(STATS_PATH)
    target_means = stats["target_means"].astype(np.float32)
    target_stds = stats["target_stds"].astype(np.float32)
    depths = test_dataset.depths.astype(np.float32)

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    print()
    print("Creating model...")

    model = TemporalUNet(
        in_channels=7,
        out_channels=15,
        base_channels=32,
    ).to(device)

    # --------------------------------------------------------
    # Load best checkpoint
    # --------------------------------------------------------

    print()
    print("Loading checkpoint...")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(f"Checkpoint validation loss: {checkpoint['val_loss']:.6f}")

    # --------------------------------------------------------
    # Accumulate predictions and targets
    # --------------------------------------------------------

    predictions_by_depth = [[] for _ in range(len(depths))]
    targets_by_depth = [[] for _ in range(len(depths))]
    masks_by_depth = [[] for _ in range(len(depths))]

    print()
    print("Running inference...")

    with torch.no_grad():
        for batch in test_loader:

            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].cpu().numpy()
            mask = batch["mask"].numpy()

            prediction = model(x).float().cpu().numpy()

            # Inverse normalization
            prediction = (
                prediction
                * target_stds[None, :, None, None]
                + target_means[None, :, None, None]
            )

            y = (
                y
                * target_stds[None, :, None, None]
                + target_means[None, :, None, None]
            )

            for d in range(len(depths)):
                predictions_by_depth[d].append(prediction[:, d])
                targets_by_depth[d].append(y[:, d])
                masks_by_depth[d].append(mask[:, d])

    # --------------------------------------------------------
    # Calculate depth-wise metrics
    # --------------------------------------------------------

    print()
    print("Calculating metrics...")

    results = []

    for d, depth in enumerate(depths):

        prediction = np.concatenate(predictions_by_depth[d], axis=0)
        target = np.concatenate(targets_by_depth[d], axis=0)
        mask = np.concatenate(masks_by_depth[d], axis=0)

        metrics = calculate_metrics(prediction, target, mask)
        metrics["Depth_m"] = float(depth)
        results.append(metrics)

    results_df = pd.DataFrame(results)

    results_df = results_df[
        ["Depth_m", "RMSE_C", "MAE_C", "Bias_C", "Correlation", "R2", "N"]
    ]

    # --------------------------------------------------------
    # Overall metrics
    # --------------------------------------------------------

    all_predictions = []
    all_targets = []

    for d in range(len(depths)):
        prediction = np.concatenate(predictions_by_depth[d], axis=0)
        target = np.concatenate(targets_by_depth[d], axis=0)
        mask = np.concatenate(masks_by_depth[d], axis=0)

        all_predictions.append(prediction[mask])
        all_targets.append(target[mask])

    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)
    overall_mask = np.ones(len(all_predictions), dtype=bool)

    overall_metrics = calculate_metrics(
        all_predictions,
        all_targets,
        overall_mask,
    )

    # --------------------------------------------------------
    # SUMMARY ACROSS DEPTHS
    # --------------------------------------------------------
    
    summary_metrics = {
        "Mean_Depth_RMSE_C": results_df["RMSE_C"].mean(),
        "Median_Depth_RMSE_C": results_df["RMSE_C"].median(),
        "Mean_Depth_MAE_C": results_df["MAE_C"].mean(),
        "Median_Depth_MAE_C": results_df["MAE_C"].median(),
        "Mean_Depth_Bias_C": results_df["Bias_C"].mean(),
        "Mean_Depth_Correlation": results_df["Correlation"].mean(),
        "Mean_Depth_R2": results_df["R2"].mean(),
    }

    summary_df = pd.DataFrame([summary_metrics])

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("DEPTH-WISE RESULTS")
    print("=" * 70)
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print()
    print("=" * 70)
    print("OVERALL RESULTS")
    print("=" * 70)
    print(f"RMSE        : {overall_metrics['RMSE_C']:.4f} °C")
    print(f"MAE         : {overall_metrics['MAE_C']:.4f} °C")
    print(f"Bias        : {overall_metrics['Bias_C']:.4f} °C")
    print(f"Correlation : {overall_metrics['Correlation']:.4f}")
    print(f"R²          : {overall_metrics['R2']:.4f}")
    print(f"Valid cells : {len(all_predictions):,}")

    print()
    print("=" * 70)
    print("DEPTH-WISE SUMMARY")
    print("=" * 70)
    for key, value in summary_metrics.items():
        print(f"{key:25s}: {value:.4f}")

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    depth_output = OUTPUT_DIR / "test_metrics_by_depth.csv"
    overall_output = OUTPUT_DIR / "test_metrics_overall.csv"
    summary_output = OUTPUT_DIR / "test_metrics_summary.csv"

    results_df.to_csv(depth_output, index=False)
    pd.DataFrame([overall_metrics]).to_csv(overall_output, index=False)
    summary_df.to_csv(summary_output, index=False)

    print()
    print(f"Depth metrics saved to:   {depth_output}")
    print(f"Overall metrics saved to: {overall_output}")
    print(f"Summary metrics saved to: {summary_output}")


if __name__ == "__main__":
    main()
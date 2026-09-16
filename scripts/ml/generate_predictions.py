"""
Generate offline predictions for OceanEmbed test dataset.

Loads trained Temporal U-Net checkpoint, runs batched inference over all test
samples returned by create_test_dataset(), inverse-normalizes raw model predictions
back to degrees Celsius, and saves to an xarray Dataset in evaluation/predictions.zarr.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import time

import numpy as np
import torch
from torch.utils.data import DataLoader
import xarray as xr

from config import (
    TIME_WINDOW,
    TEST_START,
    TEST_END,
)
from scripts.ml.dataset import create_test_dataset
from scripts.ml.models.temporal_unet import TemporalUNet


# ============================================================
# Paths & Configuration
# ============================================================

CHECKPOINT_PATH = Path("checkpoints/fully_trained_temporal_unet_best_32batch.pt")
STATS_PATH = Path("data/processed/ml/training_statistics.npz")
TARGET_ZARR_PATH = Path("data/processed/y_target_025deg.zarr")
OUTPUT_ZARR_PATH = Path("evaluation/predictions.zarr")

BATCH_SIZE = 16
NUM_WORKERS = 4


def generate_predictions():
    print("=" * 70)
    print("OCEANEMBED OFFLINE PREDICTION GENERATION")
    print("=" * 70)

    # 1. Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Inference device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 2. Checkpoint check
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {CHECKPOINT_PATH}")

    # 3. Load test dataset
    print(f"\nLoading test dataset ({TEST_START} to {TEST_END})...")
    test_dataset = create_test_dataset()
    num_samples = len(test_dataset)
    print(f"Total test samples to infer: {num_samples}")

    # Extract target dates and coordinates directly from test dataset / target zarr
    test_times = test_dataset.times[test_dataset.indices]
    depths = test_dataset.depths
    target_ds = test_dataset.Y
    latitudes = target_ds["latitude"].values
    longitudes = target_ds["longitude"].values

    print(f"Time range: {test_times[0]} -> {test_times[-1]}")
    print(f"Target depths ({len(depths)}): {depths.tolist()}")
    print(f"Grid shape: Lat={len(latitudes)}, Lon={len(longitudes)}")

    # 4. Normalization statistics
    print(f"\nLoading normalization statistics from: {STATS_PATH}")
    stats = np.load(STATS_PATH)
    target_means = stats["target_means"].astype(np.float32)  # [15]
    target_stds = stats["target_stds"].astype(np.float32)    # [15]

    # Reshape for broadcasting [1, 15, 1, 1]
    target_means_b = target_means.reshape(1, -1, 1, 1)
    target_stds_b = target_stds.reshape(1, -1, 1, 1)

    # 5. Load model
    print("\nInitializing TemporalUNet...")
    model = TemporalUNet(
        in_channels=7,
        out_channels=15,
        base_channels=32,
    ).to(device)

    print(f"Loading checkpoint weights from: {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    if "epoch" in checkpoint:
        print(f"Loaded model trained to epoch: {checkpoint['epoch']}")
    if "val_loss" in checkpoint:
        print(f"Checkpoint val_loss: {checkpoint['val_loss']:.6f}")

    # 6. Run batched inference
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    print("\nRunning inference...")
    start_time = time.time()

    all_predictions = []
    processed_count = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            x = batch["x"].to(device, non_blocking=True)
            pred_norm = model(x).float().cpu().numpy()  # [B, 15, H, W]

            # Inverse-normalize raw model prediction back to °C
            # DO NOT apply mask to raw prediction array
            pred_celsius = pred_norm * target_stds_b + target_means_b

            all_predictions.append(pred_celsius.astype(np.float32))
            processed_count += len(x)

            percent = (processed_count / num_samples) * 100
            bar = ("=" * int(percent // 5)).ljust(20)
            sys.stdout.write(f"\r[{bar}] {processed_count}/{num_samples} ({percent:.1f}%)")
            sys.stdout.flush()

    sys.stdout.write("\n")
    elapsed = time.time() - start_time
    print(f"Inference completed in {elapsed:.2f}s ({num_samples / elapsed:.1f} samples/sec)")

    pred_array = np.concatenate(all_predictions, axis=0)
    print(f"Combined predictions shape: {pred_array.shape}, dtype: {pred_array.dtype}")

    # 7. Construct xarray Dataset
    print("\nConstructing xarray Dataset for Zarr storage...")
    pred_da = xr.DataArray(
        data=pred_array,
        dims=["time", "depth", "latitude", "longitude"],
        coords={
            "time": test_times,
            "depth": depths,
            "latitude": latitudes,
            "longitude": longitudes,
        },
        name="prediction",
        attrs={
            "units": "degC",
            "long_name": "Reconstructed Subsurface Ocean Temperature",
            "standard_name": "sea_water_potential_temperature",
        },
    )

    pred_ds = xr.Dataset(
        data_vars={"prediction": pred_da},
        attrs={
            "title": "OceanEmbed Subsurface Temperature Predictions",
            "model_name": "TemporalUNet",
            "model_checkpoint": str(CHECKPOINT_PATH),
            "test_start": str(test_times[0]),
            "test_end": str(test_times[-1]),
            "time_window": int(TIME_WINDOW),
            "spatial_resolution": "0.25 degree",
            "units": "degC",
            "generated_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        },
    )

    # 8. Configure chunking
    # chunk sizes: time: 1, depth: -1 (full 15), latitude: -1 (full), longitude: -1 (full)
    encoding = {
        "prediction": {
            "chunks": (1, len(depths), len(latitudes), len(longitudes)),
            "dtype": "float32",
        }
    }

    OUTPUT_ZARR_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing prediction Zarr to: {OUTPUT_ZARR_PATH} ...")
    pred_ds.to_zarr(OUTPUT_ZARR_PATH, mode="w", encoding=encoding)
    print("Zarr write complete!")

    # 9. Automated Validation against Source Dataset
    validate_zarr(OUTPUT_ZARR_PATH, test_dataset, target_ds)


def validate_zarr(zarr_path: Path, test_dataset, target_ds):
    print("\n" + "=" * 70)
    print("ZARR VALIDATION REPORT")
    print("=" * 70)

    ds = xr.open_zarr(zarr_path)
    pred = ds["prediction"]

    # Check 1: Dimensions
    expected_shape = (
        len(test_dataset),
        len(test_dataset.depths),
        len(target_ds["latitude"]),
        len(target_ds["longitude"]),
    )
    actual_shape = pred.shape
    print(f"Shape: {actual_shape} (Expected: {expected_shape})")
    assert actual_shape == expected_shape, f"Shape mismatch: {actual_shape} != {expected_shape}"

    # Check 2: Time coordinate exact match
    expected_times = test_dataset.times[test_dataset.indices]
    actual_times = ds["time"].values
    assert len(actual_times) == len(expected_times), "Time coordinate length mismatch"
    assert np.array_equal(actual_times, expected_times), "Time coordinates do not match test dataset"
    print(f"Time coordinate: verified {len(actual_times)} dates ({actual_times[0]} to {actual_times[-1]})")

    # Check 3: Depth coordinate exact match
    expected_depths = test_dataset.depths
    actual_depths = ds["depth"].values
    assert np.allclose(actual_depths, expected_depths), "Depth coordinates mismatch"
    print(f"Depth coordinate: verified {len(actual_depths)} levels ({actual_depths.tolist()})")

    # Check 4: Latitude and Longitude match source dataset
    expected_lat = target_ds["latitude"].values
    expected_lon = target_ds["longitude"].values
    assert np.allclose(ds["latitude"].values, expected_lat), "Latitude coordinates mismatch"
    assert np.allclose(ds["longitude"].values, expected_lon), "Longitude coordinates mismatch"
    print(f"Latitude coordinate: verified {len(ds['latitude'])} values matching source target Zarr")
    print(f"Longitude coordinate: verified {len(ds['longitude'])} values matching source target Zarr")

    # Check 5: Numerical validity (check for inf / unphysical blowups)
    vals = pred.values
    inf_count = np.isinf(vals).sum()
    print(f"Infinite values count: {inf_count}")
    assert inf_count == 0, f"Found {inf_count} infinite values in predictions!"

    valid_vals = vals[~np.isnan(vals)]
    assert len(valid_vals) > 0, "All values are NaN!"
    print(f"Prediction range: Min = {valid_vals.min():.2f} deg C, Max = {valid_vals.max():.2f} deg C, Mean = {valid_vals.mean():.2f} deg C")

    # Check 6: Attributes
    assert "model_name" in ds.attrs
    assert ds.attrs["units"] == "degC"
    print("Metadata attributes: verified successfully.")

    print("=" * 70)
    print("ALL VALIDATION CHECKS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    generate_predictions()

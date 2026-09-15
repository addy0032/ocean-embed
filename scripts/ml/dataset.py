"""
PyTorch Dataset for OceanEmbed.

Creates temporal samples from the canonical processed Zarr datasets.

For TIME_WINDOW = 7:

    X: [7, 7, 101, 241]
    Y: [15, 101, 241]
    M: [15, 101, 241]

The target date is the final day of the input sequence.

Example:

    Input:
        t-6, t-5, t-4, t-3, t-2, t-1, t

    Target:
        t
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
import xarray as xr

from config import (
    INPUT_VARIABLES,
    TIME_WINDOW,
    TRAIN_START,
    TRAIN_END,
    VAL_START,
    VAL_END,
    TEST_START,
    TEST_END,
)


X_PATH = "data/processed/X_inputs_025deg_t7.zarr"
Y_PATH = "data/processed/y_target_025deg.zarr"
SURFACE_MASK_PATH = "data/processed/surface_mask_025deg.zarr"
STATS_PATH = "data/processed/ml/training_statistics.npz"


class OceanDataset(Dataset):
    """
    Dataset for one temporal split.

    Each sample contains:

        x:
            [TIME_WINDOW, 7, latitude, longitude]

        y:
            [15, latitude, longitude]

        mask:
            [15, latitude, longitude]

        time:
            target date
    """

    def __init__(
        self,
        start_date,
        end_date,
        time_window=TIME_WINDOW,
        x_path=X_PATH,
        y_path=Y_PATH,
        stats_path=STATS_PATH,
    ):
        self.start_date = np.datetime64(start_date, "ns")
        self.end_date = np.datetime64(end_date, "ns")

        self.time_window = time_window

        # --------------------------------------------------------
        # Open Zarr stores lazily.
        # --------------------------------------------------------
        self.X = xr.open_zarr(x_path)
        self.Y = xr.open_zarr(y_path)

        # --------------------------------------------------------
        # Open static master surface mask.
        #
        # [latitude, longitude]
        # --------------------------------------------------------
        self.surface_mask_ds = xr.open_zarr(SURFACE_MASK_PATH)
        self.surface_mask = self.surface_mask_ds["surface_mask"].values.astype(bool)


        # --------------------------------------------------------
        # Verify master surface mask shape.
        #
        # [latitude, longitude]
        # --------------------------------------------------------
        expected_shape = (
            len(self.Y["latitude"]),
            len(self.Y["longitude"]),
        )

        if self.surface_mask.shape != expected_shape:
            raise ValueError(
                "Surface mask shape mismatch.\n"
                f"Expected: {expected_shape}\n"
                f"Found:    {self.surface_mask.shape}"
            )


        # Loading depth mask once
        self.depth_mask = (self.Y["depth_valid_mask"].values.astype(bool))

        # Computing loss mask once
        self.loss_mask = (self.surface_mask[None, :, :]& self.depth_mask)

        # --------------------------------------------------------
        # Load normalization statistics.
        # --------------------------------------------------------
        stats = np.load(stats_path)

        self.input_means = stats["input_means"].astype(np.float32)
        self.input_stds = stats["input_stds"].astype(np.float32)

        self.target_means = stats["target_means"].astype(np.float32)
        self.target_stds = stats["target_stds"].astype(np.float32)

        # --------------------------------------------------------
        # Variable ordering.
        #
        # This MUST remain consistent between training and inference.
        # --------------------------------------------------------
        self.input_names = list(INPUT_VARIABLES.keys())

        if len(self.input_names) != 7:
            raise ValueError(
                f"Expected 7 input variables, found {len(self.input_names)}."
            )

        if len(self.input_means) != 7:
            raise ValueError(
                f"Expected 7 input means, found {len(self.input_means)}."
            )

        if len(self.input_stds) != 7:
            raise ValueError(
                f"Expected 7 input stds, found {len(self.input_stds)}."
            )

        # --------------------------------------------------------
        # Make sure statistics match variable order.
        # --------------------------------------------------------
        stats_names = [str(x) for x in stats["input_names"]]

        if stats_names != self.input_names:
            raise ValueError(
                "Input variable ordering mismatch.\n"
                f"Config: {self.input_names}\n"
                f"Stats:  {stats_names}"
            )

        # --------------------------------------------------------
        # Verify depth dimension.
        # --------------------------------------------------------
        self.depths = self.Y["depth"].values.astype(np.float32)

        if len(self.depths) != 15:
            raise ValueError(
                f"Expected 15 target depths, found {len(self.depths)}."
            )

        if len(self.target_means) != 15:
            raise ValueError(
                "Target statistics do not contain 15 depth values."
            )

        if len(self.target_stds) != 15:
            raise ValueError(
                "Target statistics do not contain 15 standard deviations."
            )

        # --------------------------------------------------------
        # Verify time coordinate.
        # --------------------------------------------------------
        x_times = self.X["time"].values.astype("datetime64[ns]")
        y_times = self.Y["time"].values.astype("datetime64[ns]")

        if not np.array_equal(x_times, y_times):
            raise ValueError("X and Y time coordinates do not match.")

        self.times = x_times

        # --------------------------------------------------------
        # Construct valid target indices.
        #
        # We define the split by TARGET DATE.
        #
        # For T=7, target day t requires:
        #     t-6 ... t
        #
        # Therefore the first 6 global days cannot be targets.
        # --------------------------------------------------------
        valid_indices = []

        for i, t in enumerate(self.times):

            if t < self.start_date or t > self.end_date:
                continue

            start_idx = i - self.time_window + 1

            if start_idx < 0:
                continue

            valid_indices.append(i)

        self.indices = np.asarray(valid_indices, dtype=np.int64)

        if len(self.indices) == 0:
            raise ValueError(
                f"No valid samples found between "
                f"{start_date} and {end_date}."
            )

        print(
            f"OceanDataset: {start_date} → {end_date} | "
            f"{len(self.indices)} samples | "
            f"T={self.time_window}"
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):

        target_idx = int(self.indices[idx])

        start_idx = target_idx - self.time_window + 1
        end_idx = target_idx + 1

        # --------------------------------------------------------
        # Read temporal input window.
        #
        # Each variable:
        #     [time, latitude, longitude]
        #
        # Result:
        #     [T, latitude, longitude]
        # --------------------------------------------------------
        input_arrays = []

        for name in self.input_names:

            da = self.X[name].isel(
                time=slice(start_idx, end_idx)
            )

            arr = da.values.astype(np.float32)

            input_arrays.append(arr)

        # --------------------------------------------------------
        # Stack variables:
        #
        # before:
        #     7 × [T, H, W]
        #
        # after:
        #     [T, 7, H, W]
        # --------------------------------------------------------
        x = np.stack(input_arrays, axis=1)

        # --------------------------------------------------------
        # Normalize inputs channel-by-channel.
        # --------------------------------------------------------
        input_means = self.input_means.reshape(1, -1, 1, 1)
        input_stds = self.input_stds.reshape(1, -1, 1, 1)

        x = (x - input_means) / input_stds

        # Replace invalid input values with zero in
        # standardized space.
        x = np.nan_to_num(x, nan=0.0)

        # --------------------------------------------------------
        # Read target for target day only.
        #
        # [depth, latitude, longitude]
        # --------------------------------------------------------
        y = self.Y["target"].isel(time=target_idx).values.astype(np.float32)

        # --------------------------------------------------------
        # Read static depth-valid mask.
        #
        # [depth, latitude, longitude]
        # --------------------------------------------------------
        depth_mask = self.depth_mask

        # --------------------------------------------------------
        # Combine:
        #
        # surface_mask:
        #     [latitude, longitude]
        #
        # depth_mask:
        #     [depth, latitude, longitude]
        #
        # loss_mask:
        #     [depth, latitude, longitude]
        #
        # A target value is usable only when:
        #   1. all surface inputs are valid at that location
        #   2. GLORYS target is valid at that depth/location
        # --------------------------------------------------------
        loss_mask = self.loss_mask

        # --------------------------------------------------------
        # Target normalization, independently per depth.
        # --------------------------------------------------------
        target_means = self.target_means.reshape(-1, 1, 1)
        target_stds = self.target_stds.reshape(-1, 1, 1)

        y = (y - target_means) / target_stds

        # Replace invalid target values with zero in
        # standardized space.
        y = np.nan_to_num(y, nan=0.0)

        # --------------------------------------------------------
        # Convert to PyTorch tensors.
        # --------------------------------------------------------
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        mask = torch.from_numpy(loss_mask)

        target_time = self.times[target_idx]

        return {
            "x": x,
            "y": y,
            "mask": mask,
            "time": str(target_time),
        }


def create_train_dataset():
    return OceanDataset(
        start_date=TRAIN_START,
        end_date=TRAIN_END,
    )


def create_val_dataset():
    return OceanDataset(
        start_date=VAL_START,
        end_date=VAL_END,
    )


def create_test_dataset():
    return OceanDataset(
        start_date=TEST_START,
        end_date=TEST_END,
    )
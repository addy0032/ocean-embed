"""
Independent in-situ validation of model predictions against Argo floats.

Argo profiles (scripts/download/download_argo.py) are quality controlled,
converted from pressure to depth (TEOS-10), interpolated onto the 15 model
depth levels and matched to the nearest 0.25 degree grid cell on the same
UTC day. The same matchups are also scored for GLORYS, which shows how much
of the error is inherited from the reanalysis the model was trained on.

QC:
    - position_qc and time_qc == 1
    - data_mode D/A: pres_adjusted / temp_adjusted with QC == 1
      data_mode R:   pres / temp with QC == 1
    - levels shallower than the first measurement are filled with it only
      when that measurement is within SURFACE_FILL_MAX_M (mixed layer)
    - a level is interpolated only if the bracketing measurements are close
      enough (max_gap), so sparse profiles are not smeared
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gsw
import numpy as np
import pandas as pd
import xarray as xr

from scripts.ml.dataset import Y_PATH
from scripts.ml.evaluate import calculate_metrics


ARGO_PATH = Path("data/raw/argo/argo_temp_2023-06_2023-12.csv")
PREDICTIONS_PATH = Path("evaluation/predictions.zarr")
OUTPUT_DIR = Path("evaluation/argo")

SURFACE_FILL_MAX_M = 5.0
MIN_LEVELS_PER_PROFILE = 10


def max_gap(depth):
    """Largest allowed spacing between bracketing measurements at a given depth."""
    return np.maximum(20.0, 0.3 * depth)


# ============================================================
# Argo loading + QC
# ============================================================

def load_argo(path):
    df = pd.read_csv(path, dtype=str)

    adjusted = df["data_mode"].isin(["D", "A"])

    pres = np.where(adjusted, df["pres_adjusted"], df["pres"]).astype(float)
    temp = np.where(adjusted, df["temp_adjusted"], df["temp"]).astype(float)
    pres_qc = np.where(adjusted, df["pres_adjusted_qc"], df["pres_qc"])
    temp_qc = np.where(adjusted, df["temp_adjusted_qc"], df["temp_qc"])

    good = (
        (df["position_qc"] == "1")
        & (df["time_qc"] == "1")
        & (pres_qc == "1")
        & (temp_qc == "1")
        & np.isfinite(pres)
        & np.isfinite(temp)
    )

    out = pd.DataFrame({
        "platform_number": df["platform_number"],
        "cycle_number": df["cycle_number"],
        "direction": df["direction"],
        "data_mode": df["data_mode"],
        "time": pd.to_datetime(df["time"]).dt.tz_localize(None),
        "latitude": df["latitude"].astype(float),
        "longitude": df["longitude"].astype(float),
        "pres": pres,
        "temp": temp,
    })[good]

    print(f"Argo levels: {len(df):,} raw -> {len(out):,} after QC")
    return out


def interpolate_profile(depth, temp, levels):
    order = np.argsort(depth)
    depth, temp = depth[order], temp[order]
    depth, unique = np.unique(depth, return_index=True)
    temp = temp[unique]

    result = np.full(len(levels), np.nan)
    if len(depth) < MIN_LEVELS_PER_PROFILE:
        return result

    idx = np.searchsorted(depth, levels)
    inside = (idx > 0) & (idx < len(depth))
    lo = np.clip(idx - 1, 0, len(depth) - 1)
    hi = np.clip(idx, 0, len(depth) - 1)
    ok = inside & ((depth[hi] - depth[lo]) <= max_gap(levels))

    result[ok] = np.interp(levels, depth, temp)[ok]

    # exact hits on a measured level
    exact = depth[hi] == levels
    result[exact] = temp[hi][exact]

    if depth[0] <= SURFACE_FILL_MAX_M:
        result[levels < depth[0]] = temp[0]

    return result


def build_profiles(levels_df, levels):
    keys = ["platform_number", "cycle_number", "direction"]
    rows = []

    for key, prof in levels_df.groupby(keys, sort=False):
        lat = prof["latitude"].iloc[0]
        depth = -gsw.z_from_p(prof["pres"].to_numpy(), lat)
        temps = interpolate_profile(depth, prof["temp"].to_numpy(), levels)

        if np.isfinite(temps).any():
            rows.append((*key, prof["data_mode"].iloc[0], prof["time"].iloc[0],
                         lat, prof["longitude"].iloc[0], temps))

    profiles = pd.DataFrame(rows, columns=[*keys, "data_mode", "time", "latitude", "longitude", "temps"])
    print(f"Argo profiles with at least one usable level: {len(profiles):,}")
    return profiles


# ============================================================
# Colocation
# ============================================================

def colocate(profiles, predictions, target, valid_mask):
    lat = predictions["latitude"].values
    lon = predictions["longitude"].values
    times = predictions["time"].values

    day = profiles["time"].dt.floor("D").to_numpy()
    in_period = np.isin(day, times)
    profiles = profiles[in_period].reset_index(drop=True)
    day = day[in_period]

    t_idx = np.searchsorted(times, day)
    lat_idx = np.abs(lat[None, :] - profiles["latitude"].to_numpy()[:, None]).argmin(axis=1)
    lon_idx = np.abs(lon[None, :] - profiles["longitude"].to_numpy()[:, None]).argmin(axis=1)

    point = dict(
        time=xr.DataArray(t_idx, dims="profile"),
        latitude=xr.DataArray(lat_idx, dims="profile"),
        longitude=xr.DataArray(lon_idx, dims="profile"),
    )

    model = predictions.isel(point).transpose("profile", "depth").values
    glorys = target.isel(point).transpose("profile", "depth").values
    mask = valid_mask.values[:, lat_idx, lon_idx].T

    return profiles, model, glorys, mask


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("OCEANEMBED ARGO IN-SITU VALIDATION")
    print("=" * 70)

    predictions = xr.open_zarr(PREDICTIONS_PATH)["prediction"]
    target_ds = xr.open_zarr(Y_PATH)
    target = target_ds["target"].sel(time=predictions["time"])
    valid_mask = target_ds["depth_valid_mask"]
    levels = predictions["depth"].values.astype(float)

    profiles = build_profiles(load_argo(ARGO_PATH), levels)
    profiles, model, glorys, mask = colocate(profiles, predictions, target, valid_mask)

    argo = np.stack(profiles["temps"].to_numpy())
    valid = mask & np.isfinite(argo) & np.isfinite(model) & np.isfinite(glorys)

    print(f"Matched profiles: {valid.any(axis=1).sum():,}")
    print(f"Matched profile levels: {valid.sum():,}")

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    by_depth = []
    for d, depth in enumerate(levels):
        for source, values in (("Model", model), ("GLORYS", glorys)):
            m = calculate_metrics(values[:, d], argo[:, d], valid[:, d])
            by_depth.append({"Source": source, "Depth_m": depth, **m})

    by_depth_df = pd.DataFrame(by_depth)[
        ["Source", "Depth_m", "RMSE_C", "MAE_C", "Bias_C", "Correlation", "R2", "N"]
    ]

    overall_df = pd.DataFrame([
        {"Source": source, **calculate_metrics(values, argo, valid)}
        for source, values in (("Model", model), ("GLORYS", glorys))
    ])

    summary_df = by_depth_df.groupby("Source", sort=False).agg(
        Mean_Depth_RMSE_C=("RMSE_C", "mean"),
        Median_Depth_RMSE_C=("RMSE_C", "median"),
        Mean_Depth_MAE_C=("MAE_C", "mean"),
        Mean_Depth_Bias_C=("Bias_C", "mean"),
        Mean_Depth_Correlation=("Correlation", "mean"),
        Mean_Depth_R2=("R2", "mean"),
    ).reset_index()

    # Per-level matchups, for plots / further analysis
    p, d = np.nonzero(valid)
    matchups = pd.DataFrame({
        "platform_number": profiles["platform_number"].to_numpy()[p],
        "cycle_number": profiles["cycle_number"].to_numpy()[p],
        "data_mode": profiles["data_mode"].to_numpy()[p],
        "time": profiles["time"].to_numpy()[p],
        "latitude": profiles["latitude"].to_numpy()[p],
        "longitude": profiles["longitude"].to_numpy()[p],
        "depth_m": levels[d],
        "argo_temp_C": argo[p, d],
        "model_temp_C": model[p, d],
        "glorys_temp_C": glorys[p, d],
    })

    # --------------------------------------------------------
    # Print + save
    # --------------------------------------------------------

    fmt = lambda x: f"{x:.4f}"
    print()
    print("=" * 70)
    print("DEPTH-WISE RESULTS (vs Argo)")
    print("=" * 70)
    print(by_depth_df.to_string(index=False, float_format=fmt))
    print()
    print("=" * 70)
    print("OVERALL RESULTS (vs Argo)")
    print("=" * 70)
    print(overall_df.to_string(index=False, float_format=fmt))
    print()
    print(summary_df.to_string(index=False, float_format=fmt))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    by_depth_df.to_csv(OUTPUT_DIR / "argo_metrics_by_depth.csv", index=False)
    overall_df.to_csv(OUTPUT_DIR / "argo_metrics_overall.csv", index=False)
    summary_df.to_csv(OUTPUT_DIR / "argo_metrics_summary.csv", index=False)
    matchups.to_csv(OUTPUT_DIR / "argo_matchups.csv", index=False)
    print(f"\nResults saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()

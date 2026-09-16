"""
OceanEmbed Dashboard - FastAPI Backend Server

Provides REST API endpoints for the React Three.js interactive dashboard:
- GET /api/metadata: Available dates, depths, grid coordinates, and model specs
- GET /api/field: Single depth/time 2D temperature or error field
- GET /api/compare: Synchronized prediction, GLORYS target, and error fields
- GET /api/metrics: Test-set evaluation metrics (overall, summary, depth-wise)
- GET /api/health: Service readiness status

NOTE: This server NEVER imports or runs PyTorch inference. It reads directly
and lazily from evaluation/predictions.zarr and data/processed/y_target_025deg.zarr.
"""

from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Any
from pathlib import Path
import json

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd
import xarray as xr

from dashboard.backend.config import (
    PREDICTIONS_ZARR_PATH,
    TARGET_ZARR_PATH,
    SURFACE_MASK_PATH,
    EVALUATION_DIR,
    CORS_ORIGINS,
)

# Global dataset references & cached metadata
STATE: Dict[str, Any] = {
    "pred_ds": None,
    "target_ds": None,
    "loss_mask": None,         # [15, 101, 241] boolean
    "depths": [],              # List[float]
    "depth_indices": {},       # Map[float, int]
    "dates": [],               # List[str] "YYYY-MM-DD"
    "date_indices": {},        # Map[str, int]
    "latitudes": [],           # List[float]
    "longitudes": [],          # List[float]
    "metrics": {},             # Pre-loaded evaluation metrics
    "metadata": {},            # Pre-computed metadata response
}


def load_metrics_data(eval_dir: Path) -> Dict[str, Any]:
    """Find and load evaluation CSV metrics produced by the pipeline."""
    metrics_result = {
        "overall": None,
        "summary": None,
        "by_depth": []
    }

    # 1. By depth
    depth_candidates = [
        eval_dir / "test_metrics_by_depth_7day_32batch.csv",
        eval_dir / "test_metrics_by_depth.csv",
    ]
    for p in depth_candidates:
        if p.exists():
            df = pd.read_csv(p)
            metrics_result["by_depth"] = df.to_dict(orient="records")
            break

    # 2. Overall
    overall_candidates = [
        eval_dir / "test_metrics_overall_7day_32batch.csv",
        eval_dir / "test_metrics_overall.csv",
    ]
    for p in overall_candidates:
        if p.exists():
            df = pd.read_csv(p)
            records = df.to_dict(orient="records")
            if records:
                metrics_result["overall"] = records[0]
            break

    # 3. Summary across depths
    summary_candidates = [
        eval_dir / "test_metrics_summary_7day_32batch.csv",
        eval_dir / "test_metrics_summary.csv",
    ]
    for p in summary_candidates:
        if p.exists():
            df = pd.read_csv(p)
            records = df.to_dict(orient="records")
            if records:
                metrics_result["summary"] = records[0]
            break

    return metrics_result


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup verification
    print("=" * 70)
    print("STARTING OCEANEMBED DASHBOARD BACKEND")
    print("=" * 70)

    if not PREDICTIONS_ZARR_PATH.exists():
        msg = (
            f"Required prediction store not found at '{PREDICTIONS_ZARR_PATH}'!\n"
            "You must run offline inference first:\n"
            "    uv run python scripts/ml/generate_predictions.py\n"
        )
        print(f"ERROR: {msg}")
        raise RuntimeError(msg)

    if not TARGET_ZARR_PATH.exists():
        raise FileNotFoundError(f"Target Zarr not found at '{TARGET_ZARR_PATH}'")

    print(f"Opening prediction store: {PREDICTIONS_ZARR_PATH}")
    pred_ds = xr.open_zarr(PREDICTIONS_ZARR_PATH)

    print(f"Opening target store:     {TARGET_ZARR_PATH}")
    target_ds = xr.open_zarr(TARGET_ZARR_PATH)

    print(f"Opening surface mask:     {SURFACE_MASK_PATH}")
    surface_mask_ds = xr.open_zarr(SURFACE_MASK_PATH)
    surface_mask = surface_mask_ds["surface_mask"].values.astype(bool)

    # Compute 3D loss_mask = surface_mask & depth_valid_mask
    depth_mask = target_ds["depth_valid_mask"].values.astype(bool)  # [15, 101, 241]
    loss_mask = (surface_mask[None, :, :] & depth_mask)

    # Coordinates
    depths = [float(d) for d in pred_ds["depth"].values]
    depth_indices = {float(d): i for i, d in enumerate(depths)}

    # Dates formatted as YYYY-MM-DD
    raw_times = pred_ds["time"].values
    dates = [pd.to_datetime(t).strftime("%Y-%m-%d") for t in raw_times]
    date_indices = {d: i for i, d in enumerate(dates)}

    latitudes = [float(lat) for lat in pred_ds["latitude"].values]
    longitudes = [float(lon) for lon in pred_ds["longitude"].values]

    # Metrics
    metrics = load_metrics_data(EVALUATION_DIR)

    # Store in global state
    STATE["pred_ds"] = pred_ds
    STATE["target_ds"] = target_ds
    STATE["loss_mask"] = loss_mask
    STATE["depths"] = depths
    STATE["depth_indices"] = depth_indices
    STATE["dates"] = dates
    STATE["date_indices"] = date_indices
    STATE["latitudes"] = latitudes
    STATE["longitudes"] = longitudes
    STATE["metrics"] = metrics

    # Build cached metadata response
    STATE["metadata"] = {
        "title": "OceanEmbed Subsurface Ocean Temperature Reconstruction",
        "dates": dates,
        "depths": depths,
        "latitude_range": [min(latitudes), max(latitudes)],
        "longitude_range": [min(longitudes), max(longitudes)],
        "latitudes": latitudes,
        "longitudes": longitudes,
        "grid": {
            "latitude": len(latitudes),
            "longitude": len(longitudes),
            "depth": len(depths),
            "time": len(dates),
        },
        "units": "°C",
        "test_period": {
            "start": dates[0] if dates else "2023-01-01",
            "end": dates[-1] if dates else "2024-12-31",
            "total_days": len(dates),
        },
        "model": {
            "name": pred_ds.attrs.get("model_name", "TemporalUNet"),
            "architecture": "Temporal U-Net (3D-2D)",
            "checkpoint": pred_ds.attrs.get("model_checkpoint", "checkpoints/fully_trained_temporal_unet_best_32batch.pt"),
            "input_variables": 7,
            "time_window_days": int(pred_ds.attrs.get("time_window", 7)),
            "depth_levels": len(depths),
            "status": "Model Ready",
        },
    }

    print(f"Ready: {len(dates)} dates ({dates[0]} to {dates[-1]}), {len(depths)} depths.")
    print("=" * 70)

    yield

    # Teardown
    print("Closing Zarr stores...")
    if STATE["pred_ds"] is not None:
        STATE["pred_ds"].close()
    if STATE["target_ds"] is not None:
        STATE["target_ds"].close()


app = FastAPI(
    title="OceanEmbed API",
    description="Subsurface Ocean Temperature Reconstruction Visualizer API",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for Vite dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health_check():
    """Health check verifying model predictions are available."""
    is_ready = STATE["pred_ds"] is not None
    return {
        "status": "ok" if is_ready else "initializing",
        "model_ready": is_ready,
        "total_test_dates": len(STATE["dates"]),
    }


@app.get("/api/metadata")
def get_metadata():
    """Return dashboard metadata including date list, depths, grid coords, and model info."""
    if not STATE["metadata"]:
        raise HTTPException(status_code=503, detail="Server metadata not initialized")
    return STATE["metadata"]


@app.get("/api/metrics")
def get_metrics():
    """Return pre-computed test-set evaluation metrics (overall, summary, by-depth)."""
    return STATE["metrics"]


def resolve_depth_index(depth: float) -> int:
    """Find the exact or nearest depth level index."""
    if depth in STATE["depth_indices"]:
        return STATE["depth_indices"][depth]
    
    # Nearest depth fallback
    depths = STATE["depths"]
    nearest_idx = int(np.argmin(np.abs(np.array(depths) - depth)))
    return nearest_idx


def resolve_date_index(date_str: str) -> int:
    """Find the exact date index, accepting YYYY-MM-DD or full timestamp."""
    # Strip any time component
    clean_date = date_str.split("T")[0]
    if clean_date in STATE["date_indices"]:
        return STATE["date_indices"][clean_date]
    
    raise HTTPException(
        status_code=404,
        detail=f"Date '{date_str}' not found in test dataset range ({STATE['dates'][0]} to {STATE['dates'][-1]})."
    )


def mask_array_to_json(arr: np.ndarray, mask_2d: np.ndarray) -> List[List[Optional[float]]]:
    """
    Convert a 2D numpy array [101, 241] to a JSON-serializable nested list.
    Invalid cells (where mask_2d is False or arr is NaN) are replaced with None (null in JSON).
    """
    # Replace invalid or NaN entries with nan
    clean_arr = arr.copy()
    clean_arr[~mask_2d] = np.nan
    clean_arr[np.isnan(clean_arr)] = np.nan
    
    # Fast conversion to list with None for NaNs
    # Using np.where to build python list
    out = []
    for row in clean_arr:
        out.append([None if np.isnan(v) else round(float(v), 3) for v in row])
    return out


@app.get("/api/field")
def get_field(
    date: str = Query(..., description="Target date (YYYY-MM-DD)"),
    depth: float = Query(..., description="Depth in meters"),
    mode: str = Query("prediction", description="Field mode: prediction, target, or error"),
):
    """
    Retrieve a single 2D depth slice [101, 241] for the selected date and depth.
    Invalid ocean / land cells are returned as null.
    """
    time_idx = resolve_date_index(date)
    d_idx = resolve_depth_index(depth)
    actual_depth = STATE["depths"][d_idx]
    actual_date = STATE["dates"][time_idx]

    # Mask for this depth level [101, 241]
    mask_2d = STATE["loss_mask"][d_idx]

    pred_ds = STATE["pred_ds"]
    target_ds = STATE["target_ds"]

    mode_lower = mode.lower()

    if mode_lower == "prediction":
        # Load single slice from prediction Zarr
        slice_vals = pred_ds["prediction"].isel(time=time_idx, depth=d_idx).values.astype(np.float32)
    elif mode_lower == "target":
        # Find corresponding target time index
        # Target Zarr contains 2015-2024, match by date string
        target_time_val = np.datetime64(actual_date, "ns")
        slice_vals = target_ds["target"].sel(time=target_time_val, depth=actual_depth).values.astype(np.float32)
    elif mode_lower == "error":
        pred_slice = pred_ds["prediction"].isel(time=time_idx, depth=d_idx).values.astype(np.float32)
        target_time_val = np.datetime64(actual_date, "ns")
        target_slice = target_ds["target"].sel(time=target_time_val, depth=actual_depth).values.astype(np.float32)
        slice_vals = np.abs(pred_slice - target_slice)
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mode '{mode}'. Must be one of: 'prediction', 'target', 'error'."
        )

    # Compute statistics over valid ocean cells only
    valid_data = slice_vals[mask_2d & ~np.isnan(slice_vals)]
    if len(valid_data) > 0:
        val_min = round(float(valid_data.min()), 2)
        val_max = round(float(valid_data.max()), 2)
        val_mean = round(float(valid_data.mean()), 2)
    else:
        val_min, val_max, val_mean = None, None, None

    # Mask invalid cells to None (null in JSON)
    formatted_values = mask_array_to_json(slice_vals, mask_2d)

    return {
        "date": actual_date,
        "depth": actual_depth,
        "mode": mode_lower,
        "min": val_min,
        "max": val_max,
        "mean": val_mean,
        "units": "°C",
        "values": formatted_values,
    }


@app.get("/api/compare")
def get_compare(
    date: str = Query(..., description="Target date (YYYY-MM-DD)"),
    depth: float = Query(..., description="Depth in meters"),
):
    """
    Retrieve synchronized prediction, GLORYS target, and absolute error fields
    in a single request for the side-by-side comparison mode.
    """
    time_idx = resolve_date_index(date)
    d_idx = resolve_depth_index(depth)
    actual_depth = STATE["depths"][d_idx]
    actual_date = STATE["dates"][time_idx]

    mask_2d = STATE["loss_mask"][d_idx]
    pred_ds = STATE["pred_ds"]
    target_ds = STATE["target_ds"]

    # 1. Prediction
    pred_slice = pred_ds["prediction"].isel(time=time_idx, depth=d_idx).values.astype(np.float32)

    # 2. GLORYS Reference Target
    target_time_val = np.datetime64(actual_date, "ns")
    target_slice = target_ds["target"].sel(time=target_time_val, depth=actual_depth).values.astype(np.float32)

    # 3. Absolute Error
    error_slice = np.abs(pred_slice - target_slice)

    # Extract depth-specific metrics if available
    depth_metrics = None
    for row in STATE["metrics"].get("by_depth", []):
        if abs(row.get("Depth_m", -999) - actual_depth) < 0.1:
            depth_metrics = row
            break

    # Stats for colorbar scaling
    valid_temp = np.concatenate([
        pred_slice[mask_2d & ~np.isnan(pred_slice)],
        target_slice[mask_2d & ~np.isnan(target_slice)]
    ])
    valid_error = error_slice[mask_2d & ~np.isnan(error_slice)]

    temp_min = round(float(valid_temp.min()), 2) if len(valid_temp) > 0 else 0.0
    temp_max = round(float(valid_temp.max()), 2) if len(valid_temp) > 0 else 35.0
    error_max = round(float(valid_error.max()), 2) if len(valid_error) > 0 else 5.0

    return {
        "date": actual_date,
        "depth": actual_depth,
        "prediction": mask_array_to_json(pred_slice, mask_2d),
        "target": mask_array_to_json(target_slice, mask_2d),
        "error": mask_array_to_json(error_slice, mask_2d),
        "temp_min": temp_min,
        "temp_max": temp_max,
        "error_max": error_max,
        "units": "°C",
        "depth_metrics": depth_metrics,
    }

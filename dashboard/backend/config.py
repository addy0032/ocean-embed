"""
Configuration for OceanEmbed Dashboard Backend.
"""

from pathlib import Path

# Root directory of the repository
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Zarr datasets
PREDICTIONS_ZARR_PATH = PROJECT_ROOT / "evaluation" / "predictions.zarr"
TARGET_ZARR_PATH = PROJECT_ROOT / "data" / "processed" / "y_target_025deg.zarr"
SURFACE_MASK_PATH = PROJECT_ROOT / "data" / "processed" / "surface_mask_025deg.zarr"

# Evaluation metrics directory
EVALUATION_DIR = PROJECT_ROOT / "evaluation"

# CORS configuration
CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "*"
]

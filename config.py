"""
Central configuration for the harmonization / dataset-build pipeline.

Fill in the real zarr paths + variable names for all 7 surface inputs.
The target (GLORYS temperature, full 15-depth profile) already ships with
its own bathymetric depth_valid_mask variable, so we consume that
directly rather than deriving one ourselves.
"""

# --- INPUT VARIABLES (surface, 7 total) ---
# key = short name you want to use downstream (becomes the channel/column name)
INPUT_VARIABLES = {
    "sst":    {"path": "data/processed/sst/sst_2015_2024_025deg.zarr", "var": "analysed_sst"},
    "sss":    {"path": "data/processed/sss/sss_2015_2024_025deg.zarr", "var": "sos"},
    "adt":    {"path": "data/processed/sealevel/sealevel_2015_2024_025deg.zarr", "var": "adt"},
    "u_curr": {"path": "data/processed/sealevel/sealevel_2015_2024_025deg.zarr", "var": "ugos"},
    "v_curr": {"path": "data/processed/sealevel/sealevel_2015_2024_025deg.zarr", "var": "vgos"},
    "u_wind": {"path": "data/processed/wind/wind_2015_2024_025deg.zarr", "var": "eastward_wind"},
    "v_wind": {"path": "data/processed/wind/wind_2015_2024_025deg.zarr", "var": "northward_wind"},
}

# --- TARGET VARIABLE (GLORYS temperature, full depth profile) ---
# "depth_mask_var" is the pre-existing bathymetric mask already stored
# alongside thetao in the same dataset (1 = valid ocean at that depth,
# 0 = land / seafloor shallower than this level). We use it as-is instead
# of recomputing it from valid fractions.
TARGET_VARIABLE = {
    "path": "data/processed/glorys/glorys_temperature_2015_2024_025deg.zarr",
    "var": "thetao",
    "depth_mask_var": "depth_valid_mask",
    "depth_dim": "depth",
    "surface_depth_value": 0,
}

COMMON_MASK_PATH = "data/processed/surface_mask_025deg.zarr"
ALIGNED_DIR = "data/processed/aligned"

# Final chunking used everywhere. Keep this consistent across every zarr
# store you write, input and target alike, or downstream merges will
# trigger expensive rechunking.
FINAL_CHUNKS = {"time": 1, "latitude": -1, "longitude": -1}
FINAL_CHUNKS_TARGET = {"time": 1, "depth": -1, "latitude": -1, "longitude": -1}


# ML CONFIGURATION

TIME_WINDOW = 7

TRAIN_START = "2015-01-01"
TRAIN_END   = "2021-12-31"

VAL_START = "2022-01-01"
VAL_END   = "2022-12-31"

TEST_START = "2023-01-01"
TEST_END   = "2024-12-31"

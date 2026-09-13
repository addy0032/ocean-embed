"""
Run this AFTER create_master_mask.py and apply_master_mask.py.

Produces three zarr stores:

  X_inputs_025deg.zarr          - single Dataset, one data variable per
                                   surface input (7 channels), dims
                                   (time, latitude, longitude).
  y_target_025deg.zarr          - the target, kept as its own Dataset
                                   (never merged into X). Dims
                                   (time, depth, latitude, longitude) --
                                   15 depth levels, plus the carried-over
                                   depth_valid_mask (depth, lat, lon).
"""

import xarray as xr

from config import INPUT_VARIABLES, TARGET_VARIABLE, ALIGNED_DIR, FINAL_CHUNKS, FINAL_CHUNKS_TARGET

X_PATH = "data/processed/X_inputs_025deg.zarr"
Y_PATH = "data/processed/y_target_025deg.zarr"


def load_aligned(name):
    return xr.open_zarr(f"{ALIGNED_DIR}/{name}_aligned_025deg.zarr")

def normalize_time(da):
    """
    Normalize daily timestamps to midnight.
    """
    dates = da["time"].values.astype("datetime64[D]").astype("datetime64[ns]")

    if len(set(dates)) != len(dates):
        raise ValueError(
            f"{da.name}: duplicate dates detected after time normalization."
        )

    return da.assign_coords(time=dates)

def normalize_horizontal_coords(da):
    """
    Keep only the coordinates relevant to the common horizontal grid.
    """
    return da.drop_vars(
        [c for c in da.coords if c not in {"time", "latitude", "longitude"}],
        errors="ignore",
    )


def main():
    # ---- load all masked/aligned inputs, renamed to their short key ----
    input_das = {}
    for name, info in INPUT_VARIABLES.items():
        ds = load_aligned(name)
        da = ds[info["var"]].rename(name)
        da = normalize_time(da)
        da = normalize_horizontal_coords(da)
        input_das[name] = da

    target_ds = load_aligned("target")
    target_da = target_ds[TARGET_VARIABLE["var"]].rename("target")
    target_da = normalize_time(target_da)                            
    depth_mask_da = target_ds[TARGET_VARIABLE["depth_mask_var"]]       

    # ---- sanity check: horizontal + time coords match across everything ----
    ref = next(iter(input_das.values()))
    for name, da in list(input_das.items()) + [("target", target_da)]:
        xr.testing.assert_allclose(da["latitude"], ref["latitude"])
        xr.testing.assert_allclose(da["longitude"], ref["longitude"])
        xr.testing.assert_allclose(da["time"], ref["time"])
    print("Coordinate check passed: all 7 inputs + target share the exact same time/lat/lon grid.")

    # ---- combine the 7 inputs into ONE dataset (this is your X) ----
    X = xr.Dataset(input_das)
    X = X.chunk(FINAL_CHUNKS)
    for v in X.variables:
        X[v].encoding = {}
    X.to_zarr(X_PATH, mode="w", zarr_format=3)
    print(f"Wrote combined 7-channel input dataset -> {X_PATH}")

    # ---- target stays a separate dataset (this is your y), full depth profile ----
    y = xr.Dataset({"target": target_da, TARGET_VARIABLE["depth_mask_var"]: depth_mask_da})
    y = y.chunk(FINAL_CHUNKS_TARGET)
    for v in y.variables:
        y[v].encoding = {}
    y.to_zarr(Y_PATH, mode="w", zarr_format=3)
    print(f"Wrote target dataset (15 depths + bathymetric mask, kept separate from X) -> {Y_PATH}")


if __name__ == "__main__":
    main()

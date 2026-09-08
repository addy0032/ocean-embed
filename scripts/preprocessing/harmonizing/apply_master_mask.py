import xarray as xr

COMMON_MASK_PATH = "data/processed/common_ocean_mask_025deg.zarr"
FINAL_CHUNKS = {"time": 1, "latitude": -1, "longitude": -1}

def apply_common_mask(processed_path, var_name, output_path):
    print(f"Masking {var_name}...")
    ds = xr.open_zarr(processed_path)
    mask = xr.open_zarr(COMMON_MASK_PATH)["common_ocean_mask"]

    # Apply the strict coastal intersection
    ds[var_name] = ds[var_name].where(mask)

    # rechunking for ml pipeline
    ds = ds.chunk(FINAL_CHUNKS)

    for v in ds.variables:
        ds[v].encoding = {}

    ds.to_zarr(output_path, mode="w", zarr_format=3)
    print(f"Successfully wrote {output_path}")

if __name__ == "__main__":
    apply_common_mask(
        "data/processed/sst_2015_2024_025deg.zarr",
        "analysed_sst",
        "data/processed/sst_aligned_025deg.zarr",
    )
    apply_common_mask(
        "data/processed/sss/sss_2015_2024_025deg.zarr",
        "sos",
        "data/processed/sss_aligned_025deg.zarr",
    )
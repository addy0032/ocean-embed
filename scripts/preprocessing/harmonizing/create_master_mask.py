import xarray as xr

# all of the processed variables to harmonize
PROCESSED_VARIABLES = {
    "sst": {
        "path": "data/processed/sst_2015_2024_025deg.zarr",
        "var": "analysed_sst",
    },
    "sss": {
        "path": "data/processed/sss/sss_2015_2024_025deg.zarr",
        "var": "sos",
    },
}

COMMON_MASK_PATH = "data/processed/common_ocean_mask_025deg.zarr"

def main():
    valid_masks = []

    for name, info in PROCESSED_VARIABLES.items():
        print(f"Checking {name}...")
        ds = xr.open_zarr(info["path"])
        da = ds[info["var"]]

        # > 0.5 threshold removes flickering edge pixels
        valid_fraction = da.notnull().mean(dim="time").compute()
        valid = valid_fraction > 0.5

        print(f"  {name}: {int(valid.sum())}/{valid.size} valid cells")
        valid_masks.append(valid)

    # intersect all masks
    common_mask = valid_masks[0]
    for m in valid_masks[1:]:
        common_mask = common_mask & m

    print(f"\nCommon mask: {int(common_mask.sum())}/{common_mask.size} cells")

    common_mask.name = "common_ocean_mask"
    common_mask.to_dataset().to_zarr(COMMON_MASK_PATH, mode="w", zarr_format=3)

if __name__ == "__main__":
    main()
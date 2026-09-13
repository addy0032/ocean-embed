import xarray as xr

from config import (
    INPUT_VARIABLES,
    TARGET_VARIABLE,
    COMMON_MASK_PATH
)


def all_time_valid_mask(da):
    return da.notnull().all(dim="time").compute()


def main():
    # SURFACE MASK (lat, lon): intersection of the 7 surface inputs + the
    # GLORYS 0m slice. This defines where you have consistent surface
    # information across every input plus the target's own surface level.
    #
    # (The target's bathymetric depth mask is NOT part of this step --
    # it already exists as depth_valid_mask inside the target dataset and
    # is applied separately, per depth level, in apply_master_mask.py.)
    valid_masks = []

    for name, info in INPUT_VARIABLES.items():
        print(f"Checking {name} ({info['var']})...")
        ds = xr.open_zarr(info["path"])
        valid = all_time_valid_mask(ds[info["var"]])
        print(f"    {int(valid.sum())}/{valid.size} cells pass")
        valid_masks.append(valid)

        ds.close()

    print(f"Checking target 0m slice ({TARGET_VARIABLE['var']})...")
    target_ds = xr.open_zarr(TARGET_VARIABLE["path"])
    target_surface = target_ds[TARGET_VARIABLE["var"]].sel(
        {TARGET_VARIABLE["depth_dim"]: TARGET_VARIABLE["surface_depth_value"]}
    )
    valid = all_time_valid_mask(target_surface)
    print(f"    {int(valid.sum())}/{valid.size} cells pass")
    valid_masks.append(valid)
    target_ds.close()

    surface_mask = valid_masks[0]
    for m in valid_masks[1:]:
        surface_mask = surface_mask & m

    print(
        f"\nSurface mask across 7 inputs + target 0m slice: "
        f"{int(surface_mask.sum())}/{surface_mask.size} cells"
    )
    surface_mask.name = "surface_mask"
    surface_mask.to_dataset().to_zarr(COMMON_MASK_PATH, mode="w", zarr_format=3)


if __name__ == "__main__":
    main()

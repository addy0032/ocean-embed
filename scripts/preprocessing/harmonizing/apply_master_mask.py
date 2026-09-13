import os

import xarray as xr

from config import (
    INPUT_VARIABLES,
    TARGET_VARIABLE,
    COMMON_MASK_PATH,
    ALIGNED_DIR,
    FINAL_CHUNKS,
    FINAL_CHUNKS_TARGET,
)


def write_zarr(ds, output_path, chunks):
    ds = ds.chunk(chunks)
    for v in ds.variables:
        ds[v].encoding = {}
    ds.to_zarr(output_path, mode="w", zarr_format=3)
    print(f"  wrote {output_path}")


def apply_to_input(processed_path, var_name, output_path, surface_mask):
    print(f"Masking {var_name} -> {output_path}")
    ds = xr.open_zarr(processed_path)
    ds[var_name] = ds[var_name].where(surface_mask)
    write_zarr(ds, output_path, FINAL_CHUNKS)


def apply_to_target(processed_path, var_name, depth_mask_var, output_path, surface_mask):
    print(f"Masking {var_name} (full profile) -> {output_path}")
    ds = xr.open_zarr(processed_path)

    # Pre-existing bathymetric mask shipped with the target dataset
    # (1 = valid ocean at this depth, 0 = land/shallower-than-this-level).
    depth_mask = ds[depth_mask_var].astype(bool)

    # Combined mask: inputs must exist here (surface_mask, broadcast over
    # depth) AND the ocean must actually be this deep here (depth_mask,
    # broadcast over time). Note thetao is already NaN wherever
    # depth_mask is 0 per the file's own metadata, so this mainly adds
    # the surface_mask restriction; re-applying depth_mask here is cheap
    # and keeps the result correct even if that assumption ever changes.
    ds[var_name] = ds[var_name].where(surface_mask).where(depth_mask)

    # keep the bathymetric mask alongside the target for convenience at
    # training time (loss masking per depth level)
    ds[depth_mask_var] = depth_mask

    write_zarr(ds, output_path, FINAL_CHUNKS_TARGET)


def main():
    surface_mask = xr.open_zarr(COMMON_MASK_PATH)["surface_mask"]
    os.makedirs(ALIGNED_DIR, exist_ok=True)

    for name, info in INPUT_VARIABLES.items():
        out_path = f"{ALIGNED_DIR}/{name}_aligned_025deg.zarr"
        apply_to_input(info["path"], info["var"], out_path, surface_mask)

    out_path = f"{ALIGNED_DIR}/target_aligned_025deg.zarr"
    apply_to_target(
        TARGET_VARIABLE["path"],
        TARGET_VARIABLE["var"],
        TARGET_VARIABLE["depth_mask_var"],
        out_path,
        surface_mask,
    )

    print("\nAll 7 inputs masked to the surface domain; target masked to surface domain x its own bathymetric mask.")


if __name__ == "__main__":
    main()

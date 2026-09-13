import os

import numpy as np
import xarray as xr
import xesmf as xe

from scipy.interpolate import PchipInterpolator
from dask.distributed import Client, performance_report


# ============================================================
# Configuration
# ============================================================

GLORYS_DIR = "data/raw/glorys"
REFERENCE_GRID_PATH = "data/processed/reference_grid_025deg.zarr"

OUTPUT_DIR = "data/processed/glorys"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "glorys_temperature_2015_2024_025deg.zarr")

WEIGHTS_DIR = os.path.join(OUTPUT_DIR, "weights")
REPORT_DIR = os.path.join(OUTPUT_DIR, "reports")

YEARS = range(2015, 2025)

SOURCE_VARIABLE = "thetao"

TARGET_DEPTHS = np.array([0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000], dtype=np.float32)

INPUT_CHUNKS = {
    "time": 60,
    "depth": -1,
    "latitude": 16,
    "longitude": -1,
}

OUTPUT_CHUNKS = {
    "time": -1,
    "depth": -1,
    "latitude": -1,
    "longitude": -1,
}


# ============================================================
# Grid helpers
# ============================================================

def cell_bounds(values):
    values = np.asarray(values, dtype=np.float64)
    return np.concatenate([
        [values[0] - (values[1] - values[0]) / 2],
        (values[:-1] + values[1:]) / 2,
        [values[-1] + (values[-1] - values[-2]) / 2],
    ])


def create_source_grid(ds):
    lat = ds["latitude"].values
    lon = ds["longitude"].values
    lat_b = cell_bounds(lat)
    lon_b = cell_bounds(lon)

    return xr.Dataset(
        coords={
            "latitude": ("latitude", lat),
            "longitude": ("longitude", lon),
            "lat_b": ("lat_b", lat_b),
            "lon_b": ("lon_b", lon_b),
        }
    )


def create_target_grid(reference_grid):
    return xr.Dataset(
        coords={
            "latitude": reference_grid["latitude"].values,
            "longitude": reference_grid["longitude"].values,
            "lat_b": reference_grid["lat_b"].values,
            "lon_b": reference_grid["lon_b"].values,
        }
    )

def create_depth_masks(thetao, source_depths):
    source_valid = thetao.isel(time=0).notnull().compute()
    bottom_index = source_valid.sum(dim="depth") - 1

    masks = {}

    for target_depth in TARGET_DEPTHS:
        if target_depth == 0:
            source_index = 0
            mask = source_valid.isel(depth=source_index)
        else:
            deeper_index = np.searchsorted(source_depths.values, target_depth, side="left")
            shallower_index = deeper_index - 1

            if deeper_index >= len(source_depths):
                mask = xr.zeros_like(source_valid.isel(depth=0), dtype=bool)
            else:
                mask = bottom_index >= deeper_index

        masks[float(target_depth)] = mask.astype(np.int8)

    return masks


# ============================================================
# PCHIP
# ============================================================

def pchip_block(values, source_depths, target_depths):
    n_time, n_lat, n_lon, n_depth = values.shape
    n_target = len(target_depths)

    output = np.full((n_time, n_lat, n_lon, n_target), np.nan, dtype=np.float32)

    flat_values = values.reshape(-1, n_depth)
    flat_output = output.reshape(-1, n_target)

    for bottom_index in range(1, n_depth + 1):
        profiles = np.all(np.isfinite(flat_values[:, :bottom_index]), axis=1) & (np.sum(np.isfinite(flat_values), axis=1) == bottom_index)

        if not profiles.any():
            continue

        profile_values = flat_values[profiles, :bottom_index]
        valid_targets = (target_depths >= source_depths[0]) & (target_depths <= source_depths[bottom_index - 1])

        if valid_targets.any():
            interpolator = PchipInterpolator(source_depths[:bottom_index], profile_values, axis=1, extrapolate=False)
            flat_output[np.where(profiles)[0][:, None], np.where(valid_targets)[0]] = interpolator(target_depths[valid_targets])

    surface_index = np.argmin(np.abs(source_depths - 0.49402499))
    surface_values = flat_values[:, surface_index]
    surface_valid = np.isfinite(surface_values)
    flat_output[surface_valid, 0] = surface_values[surface_valid]

    return output


def interpolate_vertical_pchip(thetao, source_depths):
    thetao_for_pchip = thetao.transpose("time", "latitude", "longitude", "depth")

    result = xr.apply_ufunc(
        pchip_block,
        thetao_for_pchip,
        input_core_dims=[["depth"]],
        output_core_dims=[["target_depth"]],
        kwargs={
            "source_depths": source_depths.values,
            "target_depths": TARGET_DEPTHS,
        },
        dask="parallelized",
        output_dtypes=[np.float32],
        dask_gufunc_kwargs={"output_sizes": {"target_depth": len(TARGET_DEPTHS)}},
    )

    return result.rename({"target_depth": "depth"}).assign_coords(depth=TARGET_DEPTHS).transpose("time", "depth", "latitude", "longitude")


# ============================================================
# Main
# ============================================================

def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    if os.path.exists(OUTPUT_PATH):
        raise FileExistsError(
            f"Output already exists:\n{OUTPUT_PATH}\n"
            "Delete it before running a fresh preprocessing job."
        )

    print("=" * 70)
    print("GLORYS TARGET PREPROCESSING")
    print("=" * 70)

    client = Client(
        n_workers=1,
        threads_per_worker=4,
    )

    print("\nDask client:")
    print(client)
    print("Dashboard:", client.dashboard_link)

    print("\nOpening reference grid...")

    reference_grid = xr.open_zarr(
        REFERENCE_GRID_PATH
    )

    target_grid = create_target_grid(
        reference_grid
    )

    print("\nReference grid:")
    print(reference_grid)

    print("\nTarget latitude:")
    print(reference_grid["latitude"].values)

    print("\nTarget longitude:")
    print(reference_grid["longitude"].values)

    print("\nTarget depths:")
    print(TARGET_DEPTHS)

    regridders = {}
    first_year = True
    source_grid = None

    for year in YEARS:

        print("\n" + "=" * 70)
        print(f"PROCESSING {year}")
        print("=" * 70)

        input_path = os.path.join(
            GLORYS_DIR,
            f"cmems_mod_glo_phy_my_0.083deg_P1D-m_thetao_45.00E-105.00E_5.00N-30.00N_0.49-1062.44m_"
            f"{year}-01-01T00-00-00-{year}-12-31T00-00-00.zarr",
        )

        if not os.path.exists(input_path):
            raise FileNotFoundError(
                f"GLORYS input not found:\n{input_path}"
            )

        print("\nOpening:")
        print(input_path)

        ds = xr.open_zarr(
            input_path,
            chunks=INPUT_CHUNKS,
        )

        thetao = ds[SOURCE_VARIABLE].astype(
            np.float32
        )

        source_depths = ds["depth"].astype(
            np.float32
        )

        print("\nInput dataset:")
        print(ds)

        print("\nInput thetao chunks:")
        print(thetao.chunks)

        if source_grid is None:
            source_grid = create_source_grid(ds)

        if first_year:
            print("\nCreating depth-specific source masks...")
            depth_masks = create_depth_masks(thetao, source_depths)
            print("Depth-specific source masks created.")

        # ====================================================
        # Vertical interpolation
        # ====================================================

        print("\n" + "-" * 70)
        print("VERTICAL PCHIP")
        print("-" * 70)

        print("\n36 native levels -> 15 target levels")

        with performance_report(
            filename=os.path.join(
                REPORT_DIR,
                f"glorys_vertical_{year}.html",
            )
        ):

            thetao_15 = interpolate_vertical_pchip(
                thetao,
                source_depths,
            )

        print("\nVertically interpolated thetao:")
        print(thetao_15)

        print("\nChunks:")
        print(thetao_15.chunks)

        # ====================================================
        # Horizontal regridding, depth by depth
        # ====================================================

        depth_results = []

        for depth_index, target_depth in enumerate(TARGET_DEPTHS):

            print("\n" + "-" * 70)
            print(f"DEPTH: {target_depth:g} m")
            print("-" * 70)

            thetao_depth = thetao_15.isel(
                depth=depth_index
            )

            source_mask = depth_masks[float(target_depth)]

            valid_cells = int(source_mask.sum().item())
            total_cells = int(source_mask.size)

            print(f"Valid source cells: {valid_cells:,} / {total_cells:,}")

            depth_source_grid = source_grid.copy()
            depth_source_grid["mask"] = (("latitude", "longitude"), source_mask.values)

            weight_path = os.path.join(
                WEIGHTS_DIR,
                f"glorys_{target_depth:g}m_to_025deg_conservative_normed.nc",
            )

            if target_depth not in regridders:

                print("\nCreating conservative_normed regridder...")

                regridders[target_depth] = xe.Regridder(
                    depth_source_grid,
                    target_grid,
                    method="conservative_normed",
                    filename=weight_path,
                    reuse_weights=os.path.exists(
                        weight_path
                    ),
                )

            regridder = regridders[target_depth]

            print("\nRegridding temperature...")

            with performance_report(
                filename=os.path.join(
                    REPORT_DIR,
                    f"glorys_regrid_{year}_{target_depth:g}m.html",
                )
            ):

                thetao_depth_025 = regridder(
                    thetao_depth,
                    keep_attrs=True,
                    skipna=True,
                    na_thres=0.50,
                    output_chunks={
                        "time": -1,
                        "latitude": -1,
                        "longitude": -1,
                    },
                )

            thetao_depth_025 = thetao_depth_025.astype(
                np.float32
            )

            depth_results.append(
                thetao_depth_025
            )

            print("\nCompleted depth:", target_depth)

        # ====================================================
        # Combine depths
        # ====================================================

        print("\n" + "-" * 70)
        print("COMBINING DEPTHS")
        print("-" * 70)

        thetao_025 = xr.concat(
            depth_results,
            dim=xr.DataArray(
                TARGET_DEPTHS,
                dims="depth",
                name="depth",
            ),
        ).transpose("time", "depth", "latitude", "longitude")

        thetao_025.name = "thetao"

        # ====================================================
        # Final depth-dependent mask
        # ====================================================

        print("\nCreating target depth mask...")

        target_mask = thetao_025.isel(
            time=0
        ).notnull().astype(
            np.uint8
        )

        target_mask.name = "depth_valid_mask"

        thetao_025 = thetao_025.where(
            target_mask == 1
        )

        # ====================================================
        # Metadata
        # ====================================================

        thetao_025.attrs.update({
            "long_name": "Sea water potential temperature",
            "standard_name": "sea_water_potential_temperature",
            "units": "degC",
            "source": "CMEMS GLORYS12V1",
            "source_resolution": "approximately 0.083333 degree",
            "target_resolution": "0.25 degree",
            "source_vertical_levels": 36,
            "target_vertical_levels": 15,
            "vertical_interpolation": "1D PCHIP",
            "horizontal_regridding": "xESMF conservative_normed with skipna=True and na_thres=0.50",
            "surface_depth_handling": "0 m represented by native GLORYS 0.49402499 m level",
            "vertical_extrapolation": "disabled",
            "bathymetry_handling": "depth-specific source masks",
            "missing_value_handling": "invalid bathymetry cells retained as NaN",
        })

        target_mask.attrs.update({
            "long_name": "Depth-dependent ocean validity mask",
            "description": "1 indicates valid ocean temperature at the corresponding target depth; 0 indicates "
            "land or insufficient bathymetric depth.",
        })

        # ====================================================
        # Dataset
        # ====================================================

        if first_year:

            year_ds = xr.Dataset({
                "thetao": thetao_025,
                "depth_valid_mask": target_mask,
            })

            year_ds.attrs.update({
                "title": "GLORYS subsurface temperature target",
                "processing": "36 GLORYS levels vertically interpolated to 15 standard depths using 1D PCHIP, "
                "followed by depth-specific conservative_normed horizontal regridding to 0.25 degree.",
                "target_depths_m": ", ".join(
                    str(float(x))
                    for x in TARGET_DEPTHS
                ),
            })

        else:

            year_ds = xr.Dataset({
                "thetao": thetao_025,
            })

        # ====================================================
        # Final chunking
        # ====================================================

        year_ds = year_ds.chunk(OUTPUT_CHUNKS)

        for var in year_ds.variables:
            year_ds[var].encoding = {}

        print("\nFinal yearly dataset:")
        print(year_ds)

        print("\nThetao chunks:")
        print(year_ds["thetao"].chunks)

        # ====================================================
        # Write
        # ====================================================

        if first_year:

            print("\nWriting 2015 and creating output dataset...")

            year_ds.to_zarr(
                OUTPUT_PATH,
                mode="w",
                zarr_format=3,
            )

            first_year = False

        else:

            print(f"\nAppending {year}...")

            year_ds.to_zarr(
                OUTPUT_PATH,
                mode="a",
                append_dim="time",
                zarr_format=3,
            )

        print(f"\n{year} COMPLETE")

        ds.close()

        del thetao
        del thetao_15
        del depth_results
        del thetao_025
        del target_mask
        del year_ds

    # ========================================================
    # Final inspection
    # ========================================================

    print("\n" + "=" * 70)
    print("FINAL GLORYS DATASET")
    print("=" * 70)

    final_ds = xr.open_zarr(
        OUTPUT_PATH
    )

    print(final_ds)

    print("\nDimensions:")
    print(final_ds.sizes)

    print("\nDepths:")
    print(final_ds["depth"].values)

    print("\nThetao dtype:")
    print(final_ds["thetao"].dtype)

    print("\nThetao chunks:")
    print(final_ds["thetao"].chunks)

    print("\nMask dtype:")
    print(final_ds["depth_valid_mask"].dtype)

    print("\nTime:")
    print(final_ds["time"].values[0])
    print(final_ds["time"].values[-1])

    print("\nOutput:")
    print(OUTPUT_PATH)

    print("\n" + "=" * 70)
    print("GLORYS PREPROCESSING COMPLETE")
    print("=" * 70)

    client.close()


if __name__ == "__main__":
    main()
import os

import numpy as np
import xarray as xr
import xesmf as xe

from dask.distributed import Client, performance_report, wait


# ============================================================
# Main
# ============================================================

def main():

    # ========================================================
    # 1. Dask
    # ========================================================

    print("Starting Dask client...")

    client = Client(n_workers=2, threads_per_worker=8)

    print(client)
    print("Dashboard:", client.dashboard_link)


    # ========================================================
    # 2. Paths
    # ========================================================

    WIND_PATH = "data/raw/wind/cmems_wind_2015_2024_0125deg_hourly.zarr"
    REFERENCE_GRID_PATH = "data/processed/reference_grid_025deg.zarr"

    WIND_PROCESSED_DIR = "data/processed/wind"
    os.makedirs(WIND_PROCESSED_DIR, exist_ok=True)

    OUTPUT_PATH = os.path.join(
        WIND_PROCESSED_DIR,
        "wind_2015_2024_025deg.zarr"
    )

    WEIGHTS_PATH = os.path.join(
        WIND_PROCESSED_DIR,
        "wind_to_025deg_conservative_normed_10yr.nc"
    )

    MASK_REPORT_PATH = os.path.join(
        WIND_PROCESSED_DIR,
        "wind_mask_report.html"
    )

    DAILY_REPORT_PATH = os.path.join(
        WIND_PROCESSED_DIR,
        "wind_daily_averaging_report.html"
    )

    REGRID_REPORT_PATH = os.path.join(
        WIND_PROCESSED_DIR,
        "wind_regridding_report.html"
    )


    # ========================================================
    # 3. Configuration
    # ========================================================

    # The first operation is hourly -> daily resampling.
    # A 30-day time chunk is a reasonable starting point for this.
    INPUT_CHUNKS = {
        "time": 336,
        "latitude": -1,
        "longitude": -1,
    }

    U_VAR = "eastward_wind"
    V_VAR = "northward_wind"

    # ========================================================
    # 4. Open wind dataset
    # ========================================================

    print("\nOpening wind dataset...")

    wind_ds = xr.open_zarr(
        WIND_PATH,
        chunks=INPUT_CHUNKS,
    )
    wind_ds = wind_ds.sel(time=slice("2015-01-01", "2024-12-31 23:00:00"))

    print(wind_ds)
    print("\nWind variables:")
    print(wind_ds.data_vars)


    # ========================================================
    # 5. Select wind components
    # ========================================================

    print("\nSelecting wind components...")

    u = wind_ds[U_VAR].astype("float32")
    v = wind_ds[V_VAR].astype("float32")

    print("\nEastward wind:")
    print(u)

    print("\nNorthward wind:")
    print(v)


    # ========================================================
    # 6. Check time axis
    # ========================================================

    print("\nChecking time axis...")

    time_values = wind_ds["time"].values

    print("First time :", time_values[0])
    print("Last time  :", time_values[-1])

    # ========================================================
    # 7. Daily vector averaging
    # ========================================================

    print("\nCalculating daily mean wind vectors...")

    with performance_report(filename=DAILY_REPORT_PATH):

        # Mean U and V independently.
        u_daily = u.resample(time="1D").mean(skipna=True)
        v_daily = v.resample(time="1D").mean(skipna=True)

        # Force actual computation now, once, and keep results in distributed memory
        u_daily = u_daily.persist()
        v_daily = v_daily.persist()
        wait([u_daily, v_daily])

    print("\nDaily eastward wind:")
    print(u_daily)

    print("\nDaily northward wind:")
    print(v_daily)


    # ========================================================
    # 8. Open reference grid
    # ========================================================

    print("\nOpening reference grid...")

    reference_grid = xr.open_zarr(REFERENCE_GRID_PATH)

    print(reference_grid)


    # ========================================================
    # 9. Create source grid
    # ========================================================

    print("\nCreating source grid...")

    source_lat = wind_ds["latitude"].values
    source_lon = wind_ds["longitude"].values

    # --------------------------------------------------------
    # Source latitude boundaries
    # --------------------------------------------------------

    source_lat_b = np.concatenate([
        [source_lat[0] - (source_lat[1] - source_lat[0]) / 2],
        (source_lat[:-1] + source_lat[1:]) / 2,
        [source_lat[-1] + (source_lat[-1] - source_lat[-2]) / 2],
    ])

    # --------------------------------------------------------
    # Source longitude boundaries
    # --------------------------------------------------------

    source_lon_b = np.concatenate([
        [source_lon[0] - (source_lon[1] - source_lon[0]) / 2],
        (source_lon[:-1] + source_lon[1:]) / 2,
        [source_lon[-1] + (source_lon[-1] - source_lon[-2]) / 2],
    ])

    source_grid = xr.Dataset(
        coords={
            "latitude": ("latitude", source_lat),
            "longitude": ("longitude", source_lon),
            "lat_b": ("lat_b", source_lat_b),
            "lon_b": ("lon_b", source_lon_b),
        }
    )


    # ========================================================
    # 10. Create source mask
    # ========================================================

    print("\nCreating source mask...")

    print(
        "Checking wind observations to identify source cells "
        "that contain valid data..."
    )

    with performance_report(filename=MASK_REPORT_PATH):

        source_mask = (
            u_daily.notnull().any(dim="time")
            .compute()
            .astype(np.int8)
        )

    valid_cells = int(source_mask.values.sum())
    total_cells = int(source_mask.size)

    print(
        f"Valid source cells: "
        f"{valid_cells:,} / {total_cells:,}"
    )

    source_grid["mask"] = (
        ("latitude", "longitude"),
        source_mask.values,
    )


    # ========================================================
    # 11. Create target grid
    # ========================================================

    print("\nCreating target grid...")

    target_grid = xr.Dataset(
        coords={
            "latitude": (
                "latitude",
                reference_grid["latitude"].values,
            ),
            "longitude": (
                "longitude",
                reference_grid["longitude"].values,
            ),
            "lat_b": (
                "lat_b",
                reference_grid["lat_b"].values,
            ),
            "lon_b": (
                "lon_b",
                reference_grid["lon_b"].values,
            ),
        }
    )



    with performance_report(filename=REGRID_REPORT_PATH):
        # ========================================================
        # 12. Create xESMF regridder
        # ========================================================

        print("\nCreating xESMF regridder...")

        regridder = xe.Regridder(
            source_grid,
            target_grid,
            method="conservative_normed",
            filename=WEIGHTS_PATH,
            reuse_weights=False,
        )

        print(regridder)


        # ========================================================
        # 13. Regrid daily wind
        # ========================================================

        print("\nRegridding daily wind dataset...")

        print("\nRegridding eastward wind...")

        u_025 = regridder(
            u_daily,
            keep_attrs=True,
            skipna=True,
            na_thres=0.50,
        )

        print("\nRegridding northward wind...")

        v_025 = regridder(
            v_daily,
            keep_attrs=True,
            skipna=True,
            na_thres=0.50,
        )


        # ====================================================
        # 14. Convert back to float32
        # ====================================================

        u_025 = u_025.astype("float32")
        v_025 = v_025.astype("float32")


        # ====================================================
        # 15. Metadata
        # ====================================================

        u_025.name = U_VAR
        v_025.name = V_VAR

        for wind_component in (u_025, v_025):

            wind_component.attrs["regridding_method"] = (
                "xESMF conservative_normed"
            )

            wind_component.attrs["source_resolution"] = (
                "0.125 degree"
            )

            wind_component.attrs["target_resolution"] = (
                "0.25 degree"
            )

            wind_component.attrs["temporal_processing"] = (
                "daily vector mean calculated from hourly data"
            )

            wind_component.attrs["source_grid_coordinate_convention"] = (
                "cell centers"
            )

            wind_component.attrs["target_grid_coordinate_convention"] = (
                "cell centers"
            )

            wind_component.attrs["missing_value_handling"] = (
                "daily minimum-valid-hours filtering followed by "
                "conservative_normed regridding with "
                "skipna=True and na_thres=0.50"
            )


        # ====================================================
        # 16. Convert to Dataset
        # ====================================================

        wind_025_ds = xr.Dataset({
            U_VAR: u_025,
            V_VAR: v_025,
        })


        # ====================================================
        # 17. Final ML-oriented chunking
        # ====================================================

        print("\nApplying final chunking...")

        wind_025_ds = wind_025_ds.chunk({
            "time": 1,
            "latitude": -1,
            "longitude": -1,
        })


        # ====================================================
        # 18. Remove inherited encodings
        # ====================================================

        print("\nCleaning variable encodings...")

        for var in wind_025_ds.variables:
            wind_025_ds[var].encoding = {}


        # ====================================================
        # 19. Inspect final dataset
        # ====================================================

        print("\nFinal dataset:")
        print(wind_025_ds)

        print("\nFinal U chunks:")
        print(wind_025_ds[U_VAR].chunks)

        print("\nFinal V chunks:")
        print(wind_025_ds[V_VAR].chunks)


        # ====================================================
        # 20. Write Zarr 3
        # ====================================================

        print("\nWriting Zarr 3 output...")

        wind_025_ds.to_zarr(
            OUTPUT_PATH,
            mode="w",
            zarr_format=3,
        )


    print("\nDAILY WIND REGRIDDING COMPLETE")

    print("Output:")
    print(OUTPUT_PATH)

    print("\nWeights:")
    print(WEIGHTS_PATH)

    print("\nFormat: Zarr 3")

    client.close()


if __name__ == "__main__":
    main()
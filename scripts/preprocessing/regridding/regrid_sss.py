import os

import numpy as np
import xarray as xr
import xesmf as xe

from dask.distributed import Client, performance_report


# ============================================================
# Main
# ============================================================

def main():

    # ========================================================
    # 1. Dask
    # ========================================================

    print("Starting Dask client...")

    client = Client(
        n_workers=2,
        threads_per_worker=5
    )

    print(client)
    print("Dashboard:", client.dashboard_link)


    # ========================================================
    # 2. Paths
    # ========================================================

    SSS_MY_PATH = ("data/raw/sss/multiobs_sss_my_2015_2024-12-15_0125deg.zarr")

    SSS_NRT_PATH = ("data/raw/sss/multiobs_sss_nrt_2024-12-16_2024-12-31_0125deg.zarr")

    REFERENCE_GRID_PATH = ("data/processed/reference_grid_025deg.zarr")


    SSS_PROCESSED_DIR = "data/processed/sss"
    os.makedirs(SSS_PROCESSED_DIR, exist_ok=True)

    OUTPUT_PATH = os.path.join(SSS_PROCESSED_DIR, "sss_2015_2024_025deg.zarr")

    WEIGHTS_PATH = os.path.join(SSS_PROCESSED_DIR, "sss_to_025deg_conservative_normed_10yr.nc")

    MASK_REPORT_PATH = os.path.join(SSS_PROCESSED_DIR, "sss_mask_report.html")

    REGRID_REPORT_PATH = os.path.join(SSS_PROCESSED_DIR, "sss_regridding_report.html")


    # ========================================================
    # 3. Configuration
    # ========================================================

    INPUT_CHUNKS = {
        "time": 90,
        "latitude": -1,
        "longitude": -1,
    }

    # Copernicus SSS products normally use "sos".
    # Confirm with print(sss_my.data_vars) if necessary.
    SSS_VAR = "sos"


    # ========================================================
    # 4. Open MY dataset
    # ========================================================

    print("\nOpening MY SSS...")

    sss_my_ds = xr.open_zarr(
        SSS_MY_PATH,
        chunks=INPUT_CHUNKS,
    )

    print(sss_my_ds)
    print("\nMY variables:")
    print(sss_my_ds.data_vars)


    # ========================================================
    # 5. Open NRT dataset
    # ========================================================

    print("\nOpening NRT SSS...")

    sss_nrt_ds = xr.open_zarr(
        SSS_NRT_PATH,
        chunks=INPUT_CHUNKS,
    )

    print(sss_nrt_ds)
    print("\nNRT variables:")
    print(sss_nrt_ds.data_vars)


    # ========================================================
    # 6. Select SSS variable
    # ========================================================

    print("\nSelecting SSS variable...")

    sss_my = sss_my_ds[SSS_VAR].squeeze("depth", drop=True)
    sss_nrt = sss_nrt_ds[SSS_VAR].squeeze("depth", drop=True)

    print("MY:")
    print(sss_my)

    print("\nNRT:")
    print(sss_nrt)


    # ========================================================
    # 7. Concatenate MY + NRT
    # ========================================================

    print("\nConcatenating MY + NRT along time...")

    sss = xr.concat(
        [sss_my, sss_nrt],
        dim="time",
    )

    # Sort chronologically
    sss = sss.sortby("time")

    print("\nCombined SSS:")
    print(sss)


    # ========================================================
    # 8. Check time coordinates
    # ========================================================

    print("\nChecking time axis...")

    time_values = sss["time"].values

    print("First time :", time_values[0])
    print("Last time  :", time_values[-1])
    print("Number of timesteps:", len(time_values))

    unique_times, counts = np.unique(
        time_values,
        return_counts=True
    )

    duplicate_times = unique_times[counts > 1]

    if len(duplicate_times) > 0:

        print(
            f"WARNING: Found {len(duplicate_times)} "
            "duplicate timestamps."
        )

        print(duplicate_times)

        # Keep first occurrence
        _, unique_indices = np.unique(
            time_values,
            return_index=True
        )

        unique_indices = np.sort(unique_indices)

        sss = sss.isel(time=unique_indices)

        print(
            "Duplicate timestamps removed."
        )

    else:

        print("No duplicate timestamps found.")


    # ========================================================
    # 9. Open reference grid
    # ========================================================

    print("\nOpening reference grid...")

    reference_grid = xr.open_zarr(REFERENCE_GRID_PATH)

    print(reference_grid)


    # ========================================================
    # 10. Create source grid
    # ========================================================

    print("\nCreating source grid...")

    source_lat = sss["latitude"].values
    source_lon = sss["longitude"].values


    # ========================================================
    # Source latitude boundaries
    # ========================================================

    source_lat_b = np.concatenate([
        [source_lat[0] - (source_lat[1] - source_lat[0]) / 2],
        (source_lat[:-1] + source_lat[1:]) / 2,
        [source_lat[-1] + (source_lat[-1] - source_lat[-2]) / 2],
    ])


    # ========================================================
    # Source longitude boundaries
    # ========================================================

    source_lon_b = np.concatenate([
        [source_lon[0] - (source_lon[1] - source_lon[0]) / 2],
        (source_lon[:-1] + source_lon[1:]) / 2,
        [source_lon[-1] + (source_lon[-1] - source_lon[-2]) / 2],
    ])


    source_grid = xr.Dataset(
        coords={
            "latitude": (
                "latitude",
                source_lat,
            ),

            "longitude": (
                "longitude",
                source_lon,
            ),

            "lat_b": (
                "lat_b",
                source_lat_b,
            ),

            "lon_b": (
                "lon_b",
                source_lon_b,
            ),
        }
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


    
    with performance_report(filename=MASK_REPORT_PATH):
        # convert float64 to float32
        print("\nConverting SSS to float32...")

        sss = sss.astype("float32")


        # ========================================================
        # 13. Create 10-year source mask
        # ========================================================

        print("\nCreating 10-year source mask...")
        print("Checking all SSS observations to determine which source cells contain valid data...")

        source_mask = (
            sss
            .notnull()
            .any(dim="time")
            .compute()
            .astype(np.int8)
        )

        valid_cells = int(
            source_mask.values.sum()
        )

        total_cells = int(
            source_mask.size
        )

        print(
            f"Valid source cells: "
            f"{valid_cells:,} / {total_cells:,}"
        )

        source_grid["mask"] = (
            ("latitude", "longitude"),
            source_mask.values,
        )


        # ========================================================
        # 14. Create xESMF regridder
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
    # 15. Regrid
    # ========================================================

    print("\nRegridding complete 2015-2024 SSS dataset...")

    with performance_report(filename=REGRID_REPORT_PATH):

        sss_025 = regridder(
            sss,
            keep_attrs=True,
            skipna=True,
            na_thres=0.50,
        )


        # converting back to float32 since regridding casts it to float64
        sss_025 = sss_025.astype("float32")


        # ====================================================
        # 17. Metadata
        # ====================================================

        sss_025.name = SSS_VAR

        sss_025.attrs["regridding_method"] = (
            "xESMF conservative_normed"
        )

        sss_025.attrs["source_resolution"] = (
            "0.125 degree"
        )

        sss_025.attrs["target_resolution"] = (
            "0.25 degree"
        )

        sss_025.attrs["source_grid_coordinate_convention"] = (
            "cell centers"
        )

        sss_025.attrs["target_grid_coordinate_convention"] = (
            "cell centers"
        )

        sss_025.attrs["missing_value_handling"] = (
            "adaptive masking with "
            "skipna=True, na_thres=0.50"
        )


        # ====================================================
        # 18. Convert to Dataset
        # ====================================================

        sss_025_ds = sss_025.to_dataset()


        # ====================================================
        # 19. Final ML-oriented chunking
        # ====================================================

        print("\nApplying final chunking...")

        sss_025_ds = sss_025_ds.chunk({
            "time": 1,
            "latitude": -1,
            "longitude": -1,
        })


        # ====================================================
        # 20. Remove inherited encodings
        # ====================================================

        print("\nCleaning variable encodings...")

        for var in sss_025_ds.variables:
            sss_025_ds[var].encoding = {}


        # ====================================================
        # 21. Inspect final dataset
        # ====================================================

        print("\nFinal dataset:")
        print(sss_025_ds)

        print("\nFinal Dask chunks:")
        print(sss_025_ds[SSS_VAR].chunks)


        # ====================================================
        # 22. Write Zarr 3
        # ====================================================

        print("\nWriting Zarr 3 output...")

        sss_025_ds.to_zarr(
            OUTPUT_PATH,
            mode="w",
            zarr_format=3,
        )


    print("\n10-YEAR SSS REGRIDDING COMPLETE")

    print("Output :")
    print(OUTPUT_PATH)

    print("\nWeights :")
    print(WEIGHTS_PATH)

    print("\nFormat : Zarr 3")

    client.close()


if __name__ == "__main__":
    main()
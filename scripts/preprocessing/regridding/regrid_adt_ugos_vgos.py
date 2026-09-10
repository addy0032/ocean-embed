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
        n_workers=1,
        threads_per_worker=10
    )

    print(client)
    print("Dashboard:", client.dashboard_link)


    # ========================================================
    # 2. Paths
    # ========================================================

    SEALEVEL_PATH = (
        "data/raw/sealevel/"
        "cmems_sealevel_adt_ugos_vgos_2015_2024_0125deg.zarr"
    )

    REFERENCE_GRID_PATH = (
        "data/processed/reference_grid_025deg.zarr"
    )

    SEALEVEL_PROCESSED_DIR = (
        "data/processed/sealevel"
    )

    os.makedirs(
        SEALEVEL_PROCESSED_DIR,
        exist_ok=True
    )

    OUTPUT_PATH = os.path.join(
        SEALEVEL_PROCESSED_DIR,
        "sealevel_2015_2024_025deg.zarr"
    )

    WEIGHTS_PATH = os.path.join(
        SEALEVEL_PROCESSED_DIR,
        "sealevel_to_025deg_conservative_normed_10yr.nc"
    )

    MASK_REPORT_PATH = os.path.join(
        SEALEVEL_PROCESSED_DIR,
        "sealevel_mask_report.html"
    )

    REGRID_REPORT_PATH = os.path.join(
        SEALEVEL_PROCESSED_DIR,
        "sealevel_regridding_report.html"
    )


    # ========================================================
    # 3. Configuration
    # ========================================================

    VARIABLES = [
        "adt",
        "ugos",
        "vgos",
    ]


    # ========================================================
    # 4. Open sea-level dataset
    # ========================================================

    print("\nOpening sea-level dataset...")

    sealevel_ds = xr.open_zarr(
        SEALEVEL_PATH,
        chunks={
            "time": 720,
            "latitude": -1,
            "longitude": -1,
        }
    )

    print(sealevel_ds)

    print("\nSea-level variables:")
    print(sealevel_ds.data_vars)


    # ========================================================
    # 5. Select variables
    # ========================================================

    print("\nSelecting ADT, UGOS and VGOS...")

    adt = sealevel_ds["adt"].astype("float32")
    ugos = sealevel_ds["ugos"].astype("float32")
    vgos = sealevel_ds["vgos"].astype("float32")

    print("\nADT:")
    print(adt)

    print("\nUGOS:")
    print(ugos)

    print("\nVGOS:")
    print(vgos)


    # ========================================================
    # 6. Check time axis
    # ========================================================

    print("\nChecking time axis...")

    time_values = sealevel_ds["time"].values

    print("First time :", time_values[0])
    print("Last time  :", time_values[-1])
    print("Number of timesteps:", len(time_values))


    # ========================================================
    # 7. Open reference grid
    # ========================================================

    print("\nOpening reference grid...")

    reference_grid = xr.open_zarr(
        REFERENCE_GRID_PATH
    )

    print(reference_grid)


    # ========================================================
    # 8. Create source grid
    # ========================================================

    print("\nCreating source grid...")

    source_lat = sealevel_ds["latitude"].values
    source_lon = sealevel_ds["longitude"].values

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
            "latitude": (
                "latitude",
                source_lat
            ),

            "longitude": (
                "longitude",
                source_lon
            ),

            "lat_b": (
                "lat_b",
                source_lat_b
            ),

            "lon_b": (
                "lon_b",
                source_lon_b
            ),
        }
    )


    # ========================================================
    # 9. Create target grid
    # ========================================================

    print("\nCreating target grid...")

    target_grid = xr.Dataset(
        coords={
            "latitude": (
                "latitude",
                reference_grid["latitude"].values
            ),

            "longitude": (
                "longitude",
                reference_grid["longitude"].values
            ),

            "lat_b": (
                "lat_b",
                reference_grid["lat_b"].values
            ),

            "lon_b": (
                "lon_b",
                reference_grid["lon_b"].values
            ),
        }
    )


    # ========================================================
    # 10. Create source mask
    # ========================================================

    print("\nCreating 10-year source mask...")

    print(
        "Checking ADT observations to identify source "
        "cells containing valid data..."
    )

    with performance_report(
        filename=MASK_REPORT_PATH
    ):

        # ADT is used for the static mask because it is the
        # primary sea-level field and represents ocean coverage.
        source_mask = (
            adt
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
    # 11. Create xESMF regridder
    # ========================================================

    print("\nCreating xESMF regridder...")

    with performance_report(
        filename=REGRID_REPORT_PATH
    ):

        regridder = xe.Regridder(
            source_grid,
            target_grid,
            method="conservative_normed",
            filename=WEIGHTS_PATH,
            reuse_weights=False,
        )

        print(regridder)


        # ====================================================
        # 12. Regrid ADT
        # ====================================================

        print("\nRegridding ADT...")

        adt_025 = regridder(
            adt,
            keep_attrs=True,
            skipna=True,
            na_thres=0.50,
            output_chunks={
                "time": 1,
                "latitude": -1,
                "longitude": -1,
            }
        )


        # ====================================================
        # 13. Regrid UGOS
        # ====================================================

        print("\nRegridding UGOS...")

        ugos_025 = regridder(
            ugos,
            keep_attrs=True,
            skipna=True,
            na_thres=0.50,
            output_chunks={
                "time": 1,
                "latitude": -1,
                "longitude": -1,
            }
        )


        # ====================================================
        # 14. Regrid VGOS
        # ====================================================

        print("\nRegridding VGOS...")

        vgos_025 = regridder(
            vgos,
            keep_attrs=True,
            skipna=True,
            na_thres=0.50,
            output_chunks={
                "time": 1,
                "latitude": -1,
                "longitude": -1,
            }
        )


        # ====================================================
        # 15. Convert back to float32
        # ====================================================

        print("\nConverting regridded data to float32...")

        adt_025 = adt_025.astype("float32")
        ugos_025 = ugos_025.astype("float32")
        vgos_025 = vgos_025.astype("float32")


        # ====================================================
        # 16. Metadata
        # ====================================================

        adt_025.name = "adt"
        ugos_025.name = "ugos"
        vgos_025.name = "vgos"


        for variable in (
            adt_025,
            ugos_025,
            vgos_025,
        ):

            variable.attrs["regridding_method"] = (
                "xESMF conservative_normed"
            )

            variable.attrs["source_resolution"] = (
                "0.125 degree"
            )

            variable.attrs["target_resolution"] = (
                "0.25 degree"
            )

            variable.attrs["source_grid_coordinate_convention"] = (
                "cell centers"
            )

            variable.attrs["target_grid_coordinate_convention"] = (
                "cell centers"
            )

            variable.attrs["missing_value_handling"] = (
                "conservative_normed regridding with "
                "skipna=True and na_thres=0.50"
            )


        # ====================================================
        # 17. Create output dataset
        # ====================================================

        print("\nCreating output dataset...")

        sealevel_025_ds = xr.Dataset({
            "adt": adt_025,
            "ugos": ugos_025,
            "vgos": vgos_025,
        })


        # ====================================================
        # 18. Final ML-oriented chunking
        # ====================================================

        print("\nApplying final chunking...")

        sealevel_025_ds = sealevel_025_ds.chunk({
            "time": 1,
            "latitude": -1,
            "longitude": -1,
        })


        # ====================================================
        # 19. Remove inherited encodings
        # ====================================================

        print("\nCleaning variable encodings...")

        for var in sealevel_025_ds.variables:
            sealevel_025_ds[var].encoding = {}


        # ====================================================
        # 20. Inspect final dataset
        # ====================================================

        print("\nFinal dataset:")
        print(sealevel_025_ds)

        print("\nFinal ADT chunks:")
        print(sealevel_025_ds["adt"].chunks)

        print("\nFinal UGOS chunks:")
        print(sealevel_025_ds["ugos"].chunks)

        print("\nFinal VGOS chunks:")
        print(sealevel_025_ds["vgos"].chunks)


        # ====================================================
        # 21. Write Zarr 3
        # ====================================================

        print("\nWriting Zarr 3 output...")

        sealevel_025_ds.to_zarr(
            OUTPUT_PATH,
            mode="w",
            zarr_format=3,
        )


    # ========================================================
    # 22. Done
    # ========================================================

    print("\n10-YEAR SEA-LEVEL REGRIDDING COMPLETE")

    print("\nOutput:")
    print(OUTPUT_PATH)

    print("\nWeights:")
    print(WEIGHTS_PATH)

    print("\nFormat: Zarr 3")


    client.close()


if __name__ == "__main__":
    main()
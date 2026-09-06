from dask.distributed import Client, performance_report

import numpy as np
import xarray as xr
import xesmf as xe

def graph_size_mb(obj):
    try:
        graph = obj.__dask_graph__()
        if graph is None:
            return 0
        return len(graph) / 1024 / 1024
    except AttributeError:
        return 0

# ============================================================
# Dask
# ============================================================
def main():
    print("Starting Dask client...")

    client = Client(n_workers=2, threads_per_worker=5)

    print(client)
    print("Dashboard:", client.dashboard_link)

    # ============================================================
    # Paths
    # ============================================================

    SST_PATH = "data/raw/sst/c3s_sst_2015_2024_005deg.zarr"
    REFERENCE_GRID_PATH = "data/processed/reference_grid_025deg.zarr"

    OUTPUT_PATH = "data/processed/sst_2015_2024_025deg.zarr"
    WEIGHTS_PATH = ("data/processed/sst_to_025deg_conservative_normed_10yr.nc")


    # ============================================================
    # Configuration
    # ============================================================

    INPUT_CHUNKS = {
        "time": 90,
        "latitude": -1,
        "longitude": -1,
    }


    # ============================================================
    # 1. Open SST
    # ============================================================

    print("Opening SST...")

    sst_ds = xr.open_zarr(
        SST_PATH,
        chunks=INPUT_CHUNKS,
    )

    print(sst_ds)


    # ============================================================
    # 2. Open reference grid
    # ============================================================

    print("\nOpening reference grid...")

    reference_grid = xr.open_zarr(
        REFERENCE_GRID_PATH
    )

    print(reference_grid)


    # ============================================================
    # 3. Create source grid
    # ============================================================

    print("\nCreating source grid...")

    source_lat = sst_ds["latitude"].values
    source_lon = sst_ds["longitude"].values


    # ------------------------------------------------------------
    # Source latitude cell boundaries
    # ------------------------------------------------------------

    source_lat_b = np.concatenate([
        [
            source_lat[0]
            - (source_lat[1] - source_lat[0]) / 2
        ],
        (source_lat[:-1] + source_lat[1:]) / 2,
        [
            source_lat[-1]
            + (source_lat[-1] - source_lat[-2]) / 2
        ],
    ])


    # ------------------------------------------------------------
    # Source longitude cell boundaries
    # ------------------------------------------------------------

    source_lon_b = np.concatenate([
        [
            source_lon[0]
            - (source_lon[1] - source_lon[0]) / 2
        ],
        (source_lon[:-1] + source_lon[1:]) / 2,
        [
            source_lon[-1]
            + (source_lon[-1] - source_lon[-2]) / 2
        ],
    ])


    source_grid = xr.Dataset(
        coords={
            "latitude": ("latitude", source_lat),
            "longitude": ("longitude", source_lon),
            "lat_b": ("lat_b", source_lat_b),
            "lon_b": ("lon_b", source_lon_b),
        }
    )

    with performance_report(filename="data/processed/sst_mask_report.html"):
        # convert float64 to float32
        sst_ds["analysed_sst"] = sst_ds["analysed_sst"].astype("float32")
        print("After astype:", graph_size_mb(sst_ds), "MB")

        # ============================================================
        # 4. Create 10-year static source mask
        # ============================================================

        print("\nCreating 10-year source mask...")

        print(
            "Checking all 2015-2024 SST observations "
            "to identify cells that are valid at least once..."
        )

        source_mask = (
            sst_ds["analysed_sst"]
            .notnull()
            .any(dim="time")
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


        # ============================================================
        # 5. Create target grid
        # ============================================================

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

        # ============================================================
        # 7. Create fresh xESMF regridder
        # ============================================================

        print("\nCreating FRESH xESMF regridder...")

        regridder = xe.Regridder(
            source_grid,
            target_grid,
            method="conservative_normed",
            filename=WEIGHTS_PATH,
            reuse_weights=False
        )

    print(regridder)


    # ============================================================
    # 8. Select complete 2015-2024 dataset
    # ============================================================

    print("\nPreparing complete 2015-2024 SST dataset...")

    sst = sst_ds["analysed_sst"].sel(
        time=slice("2015-01-01", "2024-12-31")
    )

    print("\nInput SST:")
    print(sst)


    # ============================================================
    # 9. Regrid entire 10-year dataset
    # ============================================================

    print("\nRegridding entire 2015-2024 dataset...")

    with performance_report(filename="data/processed/sst_regridding_report.html"):
        sst_025 = regridder(
            sst,
            keep_attrs=True,
            skipna=True,
            na_thres=0.50
        )

        print("After regrid:", graph_size_mb(sst_025), "MB")

        # converting back to float32 since regridding casts it to float64
        sst_025 = sst_025.astype("float32")
        print("After final astype:", graph_size_mb(sst_025), "MB")

        # ============================================================
        # 10. Add metadata
        # ============================================================

        sst_025.name = "analysed_sst"

        sst_025.attrs["regridding_method"] = (
            "xESMF conservative_normed"
        )

        sst_025.attrs["source_resolution"] = (
            "0.05 degree"
        )

        sst_025.attrs["target_resolution"] = (
            "0.25 degree"
        )

        sst_025.attrs["source_grid_coordinate_convention"] = (
            "cell centers"
        )

        sst_025.attrs["target_grid_coordinate_convention"] = (
            "cell centers"
        )

        sst_025.attrs["missing_value_handling"] = (
            "adaptive masking with skipna=True, na_thres=0.50"
        )


        # ============================================================
        # 11. Convert to Dataset
        # ============================================================

        sst_025_ds = sst_025.to_dataset()


        # ============================================================
        # 12. Final ML-oriented chunking
        # ============================================================

        print("\nApplying final chunking...")

        sst_025_ds = sst_025_ds.chunk({
            "time": 1,
            "latitude": -1,
            "longitude": -1,
        })

        print("After rechunk:", graph_size_mb(sst_025_ds), "MB")

        # ============================================================
        # 13. Remove inherited Zarr v2 encodings
        # ============================================================

        print("\nCleaning variable encodings...")

        for var in sst_025_ds.variables:
            sst_025_ds[var].encoding = {}


        # ============================================================
        # 14. Inspect final dataset
        # ============================================================

        print("\nFinal dataset:")
        print(sst_025_ds)

        print("\nFinal Dask chunks:")
        print(sst_025_ds["analysed_sst"].chunks)


        # ============================================================
        # 15. Write Zarr 3
        # ============================================================

        print("\nWriting Zarr 3 output...")

        sst_025_ds.to_zarr(
            OUTPUT_PATH,
            mode="w",
            zarr_format=3,
        )


    print("\n============================================")
    print("10-YEAR SST REGRIDDING COMPLETE")
    print("============================================")
    print(f"Output : {OUTPUT_PATH}")
    print(f"Weights: {WEIGHTS_PATH}")
    print("Format : Zarr 3")

    client.close()

if __name__ == "__main__":
    main()
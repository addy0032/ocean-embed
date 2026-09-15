import xarray as xr


INPUT_PATH = "data/processed/X_inputs_025deg.zarr"
OUTPUT_PATH = "data/processed/X_inputs_025deg_t7.zarr"


def main():

    print("Opening original dataset...")

    ds = xr.open_zarr(INPUT_PATH)

    print("\nOriginal chunks:")
    for name in ds.data_vars:
        print(f"{name}: {ds[name].chunks}")

    print("\nApplying 7-day temporal chunks...")

    ds = ds.chunk({
        "time": 7,
        "latitude": 101,
        "longitude": 241,
    })

    print("\nNew chunks:")
    for name in ds.data_vars:
        print(f"{name}: {ds[name].chunks}")

    print("\nWriting new Zarr dataset...")

    ds.to_zarr(
        OUTPUT_PATH,
        mode="w",
    )

    print("\nFinished.")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
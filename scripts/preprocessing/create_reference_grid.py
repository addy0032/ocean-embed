import xarray as xr
import numpy as np

# reference grid cell centers

lat = np.arange(5.0, 30.0 + 0.25, 0.25)
lon = np.arange(45.0, 105.0 + 0.25, 0.25)


# cell boundaries

lat_b = np.concatenate([
    [lat[0] - 0.125],
    (lat[:-1] + lat[1:]) / 2,
    [lat[-1] + 0.125],
])

lon_b = np.concatenate([
    [lon[0] - 0.125],
    (lon[:-1] + lon[1:]) / 2,
    [lon[-1] + 0.125],
])


# create reference grid

reference_grid = xr.Dataset(
    coords={
        "latitude": lat,
        "longitude": lon,
        "lat_b": ("lat_b", lat_b),
        "lon_b": ("lon_b", lon_b),
    },
    attrs={
        "description": "0.25 degree reference grid",
        "latitude_range": "5N to 30N",
        "longitude_range": "45E to 105E",
        "resolution": "0.25 degree",
        "coordinate_convention": "cell centers",
    }
)

# save
reference_grid.to_zarr(
    "data/processed/reference_grid_025deg.zarr",
    mode="w",
    zarr_format=3
)

print(reference_grid)
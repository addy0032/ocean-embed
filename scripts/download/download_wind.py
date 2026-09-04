import copernicusmarine

copernicusmarine.subset(
    dataset_id="cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H",
    variables=["eastward_wind", "northward_wind"],

    minimum_longitude=45,
    maximum_longitude=105,
    minimum_latitude=5,
    maximum_latitude=30,

    start_datetime="2015-01-01T00:00:00",
    end_datetime="2024-12-31T23:59:59",

    coordinates_selection_method="outside",

    file_format="zarr",
    output_filename="cmems_wind_2015_2024_0125deg_hourly.zarr",
    output_directory="data/raw/wind",

    disable_progress_bar=False,
    overwrite=False
)
import copernicusmarine

copernicusmarine.subset(
    dataset_id="cmems_obs-mob_glo_phy-sss_my_multi_P1D",
    variables=["sos"],

    minimum_longitude=45,
    maximum_longitude=105,
    minimum_latitude=5,
    maximum_latitude=30,

    start_datetime="2015-01-01T00:00:00",
    end_datetime="2024-12-15T00:00:00",

    coordinates_selection_method="outside",

    file_format="zarr",
    output_filename="multiobs_sss_my_2015_2024-12-15_0125deg.zarr",
    output_directory="data/raw/sss",

    disable_progress_bar=False,
    overwrite=False
)

copernicusmarine.subset(
    dataset_id="cmems_obs-mob_glo_phy-sss_nrt_multi_P1D",
    variables=["sos"],

    minimum_longitude=45,
    maximum_longitude=105,
    minimum_latitude=5,
    maximum_latitude=30,

    start_datetime="2024-12-16T00:00:00",
    end_datetime="2024-12-31T00:00:00",

    coordinates_selection_method="outside",

    file_format="zarr",
    output_filename="multiobs_sss_nrt_2024-12-16_2024-12-31_0125deg.zarr",
    output_directory="data/raw/sss",

    disable_progress_bar=False,
    overwrite=False
)
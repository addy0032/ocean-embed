import copernicusmarine

copernicusmarine.subset(
    dataset_id="C3S-GLO-SST-L4-REP-OBS-SST",
    variables=["analysed_sst"],

    minimum_longitude=45,
    maximum_longitude=105,
    minimum_latitude=5,
    maximum_latitude=30,

    start_datetime="2015-01-01T00:00:00",
    end_datetime="2024-12-31T00:00:00",

    coordinates_selection_method="outside",

    file_format="zarr",
    output_filename="c3s_sst_2015_2024_005deg.zarr",
    output_directory="data/raw/sst",

    disable_progress_bar=False,
    overwrite=False
)
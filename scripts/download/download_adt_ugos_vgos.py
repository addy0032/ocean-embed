import copernicusmarine

copernicusmarine.subset(
    dataset_id="cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D",
    variables=["adt", "ugos", "vgos"],

    minimum_longitude=45,
    maximum_longitude=105,
    minimum_latitude=5,
    maximum_latitude=30,

    start_datetime="2015-01-01T00:00:00",
    end_datetime="2024-12-31T00:00:00",

    coordinates_selection_method="outside",

    file_format="zarr",
    output_filename="cmems_sealevel_adt_ugos_vgos_2015_2024_0125deg.zarr",
    output_directory="data/raw/sealevel",

    disable_progress_bar=False,
    overwrite=False
)
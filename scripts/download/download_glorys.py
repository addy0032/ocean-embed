import copernicusmarine

# copernicusmarine.subset(
#     dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m",
#     variables=["thetao"],

#     minimum_longitude=45,
#     maximum_longitude=105,
#     minimum_latitude=5,
#     maximum_latitude=30,

#     start_datetime="2015-01-01T00:00:00",
#     end_datetime="2024-12-31T00:00:00",

#     minimum_depth=0.49402499198913574,
#     maximum_depth=1062.43994140625,

#     coordinates_selection_method="outside",

#     file_format="zarr",
#     output_filename="glorys_thetao_2015_2024_0083deg.zarr",
#     output_directory="data/raw/glorys",

#     chunk_size_limit=100,

#     disable_progress_bar=False,
#     overwrite=False,
# )

def main():
    copernicusmarine.subset_split_on(
        dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m",
        variables=["thetao"],

        minimum_longitude=45,
        maximum_longitude=105,
        minimum_latitude=5,
        maximum_latitude=30,

        start_datetime="2015-01-01T00:00:00",
        end_datetime="2024-12-31T00:00:00",

        minimum_depth=0.49402499198913574,
        maximum_depth=1062.43994140625,

        coordinates_selection_method="outside",

        file_format="zarr",
        output_directory="data/raw/glorys",

        on_time="year",
        concurrent_processes=2,

        chunk_size_limit=100,

        disable_progress_bar=False,
        overwrite=False,
    )


if __name__ == "__main__":
    main()
import xarray as xr

import holoviews as hv
import hvplot.xarray
import panel as pn


# sst_ds = xr.open_zarr(
#     "data/raw/sst/c3s_sst_2015_2024_005deg.zarr"
# )

# sst_ds = xr.open_zarr(
#     "data/processed/sst_test_30days_025deg.zarr"
# )

sst_ds = xr.open_zarr(
    "data/processed/sst_2015_2024_025deg.zarr"
)

sst = sst_ds["analysed_sst"]

hv.extension("bokeh")
pn.extension()

dates = sst.time.values

date_slider = pn.widgets.DiscreteSlider(
    name="Date",
    options=list(dates),
    value=dates[0]
)

def plot_day(date):

    day = (
        sst.sel(
            time=date,
            latitude=slice(5, 30),
            longitude=slice(45, 105)
        )
        .compute()
        - 273.15
    )

    day = day.assign_attrs(
        units="°C",
        long_name="Sea Surface Temperature"
    )

    return day.hvplot.image(
        x="longitude",
        y="latitude",
        cmap="Turbo",
        clim=(20, 32),
        colorbar=True,
        width=900,
        height=600
    )

plot = pn.bind(plot_day, date_slider)

pn.Column(
    date_slider,
    plot
).servable()
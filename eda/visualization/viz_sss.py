import xarray as xr

import holoviews as hv
import hvplot.xarray
import panel as pn


# sss_ds = xr.open_zarr(
#     "data/raw/sss/multiobs_sss_my_2015_2024-12-15_0125deg.zarr"
# )
# sss = sss_ds["sos"].squeeze("depth", drop=True)

sss_ds = xr.open_zarr(
    "data/processed/sss/sss_2015_2024_025deg.zarr"
)

sss = sss_ds["sos"]

hv.extension("bokeh")
pn.extension()

dates = sss.time.values

date_slider = pn.widgets.DiscreteSlider(
    name="Date",
    options=list(dates),
    value=dates[0]
)


def plot_day(date):

    day = (
        sss.sel(
            time=date,
            latitude=slice(5, 30),
            longitude=slice(45, 105)
        )
        .compute()
    )

    day = day.assign_attrs(
        units="PSU",
        long_name="Sea Surface Salinity"
    )

    return day.hvplot.image(
        x="longitude",
        y="latitude",
        cmap="Turbo",
        clim=(30, 37),
        colorbar=True,
        width=900,
        height=600
    )


plot = pn.bind(plot_day, date_slider)

pn.Column(
    date_slider,
    plot
).servable()
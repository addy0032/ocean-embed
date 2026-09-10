import xarray as xr
import numpy as np

import holoviews as hv
import hvplot.xarray
import panel as pn

# Open dataset

sealevel_ds = xr.open_zarr(
    "data/processed/sealevel/sealevel_2015_2024_025deg.zarr"
)

adt = sealevel_ds["adt"]

# Panel / HoloViews

hv.extension("bokeh")
pn.extension()

# Dates

dates = adt.time.values

# Date picker

date_picker = pn.widgets.DatePicker(
    name="Select date",
    value=dates[0].astype("datetime64[D]").astype(object),
)

# Date slider

date_slider = pn.widgets.DiscreteSlider(
    name="Browse dates",
    options=list(dates),
    value=dates[0],
)

# Keep picker and slider synchronized

_syncing = False


def slider_to_picker(event):

    global _syncing

    if _syncing:
        return

    _syncing = True

    date = np.datetime64(event.new, "D")

    date_picker.value = date.astype(object)

    _syncing = False


def picker_to_slider(event):

    global _syncing

    if _syncing or event.new is None:
        return

    _syncing = True

    selected_date = np.datetime64(event.new, "D")

    matches = np.where(
        dates.astype("datetime64[D]") == selected_date
    )[0]

    if len(matches) > 0:
        date_slider.value = dates[matches[0]]

    _syncing = False


date_slider.param.watch(
    slider_to_picker,
    "value",
)

date_picker.param.watch(
    picker_to_slider,
    "value",
)

# Plot one day

def plot_day(date):

    # Select one day and region

    adt_day = (
        adt
        .sel(
            time=date,
            latitude=slice(5, 30),
            longitude=slice(45, 105),
        )
        .compute()
    )

    # ADT plot

    adt_plot = adt_day.hvplot.image(
        x="longitude",
        y="latitude",
        cmap="Turbo",
        clim=(-1.5, 1.5),
        colorbar=True,
        clabel="Absolute dynamic topography (m)",
        width=900,
        height=600,
        xlabel="Longitude",
        ylabel="Latitude",
        title=(
            "Absolute Dynamic Topography — "
            f"{np.datetime_as_string(date, unit='D')}"
        ),
    )


    return adt_plot

# Bind plot

plot = pn.bind(
    plot_day,
    date_slider,
)

# Panel application

pn.Column(
    pn.Row(
        date_picker,
    ),
    date_slider,
    plot,
).servable()
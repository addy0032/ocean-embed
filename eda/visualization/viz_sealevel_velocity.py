import xarray as xr
import numpy as np

import holoviews as hv
import hvplot.xarray
import panel as pn

# Open dataset

sealevel_ds = xr.open_zarr(
    "data/processed/sealevel/sealevel_2015_2024_025deg.zarr"
)

ugos = sealevel_ds["ugos"]
vgos = sealevel_ds["vgos"]

# Panel / HoloViews

hv.extension("bokeh")
pn.extension()

# Dates

dates = ugos.time.values

date_strings = [
    np.datetime_as_string(date, unit="D")
    for date in dates
]

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

    # Find exact matching date
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
    
    ugos_day = (
        ugos
        .sel(
            time=date,
            latitude=slice(5, 30),
            longitude=slice(45, 105),
        )
        .compute()
    )

    vgos_day = (
        vgos
        .sel(
            time=date,
            latitude=slice(5, 30),
            longitude=slice(45, 105),
        )
        .compute()
    )

    # Velocity magnitude

    speed_day = np.sqrt(
        ugos_day**2 +
        vgos_day**2
    )

    speed_day.name = "Geostrophic velocity magnitude"

    # Downsample vectors

    step = 4

    ugos_arrow = ugos_day.isel(
        latitude=slice(None, None, step),
        longitude=slice(None, None, step),
    )

    vgos_arrow = vgos_day.isel(
        latitude=slice(None, None, step),
        longitude=slice(None, None, step),
    )

    # VectorField coordinates

    lon = ugos_arrow.longitude.values
    lat = ugos_arrow.latitude.values

    lon2d, lat2d = np.meshgrid(
        lon,
        lat,
    )

    # Direction and magnitude

    angle = np.arctan2(
        vgos_arrow.values,
        ugos_arrow.values,
    )

    magnitude = np.sqrt(
        ugos_arrow.values**2 +
        vgos_arrow.values**2
    )

    # Background magnitude plot

    speed_plot = speed_day.hvplot.image(
        x="longitude",
        y="latitude",
        cmap="Turbo",
        clim=(0, 1.5),
        colorbar=True,
        clabel="Velocity magnitude (m/s)",
        width=900,
        height=600,
        xlabel="Longitude",
        ylabel="Latitude",
        title=(
            "Surface Geostrophic Velocity — "
            f"{np.datetime_as_string(date, unit='D')}"
        ),
    )


    # Velocity vectors

    vectors = hv.VectorField(
        (
            lon2d,
            lat2d,
            angle,
            magnitude,
        ),
        vdims=[
            "Angle",
            "Magnitude",
        ],
    ).opts(
        color="black",
        line_width=1.5,
        pivot="mid",
        magnitude="Magnitude",
    )


    return speed_plot * vectors


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
import xarray as xr
import numpy as np

import holoviews as hv
import hvplot.xarray
import panel as pn

# Open wind dataset

wind_ds = xr.open_zarr("data/processed/wind/wind_2015_2024_025deg.zarr",)

u = wind_ds["eastward_wind"]
v = wind_ds["northward_wind"]

# Wind speed from daily-mean U and V
speed = np.sqrt(u**2 + v**2)

hv.extension("bokeh")
pn.extension()

# Date slider

dates = speed.time.values

date_slider = pn.widgets.DiscreteSlider(
    name="Date",
    options=list(dates),
    value=dates[0]
)

# Plot one day

def plot_day(date):
    
    # Select one day  

    u_day = (u.sel(time=date, latitude=slice(5, 30), longitude=slice(45, 105)).compute())

    v_day = (v.sel(time=date, latitude=slice(5, 30), longitude=slice(45, 105)).compute())

    speed_day = np.sqrt(u_day**2 + v_day**2)

    # Downsample arrows 

    step = 6

    u_arrow = u_day.isel(
        latitude=slice(None, None, step),
        longitude=slice(None, None, step)
    )

    v_arrow = v_day.isel(
        latitude=slice(None, None, step),
        longitude=slice(None, None, step)
    )
    
    # Create angle and magnitude for VectorField
    
    lon = u_arrow.longitude.values
    lat = u_arrow.latitude.values

    lon2d, lat2d = np.meshgrid(lon, lat)

    angle = np.arctan2(
        v_arrow.values,
        u_arrow.values
    )

    magnitude = np.sqrt(
        u_arrow.values**2 +
        v_arrow.values**2
    )

    # Wind speed background
    
    speed_plot = speed_day.hvplot.image(
        x="longitude",
        y="latitude",
        cmap="Turbo",
        clim=(0, 15),
        colorbar=True,
        clabel = "Wind speed (m/s)",
        width=900,
        height=600,
        xlabel="Longitude",
        ylabel="Latitude",
        title=f"Surface Wind — {np.datetime_as_string(date, unit='D')}"
    )

    # Wind vectors

    vectors = hv.VectorField((lon2d, lat2d, angle, magnitude), vdims=["Angle", "Magnitude"]).opts(
        color="black",
        line_width=1.5,
        pivot="mid",
        magnitude="Magnitude"
    )

    return speed_plot * vectors

# Bind slider

plot = pn.bind(plot_day, date_slider)

# Panel application

pn.Column(
    date_slider,
    plot
).servable()
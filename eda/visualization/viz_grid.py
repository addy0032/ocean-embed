import numpy as np
import holoviews as hv
import hvplot

# initialize the Bokeh extension for interactive plots
hv.extension('bokeh')

# define the grid centers and edges
x_centers = np.arange(5, 30.25, 0.25)
y_centers = np.arange(45, 105.25, 0.25)
x_edges = np.arange(4.875, 30.125 + 0.25, 0.25)
y_edges = np.arange(44.875, 105.125 + 0.25, 0.25)

# draw the grid lines using Segments
# Format: (x_start, y_start, x_end, y_end)
h_lines = [(4.875, y, 30.125, y) for y in y_edges]
v_lines = [(x, 44.875, x, 105.125) for x in x_edges]
grid_segments = hv.Segments(h_lines + v_lines).opts(
    color='gray', line_width=0.5, alpha=0.7
)

# create the center dots using a meshgrid
X, Y = np.meshgrid(x_centers, y_centers)
# Flatten the 2D arrays to plot every single center point
centers_scatter = hv.Scatter((X.flatten(), Y.flatten())).opts(
    color='red', size=2, tools=['hover'], active_tools=['wheel_zoom']
)

# combine and configure the perfect 1:1 Tensor Grid
tensor_grid = (grid_segments * centers_scatter).opts(
    title="0.25° Spatial Tensor Grid",
    xlabel="Latitude (5°N to 30°N)",
    ylabel="Longitude (45°E to 105°E)",
    invert_yaxis=True,      # Puts 45 at the top
    xaxis='top',            # Moves X-axis labels to the top
    data_aspect=1,          # Forces squares instead of rectangles!
    width=900,
    height=700,
    show_grid=False
)

# Render the plot
hvplot.show(tensor_grid)
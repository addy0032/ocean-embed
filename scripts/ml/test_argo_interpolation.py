import numpy as np

from scripts.ml.evaluate_argo import interpolate_profile

LEVELS = np.array([0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000], dtype=float)

# linear profile sampled every 4 m from 4 m to 1100 m -> every level recovered exactly
depth = np.arange(4.0, 1100.0, 4.0)
temp = 30.0 - 0.02 * depth
out = interpolate_profile(depth, temp, LEVELS)
assert np.allclose(out[1:], 30.0 - 0.02 * LEVELS[1:])
assert out[0] == temp[0]  # 0 m filled from 4 m (mixed layer)

# first measurement too deep -> no surface fill; big gap -> level left empty
depth = np.concatenate([np.arange(8.0, 400.0, 5.0), [900.0, 1050.0]])
out = interpolate_profile(depth, 30.0 - 0.02 * depth, LEVELS)
assert np.isnan(out[0]) and np.isnan(out[1])
assert np.isnan(out[LEVELS == 500]).all() and np.isnan(out[LEVELS == 700]).all()
assert np.isfinite(out[LEVELS == 1000]).all()

print("argo interpolation OK")

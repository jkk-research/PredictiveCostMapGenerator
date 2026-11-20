"""Fixed-scale colors for the longitudinal GP viewer."""

import numpy as np


def cost_colors(cost):
    stops = np.array([-1., -.25, .05, 1., 3.])
    palette = np.array([[25., 90., 190.], [30., 190., 170.], [240., 200., 35.],
                        [235., 100., 30.], [180., 30., 50.]])
    channels = [np.interp(cost, stops, palette[:, axis]).astype(np.uint32) for axis in range(3)]
    return channels[0] << 16 | channels[1] << 8 | channels[2]

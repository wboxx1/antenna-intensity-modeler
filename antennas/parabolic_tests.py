# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 15:41:41 2018

@author: wboxx
"""

import unittest
import numpy as np
from .parabolic import parameters, near_field_corrections, hazard_plot


class TestParabolic(unittest.TestCase):

    def test_parameters(self):
        params = parameters(2.4, 8.4e9, 400.0, 0.62, 20.0)
        np.testing.assert_allclose(params,
                                   (2.4, 8.4e9, 400.0, 0.62, 20.0, 0.4872, 1290.24, 2.1134, 175.929),
                                   rtol=1e-3)

    def test_near_field_corrections_function(self):
        params = parameters(2.4, 8.4e9, 400.0, 0.62, 20.0)
        fig, ax = near_field_corrections(params, 1)
        lines = ax.lines
        x_plot, y_plot = lines[0].get_data()
        np.testing.assert_allclose(y_plot[0:3], [3.3001617, 3.04134197, 3.1354443], rtol=1e-5)

    def test_hazard_plot_function(self):
        params = parameters(2.4, 8.4e9, 400.0, 0.62, 20.0)
        fig, ax = hazard_plot(params, 10.0)
        lines = ax.lines
        x_plot, y_plot = lines[0].get_data()
        np.testing.assert_allclose(y_plot[0:3], [2.13333333, 2.13333333, 2.13333333], rtol=1e-5)

if __name__ == '__main__':
    unittest.main()

# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 15:41:41 2018

@author: wboxx
"""

import unittest
import numpy as np
from .parabolic import Parabolic

class TestParabolic(unittest.TestCase):
    def setUp(self):
        self.ant = Parabolic(2.4,8.4e9,400,0.62,20)
    
    def test_near_field_corrections_function(self):
        fig, ax = self.ant.near_field_corrections(1)
        lines = ax.lines
        x_plot, y_plot = lines[0].get_data()
        np.testing.assert_allclose(y_plot[0:3], [3.3001617 , 3.04134197, 3.1354443 ],rtol=1e-5)

    def test_hazard_plot_function(self):
        fig, ax = self.ant.hazard_plot(10)
        lines = ax.lines
        x_plot, y_plot = lines[0].get_data()
        np.testing.assert_allclose(y_plot[0:3], [2.13333333, 2.13333333, 2.13333333], rtol=1e-5)
        
if __name__ == '__main__':
    unittest.main()
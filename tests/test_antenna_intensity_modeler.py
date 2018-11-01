# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 13:26:00 2018

@author: wboxx
"""

import unittest

# suite.addTests(loader.loadTestsFromModule(pb))


def test_all():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    # add tests to the test suite
    suite.addTests(loader.discover(start_dir='./src/', pattern='*tests.py'))
    runner = unittest.TextTestRunner(verbosity=3)
    runner.run(suite)


def test_parabolic():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    # add tests to the test suite
    suite.addTests(loader.discover(start_dir='.', pattern='*parabolic*tests.py'))
    runner = unittest.TextTestRunner(verbosity=3)
    runner.run(suite)

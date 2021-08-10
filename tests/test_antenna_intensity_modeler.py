#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Tests for `antenna_intensity_modeler` package."""

import pytest

from click.testing import CliRunner

from antenna_intensity_modeler import parabolic
from antenna_intensity_modeler import cli
import numpy as np


def test_command_line_interface():
    """Test the CLI."""
    runner = CliRunner()
    result = runner.invoke(cli.main)
    assert result.exit_code == 0
    assert "antenna_intensity_modeler.cli.main" in result.output
    help_result = runner.invoke(cli.main, ["--help"])
    assert help_result.exit_code == 0
    assert "--help  Show this message and exit." in help_result.output


@pytest.fixture
def get_params():
    return parabolic.parameters(2.4, 8400.0, 400.0, 0.62, 20.0)


def compare_approximate(first, second):
    """Return whether two dicts of arrays are roughly equal"""
    if first.keys() != second.keys():
        return False
    return all(np.allclose(first[key], second[key], rtol=1e-3) for key in first)


def test_parameters(get_params):
    params = get_params
    fact = {
        "radius_meters": 2.4,
        "freq_mhz": 8400.0,
        "power_watts": 400.0,
        "efficiency": 0.62,
        "side_lobe_ratio": 20.0,
        "H": 0.4872,
        "ffmin": 1290.24,
        "ffpwrden": 2.1134,
        "k": 175.929,
    }
    # np.testing.assert_allclose(params, fact, rtol=1e-3)
    assert compare_approximate(params, fact)


def test_near_field_corrections_function(get_params):
    params = get_params
    xbar = 1.0
    resolution = 1000
    power_norm = parabolic.near_field_corrections(params, xbar, resolution)
    np.testing.assert_allclose(sum(power_norm), 3715, rtol=1)


def test_hazard_plot_function():
    params = parabolic.parameters(2.4, 8400.0, 400.0, 0.62, 20.0)
    df = parabolic.hazard_plot(params, 10.0)
    rng = df.range.values
    positives = df.positives.values
    np.testing.assert_allclose(rng[0:3], [1.29024, 2.58048, 3.87072], rtol=1e-5)

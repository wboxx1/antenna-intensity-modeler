#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Tests for `antenna_intensity_modeler` package."""

import pytest

from click.testing import CliRunner

from antenna_intensity_modeler import parabolic
from antenna_intensity_modeler import cli
import numpy as np


@pytest.fixture
def response():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    # import requests
    # return requests.get('https://github.com/audreyr/cookiecutter-pypackage')


def test_content(response):
    """Sample pytest test function with the pytest fixture as an argument."""
    # from bs4 import BeautifulSoup
    # assert 'GitHub' in BeautifulSoup(response.content).title.string


def test_command_line_interface():
    """Test the CLI."""
    runner = CliRunner()
    result = runner.invoke(cli.main)
    assert result.exit_code == 0
    assert "antenna_intensity_modeler.cli.main" in result.output
    help_result = runner.invoke(cli.main, ["--help"])
    assert help_result.exit_code == 0
    assert "--help  Show this message and exit." in help_result.output


def test_parameters():
    params = parabolic.parameters(2.4, 8400.0, 400.0, 0.62, 20.0)
    test_dict = {
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
    for key in test_dict:
        np.testing.assert_almost_equal(test_dict.get(key), params.get(key), decimal=3)


def test_near_field_corrections_function():
    params = parabolic.parameters(2.4, 8400.0, 400.0, 0.62, 20.0)
    table = parabolic.near_field_corrections(params, 1.0)
    np.testing.assert_allclose(
        [table.delta.sum(), table.Pcorr.sum()], [505, 3715], rtol=1
    )


def test_hazard_plot_function():
    params = parabolic.parameters(2.4, 8400.0, 400.0, 0.62, 20.0)
    df = parabolic.hazard_plot(params, 10.0)
    rng_test = df.range
    positives = df.positives.values
    negatives = df.negatives.values

    delta = np.linspace(1.0, 0.001, 1000)
    ffmin = params.get("ffmin")
    rng_true = delta[::-1] * ffmin
    assert len(rng_test) == len(rng_true)

    pos_first_true = 1.584
    pos_last_true = 0.000
    pos_first_test = positives[0]
    pos_last_test = positives[-1]
    neg_first_true = -pos_first_true
    neg_last_true = -pos_last_true
    neg_first_test = negatives[0]
    neg_last_test = negatives[-1]

    true_array = [pos_first_true, pos_last_true, neg_first_true, neg_last_true]
    test_array = [pos_first_test, pos_last_test, neg_first_test, neg_last_test]
    np.testing.assert_array_almost_equal(true_array, test_array, decimal=3)
    # lines = ax.lines
    # x_plot, y_plot = lines[0].get_data()
    # np.testing.assert_allclose(y_plot[0:3], [2.16, 2.16, 2.16], rtol=1e-5)

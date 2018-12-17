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
    assert 'antenna_intensity_modeler.cli.main' in result.output
    help_result = runner.invoke(cli.main, ['--help'])
    assert help_result.exit_code == 0
    assert '--help  Show this message and exit.' in help_result.output


def test_parameters():
    params = parabolic.parameters(2.4, 8.4e9, 400.0, 0.62, 20.0)
    np.testing.assert_allclose(
        params,
        (2.4, 8.4e9, 400.0, 0.62, 20.0, 0.4872, 1290.24, 2.1134, 175.929),
        rtol=1e-3
    )


def test_near_field_corrections_function():
    params = parabolic.parameters(2.4, 8.4e9, 400.0, 0.62, 20.0)
    table = parabolic.near_field_corrections(params, 1.0)
    np.testing.assert_allclose(
        [table.sum()[0], table.sum()[1]],
        [3715, 505],
        rtol=1
    )


def test_hazard_plot_function():
    params = parabolic.parameters(2.4, 8.4e9, 400.0, 0.62, 20.0)
    fig, ax = parabolic.hazard_plot(params, 10.0)
    lines = ax.lines
    x_plot, y_plot = lines[0].get_data()
    np.testing.assert_allclose(
        y_plot[0:3],
        [2.13333333, 2.13333333, 2.13333333],
        rtol=1e-5
    )

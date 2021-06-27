# -*- coding: utf-8 -*-
"""Basic test suite."""

import io
from argparse import ArgumentTypeError
from os import path, remove

from slm_scorer import SLM

import pandas as pd
import pytest


class TestData:

    @pytest.fixture()
    def slm(self):
        return SLM(path.join(path.abspath(path.dirname(__file__)), 'test.gpx'))

    def test_augment(self, slm):
        slm.augment()
        assert isinstance(slm.df_raw, pd.DataFrame) and (
            'x' in slm.df_raw.columns and slm.df_raw.at[0, 'x'] == 0 and slm.df_raw.at[0, 'y'] == 0)

    def test_rotate(self, slm):
        slm.rotate()
        assert isinstance(slm.df_raw, pd.DataFrame) and (
            'x_rot' in slm.df_raw.columns and abs(slm.df_raw['y_rot'].iat[-1]) < 1E-6)

    def test_clean(self, slm):
        slm.clean()
        assert isinstance(slm.df_clean, pd.DataFrame) and len(slm.df_clean) == 830 and abs(
            slm.df_clean.at[0, 'lon'] + 2.761551272124052) < 1E-6 and abs(
            slm.df_clean.at[0, 'lat'] - 50.71145405061543) < 1E-6
        # TODO check clean actually works, currently relying on 830 rows

    def test_augment_clean(self, slm):
        slm.clean()
        slm.augment()
        assert isinstance(slm.df_clean, pd.DataFrame) and (
            'x' in slm.df_clean.columns and slm.df_clean.at[0, 'x'] == 0 and slm.df_clean.at[0, 'y'] == 0)

    def test_rotate_clean(self, slm):
        slm.clean()
        slm.rotate()
        assert isinstance(slm.df_clean, pd.DataFrame) and (
            'x_rot' in slm.df_clean.columns and abs(slm.df_clean['y_rot'].iat[-1]) < 1E-6)

    def test_normalize(self, slm):
        slm.normalize()
        assert isinstance(slm.df_raw, pd.DataFrame) and (
            'x_norm' in slm.df_raw.columns and abs(slm.df_raw['x_norm'].iat[-1] - 1) < 1E-6)

    def test_normalize_clean(self, slm):
        slm.clean()
        slm.normalize()
        assert isinstance(slm.df_clean, pd.DataFrame) and (
            'x_norm' in slm.df_clean.columns and abs(slm.df_clean['x_norm'].iat[-1] - 1) < 1E-6)

    def test_plot_latlon(self, slm):
        fn = path.join(path.abspath(path.dirname(__file__)), 'file.png')
        slm.plot_latlon(fn)
        assert path.isfile(fn)
        remove(fn)
        # TODO: anyway to check content of plot?

    def test_plot_html(self, slm):
        fn = path.join(path.abspath(path.dirname(__file__)), 'file.html')
        slm.plot_html(fn)
        assert path.isfile(fn)
        remove(fn)

    def test_plot_xy(self, slm):
        fn = path.join(path.abspath(path.dirname(__file__)), 'file.png')
        slm.plot_xy(fn)
        assert path.isfile(fn)
        remove(fn)

    def test_plot(self, slm):
        fn = path.join(path.abspath(path.dirname(__file__)), 'file.png')
        slm.plot(fn)
        assert path.isfile(fn)
        remove(fn)

    def test_plot_norm(self, slm):
        fn = path.join(path.abspath(path.dirname(__file__)), 'file.png')
        slm.plot_norm(fn)
        assert path.isfile(fn)
        remove(fn)

    def test_derivative(self, slm):
        slm.derivative()
        assert isinstance(slm.df_raw, pd.DataFrame) and 'dydx' in slm.df_raw.columns

    def test_plot_derivative(self, slm):
        fn = path.join(path.abspath(path.dirname(__file__)), 'file.png')
        slm.plot_derivative(fn)
        assert path.isfile(fn)
        remove(fn)

    def test_second_derivative(self, slm):
        slm.second_derivative()
        assert isinstance(slm.df_raw, pd.DataFrame) and 'dydxdx' in slm.df_raw.columns

    def test_plot_second_derivative(self, slm):
        fn = path.join(path.abspath(path.dirname(__file__)), 'file.png')
        slm.plot_second_derivative(fn)
        assert path.isfile(fn)
        remove(fn)

# -*- coding: utf-8 -*-
"""Basic test suite."""

import io
from argparse import ArgumentTypeError
from os import path, remove

from slm_scorer import SLM

import pandas as pd
import pytest


class TestInit:

    def test_instantiate_empty(self):
        slm = SLM()
        assert slm._useCleaned is False

    def test_instantiate_with_path(self):
        slm = SLM(path.join(path.abspath(path.dirname(__file__)), 'test.gpx'))
        assert isinstance(slm.gpx_file, io.IOBase) and len(slm.points) == 1086

    def test_set_gpx(self):
        slm = SLM()
        slm.set_gpx(path.join(path.abspath(path.dirname(__file__)), 'test.gpx'))
        assert isinstance(slm.gpx_file, io.IOBase) and len(slm.points) == 1086

    def test_file_not_file(self):
        with pytest.raises(ArgumentTypeError):
            slm = SLM('./some_gpx_file_that_cannot_be_found.gpx')

    def test_set_track_zero(self):
        slm = SLM(path.join(path.abspath(path.dirname(__file__)), 'test.gpx'))
        slm.set_track(0)
        assert isinstance(slm.gpx_file, io.IOBase) and len(slm.points) == 1086

    def test_set_track_error(self):
        with pytest.raises(IndexError):
            slm = SLM(path.join(path.abspath(path.dirname(__file__)), 'test.gpx'))
            slm.set_track(2)

    def test_load_data(self):
        slm = SLM()
        slm.set_gpx(path.join(path.abspath(path.dirname(__file__)), 'test.gpx'))
        slm.load_data()
        assert isinstance(slm.df_raw, pd.DataFrame) and len(slm.df_raw) == 1086 and abs(
            slm.df_raw.at[0, 'lon'] + 2.761551272124052) < 1E-6 and abs(
            slm.df_raw.at[0, 'lat'] - 50.71145405061543) < 1E-6

    def test_load_data_with_path(self):
        slm = SLM()
        slm.load_data(path.join(path.abspath(path.dirname(__file__)), 'test.gpx'))
        assert isinstance(slm.df_raw, pd.DataFrame) and len(slm.df_raw) == 1086 and abs(
            slm.df_raw.at[0, 'lon'] + 2.761551272124052) < 1E-6 and abs(
            slm.df_raw.at[0, 'lat'] - 50.71145405061543) < 1E-6

    def test_change_gpx(self):
        slm = SLM(path.join(path.abspath(path.dirname(__file__)), 'test.gpx'))
        slm.set_gpx(path.join(path.abspath(path.dirname(__file__)), 'test2.gpx'))
        assert isinstance(slm.gpx_file, io.IOBase) and len(slm.points) == 196

    # TODO: add test for init_args

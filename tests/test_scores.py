# -*- coding: utf-8 -*-
"""Basic test suite."""

import io
from argparse import ArgumentTypeError
from os import path, remove

from slm_scorer import SLM

import pandas as pd
import pytest


class TestScores:

    @pytest.fixture()
    def slm(self):
        return SLM(path.join(path.abspath(path.dirname(__file__)), 'test.gpx'))

    def test_max_dist(self, slm):
        assert abs(1634.864072949445 - slm.max_distance()) < 1E-6

    def test_mean_dist(self, slm):
        assert abs(791.9523847298758 - slm.mean_distance()) < 1E-6

    def test_mean_dist_time(self, slm):
        assert abs(800.7437815648872 - slm.mean_distance(True)) < 1E-6

    def test_rms_dist(self, slm):
        assert abs(900.0362127011514 - slm.rms_distance()) < 1E-6

    def test_rsquared(self, slm):
        assert abs(0.8173224796564555 - slm.rsquared()) < 1E-6

    def test_travelled(self, slm):
        t, l = slm.travelled()
        assert abs(16621.17338228421 - t) < 1E-6 and abs(12129.517033148137 - l) < 1E-6

    def test_max_dist_clean(self, slm):
        slm.clean()
        assert abs(1634.864072949445 - slm.max_distance()) < 1E-6

    def test_mean_dist_clean(self, slm):
        slm.clean()
        assert abs(781.2583822923078 - slm.mean_distance()) < 1E-6

    def test_mean_dist_time_clean(self, slm):
        slm.clean()
        assert abs(800.901990741595 - slm.mean_distance(True)) < 1E-6

    def test_rms_dist_clean(self, slm):
        slm.clean()
        assert abs(889.4260631357059 - slm.rms_distance()) < 1E-6

    def test_rsquared_clean(self, slm):
        slm.clean()
        assert abs(0.8000591472323133 - slm.rsquared()) < 1E-6

    def test_travelled_clean(self, slm):
        slm.clean()
        t, l = slm.travelled()
        assert abs(15917.477241676657 - t) < 1E-6 and abs(12129.517033148137 - l) < 1E-6

    def test_shoelace(self, slm):
        assert abs(31.28482622843 - slm.shoelace()) < 1E-6 and isinstance(slm.df_clean, pd.DataFrame) and slm._useCleaned

    def test_trapz(self, slm):
        assert abs(31.28482622843 - slm.trapz()) < 1E-6 and isinstance(slm.df_clean, pd.DataFrame) and slm._useCleaned

    def test_area_consistent(self, slm):
        assert abs(slm.shoelace() - slm.trapz()) < 0.25

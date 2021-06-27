# -*- coding: utf-8 -*-
"""Module main-file.

A module's __name__ is set equal to '__main__' when read from standard input,
a script, or from an interactive prompt.
"""
from . import SLM

slm = SLM()
slm.init_args()
slm.load_data()
slm.augment()
slm.clean()
slm.plot_latlon()
slm.plot_xy()
slm.plot()
slm.plot_html()
print('Max Distance is: {}m'.format(round(slm.max_distance() * 2, 0) / 2))
print('Mean Distance (x weighted) is: {}m'.format(round(slm.mean_distance() * 2, 0) / 2))
print('Mean Distance (time weighted) is: {}m'.format(round(slm.mean_distance(True) * 2, 0) / 2))
print('RMS Distance (x weighted) is: {}m'.format(round(slm.rms_distance() * 2, 0) / 2))
print('RMS Distance (time weighted) is: {}m'.format(round(slm.rms_distance(True) * 2, 0) / 2))
print('R-Squared value is: {:.6f}'.format(slm.rsquared()))
t, l = slm.travelled()
print('Distance travelled vs line is: {} ({}m vs {}m)'.format(
    round(t / l, 2),
    round(t * 2, 0) / 2,
    round(l * 2, 0) / 2
))
print('Area between track and line (shoelace method) is: {}'.format(round(slm.shoelace(), 1)))
print('Area between track and line (trapezium method) is: {}'.format(round(slm.trapz(), 1)))

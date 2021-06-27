# -*- coding: utf-8 -*-
"""Module init-file.

The __init__.py files are required to make Python treat directories
containing the file as packages.
"""

import argparse
import atexit
import datetime
import logging
import os
import subprocess
import sys
from collections import namedtuple
from math import copysign, floor, sqrt

import gmplot
import gpxpy
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from geopy import distance

from .utils import FileTypeWithExtensionCheck, PolyArea

# import chart_studio.plotly as py
# import plotly.graph_objs as go

Point = namedtuple('Point', ['lon', 'lat', 'alt', 'time'])


class SLM():
    def __init__(self, gpx_file=None):
        self.init_logging()
        atexit.register(self.__del__)
        self._useCleaned = False
        if gpx_file is not None:
            self.set_gpx(gpx_file)
        else:
            self.gpx_file = None

    def init_logging(self):
        log_console_format = "[%(levelname)s:%(name)s]: %(message)s"
        self.logger = logging.getLogger(__name__)
        # Two handler approach:
        #   INFO and DEBUG should go to stdout
        #   CRITICAL, ERROR and WARNING should go to stderr
        # TODO: Add file handler
        #   log_file_format = "[%(levelname)s] - %(asctime)s - %(name)s - : %(message)s in %(pathname)s:%(lineno)d"
        sout = logging.StreamHandler(sys.stdout)
        sout.setLevel(logging.DEBUG)                             # Shows DEBUG and...
        sout.addFilter(lambda msg: msg.levelno <= logging.INFO)  # ...INFO
        fmt = logging.Formatter(log_console_format)
        sout.setFormatter(fmt)
        serr = logging.StreamHandler()  # stderr by default
        serr.setLevel(logging.WARNING)  # Shows >= WARNING
        serr.setFormatter(fmt)
        self.logger.addHandler(sout)
        self.logger.addHandler(serr)

    def init_args(self):
        parser = argparse.ArgumentParser(description='Score a straight line mission based on GPX file.')
        parser.add_argument(
            'gpxfile',
            type=FileTypeWithExtensionCheck('r', '.gpx', encoding='UTF-8'),
            help='path to the GPX file to score'
        )
        parser.add_argument("-v", "--verbose", help="Enable verbose output", action="store_const",
                            dest="loglevel", const=logging.DEBUG, default=logging.INFO)
        self.args = parser.parse_args()
        self.logger.setLevel(self.args.loglevel)  # Based on -v command line flag
        self.log('Parsed arguments sucessfully')
        self.set_gpx(self.args.gpxfile)

    def __del__(self):
        self.log('Cleaning up SLM...')
        try:
            if self.args and self.args.gpxfile and not self.args.gpxfile.closed:
                self.log('Closing GPX file')
                self.args.gpxfile.close()
            if self.gpx_file and not self.gpx_file.closed:
                self.log('Closing GPX file')
                self.gpx_file.close()
        except AttributeError:
            pass
        atexit.unregister(self.__del__)

    def log(self, msg, lvl='DEBUG'):
        lvls = {
            10: "DEBUG",
            20: "INFO",
            30: "WARNING",
            40: "ERROR",
            50: "CRITICAL"
        }
        if type(lvl) == int or type(lvl) == float or str(lvl).isnumeric():
            lvl = max(10, min(floor(int(lvl) / 10), 5)) * 10
            lvl = lvls.get(lvl, "INFO")
        lvl = str(lvl).upper()
        if lvl == "CRITICAL":
            self.logger.critical(msg)
        elif lvl == "ERROR" or lvl == "ERR":
            self.logger.error(msg)
        elif lvl == "WARNING" or lvl == "WARN":
            self.logger.warn(msg)
        elif lvl == "INFO":
            self.logger.info(msg)
        else:
            self.logger.debug(msg)

    def set_gpx(self, path):
        if not hasattr(path, 'read'):
            self.log('Trying to open GPX file...')
            _path = path
            if not os.path.isfile(path):
                _path = os.path.join(os.path.dirname(__file__), path)
                if not os.path.isfile(_path):
                    _path = os.path.join(os.getcwd(), path)
            encodings = ['utf-8', 'windows-1250', 'windows-1252', 'utf-16', 'latin-1', None]
            suceeded = False
            for enc in encodings:
                try:
                    self.gpx_file = open(_path, 'r', encoding=enc)
                except OSError as e:
                    args = {'filename': path, 'error': e}
                    message = "can't open '%(filename)s': %(error)s"
                    raise argparse.ArgumentTypeError(message % args)
                except UnicodeDecodeError:
                    self.log(f'Got unicode error with {enc}, trying different encoding')
                except gpxpy.gpx.GPXXMLSyntaxException:
                    self.log(f'Got syntax error with {enc}, trying different encoding')
                else:
                    suceeded = True
                    break
            if not suceeded:
                raise Exception(f'Could not open GPX file at {path}')
        else:
            self.gpx_file = path
        self.log('Parsing GPX file...')
        try:
            self.gpx = gpxpy.parse(self.gpx_file)
        except Exception as e:
            self.log('Error occured trying to parse GPX file!', 'CRITICAL')
            raise e
        self.log('Tracks: {}, first track segments: {}, first segment points: {}'.format(
            len(self.gpx.tracks),
            len(self.gpx.tracks[0].segments) if len(self.gpx.tracks) else None,
            len(self.gpx.tracks[0].segments[0].points) if len(
                self.gpx.tracks) and len(self.gpx.tracks[0].segments) else None
        ))
        if len(self.gpx.tracks) == 0:
            raise Exception("GPX file has no tracks")
        if len(self.gpx.tracks[0].segments) == 0 or len(self.gpx.tracks[0].segments[0].points) == 0:
            raise Exception("Track in GPX file has no point data")
        self.points = [Point(
            lon=p.longitude,
            lat=p.latitude,
            alt=p.elevation,
            time=p.time,
        ) for p in self.gpx.tracks[0].segments[0].points]

    def set_track(self, i):
        num_tracks = len(self.gpx.tracks)
        if i >= num_tracks:
            raise IndexError('Parsed GPX file has no track at index {}, file contains {} track{}'.format(
                i, num_tracks, '' if num_tracks == 1 else 's'
            ))
        if len(self.gpx.tracks[i].segments) == 0 or len(self.gpx.tracks[i].segments[0].points) == 0:
            raise Exception(f"Track at index {i} in GPX file has no point data")
        self.points = self.gpx.tracks[i].segments[0].points

    def load_data(self, path=None):
        if path is not None:
            self.set_gpx(path)
        elif not hasattr(self, 'gpx_file') or (self.gpx_file.closed if hasattr(self.gpx_file, 'closed') else False):
            raise Exception("Cannot parse GPX, no file is currently set")

        self.log('Creating dataframe...')
        self.df_raw = pd.DataFrame(self.points)
        self.start_point = self.points[0]
        self.finish_point = self.points[-1]

        self.log('Augmenting point data with time and distance stats...')
        if self.df_raw.isnull().time.all():
            self.log('GPX file does not contain timing data, using today and a speed of 5kmph as a default', 'WARN')
            self.start_point = Point(
                lon=self.start_point.lon,
                lat=self.start_point.lat,
                alt=self.start_point.alt,
                time=datetime.datetime.combine(datetime.date.today(), datetime.time(9, 0)),
            )
            self.df_raw.at[0, 'time'] = self.start_point.time

        if self.df_raw.isnull().alt.all():
            self.log('GPX file does not contain elevation data, 3D distance will equal 2D distance', 'WARN')

        # self.log('\n' + self.df_raw.__repr__())

    def augment(self, useCleaned=None):
        if useCleaned is None:
            useCleaned = self._useCleaned
        if not hasattr(self, 'df_raw'):
            self.load_data()
        if useCleaned and not hasattr(self, 'df_clean'):
            self.clean()
        df = self.df_clean[['lon', 'lat', 'alt', 'time']] if useCleaned else self.df_raw[['lon', 'lat', 'alt', 'time']]
        self.log('Augmenting data...')

        # list to hold new columns
        index = [0]
        len_delta_2d = [0]
        len_2d = [0]
        alt_delta = [0]
        len_3d = [0]
        time_delta = [0]
        speed = [0]
        x = [0]
        y = [0]

        for i in range(1, len(df)):
            prev = df.iloc[i - 1]
            row = df.iloc[i]

            index.append(df.index.values[i])
            move_2d = distance.geodesic((prev.lat, prev.lon), (row.lat, row.lon)).m
            len_delta_2d.append(move_2d)
            len_2d.append(len_2d[-1] + move_2d)
            alt_dif = 0 if prev.alt is None or row.alt is None else row.alt - prev.alt
            alt_delta.append(alt_dif)
            move_3d = sqrt(move_2d**2 + (alt_dif)**2)
            len_3d.append(len_3d[-1] + move_3d)
            if row.time is None:
                df.at[i, 'time'] = prev.time + datetime.timedelta(0, move_2d / (5000 / 3600))  # 5kmph
                row = df.iloc[i]
            time_dif = (row.time - prev.time).total_seconds()
            if useCleaned and move_2d / time_dif < 0.5:
                time_dif = move_2d / (5000 / 3600)  # 5kmph
            time_delta.append(time_dif)
            speed.append(move_2d / time_dif)

            x_dist = (-1 if row.lon < self.start_point.lon else 1) * \
                distance.geodesic((self.start_point.lat, self.start_point.lon), (self.start_point.lat, row.lon)).m
            y_dist = (-1 if row.lat < self.start_point.lat else 1) * \
                distance.geodesic((self.start_point.lat, self.start_point.lon), (row.lat, self.start_point.lon)).m
            x.append(x_dist)
            y.append(y_dist)

        # add columns to df
        df = pd.concat(
            [
                df,
                pd.DataFrame({
                    'len_delta_2d': len_delta_2d,
                    'alt_delta': alt_delta,
                    'time_delta': time_delta,
                    'len_2d': len_2d,
                    'len_3d': len_3d,
                    'speed': speed,
                    'x': x,
                    'y': y,
                }, index=index)
            ], axis=1
        )
        if useCleaned:
            self.df_clean = df
        else:
            self.df_raw = df
        # self.log('\n' + df.__repr__())

    def rotate(self, useCleaned=None):
        if useCleaned is None:
            useCleaned = self._useCleaned
        if not hasattr(self, 'df_raw') or 'x' not in self.df_raw.columns:
            self.augment(False)
        if useCleaned and (not hasattr(self, 'df_clean') or 'x' not in self.df_clean.columns):
            self.augment(useCleaned)
        df = self.df_clean if useCleaned else self.df_raw

        x, y = df['x'].iloc[-1], df['y'].iloc[-1]
        theta = -1 * np.arctan(y / x)
        self.log(f'Rotating points by {(theta/(2 * np.pi))*360:.0f} degrees')
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, -s), (s, c)))
        index = [0]
        x_rot = [0]
        x_rot_delta = [0]
        y_rot = [0]
        y_rot_mean = [0]
        for i in range(1, len(df)):
            index.append(df.index.values[i])
            coord = R @ np.array((df['x'].iloc[i], df['y'].iloc[i]))
            coord[0] *= -1 if x < 0 else 1
            x_rot.append(coord.item(0))
            x_rot_delta.append(coord.item(0) - x_rot[i - 1])
            y_rot.append(coord.item(1))
            y_rot_mean.append((coord.item(1) + y_rot_mean[i - 1]) / 2)
        df = pd.concat(
            [
                df,
                pd.DataFrame({
                    'x_rot': x_rot,
                    'y_rot': y_rot,
                    'x_rot_delta': x_rot_delta,
                    'y_rot_mean': y_rot_mean
                }, index=index)
            ], axis=1
        )
        # self.log('\n' + df.__repr__())
        if useCleaned:
            self.df_clean = df
        else:
            self.df_raw = df

    def clean(self):
        if not hasattr(self, 'df_raw') or 'x_rot' not in self.df_raw.columns:
            self.rotate(False)
        self.log('Cleaning track data...')

        index = []
        lon = []
        lat = []
        alt = []
        time = []
        x_upto = 0
        max_x = self.df_raw['x_rot'].iloc[-1]
        for ind, row in self.df_raw.iterrows():
            keep = False
            if ind == 0 or ind == len(self.df_raw) - 1:
                keep = True
            elif row['x_rot'] > x_upto and row['x_rot'] < max_x and row['speed'] >= 0.5:
                keep = True
            if keep:
                index.append(ind)
                lon.append(row['lon'])
                lat.append(row['lat'])
                alt.append(row['alt'])
                time.append(row['time'])
                x_upto = row['x_rot']
        self.df_clean = pd.DataFrame({
            'lon': lon,
            'lat': lat,
            'alt': alt,
            'time': time
        }, index=index)
        self._useCleaned = True
        self.augment(True)
        # self.log('\n' + self.df_clean.__repr__())

    def useCleaned(self, b=True):
        if b and not hasattr(self, 'df_clean'):
            self.clean()
        self._useCleaned = bool(b)

    def normalize(self, useCleaned=None):
        if useCleaned is None:
            useCleaned = self._useCleaned
        if not hasattr(self, 'df_raw') or 'x_rot' not in self.df_raw.columns:
            self.rotate(False)
        if useCleaned and (not hasattr(self, 'df_clean') or 'x_rot' not in self.df_clean.columns):
            self.rotate(useCleaned)
        df = self.df_clean if useCleaned else self.df_raw

        L = df['x_rot'].iloc[-1]
        self.log(f'Normalising data from length of {L:.0f} meters to 1 unit')
        index = [0]
        x_norm = [0]
        y_norm = [0]
        for i in range(1, len(df)):
            index.append(df.index.values[i])
            x_norm.append(df['x_rot'].iloc[i] / L)
            y_norm.append(df['y_rot'].iloc[i] / L)
        df = pd.concat(
            [
                df,
                pd.DataFrame({
                    'x_norm': x_norm,
                    'y_norm': y_norm,
                }, index=index)
            ], axis=1
        )
        # self.log('\n' + df.__repr__())
        if useCleaned:
            self.df_clean = df
        else:
            self.df_raw = df

    def plot_latlon(self, filename='track.png'):
        if self._useCleaned and not hasattr(self, 'df_clean'):
            self.clean()
        elif not hasattr(self, 'df_raw'):
            self.load_data()
        df = self.df_clean if self._useCleaned else self.df_raw
        plt.cla()
        plt.figure(figsize=(8, 6), dpi=96)
        plt.gca().set_aspect('equal', 'datalim')
        plt.plot(df['lon'], df['lat'])
        plt.plot(
            [self.start_point.lon, self.finish_point.lon],
            [self.start_point.lat, self.finish_point.lat],
            '--',
            color='gray'
        )
        if matplotlib.is_interactive():
            plt.show()
        plt.savefig(filename)
        self.log(f'Plot of latlon data saved as {filename}', 'INFO')

    def plot_html(self, filename='track.html'):
        if self._useCleaned and not hasattr(self, 'df_clean'):
            self.clean()
        elif not hasattr(self, 'df_raw'):
            self.load_data()
        df = self.df_clean if self._useCleaned else self.df_raw

        min_lon, max_lon = min(df['lon']), max(df['lon'])
        min_lat, max_lat = min(df['lat']), max(df['lat'])
        try:
            k = os.environ['GOOGLE_API_KEY']
        except KeyError:
            k = None
        gmplotter = gmplot.GoogleMapPlotter(
            min_lat + (max_lat - min_lat) / 2,  # Center lat
            min_lon + (max_lon - min_lon) / 2,  # Center lon
            16,  # Zoom level
            apikey=k
        )
        gmplotter.plot(df['lat'], df['lon'], 'red', edge_width=3)
        gmplotter.draw(filename)
        self.log(f'HTML plot saved as {filename}', 'INFO')

    def open_html(self, filename='track.html'):
        try:
            os.startfile(filename)  # should work on Windows
        except AttributeError:
            try:  # should work on MacOS and most linux versions
                subprocess.call(['open', filename])
            except Exception as e:
                self.log(f'Could not open `{filename}`: {e}', 'ERROR')

    def plot_xy(self, filename='track_xy.png'):
        if self._useCleaned and (not hasattr(self, 'df_clean') or 'x' not in self.df_clean.columns):
            self.augment()
        elif not hasattr(self, 'df_raw') or 'x' not in self.df_raw.columns:
            self.augment()
        df = self.df_clean if self._useCleaned else self.df_raw

        plt.cla()
        plt.figure(figsize=(8, 6), dpi=96)
        plt.gca().set_aspect('equal', 'datalim')
        plt.plot(df['x'], df['y'])
        plt.plot(
            [0, df['x'].iloc[-1]],
            [0, df['y'].iloc[-1]],
            '--',
            color='gray'
        )
        if matplotlib.is_interactive():
            plt.show()
        plt.savefig(filename)
        self.log(f'Plot of xy data saved as {filename}', 'INFO')

    def plot(self, filename='slm.png'):
        if self._useCleaned and (not hasattr(self, 'df_clean') or 'x_rot' not in self.df_clean.columns):
            self.rotate()
        elif not hasattr(self, 'df_raw') or 'x_rot' not in self.df_raw.columns:
            self.rotate()
        df = self.df_clean if self._useCleaned else self.df_raw

        plt.cla()
        plt.figure(figsize=(20, 2.5), dpi=96)
        plt.gca().set_aspect('equal', 'datalim')
        plt.plot(df['x_rot'], df['y_rot'])
        plt.plot(
            [0, df['x_rot'].iloc[-1]],
            [0, 0],
            '--',
            color='gray'
        )
        if matplotlib.is_interactive():
            plt.show()
        plt.savefig(filename)
        self.log(f'Plot of rotated data saved as {filename}', 'INFO')

    def plot_norm(self, filename='norm.png'):
        if self._useCleaned and (not hasattr(self, 'df_clean') or 'x_norm' not in self.df_clean.columns):
            self.normalize()
        elif not hasattr(self, 'df_raw') or 'x_norm' not in self.df_raw.columns:
            self.normalize()
        df = self.df_clean if self._useCleaned else self.df_raw

        plt.cla()
        plt.figure(figsize=(20, 2.5), dpi=96)
        plt.gca().set_aspect('equal', 'datalim')
        plt.plot(df['x_norm'], df['y_norm'])
        plt.plot(
            [0, df['x_norm'].iloc[-1]],
            [0, 0],
            '--',
            color='gray'
        )
        if matplotlib.is_interactive():
            plt.show()
        plt.savefig(filename)
        self.log(f'Plot of normalized data saved as {filename}', 'INFO')

    def max_distance(self):
        if self._useCleaned and (not hasattr(self, 'df_clean') or 'x_rot' not in self.df_clean.columns):
            self.rotate()
        elif not hasattr(self, 'df_raw') or 'x_rot' not in self.df_raw.columns:
            self.rotate()
        return self.df_clean['y_rot'].abs().max() if self._useCleaned else self.df_raw['y_rot'].abs().max()

    def mean_distance(self, time_weighted=False):
        if self._useCleaned and (not hasattr(self, 'df_clean') or 'x_rot' not in self.df_clean.columns):
            self.rotate()
        elif not hasattr(self, 'df_raw') or 'x_rot' not in self.df_raw.columns:
            self.rotate()
        df = self.df_clean if self._useCleaned else self.df_raw
        if time_weighted:
            return np.average(
                df.loc[df['speed'] >= 0.5, ['y_rot_mean']].abs(),
                weights=df.loc[df['speed'] >= 0.5, ['time_delta']]
            )
        else:
            return np.average(
                df['y_rot_mean'].abs(),
                weights=df['x_rot_delta'].abs()
            )

    def rms_distance(self, time_weighted=False):
        if self._useCleaned and (not hasattr(self, 'df_clean') or 'x_rot' not in self.df_clean.columns):
            self.rotate()
        elif not hasattr(self, 'df_raw') or 'x_rot' not in self.df_raw.columns:
            self.rotate()
        df = self.df_clean if self._useCleaned else self.df_raw
        if time_weighted:
            return sqrt(np.average(
                df.loc[df['speed'] >= 0.5, ['y_rot_mean']]**2,
                weights=df.loc[df['speed'] >= 0.5, ['time_delta']]
            ))
        else:
            return sqrt(np.average(
                df['y_rot_mean']**2,
                weights=df['x_rot_delta'].abs()
            ))

    def rsquared(self):
        if self._useCleaned and (not hasattr(self, 'df_clean') or 'x_rot' not in self.df_clean.columns):
            self.rotate()
        elif not hasattr(self, 'df_raw') or 'x_rot' not in self.df_raw.columns:
            self.rotate()
        df = self.df_clean if self._useCleaned else self.df_raw

        theta = np.pi / 4  # 45 degrees
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, -s), (s, c)))
        x_rot = [0]
        y_rot = [0]
        for i in range(1, len(df)):
            coord = R @ np.array((df['x_rot'].iloc[i], df['y_rot'].iloc[i]))
            x_rot.append(coord.item(0))
            y_rot.append(coord.item(1))

        m = y_rot[-1] / x_rot[-1]
        y_bar = y_rot[-1] / 2
        SST = 0
        SSE = 0
        for i in range(0, len(x_rot)):
            SST += (y_rot[i] - y_bar)**2
            SSE += (y_rot[i] - m * x_rot[i])**2
        return 1 - SSE / SST

    def travelled(self):
        if self._useCleaned and (not hasattr(self, 'df_clean') or 'x_rot' not in self.df_clean.columns):
            self.rotate()
        elif not hasattr(self, 'df_raw') or 'x_rot' not in self.df_raw.columns:
            self.rotate()
        df = self.df_clean if self._useCleaned else self.df_raw
        return (df['len_2d'].iloc[-1], df['x_rot'].iloc[-1])

    def derivative(self, useCleaned=None):
        if useCleaned is None:
            useCleaned = self._useCleaned
        if not hasattr(self, 'df_raw') or 'x_rot' not in self.df_raw.columns:
            self.rotate(False)
        if useCleaned and (not hasattr(self, 'df_clean') or 'x_rot' not in self.df_clean.columns):
            self.rotate(useCleaned)
        df = self.df_clean if useCleaned else self.df_raw

        index = [0]
        derivative = [0]
        for i in range(1, len(df)):
            prev = df.iloc[i - 1]
            row = df.iloc[i]
            dydx = (row.y_rot - prev.y_rot) / (row.x_rot - prev.x_rot)
            index.append(df.index.values[i])
            derivative.append(dydx)
        df = pd.concat(
            [
                df,
                pd.DataFrame({
                    'dydx': derivative
                }, index=index)
            ], axis=1
        )
        # self.log('\n' + df.__repr__())
        if useCleaned:
            self.df_clean = df
        else:
            self.df_raw = df

    def plot_derivative(self, filename='dydx.png'):
        if self._useCleaned and (not hasattr(self, 'df_clean') or 'dydx' not in self.df_clean.columns):
            self.derivative()
        elif not hasattr(self, 'df_raw') or 'dydx' not in self.df_raw.columns:
            self.derivative()
        df = self.df_clean if self._useCleaned else self.df_raw

        plt.cla()
        plt.figure(figsize=(8, 6), dpi=96)
        plt.plot(df['x_rot'], df['dydx'])
        plt.plot(
            [0, df['x_rot'].iloc[-1]],
            [0, 0],
            '--',
            color='gray'
        )
        if matplotlib.is_interactive():
            plt.show()
        plt.savefig(filename)
        self.log(f'Plot of dydx saved as {filename}', 'INFO')

    def second_derivative(self, useCleaned=None):
        if useCleaned is None:
            useCleaned = self._useCleaned
        if not hasattr(self, 'df_raw') or 'dydx' not in self.df_raw.columns:
            self.derivative(False)
        if useCleaned and (not hasattr(self, 'df_clean') or 'dydx' not in self.df_clean.columns):
            self.derivative(useCleaned)
        df = self.df_clean if useCleaned else self.df_raw

        index = [0]
        second_derivative = [0]
        for i in range(1, len(df)):
            prev = df.iloc[i - 1]
            row = df.iloc[i]
            dydxdx = (row.dydx - prev.dydx) / (row.x_rot - prev.x_rot)
            index.append(df.index.values[i])
            second_derivative.append(dydxdx)
        df = pd.concat(
            [
                df,
                pd.DataFrame({
                    'dydxdx': second_derivative
                }, index=index)
            ], axis=1
        )
        self.log('\n' + df.__repr__())
        if useCleaned:
            self.df_clean = df
        else:
            self.df_raw = df

    def plot_second_derivative(self, filename='dydxdx.png'):
        if self._useCleaned and (not hasattr(self, 'df_clean') or 'dydx' not in self.df_clean.columns):
            self.second_derivative()
        elif not hasattr(self, 'df_raw') or 'dydx' not in self.df_raw.columns:
            self.second_derivative()
        df = self.df_clean if self._useCleaned else self.df_raw

        plt.cla()
        plt.figure(figsize=(8, 6), dpi=96)
        plt.plot(df['x_rot'], df['dydxdx'])
        plt.plot(
            [0, df['x_rot'].iloc[-1]],
            [0, 0],
            '--',
            color='gray'
        )
        if matplotlib.is_interactive():
            plt.show()
        plt.savefig(filename)
        self.log(f'Plot of dydxdx saved as {filename}', 'INFO')

    def shoelace(self):
        self.useCleaned(True)
        if 'x_rot' not in self.df_clean.columns:
            self.rotate()
        df = self.df_clean

        polys = []
        x = [0]
        y = [0]
        sign = copysign(1, df['y_rot'].iloc[1])
        for i in range(1, len(df)):
            row = df.iloc[i]
            s = copysign(1, row.y_rot)
            if s != sign:
                # we have crossed 0, close poly
                last_x = x[len(x) - 1]
                last_y = y[len(y) - 1]
                zero_x = last_x + (abs(row.y) / abs(row.y_rot - last_y)) * (row.x_rot - last_x)
                x.append(zero_x)
                y.append(0)
                polys.append([x, y])
                # start new poly
                x = [zero_x]
                y = [0]
                sign = s
            x.append(row.x_rot)
            y.append(row.y_rot)
        if len(x) > 1:
            polys.append([x, y])

        area = 0
        for x, y in polys:
            area += PolyArea(x, y)
        return area / (25 * df['x_rot'].iloc[-1])

    def trapz(self):
        self.useCleaned(True)
        if 'x_rot' not in self.df_clean.columns:
            self.rotate()
        df = self.df_clean

        polys = []
        x = [0]
        y = [0]
        sign = copysign(1, df['y_rot'].iloc[1])
        for i in range(1, len(df)):
            row = df.iloc[i]
            s = copysign(1, row.y_rot)
            if s != sign:
                # we have crossed 0, close poly
                last_x = x[len(x) - 1]
                last_y = y[len(y) - 1]
                zero_x = last_x + (abs(row.y) / abs(row.y_rot - last_y)) * (row.x_rot - last_x)
                x.append(zero_x)
                y.append(0)
                polys.append([x, y])
                # start new poly
                x = [zero_x]
                y = [0]
                sign = s
            x.append(row.x_rot)
            y.append(row.y_rot)
        if len(x) > 1:
            polys.append([x, y])

        area = 0
        for x, y in polys:
            area += abs(np.trapz(y, x=x))
        return area / (25 * df['x_rot'].iloc[-1])

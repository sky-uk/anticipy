# -*- coding: utf-8 -*-
#
# License:          This module is released under the terms of the LICENSE file
#                   contained within this applications INSTALL directory

"""
Unittests for plotting functions
"""

# -- Coding Conventions
#    http://www.python.org/dev/peps/pep-0008/   -   Use the Python style guide
# http://sphinx.pocoo.org/rest.html          -   Use Restructured Text for
# docstrings

# -- Public Imports

import logging
import numpy as np
import os
import pandas as pd

# -- Private Imports
from anticipy.utils_test import PandasTest
from anticipy import forecast_plot

# -- Globals
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Set base folder and samples folder paths
base_folder = os.path.join(os.path.dirname(__file__), 'test_plots')
samples_folder = os.path.join(os.path.dirname(__file__), 'data')


def get_file_path(folder, name):
    path = os.path.join(folder, name)
    return path


# # Forecasting output samples
# Sample dataframe without prediction intervals and a single source
df_forecast = (
    pd.concat([
        pd.DataFrame({'date': pd.date_range('2018-01-01', periods=6,
                                            freq='D'),
                      'model': 'actuals',
                      'y': 1000 * np.arange(0., 6.),
                      'is_actuals': True
                      }),
        pd.DataFrame({'date': pd.date_range('2018-01-01', periods=10,
                                            freq='D'),
                      'model': 'forecast',
                      'y': 1000 * np.full(10, 5.),
                      'is_actuals': False
                      }),

    ], sort=False, ignore_index=True)
)


# Dataframe with no prediction intervals and different data sources, to plot
# with faceting
df_forecast_p1 = df_forecast.copy()
df_forecast_p2 = df_forecast.copy()
df_forecast_p3 = df_forecast.copy()
df_forecast_p4 = df_forecast.copy()
df_forecast_p5 = df_forecast.copy()
df_forecast_p1['source'] = 'ts1'
df_forecast_p2['source'] = 'ts2'
df_forecast_p3['source'] = 'ts3'
df_forecast_p4['source'] = 'ts4'
df_forecast_p5['source'] = 'ts5'
df_forecast_facet = pd.concat([df_forecast_p1,
                               df_forecast_p2,
                               df_forecast_p3,
                               df_forecast_p4,
                               df_forecast_p5],
                              sort=False,
                              ignore_index=True)

# Sample dataframe with prediction intervals and single data source
df_forecast_pi = pd.read_csv(get_file_path(samples_folder,
                                           'df_test_forecast.csv'))

# Dataframe with prediction intervals and multiple data sources (mds), to plot
# with faceting
df_forecast_pi_facet = pd.read_csv(get_file_path(samples_folder,
                                                 'df_test_forecast_mds.csv'))


class TestForecastPlot(PandasTest):

    def test_plot_foracast_png(self):
        path = get_file_path(base_folder, 'test_mpl')
        forecast_plot.plot_forecast(df_forecast, path, 'png', 900, 600,
                                    'Test Plot', show_legend=False,
                                    auto_open=False)
        self.assertTrue(os.path.isfile('{}.png'.format(path)))

        path = get_file_path(base_folder, 'test_facet_mpl')
        forecast_plot.plot_forecast(df_forecast_facet, path, 'png', 1200, 900,
                                    'Test Plot', show_legend=True,
                                    auto_open=False)
        self.assertTrue(os.path.isfile('{}.png'.format(path)))

        # Repeat test with prediction intervals
        path = get_file_path(base_folder, 'test_pi_mpl')
        forecast_plot.plot_forecast(df_forecast_pi, path, 'png', 900, 600,
                                    'Test Plot', show_legend=True,
                                    auto_open=False)
        self.assertTrue(os.path.isfile('{}.png'.format(path)))

        path = get_file_path(base_folder, 'test_pi_facet_mpl')
        forecast_plot.plot_forecast(df_forecast_pi_facet, path, 'png', 1200,
                                    900, 'Test Plot', show_legend=True,
                                    auto_open=False)
        self.assertTrue(os.path.isfile('{}.png'.format(path)))

    def test_plot_foracast_html(self):
        path = get_file_path(base_folder, 'test_plotly')
        forecast_plot.plot_forecast(df_forecast, path, 'html', 900, 600,
                                    'Test Plot', show_legend=False,
                                    auto_open=False)
        self.assertTrue(os.path.isfile('{}.html'.format(path)))

        path = get_file_path(base_folder, 'test_facet_plolty')
        forecast_plot.plot_forecast(df_forecast_facet, path, 'html', 1900,
                                    1200, 'Test Plot', show_legend=False,
                                    auto_open=False)
        self.assertTrue(os.path.isfile('{}.html'.format(path)))

        # Repeat test with prediction intervals
        path = get_file_path(base_folder, 'test_pi_plolty')
        forecast_plot.plot_forecast(df_forecast_pi, path, 'html', 900, 600,
                                    'Test Plot', show_legend=False,
                                    auto_open=False)
        self.assertTrue(os.path.isfile('{}.html'.format(path)))

        path = get_file_path(base_folder, 'test_pi_facet_plotly')
        forecast_plot.plot_forecast(df_forecast_pi_facet, path, 'html', 1900,
                                    1200, 'Test Plot', show_legend=False,
                                    auto_open=False)
        self.assertTrue(os.path.isfile('{}.html'.format(path)))

    def test_plot_forecast_jupyter(self):
        forecast_plot.plot_forecast(df_forecast_pi_facet,
                                    path=None,
                                    output='jupyter',
                                    width=1900,
                                    height=1200,
                                    title='Test Plot',
                                    show_legend=False,
                                    auto_open=False)
        # Todo: add checks to validate py.offline.iplot

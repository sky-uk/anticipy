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
if not os.path.exists(base_folder):
    os.makedirs(base_folder)


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

    def test_plot_forecast_png(self):
        if not forecast_plot._matplotlib_imported:
            self.skipTest('Test skipped as Matplotlib is not installed...')
        path = get_file_path(base_folder, 'test_mpl')
        result = forecast_plot.plot_forecast(
            df_forecast, 'png', path, 900, 600,
            'Test Plot', show_legend=False,
            auto_open=False)
        self.assertEquals(result, 0)
        self.assertTrue(os.path.isfile('{}.png'.format(path)))

        path = get_file_path(base_folder, 'test_facet_mpl')
        forecast_plot.plot_forecast(df_forecast_facet, 'png', path, 1200, 900,
                                    'Test Plot', show_legend=True,
                                    auto_open=False)
        self.assertTrue(os.path.isfile('{}.png'.format(path)))

        # Repeat test with prediction intervals
        path = get_file_path(base_folder, 'test_pi_mpl')
        forecast_plot.plot_forecast(df_forecast_pi, 'png', path, 900, 600,
                                    'Test Plot', show_legend=True,
                                    auto_open=False,
                                    include_interval=True)
        self.assertTrue(os.path.isfile('{}.png'.format(path)))

        path = get_file_path(base_folder, 'test_pi_facet_mpl')
        forecast_plot.plot_forecast(df_forecast_pi_facet, 'png', path, 1200,
                                    900, 'Test Plot', show_legend=True,
                                    auto_open=False,
                                    include_interval=True)
        self.assertTrue(os.path.isfile('{}.png'.format(path)))

        # Test the case where a 'None' or an empty path is provided
        self.assertTrue(forecast_plot.plot_forecast(df_forecast_pi_facet,
                                                    'png', None, 1200, 900,
                                                    'Test Plot',
                                                    show_legend=True,
                                                    auto_open=False))

        self.assertTrue(forecast_plot.plot_forecast(df_forecast_pi_facet,
                                                    'png', '', 1200, 900,
                                                    'Test Plot',
                                                    show_legend=True,
                                                    auto_open=False))

    def test_plot_forecast_html(self):
        if not forecast_plot._plotly_imported:
            self.skipTest('Test skipped as Plotly is not installed...')

        path = get_file_path(base_folder, 'test_plotly')
        forecast_plot.plot_forecast(df_forecast, 'html', path, 600, 400,
                                    'Test Plot', show_legend=False,
                                    auto_open=False)
        self.assertTrue(os.path.isfile('{}.html'.format(path)))

        path = get_file_path(base_folder, 'test_plotly_legend')
        forecast_plot.plot_forecast(df_forecast, 'html', path, 600, 400,
                                    'Test Plot', show_legend=True,
                                    auto_open=False)
        self.assertTrue(os.path.isfile('{}.html'.format(path)))

        path = get_file_path(base_folder, 'test_facet_plotly')
        forecast_plot.plot_forecast(df_forecast_facet, 'html', path, 1200,
                                    900, 'Test Plot', show_legend=False,
                                    auto_open=False)
        self.assertTrue(os.path.isfile('{}.html'.format(path)))

        # Repeat test with prediction intervals
        path = get_file_path(base_folder, 'test_pi_plotly')
        forecast_plot.plot_forecast(df_forecast_pi, 'html', path, 600, 400,
                                    'Test Plot', show_legend=False,
                                    auto_open=False,
                                    include_interval=True)
        self.assertTrue(os.path.isfile('{}.html'.format(path)))

        path = get_file_path(base_folder, 'test_pi_facet_plotly')
        forecast_plot.plot_forecast(df_forecast_pi_facet, 'html', path, 1200,
                                    900, 'Test Plot', show_legend=False,
                                    auto_open=False,
                                    include_interval=True)
        self.assertTrue(os.path.isfile('{}.html'.format(path)))

        path = get_file_path(base_folder, 'test_pi_facet_slider_plotly')
        forecast_plot.plot_forecast(df_forecast_pi_facet, 'html', path, 1200,
                                    900, 'Test Plot', show_legend=False,
                                    auto_open=False,
                                    include_interval=True)
        self.assertTrue(os.path.isfile('{}.html'.format(path)))

        # Test the case where a 'None' or an empty path is provided
        self.assertTrue(forecast_plot.plot_forecast(df_forecast_pi_facet,
                                                    'html', None, 1200, 900,
                                                    'Test Plot',
                                                    show_legend=False,
                                                    auto_open=False))

        self.assertTrue(forecast_plot.plot_forecast(df_forecast_pi_facet,
                                                    'html', '', 1900, 1200,
                                                    'Test Plot',
                                                    show_legend=False,
                                                    auto_open=False))

    def test_plot_forecast_jupyter(self):
        if (not forecast_plot._plotly_imported) or \
                (not forecast_plot._ipython_imported):
            self.skipTest('Test skipped as either plotly or IPython is '
                          'not installed...')

        forecast_plot.plot_forecast(df_forecast_pi_facet,
                                    output='jupyter',
                                    width=1900,
                                    height=1200,
                                    title='Test Plot',
                                    show_legend=False,
                                    auto_open=False)
        # Todo: add checks to validate py.offline.iplot

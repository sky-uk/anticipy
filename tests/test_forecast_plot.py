# -- Public Imports

import logging
import unittest
from itertools import chain, repeat
import matplotlib.pyplot as plt
import numpy as np
import os
import itertools
import pandas as pd
from unittest import TestCase

# -- Private Imports
from anticipy.utils_test import PandasTest
from anticipy import forecast_plot

# -- Globals

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def logger_info(msg, data):
    logger.info(msg + '\n%s', data)


base_folder = os.path.join(os.path.dirname(__file__), 'test_plots')

def get_path_test_plot(name, is_matplotlib=False):
    if is_matplotlib:
        name = name+'_mpl'
    file_name = '{}.png'.format(name)
    return os.path.join(base_folder, file_name)


df_forecast = (
    pd.concat([
        pd.DataFrame({'date': pd.date_range('2018-01-01', periods=6, freq='D'),
                      'model': 'actuals',
                      'y': 1000 * np.arange(0., 6.),
                      'is_actuals': True
                      }),
        pd.DataFrame({'date': pd.date_range('2018-01-01', periods=10, freq='D'),
                      'model': 'forecast',
                      'y': 1000 * np.full(10, 5.),
                      'is_actuals': False
                      }),

    ], sort=False, ignore_index=True)
)

df_forecast_pi = (
    pd.concat([
        pd.DataFrame({'date': pd.date_range('2018-01-01', periods=6, freq='D'),
                      'model': 'actuals',
                      'y': 1000 * np.arange(0., 6.),
                      'is_actuals': True
                      }),
        pd.DataFrame({'date': pd.date_range('2018-01-01', periods=6, freq='D'),
                      'model': 'forecast',
                      'y': 1000 * np.full(6, 5.),
                      'is_actuals': False
                      }),
        pd.DataFrame({'date': pd.date_range('2018-01-07', periods=4, freq='D'),
                      'model': 'forecast',
                      'y': 1000 * np.full(4, 5.),
                      'is_actuals': False,
                      'q5': 1000 * np.full(4, 4.),
                      'q20': 1000 * np.full(4, 4.5),
                      'q80': 1000 * np.full(4, 5.5),
                      'q95': 1000 * np.full(4, 6.),
                      }),

    ], sort=False, ignore_index=True)
)

# Dataframe with different data sources, to plot with faceting
df_forecast_p1 = df_forecast.copy()
df_forecast_p2 = df_forecast.copy()
df_forecast_p1['source'] = 'ts1'
df_forecast_p2['source'] = 'ts2'
df_forecast_facet = pd.concat([df_forecast_p1, df_forecast_p2], sort=False, ignore_index=True)

df_forecast_p3 = df_forecast.copy()
df_forecast_p4 = df_forecast.copy()
df_forecast_p5 = df_forecast.copy()
df_forecast_p3['source'] = 'ts3'
df_forecast_p4['source'] = 'ts4'
df_forecast_p5['source'] = 'ts5'
df_forecast_facet_5 = pd.concat([df_forecast_p1,
                                 df_forecast_p2,
                                 df_forecast_p3,
                                 df_forecast_p4,
                                 df_forecast_p5], sort=False, ignore_index=True)


# As above, with prediction interval
# Dataframe with different data sources, to plot with faceting
df_forecast_p1_pi = df_forecast_pi.copy()
df_forecast_p2_pi = df_forecast_pi.copy()
df_forecast_p1_pi['source'] = 'ts1'
df_forecast_p2_pi['source'] = 'ts2'
df_forecast_facet_pi = pd.concat([df_forecast_p1_pi, df_forecast_p2_pi], sort=False, ignore_index=True)



class TestForecastPlot(PandasTest):

    def test_ggplot_fcast_save(self):
        is_matplotlib=True
        path = get_path_test_plot('test',is_matplotlib)
        forecast_plot.plot_forecast_save(df_forecast, path, 400, 300, 'Test Plot')
        logger_info('plot saved to :', path)

        path = get_path_test_plot('test_k',is_matplotlib)
        forecast_plot.plot_forecast_save(df_forecast, path, 400, 300, 'Test Plot', scale='k')
        logger_info('plot saved to :', path)

        path = get_path_test_plot('test_m',is_matplotlib)
        forecast_plot.plot_forecast_save(df_forecast, path, 400, 300, 'Test Plot', scale='M')
        logger_info('plot saved to :', path)

        # Todo: add checks about file creation, cleanup after running

        logger_info('debug - df_forecast_facet', df_forecast_facet)

        path = get_path_test_plot('test_facet',is_matplotlib)
        forecast_plot.plot_forecast_save(df_forecast_facet, path, 400, 300, 'Test Plot')
        logger_info('plot saved to :', path)

        ## Repeat test with prediction intervals
        # TODO: ADD _PI TO PATH NAME

        path = get_path_test_plot('test',is_matplotlib)
        forecast_plot.plot_forecast_save(df_forecast, path, 400, 300, 'Test Plot')
        logger_info('plot saved to :', path)

        path = get_path_test_plot('test_k',is_matplotlib)
        forecast_plot.plot_forecast_save(df_forecast, path, 400, 300, 'Test Plot', scale='k')
        logger_info('plot saved to :', path)

        path = get_path_test_plot('test_m',is_matplotlib)
        forecast_plot.plot_forecast_save(df_forecast, path, 400, 300, 'Test Plot', scale='M')
        logger_info('plot saved to :', path)

        # Todo: add checks about file creation, cleanup after running

        logger_info('debug - df_forecast_facet', df_forecast_facet)

        path = get_path_test_plot('test_facet')
        forecast_plot.plot_forecast_save(df_forecast_facet, path, 400, 300, 'Test Plot')
        logger_info('plot saved to :', path)

    def test_plot_forecast(self):
        i = forecast_plot.plot_forecast(df_forecast, 400, 300, 'Test Plot')
        logger_info('plot output:', repr(i))
        # Todo: add checks to validate Ipython.Image instance

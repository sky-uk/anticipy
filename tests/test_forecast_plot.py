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


def logger_info(msg, data):
    logger.info(msg + '\n%s', data)


base_folder = os.path.join(os.path.dirname(__file__), 'test_plots')


def get_path_test_plot(name):
    path = os.path.join(base_folder, name)
    return path


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
df_forecast_facet = pd.concat(
    [df_forecast_p1, df_forecast_p2], sort=False, ignore_index=True)

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
                                 df_forecast_p5],
                                sort=False,
                                ignore_index=True)


# As above, with prediction interval
# Dataframe with different data sources, to plot with faceting
df_forecast_p1_pi = df_forecast_pi.copy()
df_forecast_p2_pi = df_forecast_pi.copy()
df_forecast_p1_pi['source'] = 'ts1'
df_forecast_p2_pi['source'] = 'ts2'
df_forecast_facet_pi = pd.concat(
    [df_forecast_p1_pi, df_forecast_p2_pi], sort=False, ignore_index=True)

class TestForecastPlot(PandasTest):

    def test_plot_foracast_png(self):
        path = get_path_test_plot('test_mpl')
        forecast_plot.plot_forecast(df_forecast, path, 'png', 400, 300,
                                    'Test Plot')
        logger_info('plot saved to :', path)

        # Todo: add checks about file creation, cleanup after running

        logger_info('debug - df_forecast_facet', df_forecast_facet)

        path = get_path_test_plot('test_facet_mpl')
        forecast_plot.plot_forecast(df_forecast_pi, path, 'png', 400, 300,
                                    'Test Plot', show_legend=True,
                                    auto_open=False)
        logger_info('plot saved to :', path)

        # Repeat test with prediction intervals
        path = get_path_test_plot('test_pi_mpl')
        forecast_plot.plot_forecast(df_forecast, path, 'png', 400, 300,
                                    'Test Plot', show_legend=True,
                                    auto_open=False)
        logger_info('plot saved to :', path)

        # Todo: add checks about file creation, cleanup after running

        logger_info('debug - df_forecast_facet', df_forecast_facet)

        path = get_path_test_plot('test_pi_facet_mpl')
        forecast_plot.plot_forecast(df_forecast_facet, path, 'png', 400,
                                    300, 'Test Plot', show_legend=True,
                                    auto_open=False)

        logger_info('plot saved to :', path)

    def test_plot_forecast_html(self):
        #TODO
        logger.info('needs to be completed')

    def test_plot_forecast_jupyter(self):
        i = forecast_plot.plot_forecast(df_forecast,
                                        path=None,
                                        output='jupyter',
                                        width=1600,
                                        height=900,
                                        title='Test Plot',
                                        show_legend=False,
                                        auto_open=False)
        logger_info('plot output:', repr(i))
        # Todo: add checks to validate Ipython.Image instance

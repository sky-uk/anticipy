"""

Author: Pedro Capelastegui
Created on 04/12/2015

"""

import platform
import os
import logging
import unittest
import pandas as pd, numpy as np

from anticipy.model_utils import interpolate_df
from anticipy.utils_test import PandasTest
from anticipy.forecast import *

# Dask dependencies - not currently used
# from dask import delayed
# from dask import compute
# from dask.distributed import Client
# from dask.diagnostics import Profiler, ResourceProfiler, CacheProfiler
# from dask.diagnostics import visualize

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def logger_info(msg, data):
    logger.info(msg + '\n%s\n', data)


base_folder = os.path.join(os.path.dirname(__file__), 'data')

pd.set_option('display.max_columns', 40)
pd.set_option('display.max_rows', 200)
pd.set_option('display.width', 1000)


def list_to_str(l):
    if isinstance(l, list):
        return str([str(i) for i in l])
    else:
        return str(l)

def array_ones_in_indices(n, l_indices):
    return np.isin(np.arange(0, n), l_indices).astype(float)

def array_zeros_in_indices(n, l_indices):
    return (~np.isin(np.arange(0, n), l_indices)).astype(float)

def print_forecast_driver_output(fcast_driver_output, log_first_line=None):
    if fcast_driver_output.empty:
        logger.info('Error: empty output')
    else:
        if log_first_line is not None:
            log_first_line = '\r\n' + log_first_line
        else:
            log_first_line = ''
        logger.info(log_first_line + '\r\nAIC_C:' + str(fcast_driver_output.dict_aic_c))
        # logger_info('AIC_C:',fcast_driver_output[0])

# usage:
# compute_prof(l_dict_result2_d, scheduler = 'processes', num_workers=4, title='Test figure')
def compute_prof(*args, **kwargs ):
    with Profiler() as prof, ResourceProfiler(dt=0.25) as rprof:
        out = compute(*args, **kwargs)
    visualize([prof, rprof,# cprof
               ], show=True)
    return out


class TestForecast(PandasTest):
    def setUp(self):
        pass

    def test_normalize_df(self):

        def run_test(df, df_expected, **kwargs):
            df_out = normalize_df(df, **kwargs)
            logger_info('df_out:', df_out.tail(10))
            self.assert_frame_equal(df_out, df_expected)

        a_y = np.full(10, 0.0)
        a_x = np.arange(0, 10).astype(np.int64)
        a_x2 = np.tile(np.arange(0, 5), 2).astype(np.int64)
        a_x2_out = np.repeat(np.arange(0, 5), 2).astype(np.int64)
        a_source = ['s1'] * 5 + ['s2'] * 5
        a_weight = np.full(10, 1.0)
        a_date = pd.date_range('2014-01-01', periods=10, freq='D')
        a_date2 = np.tile(pd.date_range('2014-01-01', periods=5, freq='D'), 2)
        a_date2_out = np.repeat(pd.date_range('2014-01-01', periods=5, freq='D'), 2)

        logger_info('DEBUG: ', a_date2)

        # Test 0: Empty input

        self.assertIsNone(normalize_df(pd.DataFrame))

        # Test 1: Output with x,y columns
        df_expected = pd.DataFrame({'y': a_y, 'x': a_x, })[['x', 'y']]

        l_input = [
            [pd.DataFrame({'y': a_y}), {}],
            [pd.DataFrame({'y': a_y, 'x': a_x}), {}],
            [pd.DataFrame({'y_test': a_y, 'x_test': a_x}), {'col_name_y': 'y_test', 'col_name_x': 'x_test'}]
        ]
        for df, kwargs in l_input:
            run_test(df, df_expected, **kwargs)

        # Test 2: Output with x,y,weight columns
        df_expected = pd.DataFrame({'y': a_y, 'x': a_x, 'weight': a_weight})[['x', 'y', 'weight']]

        l_input = [
            [pd.DataFrame({'y': a_y, 'weight': a_weight}), {}],
            [pd.DataFrame({'y': a_y, 'x': a_x, 'weight': a_weight}), {}],
            [pd.DataFrame({'y_test': a_y, 'x_test': a_x, 'weight_test': a_weight}),
             {'col_name_y': 'y_test', 'col_name_x': 'x_test', 'col_name_weight': 'weight_test'}]
        ]
        for df, kwargs in l_input:
            run_test(df, df_expected, **kwargs)

        # Test 3: Output with x,y,weight,date columns
        df_expected = pd.DataFrame({'y': a_y, 'x': a_x, 'weight': a_weight, 'date': a_date})[
            ['date', 'x', 'y', 'weight']]

        l_input = [
            [pd.DataFrame({'y': a_y, 'weight': a_weight, 'date': a_date}), {}],
            [pd.DataFrame({'y': a_y, 'weight': a_weight}, index=a_date), {}],
            [pd.DataFrame({'y': a_y, 'x': a_x, 'weight': a_weight, 'date': a_date}), {}],
            [pd.DataFrame({'y_test': a_y, 'x_test': a_x, 'weight_test': a_weight, 'date_test': a_date}),
             {'col_name_y': 'y_test', 'col_name_x': 'x_test', 'col_name_weight': 'weight_test',
              'col_name_date': 'date_test'}]
        ]
        for df, kwargs in l_input:
            run_test(df, df_expected, **kwargs)

        # Test 4: Input series
        df_expected = pd.DataFrame({'y': a_y, 'x': a_x, })[['x', 'y']]

        l_input = [
            [pd.Series(a_y, name='y'), {}],
            [pd.Series(a_y, name='y', index=a_x), {}],
            [pd.Series(a_y, name='y_test'), {'col_name_y': 'y_test'}],
            # [pd.DataFrame({'y_test': a_y, 'x_test': a_x}), {'col_name_y':'y_test','col_name_x':'x_test'}]
        ]
        for df, kwargs in l_input:
            run_test(df, df_expected, **kwargs)

        # Test 5: Input series with datetimeindex
        df_expected = pd.DataFrame({'y': a_y, 'x': a_x, 'date': a_date})[['date', 'x', 'y']]

        l_input = [
            [pd.Series(a_y, name='y', index=a_date), {}],
            [pd.Series(a_y, name='y_test', index=a_date), {'col_name_y': 'y_test'}],
            # [pd.DataFrame({'y_test': a_y, 'x_test': a_x}), {'col_name_y':'y_test','col_name_x':'x_test'}]
        ]
        for df, kwargs in l_input:
            run_test(df, df_expected, **kwargs)

        # Test 6: Input df, output with x, y, weight, date, source columns
        df_expected = (
            pd.DataFrame({'y': a_y, 'x':a_x2, 'source': a_source, 'weight':a_weight,'date':a_date2})
            [['date','source','x','y','weight']]
        )

        l_input = [
            [pd.DataFrame({'y': a_y, 'weight':a_weight, 'date':a_date2, 'source': a_source}),{}],
            # Datetime index not supported with source - could be added back with multindex
            #[pd.DataFrame({'y': a_y, 'weight': a_weight},index = a_date), {}],
            [pd.DataFrame({'y': a_y, 'x': a_x2, 'weight': a_weight,'source':a_source,'date':a_date2}), {}],
            [pd.DataFrame({'y_test': a_y, 'x_test': a_x2, 'weight_test':a_weight, 'date_test':a_date2,
                           'source_test':a_source}),
             {'col_name_y':'y_test','col_name_x':'x_test','col_name_weight':'weight_test', 'col_name_date':'date_test',
              'col_name_source':'source_test'}]
        ]
        for df, kwargs in l_input:
            run_test(df, df_expected, **kwargs)

        # Test 7: Input df has multiple values per date per source
        df_expected = (
            pd.DataFrame({'y': a_y, 'x':a_x2_out, 'weight':a_weight,'date':a_date2_out})
            [['date','x','y','weight']]
        )

        l_input = [
            [pd.DataFrame({'y': a_y, 'weight': a_weight, 'date': a_date2}), {}],
            # Datetime index not supported with source - could be added back with multindex
            [pd.DataFrame({'y': a_y, 'x': a_x2, 'weight': a_weight, 'date': a_date2}), {}],
            [pd.DataFrame({'y_test': a_y, 'x_test': a_x2, 'weight_test': a_weight, 'date_test': a_date2,
                           }),
             {'col_name_y': 'y_test', 'col_name_x': 'x_test', 'col_name_weight': 'weight_test',
              'col_name_date': 'date_test',
              }]
        ]
        for df, kwargs in l_input:
            run_test(df, df_expected, **kwargs)

        # Test 8: input df has date column in string form
        a_date_str = a_date2.astype(str)
        df_expected = (
            pd.DataFrame({'y': a_y, 'x':a_x2, 'source': a_source, 'weight':a_weight,'date':a_date2})
            [['date','source','x','y','weight']]
        )

        l_input = [
            [pd.DataFrame({'y': a_y, 'weight': a_weight, 'date': a_date_str, 'source': a_source}), {}],
            [pd.DataFrame({'y': a_y, 'x': a_x2, 'weight': a_weight, 'source': a_source, 'date': a_date_str}), {}],
            [pd.DataFrame({'y_test': a_y, 'x_test': a_x2, 'weight_test': a_weight, 'date_test': a_date_str,
                           'source_test': a_source}),
             {'col_name_y': 'y_test', 'col_name_x': 'x_test', 'col_name_weight': 'weight_test',
              'col_name_date': 'date_test',
              'col_name_source': 'source_test'}]
        ]
        for df, kwargs in l_input:
            run_test(df, df_expected, **kwargs)

        # Test 9: unordered input df

        df_expected = pd.DataFrame({'y': a_y, 'x':a_x,})[['x','y']]

        l_input = [
            [pd.DataFrame({'y': a_y[::-1]}),{}],
            [pd.DataFrame({'y': a_y[::-1], 'x': a_x[::-1]}), {}],
        ]
        for df, kwargs in l_input:
            run_test(df, df_expected, **kwargs)

        # Test 10: candy production dataset
        path_candy = os.path.join(base_folder, 'candy_production.csv')
        df_candy_raw = pd.read_csv(path_candy)
        df_candy = df_candy_raw.pipe(normalize_df,
                                     col_name_y='IPG3113N', col_name_date='observation_date')
        logger_info('df_candy:', df_candy.tail())

        # Test 11: test_normalize.csv

        path_file = os.path.join(base_folder, 'test_normalize.csv')
        df_test_raw = pd.read_csv(path_file)
        df_test = df_test_raw.pipe(normalize_df,)
        logger_info('df_test:', df_test.x.diff().loc[df_test.x.diff()>1.0])
        self.assertFalse((df_test.x.diff()>1.0).any())

        # Test 11b: test_normalize.csv, with gaps

        path_file = os.path.join(base_folder, 'test_normalize.csv')
        df_test_raw = pd.read_csv(path_file)
        df_test_raw = pd.concat([df_test_raw.head(10), df_test_raw.tail(10)])
        df_test = df_test_raw.pipe(normalize_df,)
        logger_info('df_test:',df_test)
        logger_info('df_test:', df_test.x.diff().loc[df_test.x.diff()>1.0])
        self.assertTrue((df_test.x.max()==43))

    def test_interpolate_df(self):

        # # Test 1: DF with date column, gap
        # a_y = np.arange(0,10.)
        # a_date = pd.date_range(start='2018-01-01', periods=len(a_y), freq='D')
        # df_expected = pd.DataFrame({'y': a_y, 'date': a_date}).pipe(normalize_df)
        # df = pd.concat([df_expected.head(5), df_expected.tail(-6)]).pipe(normalize_df)
        #
        # df_result = df.pipe(interpolate_df)
        # logger_info('df_result:', df_result)
        # self.assert_frame_equal(df_result, df_expected)
        #
        # df_result = df.pipe(interpolate_df, include_mask=True)
        #
        # # Test 1: DF with no date column, gap
        # a_y = np.arange(0,10.)
        # a_date = pd.date_range(start='2018-01-01', periods=len(a_y), freq='D')
        # df_expected = pd.DataFrame({'y': a_y}).pipe(normalize_df)
        # df = pd.concat([df_expected.head(5), df_expected.tail(-6)]).pipe(normalize_df)
        #
        # df_result = df.pipe(interpolate_df)
        # logger_info('df_result:', df_result)
        # self.assert_frame_equal(df_result, df_expected)
        #
        # df_result = df.pipe(interpolate_df, include_mask=True)
        # logger_info('df_result:', df_result)


        # Test 2: Sparse series with date gaps
        df_test = pd.DataFrame({'date': pd.to_datetime(['2018-08-01', '2018-08-09']), 'y': [1., 2.]})
        df_result = df_test.pipe(interpolate_df, include_mask=True)
        logger_info('df_result:', df_result)
        self.assertEqual(df_result.index.size,9)




    def test_forecast_input(self):
        y_values1 = pd.DataFrame({'a': np.full(100, 0.0),
                                  'b': np.round(np.arange(-0.5, 0.5, 0.01), 2), },
                                 index=pd.date_range('2014-01-01', periods=100, freq='D'))
        # Too few samples
        n = 4
        y_values1b = pd.DataFrame({'a': np.full(n, 0.0)},
                                  index=pd.date_range('2014-01-01', periods=n, freq='D'))

        y_values2 = pd.DataFrame({'a': np.full(100, 0.0)},
                                 index=pd.date_range('2014-01-01', periods=100, freq='D'))

        # SolverConfig with trend
        conf1 = ForecastInput(
            source_id='source1',
            l_model_trend=[forecast_models.model_constant, forecast_models.model_linear],
            l_model_season=None, df_y=y_values1,
            weights_y_values=1.0, date_start_actuals=None
        )
        logger_info('Solver config:', conf1)

    def test_get_residuals(self):
        # Linear model
        model = forecast_models.model_linear
        a_y = np.arange(10.0)
        a_x = np.arange(10.0)
        a_date = None
        # Using parameter(0,0)
        residuals = get_residuals([0, 0], model, a_x, a_y, a_date)
        l_expected1 = np.arange(10.0)
        logger_info('residuals:', residuals)
        self.assert_array_equal(residuals, l_expected1)

        # Test - If input array is not 1-dimensional, throw Exception
        model = forecast_models.model_linear
        a_y = pd.DataFrame({'a': np.arange(10.0), 'b': -np.arange(10.0)}).values
        a_x = np.arange(10.0)
        with self.assertRaises(AssertionError):
            residuals = get_residuals([0, 0], model, a_x, a_y, a_date)

        # Test - multiple values per sample
        a_y = np.concatenate([np.arange(10.0), -np.arange(10.0)])
        a_x = np.tile(np.arange(10.0), 2)

        residuals = get_residuals([0, 0], model, a_x, a_y, a_date)
        logger_info('residuals:', residuals)
        l_expected2 = np.concatenate([np.arange(10.0), np.arange(10.0)])
        self.assert_array_equal(residuals, l_expected2)

        # As above, but applying weights to input time series [1.0, 0]
        residuals = get_residuals([0, 0], model, a_x, a_y, a_date,
                                  a_weights=np.repeat([1.0, 0], 10))
        l_expected2b = np.concatenate([np.arange(10.0), np.full(10, 0)])
        logger_info('residuals:', residuals)
        self.assert_array_equal(residuals, l_expected2b)

        # TODO: MORE TESTS WITH WEIGHTS_Y_VALUES

        # New test, different parameters
        residuals = get_residuals([0, 5], model, a_x, a_y, a_date)
        logger_info('residuals:', residuals)
        self.assert_array_equal(residuals,
                                [5., 4., 3., 2., 1., 0., 1., 2., 3., 4., 5., 6., 7.,
                                 8., 9., 10., 11., 12., 13., 14.])

        # Test - Use a_weights to weight residuals based on time
        # Using parameter(0,0)
        a_y = np.arange(10.0)
        a_x = np.arange(10.0)
        a_weights = np.linspace(1., 2., 10)
        logger_info('a_y: ', a_y)
        logger_info('a_weights: ', a_weights)
        residuals = get_residuals([0, 0], model, a_x, a_y, a_date, a_weights=a_weights)
        self.assert_array_equal(residuals, np.arange(10.0) * a_weights)
        logger_info('residuals:', residuals)

    def test_optimize_least_squares(self):
        # Setup
        a_x = pd.np.arange(100.0)
        a_y = np.arange(100.0)

        a_x_long = np.tile(a_x, 2)
        a_y_long = np.concatenate([np.full(100, 0.0),
                                   np.round(np.arange(-0.5, 0.5, 0.01), 2)])
        a_date = None

        l_model = [
            forecast_models.model_linear,
            forecast_models.model_constant
        ]

        def print_result(result):
            logger.info('result cost: %s, shape: %s, x: %s, message: %s',
                        result.cost, result.fun.shape, result.x, result.message)

        for model in l_model:
            logger.info('#### Model function: %s', model.name)

            df_result = optimize_least_squares(model, a_x, a_y, a_date)
            logger_info('result:', df_result)
            self.assertTrue(df_result.success.any())
            # logger_info('result.x:',res_trend.x)

            df_result = optimize_least_squares(model, a_x_long, a_y_long, a_date)
            logger_info('result:', df_result)
            self.assertTrue(df_result.success.any())

    def test_fit_model(self):
        # Input dataframes must have an y column, and may have columns x,date, weight

        # Setup
        # TODO: Use pre-normalized input dfs, rather than callling normalize_df()
        dict_df_y = {
            # Single ts
            'df_1ts_nodate': pd.DataFrame({'y': np.full(100, 0.0),
                                           'weight_test': np.full(100, 1.0)}),
            # 2 ts
            'df_2ts_nodate': pd.DataFrame({'y': np.concatenate([np.full(100, 0.0),
                                                                np.round(np.arange(-0.5, 0.5, 0.01), 2)]),
                                           'weight_test': np.concatenate([np.full(100, 0.1), np.full(100, 1.0)]),
                                           'x': np.tile(np.arange(0, 100), 2),
                                           }),
            # 1 ts with datetime index
            'df_1ts_w': pd.DataFrame({'y': np.full(100, 0.0),
                                      'weight_test': np.full(100, 1.0)
                                      },
                                     index=pd.date_range('2014-01-01', periods=100, freq='W')),
            # 2 ts with datetime index
            'df_2ts_w': pd.DataFrame({'y': np.concatenate([np.full(100, 0.0), np.round(np.arange(-0.5, 0.5, 0.01), 2)]),
                                      'weight_test': np.concatenate([np.full(100, 0.1), np.full(100, 1.0)]),
                                      },
                                     index=np.tile(pd.date_range('2014-01-01', periods=100, freq='W'), 2)),
            # Single ts, freq=D
            'df_1ts_d': pd.DataFrame({'y': np.full(100, 0.0),
                                      'weight_test': np.full(100, 1.0)},
                                     index=pd.date_range('2014-01-01', periods=100, freq='D')),
            # 2 ts with datetime index, freq=D
            'df_2ts_d': pd.DataFrame({'y': np.concatenate([np.full(100, 0.0), np.round(np.arange(-0.5, 0.5, 0.01), 2)]),
                                      'weight_test': np.concatenate([np.full(100, 0.1), np.full(100, 1.0)])},
                                     index=np.tile(pd.date_range('2014-01-01', periods=100, freq='D'), 2))
        }
        l_source1 = ['df_1ts_nodate', 'df_2ts_nodate', 'df_1ts_w', 'df_1ts_w', 'df_2ts_d', 'df_2ts_d']
        l_source2 = ['df_1ts_d', 'df_2ts_d']

        # Naive trend models - cannot add seasonality
        l_model1a = [
            forecast_models.model_naive, # model_naive never actually goes to fit_model
            # TODO: add assert check on fit model re: validity of input model

        ]

        l_model1b = [
            forecast_models.model_snaive_wday
            # TODO: add assert check on fit model re: validity of input model

        ]

        l_model1c = [
            forecast_models.model_linear,
            forecast_models.model_constant
        ]
        # All trend models
        l_model1 = l_model1a+l_model1b+l_model1c

        l_model2 = [
            forecast_models.model_season_wday,
            forecast_models.model_season_wday_2,
            forecast_models.model_season_month
        ]
        l_model3 = get_list_model(l_model1c, l_model2)

        l_results = []
        l_optimize_info = []

        l_add_weight = [False, True]

        def run_test_logic(source, model, add_weight):
            df_y = dict_df_y[source].copy()
            if add_weight:  # Enable weight column
                df_y['weight'] = df_y['weight_test']
            df_y = df_y.pipe(normalize_df)
            logger.info('Fitting src: %s , mod: %s, add_weight: %s', source, model, add_weight)
            dict_fit_model = fit_model(model, df_y,  source=source, df_actuals = df_y)
            return dict_fit_model
            # logger_info('Result: ',result)

        # Test - single solver type, return best fit
        for (source, model, add_weight) in itertools.product(
                l_source1, l_model1a+l_model1c, l_add_weight):
            dict_fit_model = run_test_logic(source, model, add_weight)
            result_tmp = dict_fit_model['metadata']
            info_tmp = dict_fit_model['optimize_info']
            l_results += [result_tmp]
            l_optimize_info += [info_tmp]

        # Now for models that require datetimeindex
        for (source, model, add_weight) in itertools.product(
                l_source2, l_model1b+l_model2, l_add_weight):
            dict_fit_model = run_test_logic(source, model, add_weight)
            result_tmp = dict_fit_model['metadata']
            info_tmp = dict_fit_model['optimize_info']
            l_results += [result_tmp]
            l_optimize_info += [info_tmp]

        # Finally, we use trend+seasonality with all models
        for (source, model, add_weight) in itertools.product(
                l_source2, l_model3, l_add_weight):
            dict_fit_model = run_test_logic(source, model, add_weight)
            result_tmp = dict_fit_model['metadata']
            info_tmp = dict_fit_model['optimize_info']
            l_results += [result_tmp]
            l_optimize_info += [info_tmp]

        df_result = pd.concat(l_results, sort=False, ignore_index=True)
        df_optimize_info = pd.concat(l_optimize_info, sort=False, ignore_index=True)

        self.assertFalse(df_result.cost.pipe(pd.isnull).any())

        logger_info('Result summary:', df_result)
        logger_info('Optimize info summary:', df_optimize_info)

    @unittest.skip('Dask not supported yet')
    def test_fit_model_dask(self):
        # Input dataframes must have an y column, and may have columns x,date, weight

        # Setup
        # TODO: Use pre-normalized input dfs, rather than callling normalize_df()
        dict_df_y = {
            # Single ts
            'df_1ts_nodate': pd.DataFrame({'y': np.full(100, 0.0),
                                           'weight_test': np.full(100, 1.0)}),
            # 2 ts
            'df_2ts_nodate': pd.DataFrame({'y': np.concatenate([np.full(100, 0.0),
                                                                np.round(np.arange(-0.5, 0.5, 0.01), 2)]),
                                           'weight_test': np.concatenate([np.full(100, 0.1), np.full(100, 1.0)]),
                                           'x': np.tile(np.arange(0, 100), 2),
                                           }),
            # 1 ts with datetime index
            'df_1ts_w': pd.DataFrame({'y': np.full(100, 0.0),
                                      'weight_test': np.full(100, 1.0)
                                      },
                                     index=pd.date_range('2014-01-01', periods=100, freq='W')),
            # 2 ts with datetime index
            'df_2ts_w': pd.DataFrame({'y': np.concatenate([np.full(100, 0.0), np.round(np.arange(-0.5, 0.5, 0.01), 2)]),
                                      'weight_test': np.concatenate([np.full(100, 0.1), np.full(100, 1.0)]),
                                      },
                                     index=np.tile(pd.date_range('2014-01-01', periods=100, freq='W'), 2)),
            # Single ts, freq=D
            'df_1ts_d': pd.DataFrame({'y': np.full(100, 0.0),
                                      'weight_test': np.full(100, 1.0)},
                                     index=pd.date_range('2014-01-01', periods=100, freq='D')),
            # 2 ts with datetime index, freq=D
            'df_2ts_d': pd.DataFrame({'y': np.concatenate([np.full(100, 0.0), np.round(np.arange(-0.5, 0.5, 0.01), 2)]),
                                      'weight_test': np.concatenate([np.full(100, 0.1), np.full(100, 1.0)])},
                                     index=np.tile(pd.date_range('2014-01-01', periods=100, freq='D'), 2))
        }
        l_source1 = ['df_1ts_nodate', 'df_2ts_nodate', 'df_1ts_w', 'df_1ts_w', 'df_2ts_d', 'df_2ts_d']
        l_source2 = ['df_1ts_d', 'df_2ts_d']

        l_model1 = [
            forecast_models.model_naive, # model_naive never actually goes to fit_model
            # TODO: add assert check on fit model re: validity of input model
            forecast_models.model_linear,
            forecast_models.model_constant
        ]
        l_model2 = [
            forecast_models.model_season_wday,
            forecast_models.model_season_wday_2,
            forecast_models.model_season_month
        ]
        l_model3 = get_list_model(l_model1, l_model2)

        l_add_weight = [False, True]

        def run_test_logic(df_y, source, model, add_weight):
            #df_y = dict_df_y[source].copy()
            #if add_weight:  # Enable weight column
            #    df_y['weight']=df_y['weight_test']
            col_name_weight = 'weight' if add_weight==True else 'no-weight'
            df_y = df_y.pipe(normalize_df, col_name_weight=col_name_weight)
            #logger.info('Fitting src: %s , mod: %s, add_weight: %s', source, model, add_weight)
            #dict_fit_model = delayed(fit_model)(model, df_y,  source=source, df_actuals = df_y)
            dict_fit_model = fit_model(model, df_y, source=source, df_actuals=df_y)
            return dict_fit_model
            # logger_info('Result: ',result)

        def aggregate_dict_fit_model(l_dict_fit_model):
            l_results = []
            l_optimize_info = []
            for dict_fit_model in l_dict_fit_model:
                result_tmp = dict_fit_model['metadata']
                info_tmp = dict_fit_model['optimize_info']
                l_results += [result_tmp]
                l_optimize_info += [info_tmp]
            df_metadata = pd.concat(l_results, sort=False, ignore_index=True)
            df_optimize_info = pd.concat(l_optimize_info, sort=False, ignore_index=True)
            return df_metadata, df_optimize_info

        l_dict_fit_model_d = []

        # Test - single solver type, return best fit
        for (source, model, add_weight) in itertools.product(
                l_source1, l_model1, l_add_weight):
            l_dict_fit_model_d += [delayed(run_test_logic)(dict_df_y[source].copy(), source, model, add_weight)]


        # Now for models that require datetimeindex
        for (source, model, add_weight) in itertools.product(
                l_source2, l_model2, l_add_weight):
            l_dict_fit_model_d += [delayed(run_test_logic)(dict_df_y[source].copy(), source, model, add_weight)]

        # Finally, we use trend+seasonality with all models
        for (source, model, add_weight) in itertools.product(
                l_source2, l_model3, l_add_weight):
            l_dict_fit_model_d += [delayed(run_test_logic)(dict_df_y[source].copy(), source, model, add_weight)]

        logger.info('generated delayed')

        #client = Client()
        #logger_info('client:',client)
        #l_dict_fit_model, = compute(l_dict_fit_model_d)
        l_dict_fit_model, = compute_prof(l_dict_fit_model_d, scheduler='processes', num_workers=4)
        #l_dict_fit_model, = compute(l_dict_fit_model_d, scheduler='processes', num_workers=4)
        #l_dict_fit_model, = compute(l_dict_fit_model_d, scheduler='distributed', num_workers=4)
        #l_dict_fit_model, = compute(l_dict_fit_model_d, scheduler='threads', num_workers=4)
        #l_dict_fit_model = l_dict_fit_model_d

        df_metadata, df_optimize_info = aggregate_dict_fit_model(l_dict_fit_model)
        #result_d = delayed(aggregate_dict_fit_model)(l_dict_fit_model_d)
        #result_d = delayed(aggregate_dict_fit_model)(l_dict_fit_model_d)
        #(df_metadata, df_optimize_info), = compute(result_d)
        # result, = compute(result_d)
        logger_info('Result summary:', df_metadata)
        logger_info('Optimize info summary:', df_optimize_info)
        #client.close()

    @unittest.skip('Dask not supported yet')
    def test_fit_model_dask2(self):
        # Input dataframes must have an y column, and may have columns x,date, weight

        # Setup
        # TODO: Use pre-normalized input dfs, rather than callling normalize_df()
        dict_df_y = {
            # Single ts
            'df_1ts_nodate': pd.DataFrame({'y': np.full(100, 0.0),
                                           'weight_test': np.full(100, 1.0)}),
            # 2 ts
            'df_2ts_nodate': pd.DataFrame({'y': np.concatenate([np.full(100, 0.0),
                                                                np.round(np.arange(-0.5, 0.5, 0.01), 2)]),
                                           'weight_test': np.concatenate([np.full(100, 0.1), np.full(100, 1.0)]),
                                           'x': np.tile(np.arange(0, 100), 2),
                                           }),
            # 1 ts with datetime index
            'df_1ts_w': pd.DataFrame({'y': np.full(100, 0.0),
                                      'weight_test': np.full(100, 1.0)
                                      },
                                     index=pd.date_range('2014-01-01', periods=100, freq='W')),
            # 2 ts with datetime index
            'df_2ts_w': pd.DataFrame({'y': np.concatenate([np.full(100, 0.0), np.round(np.arange(-0.5, 0.5, 0.01), 2)]),
                                      'weight_test': np.concatenate([np.full(100, 0.1), np.full(100, 1.0)]),
                                      },
                                     index=np.tile(pd.date_range('2014-01-01', periods=100, freq='W'), 2)),
            # Single ts, freq=D
            'df_1ts_d': pd.DataFrame({'y': np.full(100, 0.0),
                                      'weight_test': np.full(100, 1.0)},
                                     index=pd.date_range('2014-01-01', periods=100, freq='D')),
            # 2 ts with datetime index, freq=D
            'df_2ts_d': pd.DataFrame({'y': np.concatenate([np.full(100, 0.0), np.round(np.arange(-0.5, 0.5, 0.01), 2)]),
                                      'weight_test': np.concatenate([np.full(100, 0.1), np.full(100, 1.0)])},
                                     index=np.tile(pd.date_range('2014-01-01', periods=100, freq='D'), 2))
        }
        l_source1 = ['df_1ts_nodate', 'df_2ts_nodate', 'df_1ts_w', 'df_1ts_w', 'df_2ts_d', 'df_2ts_d']
        l_source2 = ['df_1ts_d', 'df_2ts_d']

        l_model1 = [
            forecast_models.model_naive, # model_naive never actually goes to fit_model
            # TODO: add assert check on fit model re: validity of input model
            forecast_models.model_linear,
            forecast_models.model_constant
        ]
        l_model2 = [
            forecast_models.model_season_wday,
            forecast_models.model_season_wday_2,
            forecast_models.model_season_month
        ]
        l_model3 = get_list_model(l_model1, l_model2)

        l_weight = ['no-weight', 'weight_test']

        def run_test_logic(df_y, source, model, add_weight):
            #df_y = dict_df_y[source].copy()
            if add_weight:  # Enable weight column
                df_y['weight']=df_y['weight_test']
            df_y = df_y.pipe(normalize_df)
            #logger.info('Fitting src: %s , mod: %s, add_weight: %s', source, model, add_weight)
            #dict_fit_model = delayed(fit_model)(model, df_y,  source=source, df_actuals = df_y)
            dict_fit_model = fit_model(model, df_y, source=source, df_actuals=df_y)
            return dict_fit_model
            # logger_info('Result: ',result)

        def aggregate_dict_fit_model(l_dict_fit_model):
            l_results = []
            l_optimize_info = []
            for dict_fit_model in l_dict_fit_model:
                result_tmp = dict_fit_model['metadata']
                info_tmp = dict_fit_model['optimize_info']
                l_results += [result_tmp]
                l_optimize_info += [info_tmp]
            df_metadata = delayed(pd.concat)(l_results, sort=False, ignore_index=False)
            df_optimize_info = delayed(pd.concat)(l_optimize_info, sort=False, ignore_index=False)
            return df_metadata, df_optimize_info

        l_dict_fit_model_d = []

        # Test - single solver type, return best fit


        l_dict_fit_model_d += [
            delayed(fit_model)(model,
                               dict_df_y[source].pipe(delayed(normalize_df), col_name_weight=weight),
                               source=source, df_actuals=dict_df_y[source].pipe(delayed(normalize_df), col_name_weight=weight))
            for (source, model, weight) in itertools.product(l_source1, l_model1, l_weight)]

        # Now for models that require datetimeindex
        l_dict_fit_model_d += [
            delayed(fit_model)(model,
                               dict_df_y[source].pipe(delayed(normalize_df), col_name_weight=weight),
                               source=source, df_actuals=dict_df_y[source].pipe(delayed(normalize_df), col_name_weight=weight))
            for (source, model, weight) in itertools.product(l_source2, l_model2, l_weight)]

        # Finally, we use trend+seasonality with all models
        l_dict_fit_model_d += [
            delayed(fit_model)(model,
                               dict_df_y[source].pipe(delayed(normalize_df), col_name_weight=weight),
                               source=source, df_actuals=dict_df_y[source].pipe(delayed(normalize_df), col_name_weight=weight))
            for (source, model, weight) in itertools.product( l_source2, l_model3, l_weight)]

        # # Finally, we use trend+seasonality with all models
        # for (source, model, weight) in itertools.product(
        #         l_source2, l_model3, l_weight):
        #     df_y = dict_df_y[source].pipe(normalize_df, col_name_weight=weight)
        #     l_dict_fit_model_d += [delayed(fit_model)(model, df_y, source=source, df_actuals=df_y)]

        logger.info('generated delayed')

        #l_dict_fit_model, = compute(l_dict_fit_model_d)
        l_dict_fit_model = l_dict_fit_model_d

        #df_metadata, df_optimize_info = aggregate_dict_fit_model(l_dict_fit_model)
        result_d = delayed(aggregate_dict_fit_model)(l_dict_fit_model_d)
        (df_metadata, df_optimize_info), = compute(result_d)
        # result, = compute(result_d)
        logger_info('Result summary:', df_metadata)
        logger_info('Optimize info summary:', df_optimize_info)

    @unittest.skip('Dask not supported yet')
    def test_dask(self):
        def aggregate_result(l_dict_result):
            l_metadata = []
            l_opt = []
            for dict_result in l_dict_result:
                l_metadata += [dict_result['metadata']]
                l_opt += [dict_result['optimize_info']]
            return pd.concat(l_metadata, sort=False, ignore_index=False), pd.concat(l_opt, sort=False,
                                                                                    ignore_index=False)

        model = forecast_models.model_linear
        df_y = pd.DataFrame({'y': np.full(100, 0.0), 'weight_test': np.full(100, 1.0)}).pipe(normalize_df)

        l_dict_result2_d = [delayed(fit_model)(model, df_y, source=i, df_actuals=df_y) for i in np.arange(0, 20)]
        result_d = delayed(aggregate_result)(l_dict_result2_d)
        #result = compute(result_d,scheduler='processes',num_workers=2)
        result = compute(result_d)
        logger_info('result',result)


    def test_fit_model_date_gaps(self):
        # Setup
        # 2 ts with datetime index, freq=D
        df_2ts_d = pd.DataFrame({'y': np.concatenate([np.full(100, 0.0), np.round(np.arange(-0.5, 0.5, 0.01), 2)]),
                                 'weight': np.concatenate([np.full(100, 0.1), np.full(100, 1.0)])},
                                index=np.tile(pd.date_range('2014-01-01', periods=100, freq='D'), 2))

        df_y = pd.concat([df_2ts_d.head(), df_2ts_d.tail()])

        model = forecast_models.model_linear

        l_col_name_weight = [None, 'weight']

        l_results = []
        l_optimize_info = []

        def run_test_logic(col_name_weight):
            logger.info('Fitting col_w: %s', col_name_weight)
            df_y_tmp = df_y.pipe(normalize_df,col_name_weight=col_name_weight)
            dict_fit_model = fit_model(model, df_y_tmp, source='test')
            return dict_fit_model
            # logger_info('Result: ',result)

        # Test - single solver type, return best fit
        for col_name_weight in l_col_name_weight:
            dict_fit_model = run_test_logic(col_name_weight)
            result_tmp = dict_fit_model['metadata']
            info_tmp = dict_fit_model['optimize_info']
            l_results += [result_tmp]
            l_optimize_info += [info_tmp]

        df_result = pd.concat(l_results)
        df_optimize_info = pd.concat(l_optimize_info)
        logger_info('Result summary:', df_result)
        logger_info('Optimize info summary:', df_optimize_info)

    def test_get_list_model(self):
        l1 = [
            forecast_models.model_linear,
            forecast_models.model_constant
        ]
        l2 = [
            forecast_models.model_season_wday_2,
            forecast_models.model_null
        ]
        l_result_add = get_list_model(l1, l2, 'add')
        l_result_mult = get_list_model(l1, l2, 'mult')
        l_result_both = get_list_model(l1, l2, 'both')

        l_expected_add = [
            l1[0] + l2[0],
            l1[0] + l2[1],
            l1[1] + l2[0],
            l1[1] + l2[1],
        ]

        l_expected_mult = [
            l1[0] * l2[0],
            l1[0] * l2[1],
            l1[1] * l2[0],
            l1[1] * l2[1],
        ]
        l_expected_both = [
            l1[0] + l2[0],
            l1[0] + l2[1],
            l1[1] + l2[0],
            l1[1] + l2[1],
            l1[0] * l2[0],
            # l1[0] * l2[1], # This is a duplicate: linear*null = linear+null = linear
            l1[1] * l2[0],
            # l1[1] * l2[1], # This is a duplicate: constant*null = constant+null = constant
        ]
        logger_info('Result add:', l_result_add)
        logger_info('Expected add:', l_expected_add)
        self.assertListEqual(l_result_add, l_expected_add)

        logger_info('Result mult:', l_result_mult)
        logger_info('Expected mult:', l_expected_mult)
        self.assertListEqual(l_result_mult, l_expected_mult)

        logger_info('Result both:', l_result_both)
        logger_info('Expected both:', l_expected_both)
        self.assertListEqual(l_result_both, l_expected_both)

    def test_fit_model_trend_season_wday_mult(self):
        # Test Specific model combination that doesn't fit

        # Setup
        n_iterations = 10

        # Setup
        dict_df_y = {
            # Single ts
            'df_1ts_nodate': pd.DataFrame({'y': np.full(100, 0.0),
                                           'weight': np.full(100, 1.0)}),
            # 2 ts
            'df_2ts_nodate': pd.DataFrame({'y': np.concatenate([np.full(100, 0.0),
                                                                np.round(np.arange(-0.5, 0.5, 0.01), 2)]),
                                           'weight': np.concatenate([np.full(100, 0.1), np.full(100, 1.0)]),
                                           'x': np.tile(np.arange(0, 100), 2),
                                           }),
            # 1 ts with datetime index
            'df_1ts_w': pd.DataFrame({'y': np.full(100, 0.0),
                                      'weight': np.full(100, 1.0)
                                      },
                                     index=pd.date_range('2014-01-01', periods=100, freq='W')),

            # 2 ts with datetime index
            'df_2ts_w': pd.DataFrame({'y': np.concatenate([np.full(100, 0.0), np.round(np.arange(-0.5, 0.5, 0.01), 2)]),
                                      'weight': np.concatenate([np.full(100, 0.1), np.full(100, 1.0)]),
                                      },
                                     index=np.tile(pd.date_range('2014-01-01', periods=100, freq='W'), 2)),
            # Single ts, freq=D
            'df_1ts_d': pd.DataFrame({'y': np.full(100, 0.0),
                                      'weight': np.full(100, 1.0)},
                                     index=pd.date_range('2014-01-01', periods=100, freq='D')),

            # Single ts, freq=D , index named 'date
            'df_1ts_d2': pd.DataFrame({'y': np.full(100, 0.0),
                                       'weight': np.full(100, 1.0)},
                                      index=pd.date_range('2014-01-01', periods=100, freq='D', name='date'))
                .reset_index()
            ,

            # 2 ts with datetime index, freq=D
            'df_2ts_d': pd.DataFrame({'y': np.concatenate([np.full(100, 0.0), np.round(np.arange(-0.5, 0.5, 0.01), 2)]),
                                      'weight': np.concatenate([np.full(100, 0.1), np.full(100, 1.0)])},
                                     index=np.tile(pd.date_range('2014-01-01', periods=100, freq='D'), 2))
        }

        l_source_d = ['df_1ts_d', 'df_2ts_d','df_1ts_d2']
        l_source_w = ['df_1ts_2', 'df_2ts_2']

        l_model_trend = [
            forecast_models.model_linear,
        ]
        l_model_season = [
            # forecast_models.model_season_wday,
            # forecast_models.model_season_wday,
            forecast_models.model_season_wday_2,
            forecast_models.model_null
        ]

        l_col_name_weight = [  # None,
            'weight']

        l_results = []
        l_optimize_info = []

        # Fit , run n iterations, freq='D'
        for (source, col_name_weight, model) in itertools.product(
                l_source_d, l_col_name_weight, get_list_model(l_model_trend, l_model_season, 'both')):
            df_y = dict_df_y[source].copy().pipe(normalize_df,col_name_weight=col_name_weight)
            logger.info('Fitting src: %s , mod: %s, col_w: %s', source, model, col_name_weight)
            for i in np.arange(0, n_iterations):
                dict_fit_model = fit_model(model, df_y, source=source, freq='D')
                l_results += [dict_fit_model['metadata']]
                l_optimize_info += [dict_fit_model['optimize_info']]

        # Fit , run n iterations, freq='D' - test function composition in different order
        for (source, col_name_weight, model) in itertools.product(
                l_source_d, l_col_name_weight, get_list_model(l_model_season, l_model_trend, 'both')):
            df_y = dict_df_y[source].copy().pipe(normalize_df, col_name_weight=col_name_weight)
            logger.info('Fitting src: %s , mod: %s, col_w: %s', source, model, col_name_weight)
            for i in np.arange(0, n_iterations):
                dict_fit_model = fit_model(model, df_y, source=source, freq='D')
                l_results += [dict_fit_model['metadata']]
                l_optimize_info += [dict_fit_model['optimize_info']]

        df_result = pd.concat(l_results)
        df_optimize_info = pd.concat(l_optimize_info)
        logger_info('Result summary:', df_result)
        logger_info('Optimize info summary:', df_optimize_info)

    def test_extrapolate_model(self):
        # with freq=None, defaults to W
        df_y_forecast = extrapolate_model(forecast_models.model_constant, [1.0],
                                          '2017-01-01', '2017-01-01', freq=None, extrapolate_years=1.0)
        logger_info('df_y_forecast', df_y_forecast.tail(1))
        logger_info('Result length:', df_y_forecast.index.size)
        self.assertEquals(df_y_forecast.index.size, 53)

        df_y_forecast = extrapolate_model(forecast_models.model_constant, [1.0],
                                          '2017-01-01', '2017-12-31', freq='D', extrapolate_years=1.0)
        logger_info('df_y_forecast', df_y_forecast.tail(1))
        logger_info('Result length:', df_y_forecast.index.size)
        self.assertEquals(df_y_forecast.index.size, 365 * 2)

        df_y_forecast = extrapolate_model(forecast_models.model_constant, [1.0],
                                          '2017-01-01', '2017-12-31', freq='MS', extrapolate_years=1.0)
        logger_info('df_y_forecast', df_y_forecast.tail(1))
        logger_info('Result length:', df_y_forecast.index.size)
        self.assertEquals(df_y_forecast.index.size, 12 * 2)

        df_y_forecast = extrapolate_model(forecast_models.model_constant, [1.0],
                                          '2000-01-01', '2009-01-01', freq='YS', extrapolate_years=10.0)
        logger_info('df_y_forecast', df_y_forecast.tail(20))
        logger_info('Result length:', df_y_forecast.index.size)
        self.assertEquals(df_y_forecast.index.size, 20)

        # TODO: Test other time frequencies, e.g. Q, H, Y.

    def test_get_df_actuals_clean(self):
        dict_df_y = {
            # Single ts
            'df_1ts_nodate': pd.DataFrame({'y': np.full(100, 0.0),
                                           'weight': np.full(100, 1.0)}),
            # 2 ts
            'df_2ts_nodate': pd.DataFrame({'y': np.concatenate([np.full(100, 0.0),
                                                                np.round(np.arange(-0.5, 0.5, 0.01), 2)]),
                                           'weight': np.concatenate([np.full(100, 0.1), np.full(100, 1.0)]),
                                           'x': np.tile(np.arange(0, 100), 2),
                                           }),
            # 1 ts with datetime index
            'df_1ts_w': pd.DataFrame({'y': np.full(100, 0.0),
                                      'weight': np.full(100, 1.0)
                                      },
                                     index=pd.date_range('2014-01-01', periods=100, freq='W')),
            # 1 ts with datetime index named 'date
            'df_1ts_w-2': pd.DataFrame({'y': np.full(100, 0.0),
                                        'weight': np.full(100, 1.0)
                                        },
                                       index=pd.date_range('2014-01-01', periods=100, freq='W', name='date')),
            # 1 ts with datetime column
            'df_1ts_w-3': pd.DataFrame({'y': np.full(100, 0.0),
                                        'weight': np.full(100, 1.0),
                                        'date': pd.date_range('2014-01-01', periods=100, freq='W')
                                        })
        }
        # Simple test - check for crashes

        for k in dict_df_y.keys():
            logger.info('Input: %s', k)
            df_in = dict_df_y.get(k).pipe(normalize_df)
            logger_info('DF_IN',df_in.tail(3))
            df_result = get_df_actuals_clean(df_in,'test','test')
            logger_info('Result:', df_result.tail(3))
            unique_models = df_result.model.drop_duplicates().reset_index(drop=True)
            self.assert_series_equal(unique_models, pd.Series(['actuals']))
            logger_info('Models:', df_result.model.drop_duplicates())

    def _test_run_forecast_basic_tests_new_api(self, n_sources=1, **kwargs):
        # Both additive and multiplicative
        dict_result = run_forecast(simplify_output=False, **kwargs)

        df_data = dict_result['data']
        df_metadata = dict_result['metadata']
        df_optimize_info = dict_result['optimize_info']

        logger_info('df_metadata:', df_metadata)
        logger_info('df_optimize_info:', df_optimize_info)
        logger_info('df_data:', df_data.groupby(['source', 'model']).tail(1))

        l_sources = df_metadata.source_long.unique()
        include_all_fits = kwargs.get('include_all_fits')
        if not include_all_fits:  # In this case, there should be only one model fitted per data source
            self.assertTrue(df_metadata.is_best_fit.all())
            self.assertTrue((df_data.is_best_fit | df_data.is_actuals).all())
            # The following may not be true if a model doesn't converge
            self.assertEquals(df_metadata.index.size, n_sources)
            self.assertEquals(df_data.loc[~df_data.is_actuals].drop_duplicates('source_long').index.size,
                              n_sources)

        # Check that actuals are included
        self.assertTrue((df_data.is_actuals.any()))

        # Check that dtype is not corrupted
        self.assertTrue(np.issubdtype(df_data.y.astype(float), np.float64))

    def _test_run_forecast_check_length_new_api(self, **kwargs):
        freq = kwargs.get('freq', 'D')

        freq = detect_freq(kwargs.get('df_y').pipe(normalize_df))

        freq_short = freq[0:1] if freq is not None else None  # Changes e.g. W-MON to W
        freq_units_per_year = 52.0 if freq_short == 'W' else 365.0  # Todo: change to dict to support more frequencies

        extrapolate_years = kwargs.get('extrapolate_years', 1.0)

        # Both additive and multiplicative
        dict_result = run_forecast(simplify_output=False, extrapolate_years=extrapolate_years, **kwargs)

        df_data = dict_result['data']
        df_metadata = dict_result['metadata']

        logger_info('df_metadata:', df_metadata)
        logger_info('df_data:', df_data.groupby(['source', 'model']).tail(1))

        df_data_size = df_data.groupby(['source', 'model', 'is_actuals']).size().rename('group_size').reset_index()
        df_data_size_unique = (
            df_data.drop_duplicates(['source', 'model', 'is_actuals', 'date'])
                .groupby(['source', 'model', 'is_actuals']).size().rename('group_size').reset_index()
        )
        logger_info('df_data_size:', df_data_size)
        logger_info('df_data_size_unique:', df_data_size_unique)

        df_y = kwargs.get('df_y')
        assert df_y is not None

        # Normalize df_y
        df_y = normalize_df(df_y,
                            kwargs.get('col_name_y', 'y'),
                            kwargs.get('col_name_weight', 'weight'),
                            kwargs.get('col_name_x', 'x'),
                            kwargs.get('col_name_date', 'date'),
                            kwargs.get('col_name_source', 'source'))
        if 'source' not in df_y.columns:
            df_y['source'] = kwargs.get('source_id', 'source')

        l_sources = df_y.source.drop_duplicates()

        for source in l_sources:
            df_y_tmp = df_y.loc[df_y.source == source]

            size_actuals_unique_tmp = df_y_tmp.drop_duplicates('x').index.size
            size_actuals_tmp = df_y_tmp.index.size

            df_data_size_tmp = df_data_size.loc[df_data_size.source == source]
            df_data_size_actuals = df_data_size_tmp.loc[df_data_size_tmp.is_actuals]
            df_data_size_fcast = df_data_size_tmp.loc[~df_data_size_tmp.is_actuals]

            # logger.info('DEBUG: group size: %s',100 + extrapolate_years*freq_units_per_year)
            # This assert doesn't work for all years - some have 365 days, some 366. Currently running with 365-day year

            logger.info('DEBUG: df_data_size_fcast.group_size %s , size_actuals_tmp %s, total %s',
                        df_data_size_fcast.group_size.values, size_actuals_tmp,
                        size_actuals_tmp + extrapolate_years * freq_units_per_year)

            self.assertTrue((df_data_size_actuals.group_size == size_actuals_tmp).all())

            self.assert_array_equal(df_data_size_fcast.group_size,
                                    size_actuals_unique_tmp + extrapolate_years * freq_units_per_year)
            self.assertFalse(df_data_size_fcast.empty)

    def _test_run_forecast(self, freq='D'):
        # freq_short = freq[0:1]  # Changes e.g. W-MON to W
        # freq_units_per_year = 52.0 if freq_short == 'W' else 365.0  # Todo: change to dict to support more frequencies

        # Input dataframe without date column
        df_y0 = pd.DataFrame({'y': np.concatenate([np.full(100, 0.0), np.round(np.arange(-0.5, 0.5, 0.01), 2)]),
                              'weight': np.concatenate([np.full(100, 0.1), np.full(100, 1.0)]),
                              },
                             )

        df_y1 = pd.DataFrame({'y': np.concatenate([np.full(100, 0.0), np.round(np.arange(-0.5, 0.5, 0.01), 2)]),
                              'weight': np.concatenate([np.full(100, 0.1), np.full(100, 1.0)]),
                              },
                             index=np.tile(pd.date_range('2014-01-01', periods=100, freq=freq), 2))

        # Too few samples
        n = 4
        df_y1b = pd.DataFrame({'y': np.full(n, 0.0)},
                              index=pd.date_range('2017-01-01', periods=n, freq=freq))

        df_y2 = pd.DataFrame({'y': np.full(100, 0.0)},
                             index=pd.date_range('2017-01-01', periods=100, freq=freq))

        # Df with source column
        df_y3 = pd.DataFrame({'y': np.concatenate([np.full(100, 0.0), np.round(np.arange(-0.5, 0.5, 0.01), 2)]),
                              'weight': np.concatenate([np.full(100, 0.1), np.full(100, 1.0)]),
                              'source': ['src1'] * 100 + ['src2'] * 100
                              },
                             index=np.tile(pd.date_range('2014-01-01', periods=100, freq=freq), 2))
        # As above, with renamed columns
        df_y3b = pd.DataFrame({'y_test': np.concatenate([np.full(100, 0.0), np.round(np.arange(-0.5, 0.5, 0.01), 2)]),
                               'weight_test': np.concatenate([np.full(100, 0.1), np.full(100, 1.0)]),
                               'source_test': ['src1'] * 100 + ['src2'] * 100,
                               'date_test': np.tile(pd.date_range('2014-01-01', periods=100, freq=freq), 2)
                               })

        # # Model lists
        l_model_trend1 = [forecast_models.model_linear]
        l_model_trend1b = [forecast_models.model_linear, forecast_models.model_season_wday_2]
        l_model_trend2 = [forecast_models.model_linear, forecast_models.model_exp]

        l_model_season1 = [forecast_models.model_season_wday_2]
        l_model_season2 = [forecast_models.model_season_wday_2, forecast_models.model_null]
        #
        # # # Test input with source column, multiple sources
        # self._test_run_forecast_basic_tests_new_api(df_y=df_y3, include_all_fits=True,
        #                                             l_model_trend=l_model_trend2, l_model_season=l_model_season2)
        # self._test_run_forecast_basic_tests_new_api(df_y=df_y3b, include_all_fits=True,
        #                                             l_model_trend=l_model_trend2, l_model_season=l_model_season2,
        #                                             col_name_y='y_test', col_name_date='date_test',
        #                                             col_name_source='source_test', col_name_weight='weight_test')

        ## New test - forecast length
        logger.info('Testing Output Length')
        self._test_run_forecast_check_length_new_api(df_y=df_y1, include_all_fits=False,
                                                     l_model_trend=l_model_trend1b, source_id='source1')
        self._test_run_forecast_check_length_new_api(df_y=df_y2, include_all_fits=False,
                                                     l_model_trend=l_model_trend2, l_model_season=l_model_season2,
                                                     source_id='source2')

    def test_runforecast(self):
        for freq in ['D',
                     'W']:
            self._test_run_forecast(freq=freq)

    def test_run_forecast_simple_linear_model(self):
        df1 = pd.DataFrame({'y': np.arange(0, 10.)},
                           index=pd.date_range('2014-01-01', periods=10, freq='D'))
        dict_result = run_forecast(simplify_output=False, df_y=df1, l_model_trend=[forecast_models.model_linear])

        df_data = dict_result['data']
        df_metadata = dict_result['metadata']
        df_optimize_info = dict_result['optimize_info']

        logger_info('df_metadata:', df_metadata)
        logger_info('df_optimize_info:', df_optimize_info)
        logger_info('df_data:', df_data.groupby(['source', 'model']).tail(30))

        df2 = pd.DataFrame({'y': np.arange(0, 10.), 'source': ['src1'] * 5 + ['src2'] * 5},
                           index=pd.date_range('2014-01-01', periods=10, freq='D'))
        dict_result = run_forecast(simplify_output=False, df_y=df2, l_model_trend=[forecast_models.model_linear])

        df_data = dict_result['data']
        df_metadata = dict_result['metadata']
        df_optimize_info = dict_result['optimize_info']

        logger_info('df_metadata:', df_metadata)
        logger_info('df_optimize_info:', df_optimize_info)
        logger_info('df_data:', df_data.groupby(['source', 'model']).tail(60))

    def test_run_forecast_naive(self):
        # # Test 1 - linear series, 1 source
        # df1 = pd.DataFrame({'y': np.arange(0,10.)},
        #                     index=pd.date_range('2014-01-01', periods=10, freq='D'))
        # dict_result = run_forecast(simplify_output=False, df_y=df1, l_model_trend = [forecast_models.model_naive],
        #                            extrapolate_years=10./365)
        #
        # df_data = dict_result['data']
        # df_metadata = dict_result['metadata']
        # df_optimize_info = dict_result['optimize_info']
        #
        # logger_info('df_metadata:', df_metadata)
        # logger_info('df_optimize_info:', df_optimize_info)
        # logger_info('df_data:', df_data.groupby(['source', 'model']).tail(40))
        #
        # # Test 2 - 2 sources
        # df2 = pd.DataFrame({'y': np.arange(0,10.),'source' : ['src1']*5 + ['src2']*5},
        #                     index=pd.date_range('2014-01-01', periods=10, freq='D'))
        # dict_result = run_forecast(simplify_output=False, df_y=df2, l_model_trend = [forecast_models.model_naive],
        #                            extrapolate_years=10./365)
        #
        # df_data = dict_result['data']
        # df_metadata = dict_result['metadata']
        # df_optimize_info = dict_result['optimize_info']
        #
        # logger_info('df_metadata:', df_metadata)
        # logger_info('df_optimize_info:', df_optimize_info)
        # logger_info('df_data:', df_data.groupby(['source', 'model']).tail(60))
        #
        # # test 3: weight column
        # df1 = pd.DataFrame({'y': np.arange(0, 10.), 'weight': array_zeros_in_indices(10,[5,6])},
        #                    index=pd.date_range('2014-01-01', periods=10, freq='D'))
        # dict_result = run_forecast(simplify_output=False, df_y=df1, l_model_trend=[forecast_models.model_naive],
        #                            extrapolate_years=10. / 365)
        #
        # df_data = dict_result['data']
        # df_metadata = dict_result['metadata']
        # df_optimize_info = dict_result['optimize_info']
        #
        # logger_info('df_metadata:', df_metadata)
        # logger_info('df_optimize_info:', df_optimize_info)
        # logger_info('df_data:', df_data.groupby(['source', 'model']).tail(60))
        #
        # a_y_result = df_data.loc[df_data.model=='naive'].y.values
        # logger_info('a_y_result:', a_y_result)
        # self.assert_array_equal(a_y_result,
        #     np.concatenate([
        #     np.array([0., 0., 1., 2., 3., 4., 4.,4.,7.,8., 9.,]),
        #              np.full(9, 9.)
        #     ]
        # ))
        #
        # df_forecast = dict_result['forecast']
        # logger_info('df_forecast',df_forecast)
        #
        # # Test 3b: weight column, season_add_mult = 'both'
        #
        # df1 = pd.DataFrame({'y': np.arange(0, 10.), 'weight': array_zeros_in_indices(10, [5, 6])},
        #                    index=pd.date_range('2014-01-01', periods=10, freq='D'))
        # dict_result = run_forecast(simplify_output=False, df_y=df1, l_model_trend=[forecast_models.model_naive],
        #                            extrapolate_years=10. / 365,
        #                            season_add_mult='both')
        #
        # df_data = dict_result['data']
        # df_metadata = dict_result['metadata']
        # df_optimize_info = dict_result['optimize_info']
        #
        # logger_info('df_metadata:', df_metadata)
        # logger_info('df_optimize_info:', df_optimize_info)
        # logger_info('df_data:', df_data.groupby(['source', 'model']).tail(60))
        #
        # a_y_result = df_data.loc[df_data.model == 'naive'].y.values
        # logger_info('a_y_result:', a_y_result)
        # self.assert_array_equal(a_y_result,
        #                         np.concatenate([
        #                             np.array([0., 0., 1., 2., 3., 4., 4., 4., 7., 8., 9., ]),
        #                             np.full(9, 9.)
        #                         ]
        #                         ))
        #
        # df_forecast = dict_result['forecast']
        # logger_info('df_forecast', df_forecast)
        #
        # # Test 4: find_outliers
        #
        # df1 = pd.DataFrame({'y': np.arange(0, 10.)+10*array_ones_in_indices(10,[5,6])},
        #                    index=pd.date_range('2014-01-01', periods=10, freq='D'))
        # dict_result = run_forecast(simplify_output=False, df_y=df1, l_model_trend=[forecast_models.model_naive],
        #                            extrapolate_years=10. / 365, find_outliers=True)
        #
        # df_data = dict_result['data']
        # df_metadata = dict_result['metadata']
        # df_optimize_info = dict_result['optimize_info']
        #
        # logger_info('df_metadata:', df_metadata)
        # logger_info('df_optimize_info:', df_optimize_info)
        # logger_info('df_data:', df_data.groupby(['source', 'model']).tail(60))
        #
        # a_y_result = df_data.loc[df_data.model=='naive'].y.values
        # logger_info('a_y_result:', a_y_result)
        # self.assert_array_equal(a_y_result,
        #     np.concatenate([
        #     np.array([0., 0., 1., 2., 3., 4., 4.,4.,7.,8., 9.,]),
        #              np.full(9, 9.)
        #     ]
        # ))
        #
        # df_forecast = dict_result['forecast']
        # logger_info('df_forecast',df_forecast)
        #
        # # Test 4b: find_outliers, season_add_mult = 'both'
        #
        # df1 = pd.DataFrame({'y': np.arange(0, 10.)+10*array_ones_in_indices(10,[5,6])},
        #                    index=pd.date_range('2014-01-01', periods=10, freq='D'))
        # dict_result = run_forecast(simplify_output=False, df_y=df1, l_model_trend=[forecast_models.model_naive],
        #                            extrapolate_years=10. / 365, find_outliers=True, season_add_mult='both')
        #
        # df_data = dict_result['data']
        # df_metadata = dict_result['metadata']
        # df_optimize_info = dict_result['optimize_info']
        #
        # logger_info('df_metadata:', df_metadata)
        # logger_info('df_optimize_info:', df_optimize_info)
        # logger_info('df_data:', df_data.groupby(['source', 'model']).tail(60))
        #
        # a_y_result = df_data.loc[df_data.model=='naive'].y.values
        # logger_info('a_y_result:', a_y_result)
        # self.assert_array_equal(a_y_result,
        #     np.concatenate([
        #     np.array([0., 0., 1., 2., 3., 4., 4.,4.,7.,8., 9.,]),
        #              np.full(9, 9.)
        #     ]
        # ))
        #
        # df_forecast = dict_result['forecast']
        # logger_info('df_forecast',df_forecast)

        # Test 5: Series with gap

        # df1 = (
        #           pd.DataFrame({'y': np.arange(0, 10.),
        #                     #'weight': array_zeros_in_indices(10, [5, 6]),
        #                     'date': pd.date_range('2014-01-01', periods=10, freq='D')},
        #                    )
        #
        # )
        #
        # df1 = pd.concat([df1.head(5), df1.tail(3)], sort=False, ignore_index=False).pipe(normalize_df)
        #
        # dict_result = run_forecast(simplify_output=False, df_y=df1,
        #                            l_model_trend=[],
        #                            l_model_naive=[forecast_models.model_naive, forecast_models.model_snaive_wday],
        #                            extrapolate_years=10. / 365,
        #                            season_add_mult='both')
        #
        #
        # df_data = dict_result['data']
        # df_metadata = dict_result['metadata']
        # df_optimize_info = dict_result['optimize_info']
        #
        # logger_info('df_metadata:', df_metadata)
        # logger_info('df_optimize_info:', df_optimize_info)
        # logger_info('df_data:', df_data.groupby(['source', 'model']).tail(60))
        #
        # a_y_result = df_data.loc[df_data.model == 'naive'].y.values
        # logger_info('a_y_result:', a_y_result)
        # self.assert_array_equal(a_y_result,
        #                         np.concatenate([
        #                             np.array([0., 0., 1., 2., 3., 4., 4., 4., 7., 8., 9., ]),
        #                             np.full(9, 9.)
        #                         ]
        #                         ))
        #
        # df_forecast = dict_result['forecast']
        # logger_info('df_forecast', df_forecast)

        # Test 6: Series with spike, find_outliers=True, use model_snaive_wday

        df1 = (
                  pd.DataFrame({'y': np.arange(0, 21.) + 10*array_ones_in_indices(21, 7),
                            #'weight': array_zeros_in_indices(10, [5, 6]),
                            'date': pd.date_range('2014-01-01', periods=21, freq='D')},
                           )

        )

        #array_ones_in_indices(n, l_indices)

        dict_result = run_forecast(simplify_output=False, df_y=df1,
                                   l_model_trend=[],
                                   l_model_season=[],
                                   l_model_naive=[forecast_models.model_snaive_wday],
                                   extrapolate_years=20. / 365,
                                   season_add_mult='both', find_outliers=True)


        df_data = dict_result['data']
        df_metadata = dict_result['metadata']
        df_optimize_info = dict_result['optimize_info']

        logger_info('df_metadata:', df_metadata)
        logger_info('df_optimize_info:', df_optimize_info)
        df_data['wday']=df_data.date.dt.weekday
        logger_info('df_data:', df_data.groupby(['source', 'model']).tail(60))

        a_y_result = df_data.loc[df_data.model == 'snaive_wday'].y.values
        logger_info('a_y_result:', a_y_result)
        self.assert_array_equal(a_y_result,
                                np.array([0., 1., 2., 3., 4., 5., 6., 17., 8., 9., 10., 11., 12.,
                                 13., 14., 15., 16., 17., 18., 19., 20., 14., 15., 16., 17., 18.,
                                 19., 20., 14., 15., 16., 17., 18., 19.])
                                )

        df_forecast = dict_result['forecast']
        logger_info('df_forecast', df_forecast)

    def test_run_forecast_naive2(self):
        # Test 1: run forecast with  naive model, find_outliers, season_add_mult = 'add', weekly samples
        path_df_naive = os.path.join(base_folder, 'df_test_naive.csv')
        df_test_naive = pd.read_csv(path_df_naive)

        l_season_yearly = [
            forecast_models.model_season_month,
            # model_season_fourier_yearly,
            forecast_models.model_null]

        l_season_weekly = [  # forecast_models.model_season_wday_2,
            forecast_models.model_season_wday, forecast_models.model_null]

        dict_result = run_forecast(simplify_output=False, df_y=df_test_naive,
                                   #l_model_trend=[forecast_models.model_naive],
                                   l_model_naive=[forecast_models.model_naive],
                                    l_season_yearly=l_season_yearly,
                                    l_season_weekly=l_season_weekly,
                                   extrapolate_years=75. / 365, find_outliers=True, season_add_mult='add')

        df_data = dict_result['data']
        df_metadata = dict_result['metadata']
        df_optimize_info = dict_result['optimize_info']

        logger_info('df_metadata:', df_metadata)
        logger_info('df_optimize_info:', df_optimize_info)
        logger_info('df_data:', df_data.loc[(df_data.date>'2017-12-01') & (df_data.date<'2018-02-01')])

        a_y_result = df_data.loc[df_data.model == 'naive'].y.values
        #logger_info('a_y_result:', a_y_result)

        df_forecast = dict_result['forecast']
        logger_info('df_forecast',df_forecast.loc[(df_forecast.date>'2017-12-01')& (df_forecast.date<'2018-02-01')])

        # After first spike, naive forecast and actuals start matching, only if season_add_mult='both'
        self.assertNotEqual(df_data.loc[(df_data.date == '2018-01-07') & (df_data.model=='naive')].y.iloc[0],
                            df_data.loc[(df_data.date == '2018-01-07') & (df_data.model == 'actuals')].y.iloc[0])


        # Test 2: run forecast with  naive model, find_outliers, season_add_mult = 'both', weekly samples
        #path_df_naive = os.path.join(base_folder, 'df_test_naive.csv')
        #df_test_naive = pd.read_csv(path_df_naive)

        l_season_yearly = [
            forecast_models.model_season_month,
            # model_season_fourier_yearly,
            forecast_models.model_null]

        l_season_weekly = [  # forecast_models.model_season_wday_2,
            forecast_models.model_season_wday, forecast_models.model_null]

        dict_result = run_forecast(simplify_output=False, df_y=df_test_naive,
                                   #l_model_trend=[forecast_models.model_naive],
                                   l_model_naive=[forecast_models.model_naive],
                                    l_season_yearly=l_season_yearly,
                                    l_season_weekly=l_season_weekly,
                                   extrapolate_years=75. / 365, find_outliers=True, season_add_mult='both')

        df_data = dict_result['data']
        df_metadata = dict_result['metadata']
        df_optimize_info = dict_result['optimize_info']

        logger_info('df_metadata:', df_metadata)
        logger_info('df_optimize_info:', df_optimize_info)
        logger_info('df_data:', df_data.loc[(df_data.date>'2017-12-01') & (df_data.date<'2018-02-01')])

        a_y_result = df_data.loc[df_data.model == 'naive'].y.values
        #logger_info('a_y_result:', a_y_result)

        df_forecast = dict_result['forecast']
        logger_info('df_forecast',df_forecast.loc[(df_forecast.date>'2017-12-01')& (df_forecast.date<'2018-02-01')])

        # After first spike, naive forecast and actuals start matching, only if season_add_mult='both'
        self.assertNotEqual(df_data.loc[(df_data.date == '2018-01-07') & (df_data.model=='naive')].y.iloc[0],
                            df_data.loc[(df_data.date == '2018-01-07') & (df_data.model == 'actuals')].y.iloc[0])


        # Test 3 - multiple model_naive runs
        path_df_naive = os.path.join(base_folder, 'df_test_naive.csv')
        df_test_naive = pd.read_csv(path_df_naive)

        model_naive2 = forecast_models.ForecastModel('naive2', 0, forecast_models._f_model_naive)

        l_model_naive = [forecast_models.model_naive,model_naive2]

        dict_result = run_forecast(simplify_output=False, df_y=df_test_naive,
                                   l_model_trend=[],
                                    l_season_yearly=l_season_yearly,
                                    l_season_weekly=l_season_weekly,
                                   l_model_naive= l_model_naive,
                                   extrapolate_years=75. / 365, find_outliers=True, season_add_mult='add', )

        df_data = dict_result['data']
        df_metadata = dict_result['metadata']
        df_optimize_info = dict_result['optimize_info']

        logger_info('df_metadata:', df_metadata)
        logger_info('df_optimize_info:', df_optimize_info)
        logger_info('df_data:', df_data.loc[(df_data.date>'2017-12-01') & (df_data.date<'2018-02-01')])

        a_y_result = df_data.loc[df_data.model == 'naive'].y.values
        #logger_info('a_y_result:', a_y_result)

        df_forecast = dict_result['forecast']
        logger_info('df_forecast',df_forecast.loc[(df_forecast.date>'2017-12-01')& (df_forecast.date<'2018-02-01')])

        # After first spike, naive forecast and actuals start matching, only if season_add_mult='both'
        self.assertNotEqual(df_data.loc[(df_data.date == '2018-01-07') & (df_data.model=='naive')].y.iloc[0],
                            df_data.loc[(df_data.date == '2018-01-07') & (df_data.model == 'actuals')].y.iloc[0])

    def test_run_forecast_sparse_with_gaps(self):
        df_test = pd.DataFrame({'date': pd.to_datetime(['2018-08-01', '2018-08-09']), 'y': [1., 2.]})
        df_out = run_forecast(df_test, extrapolate_years=1.0)
        logger_info('df_out', df_out)
    def test_run_forecast_output_options(self):
        freq = 'D'
        freq_short = freq[0:1]  # Changes e.g. W-MON to W
        freq_units_per_year = 52.0 if freq_short == 'W' else 365.0  # Todo: change to dict to support more frequencies

        df_y = pd.DataFrame({'y': np.full(100, 0.0)},
                            index=pd.date_range('2014-01-01', periods=100, freq=freq))

        # SolverConfig with trend
        conf1 = ForecastInput(
            source_id='source1',
            l_model_trend=[forecast_models.model_linear, forecast_models.model_constant],
            l_model_season=None, df_y=df_y, date_start_actuals=None
        )

        logger.info('Testing run forecast - default settings')

        dict_result = run_l_forecast([conf1])

        df_data = dict_result['data']
        df_metadata = dict_result['metadata']
        df_optimize_info = dict_result['optimize_info']

        logger_info('df_metadata:', df_metadata)
        logger_info('df_optimize_info:', df_optimize_info)
        logger_info('df_data:', df_data.groupby(['source', 'model']).tail(1))

        for include_all_fits in [False, True]:
            logger.info('Testing run forecast - include_all_fits=%s',
                        include_all_fits)

            dict_result = run_l_forecast([conf1],
                                         include_all_fits=include_all_fits)

            df_data = dict_result['data']
            df_metadata = dict_result['metadata']
            df_optimize_info = dict_result['optimize_info']

            logger_info('df_metadata:', df_metadata)
            logger_info('df_optimize_info:', df_optimize_info)
            logger_info('df_data:', df_data.groupby(['source', 'model']).tail(1))
            # TODO: ADD ASSERTS

    def test_run_forecast_step(self):
        # Setup
        freq = 'D'
        df_y1 = pd.DataFrame({'y': 5 * [10.0] + 5 * [20.0]},
                             index=pd.date_range('2014-01-01', periods=10, freq=freq))

        # SolverConfig with trend
        conf1 = ForecastInput(
            source_id='source1',
            l_model_trend=[forecast_models.model_constant,
                           forecast_models.model_constant + forecast_models.model_step],
            l_model_season=None, df_y=df_y1, weights_y_values=1.0, date_start_actuals=None
        )

        dict_result = run_l_forecast([conf1], include_all_fits=True)
        df_data = dict_result['data']
        df_metadata = dict_result['metadata']
        df_optimize_info = dict_result['optimize_info']

        logger_info('df_metadata:', df_metadata)
        logger_info('df_optimize_info:', df_optimize_info)
        logger_info('df_data:', df_data.groupby(['source', 'model']).tail(1))

        # Test 2 : 2 steps

        # Setup
        freq = 'D'
        df_y1 = pd.DataFrame({'y': [1., 1., 1., 1., 1., 1., 5., 5., 6., 6.]},
                             index=pd.date_range('2014-01-01', periods=10, freq=freq))

        # SolverConfig with trend
        conf1 = ForecastInput(
            source_id='source1',
            l_model_trend=[forecast_models.model_constant + forecast_models.model_two_steps],
            l_model_season=None, df_y=df_y1, weights_y_values=1.0, date_start_actuals=None
        )

        dict_result = run_l_forecast([conf1], include_all_fits=True)
        df_data = dict_result['data']
        df_metadata = dict_result['metadata']
        df_optimize_info = dict_result['optimize_info']

        logger_info('df_metadata:', df_metadata)
        logger_info('df_optimize_info:', df_optimize_info)
        logger_info('df_data:', df_data.groupby(['source', 'model']).tail(1))

    def test_run_forecast_sigmoid_step(self):
        # Setup
        freq = 'D'
        df_y1 = pd.DataFrame({'y': [10., 10.1, 10.2, 10.3, 10.4, 20.0, 20.1, 20.2, 20.3, 20.4, 20.5, 20.6]},
                             index=pd.date_range('2014-01-01', periods=12, freq=freq))

        # SolverConfig with trend
        conf1 = ForecastInput(
            source_id='source1',
            l_model_trend=[forecast_models.model_constant,
                           forecast_models.model_sigmoid_step,
                           forecast_models.model_constant + forecast_models.model_sigmoid_step,
                           forecast_models.model_linear + forecast_models.model_sigmoid_step,
                           forecast_models.model_linear * forecast_models.model_sigmoid_step],
            l_model_season=None, df_y=df_y1, weights_y_values=1.0, date_start_actuals=None
        )

        dict_result = run_l_forecast([conf1], include_all_fits=True)
        df_data = dict_result['data']
        df_metadata = dict_result['metadata']
        df_optimize_info = dict_result['optimize_info']

        logger_info('df_metadata:', df_metadata)
        logger_info('df_optimize_info:', df_optimize_info)
        logger_info('df_data:', df_data.groupby(['source', 'model']).tail(1))

        # Same with negative step
        df_y1 = pd.DataFrame({'y': [20.0, 20.1, 20.2, 20.3, 20.4, 20.5, 20.6, 10., 10.1, 10.2, 10.3, 10.4]},
                             index=pd.date_range('2014-01-01', periods=12, freq=freq))

        conf1 = ForecastInput(
            source_id='source1',
            l_model_trend=[forecast_models.model_constant,
                           forecast_models.model_sigmoid_step,
                           forecast_models.model_constant + forecast_models.model_sigmoid_step,
                           forecast_models.model_linear + forecast_models.model_sigmoid_step,
                           forecast_models.model_linear * forecast_models.model_sigmoid_step],
            l_model_season=None, df_y=df_y1, weights_y_values=1.0, date_start_actuals=None
        )

        dict_result = run_l_forecast([conf1], include_all_fits=True)
        df_data = dict_result['data']
        df_metadata = dict_result['metadata']
        df_optimize_info = dict_result['optimize_info']

        logger_info('df_metadata:', df_metadata)
        logger_info('df_optimize_info:', df_optimize_info)
        logger_info('df_data:', df_data.groupby(['source', 'model']).tail(1))

    def test_run_forecast_fourier_yearly(self):
        # Yearly sinusoidal function

        # With daily samples
        length = 2 * 365
        # size will be +-10 +- uniform error
        a_date = pd.date_range(start='2018-01-01', freq='D', periods=length)
        a_y = (10 + np.random.uniform(low=0, high=1, size=length) +
               10 * (np.sin(np.linspace(-4 * np.pi, 4 * np.pi, length))))
        df_y = pd.DataFrame({'y': a_y}, index=a_date)

        conf = ForecastInput(
            source_id='source',
            l_model_trend=[
                forecast_models.model_constant,
                forecast_models.model_season_fourier_yearly,
                forecast_models.model_constant +
                forecast_models.model_season_fourier_yearly],
            l_model_season=[forecast_models.model_null], df_y=df_y, weights_y_values=1.0, date_start_actuals=None
        )
        dict_result = run_l_forecast([conf], include_all_fits=True)
        df_data = dict_result['data']
        df_metadata = dict_result['metadata']
        df_optimize_info = dict_result['optimize_info']

        logger_info('df_metadata:', df_metadata)
        logger_info('df_optimize_info:', df_optimize_info)
        logger_info('df_data:', df_data.groupby(['source', 'model']).tail(1))
        df = df_data.loc[(df_data.model == 'a') | df_data.is_best_fit,
                         ['y', 'date', 'model']]

        df = df.pivot(values='y', columns='model', index='date')
        if platform.system() != 'Darwin':  # matplotlib tests don't work on mac # matplotlib tests don't work on mac
            df.plot()

        length = 1 * 365
        # size will be +-10 +- uniform error
        a_date = pd.date_range(start='2018-01-01', freq='D', periods=length)
        a_y = (10 + np.random.uniform(low=0, high=1, size=length) +
               10 * (np.sin(np.linspace(-4 * np.pi, 4 * np.pi, length))) +
               5 * (np.cos(np.linspace(-6 * np.pi, 6 * np.pi, length))))
        df_y = pd.DataFrame({'y': a_y}, index=a_date)

        conf = ForecastInput(
            source_id='source',
            l_model_trend=[
                forecast_models.model_constant,
                forecast_models.model_season_fourier_yearly,
                forecast_models.model_constant +
                forecast_models.model_season_fourier_yearly],
            l_model_season=[forecast_models.model_null], df_y=df_y, weights_y_values=1.0, date_start_actuals=None
        )
        dict_result = run_l_forecast([conf], include_all_fits=True)
        df_data = dict_result['data']
        df_metadata = dict_result['metadata']
        df_optimize_info = dict_result['optimize_info']

        logger_info('df_metadata:', df_metadata)
        logger_info('df_optimize_info:', df_optimize_info)
        logger_info('df_data:', df_data.groupby(['source', 'model']).tail(1))
        df = df_data.loc[(df_data.model == 'a') | df_data.is_best_fit,
                         ['y', 'date', 'model']]

        df = df.pivot(values='y', columns='model', index='date')
        if platform.system() != 'Darwin':  # matplotlib tests don't work on mac # matplotlib tests don't work on mac
            df.plot()
        # TODO find a better assertion test
        pass

    def test_run_forecast_sigmoid(self):
        # Input parameters
        b_in = 100.
        c_in = 40.
        d_in = 1.
        # linear params
        a_lin = 0.01
        b_lin = 0.05

        is_mult_l = [False, True]

        def sigmoid(x, a, b, c, d):
            y = a + (b - a) / (1 + np.exp(- d * (x - c)))
            return y

        a_x = np.arange(1, 100)
        # linear to find

        for is_mult in is_mult_l:

            if is_mult:
                a_in = 1
                model = forecast_models.model_linear * forecast_models.model_sigmoid
                y_lin = a_lin * a_x + b_lin
                y_in = sigmoid(a_x, a_in, b_in, c_in, d_in) * y_lin
                input_params = [a_lin, b_lin]
                y_rand = np.random.uniform(low=0.001, high=0.1 * b_in, size=len(a_x)) * y_lin
            else:
                a_in = 30  # the constant
                model = forecast_models.model_constant + forecast_models.model_sigmoid
                y_in = sigmoid(a_x, a_in, b_in, c_in, d_in)
                input_params = [a_in]
                y_rand = np.random.uniform(low=0.001, high=0.1 * b_in, size=len(a_x))

            input_params = input_params + [b_in - a_in, c_in, d_in]

            y_in = y_rand + y_in
            df_y = pd.DataFrame({'y': y_in}, index=a_x)
            # SolverConfig with trend
            conf1 = ForecastInput(
                source_id='source1',
                l_model_trend=[
                    forecast_models.model_constant,
                    # forecast_models.model_sigmoid,
                    model,
                    # forecast_models.model_linear + forecast_models.model_sigmoid,
                    # forecast_models.model_linear * forecast_models.model_sigmoid
                ],
                l_model_season=None, df_y=df_y, weights_y_values=1.0, date_start_actuals=None
            )

            dict_result = run_l_forecast([conf1],
                                         include_all_fits=True)
            df_data = dict_result['data']
            df_metadata = dict_result['metadata']
            # df_optimize_info = dict_result['optimize_info']

            df = df_data.loc[:, ['y', 'date', 'model']]

            df = df.pivot(values='y', columns='model', index='date')
            if platform.system() != 'Darwin':  # matplotlib tests don't work on mac  # matplotlib tests don't work on mac
                df.plot()
            output_params = df_metadata.loc[df_metadata.is_best_fit, 'params_str']
            logger.info('Input parameters: %s, Output parameters: %s',
                        input_params, output_params.iloc[0])
            pass  # to see the plot

    def test_auto_find_sigmoid_step(self):
        # Setup

        # First do it manually
        freq = 'D'
        a_y = [19.8, 19.9, 20.0, 20.1, 20.2, 20.3, 20.4, 20.5,
               20.6, 10., 10.1, 10.2, 10.3, 10.4,
               10.5, 10.6, 10.7, 10.8, 10.9]
        a_date = pd.date_range(start='2018-01-01', periods=len(a_y), freq='D')
        df_y = pd.DataFrame({'y': a_y}, index=a_date)
        a_x = np.arange(0, len(a_y))

        steps, spikes = forecast_models.find_steps_and_spikes(a_x, a_y, a_date)
        assert len(steps) == 1
        assert len(spikes) == 0
        step_model = steps[0]
        trend_models = [forecast_models.model_linear + step_model,
                        forecast_models.model_linear + forecast_models.model_sigmoid_step,
                        forecast_models.model_linear]

        # SolverConfig with trend
        conf1 = ForecastInput(
            source_id='source1',
            l_model_trend=trend_models,
            l_model_season=None,
            df_y=df_y,
            weights_y_values=1.0,
            date_start_actuals=None
        )

        dict_result = run_l_forecast([conf1],
                                     include_all_fits=True)
        df_data = dict_result['data']
        df_metadata = dict_result['metadata']
        df_optimize_info = dict_result['optimize_info']

        logger_info('df_metadata:', df_metadata)
        logger_info('df_optimize_info:', df_optimize_info)
        logger_info('df_data:', df_data.groupby(['source', 'model']).tail(1))

        # Then do it automatically
        trend_models = [forecast_models.model_linear]

        # SolverConfig with trend
        conf2 = ForecastInput(
            source_id='source1',
            l_model_trend=trend_models,
            l_model_season=None,
            df_y=df_y,
            weights_y_values=1.0,
            date_start_actuals=None
        )

        dict_result = run_l_forecast([conf2],
                                     include_all_fits=True, do_find_steps_and_spikes=True)
        df_data = dict_result['data']
        df_metadata = dict_result['metadata']
        df_optimize_info = dict_result['optimize_info']

        logger_info('df_metadata:', df_metadata)
        logger_info('df_optimize_info:', df_optimize_info)
        logger_info('df_data:', df_data.groupby(['source', 'model']).tail(1))

        # Two changes
        a_y = np.concatenate((np.arange(-1, 31), [50], np.arange(51, 70), [0], np.arange(1, 30)),
                             axis=0)
        a_date = pd.date_range(start='2018-01-01', periods=len(a_y), freq='D')
        df_y = pd.DataFrame({'y': a_y}, index=a_date)
        trend_models = [forecast_models.model_linear]

        # SolverConfig with trend
        conf3 = ForecastInput(
            source_id='source1',
            l_model_trend=trend_models,
            l_model_season=None,
            df_y=df_y,
            weights_y_values=1.0,
            date_start_actuals=None
        )

        dict_result = run_l_forecast([conf3],
                                     include_all_fits=True, do_find_steps_and_spikes=True)
        df_data = dict_result['data']
        df_metadata = dict_result['metadata']
        df_optimize_info = dict_result['optimize_info']
        logger_info('df_metadata:', df_metadata)
        logger_info('df_optimize_info:', df_optimize_info)
        logger_info('df_data:', df_data.groupby(['source', 'model']).tail(1))

    def test_auto_find_sigmoid_spike(self):
        # Setup

        # First do it manually
        freq = 'D'
        a_y = np.concatenate((np.arange(-1, 30), [50, 51], np.arange(31, 50)),
                             axis=0)
        a_date = pd.date_range(start='2018-01-01', periods=len(a_y), freq='D')
        df_y = pd.DataFrame({'y': a_y}, index=a_date).pipe(normalize_df)
        a_x = np.arange(0, len(a_y))

        steps, spikes = forecast_models.find_steps_and_spikes(a_x, a_y, a_date)
        assert len(steps) == 0
        assert len(spikes) == 1
        spike_model = spikes[0]
        trend_models = [forecast_models.model_linear * spike_model,
                        forecast_models.model_linear]

        # SolverConfig with trend
        conf1 = ForecastInput(
            source_id='source1',
            l_model_trend=trend_models,
            l_model_season=None,
            df_y=df_y,
            weights_y_values=1.0,
            date_start_actuals=None
        )

        dict_result = run_l_forecast([conf1],
                                     include_all_fits=True, do_find_steps_and_spikes=False)

        df_data = dict_result['data']
        df_metadata = dict_result['metadata']
        df_optimize_info = dict_result['optimize_info']
        logger_info('df_metadata:', df_metadata)
        logger_info('df_optimize_info:', df_optimize_info)
        logger_info('df_data:', df_data.groupby(['source', 'model']).tail(1))

        # Same automatically
        trend_models = [forecast_models.model_linear]
        # SolverConfig with trend
        conf3 = ForecastInput(
            source_id='source1',
            l_model_trend=trend_models,
            l_model_season=None,
            df_y=df_y,
            weights_y_values=1.0,
            date_start_actuals=None
        )

        dict_result = run_l_forecast([conf3],
                                     include_all_fits=True,
                                     do_find_steps_and_spikes=True)
        df_data = dict_result['data']
        df_metadata = dict_result['metadata']
        df_optimize_info = dict_result['optimize_info']
        logger_info('df_metadata:', df_metadata)
        logger_info('df_optimize_info:', df_optimize_info)
        logger_info('df_data:', df_data.groupby(['source', 'model']).tail(1))

    def test_run_forecast_get_outliers(self):

        # Test 1 - no outliers
        a_y = [20.0, 20.1, 20.2, 20.3, 20.4, 20.5]
        a_date = pd.date_range(start='2018-01-01', periods=len(a_y), freq='D')
        df = pd.DataFrame({'y': a_y})

        dict_result = run_forecast(df, find_outliers=True, simplify_output=False, include_all_fits=True,
                                   season_add_mult='add')
        logger_info('Metadata', dict_result['metadata'])
        logger_info('data', dict_result['data'].tail(3))

        # Check that dtype of y is not corrupted by None values from weight mask - this happens when no spikes found
        self.assertTrue(np.issubdtype(dict_result['data'].y, np.float64))

        # Test 2 - Single step
        a_y = [19.8, 19.9, 20.0, 20.1, 20.2, 20.3, 20.4, 20.5,
               20.6, 10., 10.1, 10.2, 10.3, 10.4,
               10.5, 10.6, 10.7, 10.8, 10.9]
        a_date = pd.date_range(start='2018-01-01', periods=len(a_y), freq='D')
        df = pd.DataFrame({'y': a_y})

        dict_result = run_forecast(df, find_outliers=True, simplify_output=False, include_all_fits=True,
                                   season_add_mult='add')
        logger_info('Metadata', dict_result['metadata'])
        logger_info('data', dict_result['data'].tail(3))

        # Check that dtype of y is not corrupted by None values from weight mask - this happens when no spikes found
        self.assertTrue(np.issubdtype(dict_result['data'].y, np.float64))

        # Test 3 - Single spike

        a_y = [19.8, 19.9, 20.0, 20.1, 20.2, 20.3, 20.4, 20.5,
               20.6, 10., 20.7, 20.8, 20.9, 21.0,
               21.1, 21.2, 21.3, 21.4, 21.5]
        a_date = pd.date_range(start='2018-01-01', periods=len(a_y), freq='D')
        df_spike = pd.DataFrame({'y': a_y})

        dict_result = run_forecast(df_spike, find_outliers=True,
                                   simplify_output=False, include_all_fits=True,
                                   season_add_mult='add')
        df_data = dict_result['data']
        mask = df_data.loc[df_data.model == 'actuals'].weight
        self.assert_array_equal(mask, [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, ])

        # Test 5 - 2 spikes and 1 step
        a_y = [19.8, 19.9, 30.0, 30.1, 20.2, 20.3, 20.4, 20.5,
               20.6, 10., 10.1, 10.2, 10.3, 10.4,
               10.5, 10.6, 30.7, 10.8, 10.9]

        df = pd.DataFrame({'y': a_y}).pipe(normalize_df)

        dict_result = run_forecast(df, find_outliers=True, simplify_output=False, include_all_fits=True,
                                   season_add_mult='add')
        logger_info('Metadata', dict_result['metadata'])
        df_result = dict_result['data']
        logger_info('data', df_result.tail(3))
        mask = df_result.loc[df_result.model=='actuals'].weight
        self.assert_array_equal(mask, [1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1])

    def test_run_forecast_auto_season(self):
        # Yearly sinusoidal function

        # With daily samples
        length = 2 * 365
        # size will be +-10 +- uniform error
        a_date = pd.date_range(start='2018-01-01', freq='D', periods=length)
        a_y = (10 + np.random.uniform(low=0, high=1, size=length) +
               10 * (np.sin(np.linspace(-4 * np.pi, 4 * np.pi, length))))
        df_y = pd.DataFrame({'y': a_y}, index=a_date)

        dict_result = run_forecast(df_y, season_add_mult='add', simplify_output=False, include_all_fits=True,
                                   l_model_trend=[forecast_models.model_linear])
        df_metadata = dict_result['metadata']

        l_model_expected = ['linear', '(linear+(season_wday+season_fourier_yearly))',
                            '(linear+season_wday)', '(linear+season_fourier_yearly)']

        self.assert_array_equal(df_metadata.model, l_model_expected)
        logger_info('df_metadata:', df_metadata)

        # As above, with additive and multiplicative seasonality

        dict_result = run_forecast(df_y, season_add_mult='both', simplify_output=False, include_all_fits=True,
                                   l_model_trend=[forecast_models.model_linear])
        df_metadata = dict_result['metadata']

        l_model_expected = [
            '(linear*(season_wday*season_fourier_yearly))',
            '(linear*season_fourier_yearly)',
            '(linear*season_wday)',
            '(linear+(season_wday+season_fourier_yearly))',
            '(linear+season_fourier_yearly)',
            '(linear+season_wday)',
            'linear' ]

        self.assert_array_equal(df_metadata.model.values, l_model_expected)
        logger_info('df_metadata:', df_metadata)

    def test_run_forecast_with_weight(self):
        df1 = pd.DataFrame({'y': np.arange(0, 10.),
                            'date': pd.date_range('2014-01-01', periods=10, freq='D'),
                            'weight': 1.})
        dict_result = run_forecast(simplify_output=False, df_y=df1, l_model_trend = [forecast_models.model_linear],
                                   extrapolate_years=10./365)

        df_forecast = dict_result['forecast']
        df_data = dict_result['data']
        df_metadata = dict_result['metadata']
        df_optimize_info = dict_result['optimize_info']

        logger_info('df_forecast:', df_forecast.groupby(['source', 'model']).tail(30))
        logger_info('df_metadata:', df_metadata)
        logger_info('df_optimize_info:', df_optimize_info)
        logger_info('df_data:', df_data.groupby(['source', 'model']).tail(30))

        df_forecast_filtered = df_forecast.loc[~df_forecast.is_actuals & (df_forecast.date>'2014-01-10')]
        self.assert_series_equal(df_forecast_filtered.y, df_forecast_filtered.q5)

        df1b = df1.copy()
        df1b.loc[0,'weight']=0.



        dict_result = run_forecast(simplify_output=False, df_y=df1b, l_model_trend = [forecast_models.model_linear],
                                   extrapolate_years=10./365)

        df_forecast = dict_result['forecast']
        df_data = dict_result['data']
        df_metadata = dict_result['metadata']
        df_optimize_info = dict_result['optimize_info']

        logger_info('df_forecast:', df_forecast.groupby(['source', 'model']).tail(30))
        logger_info('df_metadata:', df_metadata)
        logger_info('df_optimize_info:', df_optimize_info)
        logger_info('df_data:', df_data.groupby(['source', 'model']).tail(30))

        len_forecast=df_data.loc[~df_data.is_actuals].index.size
        self.assertEquals(len_forecast,19) # First sample shouldn't be included due to weight=0

        # Since fit is perfect, prediction interval should be equal to point forecast
        df_forecast_filtered = df_forecast.loc[~df_forecast.is_actuals & (df_forecast.date>'2014-01-10')]
        self.assert_series_equal(df_forecast_filtered.y, df_forecast_filtered.q5)

        # Test with model_ramp
        # Param A of model_ramp needs to be within the 15-85 percentile of valid x values
        # Before a bugfix, we would get initial guesses of A=2, with boundaries (5.6, 8.4)
        # Note: somehow validate bounds doesn't catch this!

        df1c = df1.copy()
        df1c.loc[0:4, 'weight'] = 0.

        dict_result = run_forecast(simplify_output=False, df_y=df1c, l_model_trend=[forecast_models.model_ramp],
                                   extrapolate_years=10. / 365)

        df_forecast = dict_result['forecast']
        df_data = dict_result['data']
        df_metadata = dict_result['metadata']
        df_optimize_info = dict_result['optimize_info']

        logger_info('df_forecast:', df_forecast.groupby(['source', 'model']).tail(30))
        logger_info('df_metadata:', df_metadata)
        logger_info('df_optimize_info:', df_optimize_info)
        logger_info('df_data:', df_data.groupby(['source', 'model']).tail(30))

        len_forecast=df_data.loc[~df_data.is_actuals].index.size
        self.assertEquals(len_forecast,15) # First 5 samples shouldn't be included due to weight=0

        # # Since fit is perfect, prediction interval should be equal to point forecast
        # df_forecast_filtered = df_forecast.loc[~df_forecast.is_actuals & (df_forecast.date>'2014-01-10')]
        # self.assert_series_equal(df_forecast_filtered.y, df_forecast_filtered.q5)


    def test_detect_freq(self):

        # Initial test - what happens with single sample input?
        a_date = pd.a_date = pd.date_range('2014-01-01', periods=1, freq='H')
        result = detect_freq(a_date)
        #self.assertEquals(result, 'H')

        a_date = pd.a_date = pd.date_range('2014-01-01', periods=24*7, freq='H')
        result = detect_freq(a_date)
        self.assertEquals(result, 'H')

        a_date = pd.a_date = pd.date_range('2014-01-01', periods=4 * 365, freq='D')
        result = detect_freq(a_date)
        self.assertEquals(result, 'D')

        l_freq_wday = ['W-MON', 'W-TUE', 'W-WED', 'W-THU', 'W-FRI', 'W-SAT', 'W-SUN']
        for freq_wday in l_freq_wday:
            a_date = pd.a_date = pd.date_range('2014-01-01', periods=4 * 52, freq=freq_wday)
            result = detect_freq(a_date)
            self.assertEquals(result, freq_wday)

        a_date = pd.a_date = pd.date_range('2014-01-01', periods=4 * 12, freq='M')
        result = detect_freq(a_date)
        self.assertEquals(result, 'M')

        a_date = pd.a_date = pd.date_range('2014-01-01', periods=4 * 12, freq='MS')
        result = detect_freq(a_date)
        self.assertEquals(result, 'MS')

        a_date = pd.a_date = pd.date_range('2014-01-01', periods=4 * 12, freq='Q')
        result = detect_freq(a_date)
        self.assertEquals(result, 'Q')

        a_date = pd.a_date = pd.date_range('2014-01-01', periods=4 * 12, freq='Y')
        result = detect_freq(a_date)
        self.assertEquals(result, 'Y')

        # Test with input dataframe

        a_date = pd.a_date = pd.date_range('2014-01-01', periods=24 * 7, freq='H')
        df_y = pd.DataFrame({'date': a_date})
        result = detect_freq(df_y)
        self.assertEquals(result, 'H')

        a_date = pd.a_date = pd.date_range('2014-01-01', periods=4 * 365, freq='D')
        df_y = pd.DataFrame({'date': a_date})
        result = detect_freq(df_y)
        self.assertEquals(result, 'D')

        a_date = pd.a_date = pd.date_range('2014-01-01', periods=4 * 12, freq='M')
        df_y = pd.DataFrame({'date': a_date})
        result = detect_freq(df_y)
        self.assertEquals(result, 'M')

        a_date = pd.a_date = pd.date_range('2014-01-01', periods=4 * 12, freq='Q')
        df_y = pd.DataFrame({'date': a_date})
        result = detect_freq(df_y)
        self.assertEquals(result, 'Q')

        a_date = pd.a_date = pd.date_range('2014-01-01', periods=4 * 12, freq='Y')
        df_y = pd.DataFrame({'date': a_date})
        result = detect_freq(df_y)
        self.assertEquals(result, 'Y')

        a_date = pd.a_date = pd.date_range('2014-01-01', periods=4 * 12, freq='YS')
        df_y = pd.DataFrame({'date': a_date})
        result = detect_freq(df_y)
        self.assertEquals(result, 'YS')

        # Test with sparse input series
        a_date = pd.to_datetime(['2018-08-01', '2018-08-09'])
        df_y = pd.DataFrame({'date':a_date})
        result = detect_freq(df_y)
        self.assertEquals(result, 'D')

    # TODO: ADD TEST WITH NULL VALUES, E.G. MODEL_NAIVE_WDAY
    def test_get_pi(self):

        def check_result(df_result):
            self.assertTrue('q5' in df_result.columns)
            df_result_actuals = df_result.loc[df_result.is_actuals]
            if 'is_weight' in df_result_actuals.columns:
                df_result_actuals = df_result_actuals.loc[~df_result_actuals.is_weight]
            date_max_actuals = df_result_actuals.date.max()
            logger_info('debug: date max actuals', date_max_actuals)

            df_result_forecast = df_result.loc[~df_result.is_actuals & (df_result.date > date_max_actuals)]
            self.assertFalse(df_result_forecast.q5.isnull().any())

        # First test with single source
        # then test applied function on df grouped by source

        a_date_actuals = pd.date_range('2014-01-01', periods=10, freq='W')
        a_y_actuals = np.arange(0, 10.)
        df_actuals = (
            pd.DataFrame({'date': a_date_actuals, 'y': a_y_actuals,
                          'source': 's1', 'is_actuals': True, 'is_best_fit': False, 'model': 'actuals'})
        )

        a_date = pd.date_range('2014-01-01', periods=20, freq='W')
        a_y = np.arange(0, 20.) + (np.tile([-1, 1], (10)) * np.arange(2, 0., -0.1))

        df_fcast = (
            pd.DataFrame({'date': a_date, 'y': a_y,
                          'source': 's1', 'is_actuals': False, 'is_best_fit': True, 'model': 'linear'})
        )

        df1 = pd.concat([df_actuals, df_fcast], ignore_index=True, sort=False)

        df_result = get_pi(df1, n=100)
        df_result0 = df_result
        # logger_info('df_result1:', df_result1)
        logger_info('df_result1:', df_result.groupby(['source', 'model']).head(1))
        logger_info('df_result1:', df_result.groupby(['source', 'model']).tail(1))
        # TODO: Add checks
        check_result(df_result)

        # Test 1b - input dataframe without is_best_fit column, source column
        df1c = df1[['date', 'is_actuals', 'model', 'y']]
        df_result = get_pi(df1c, n=100)
        # logger_info('df_result1:', df_result1)
        logger_info('df_result1:', df_result.groupby(['model']).head(1))
        logger_info('df_result1:', df_result.groupby(['model']).tail(1))

        check_result(df_result)

        # Test 2 - 2 sources

        df1b = df1.copy()
        df1b.source = 's2'
        df2 = pd.concat([df1, df1b], sort=False)

        # df_result2 = df2.groupby('source').apply(get_pi, n=100).reset_index(drop=True)
        df_result = get_pi(df2, n=100)
        # logger_info('df_result2:', df_result2)
        logger_info('df_result2:', df_result.groupby(['source', 'model']).head(1))
        logger_info('df_result2:', df_result.groupby(['source', 'model']).tail(1))
        # TODO: Add checks

        check_result(df_result)

        # Test 3 - Input has actuals but no forecast - can happen if fit not possible

        df3 = df_actuals
        df_result = get_pi(df3, n=100)
        self.assertIsNotNone(df3)
        self.assertFalse('q5' in df_result.columns)
        # logger_info('df_result1:', df_result1)
        logger_info('df_result3:', df_result.groupby(['source', 'model']).head(1))
        logger_info('df_result3:', df_result.groupby(['source', 'model']).tail(1))
        #
        # Test 4 - Input has null values at the end
        a_date_actuals = pd.date_range('2014-01-01', periods=10, freq='W')
        a_y_actuals = np.arange(0, 10.)
        df_actuals = (
            pd.DataFrame({'date': a_date_actuals, 'y': a_y_actuals,
                          'source': 's1', 'is_actuals': True, 'is_best_fit': False, 'model': 'actuals'})
        )

        a_date = pd.date_range('2014-01-01', periods=20, freq='W')
        a_y = np.arange(0, 20.) + (np.tile([-1, 1], (10)) * np.arange(2, 0., -0.1))

        df_fcast = (
            pd.DataFrame({'date': a_date, 'y': a_y,
                          'source': 's1', 'is_actuals': False, 'is_best_fit': True, 'model': 'linear'})
        )

        df1 = pd.concat([df_actuals, df_fcast], ignore_index=True, sort=False)
        df_result = get_pi(df1, n=100)

        a_date_actuals_withnull = pd.date_range('2014-01-01', periods=20, freq='W')
        a_y_actuals_withnull = np.concatenate([np.arange(0, 10.), np.full(10, np.NaN)])
        df_actuals_withnull = (
            pd.DataFrame({'date': a_date_actuals, 'y': a_y_actuals,
                          'source': 's1', 'is_actuals': True, 'is_best_fit': False, 'model': 'actuals'})
        )

        a_date_withnull = pd.date_range('2014-01-01', periods=20, freq='W')

        df1_withnull = pd.concat([df_actuals_withnull, df_fcast], ignore_index=True, sort=False)
        df_result_withnull = get_pi(df1_withnull, n=100)

        logger_info('df_result:', df_result.groupby(['source', 'model']).tail(3))
        logger_info('df_result with null:', df_result_withnull.groupby(['source', 'model']).tail(3))
        # Prediction intervals are random, so we need to exclude them from comparison
        self.assert_frame_equal(df_result[['date', 'source', 'is_actuals', 'model', 'y']],
                                df_result_withnull[['date', 'source', 'is_actuals', 'model', 'y']])

        # Test 4b - Input with null values at the end, weight column
        df_weight = (
            pd.DataFrame({'date': a_date, 'y': 1,
                          'source': 's1', 'is_actuals': False, 'is_best_fit': True, 'model': 'linear',
                          'is_weight': True})
        )
        df_weight_withnull = (
            pd.DataFrame({'date': a_date_withnull, 'y': 1,
                          'source': 's1', 'is_actuals': False, 'is_best_fit': True, 'model': 'linear',
                          'is_weight': True})
        )

        df1['is_weight'] = False
        df1_withnull['is_weight'] = False

        df1b = pd.concat([df1, df_weight], ignore_index=True, sort=False)
        df1b_withnull = pd.concat([df1_withnull, df_weight_withnull], ignore_index=True, sort=False)

        df_result_b = get_pi(df1b, n=100)
        df_result_b_withnull = get_pi(df1b_withnull, n=100)

        logger_info('df_result b :', df_result_b.groupby(['source', 'model']).tail(3))
        logger_info('df_result b with null:', df_result_b_withnull.groupby(['source', 'model']).tail(3))
        # Prediction intervals are random, so we need to exclude them from comparison
        self.assert_frame_equal(df_result_b[['date', 'source', 'is_actuals', 'model', 'y']],
                                df_result_b_withnull[['date', 'source', 'is_actuals', 'model', 'y']])

        check_result(df_result_b)
        check_result(df_result_b_withnull)

        # Test 4C - Input has null values at the start of actuals series
        a_date_actuals = pd.date_range('2014-01-01', periods=10, freq='W')
        a_y_actuals = np.arange(0, 10.)
        df_actuals = (
            pd.DataFrame({'date': a_date_actuals, 'y': a_y_actuals,
                          'source': 's1', 'is_actuals': True, 'is_best_fit': False, 'model': 'actuals'})
        )

        a_date = pd.date_range('2014-01-01', periods=20, freq='W')
        a_y = np.arange(0, 20.) + (np.tile([-1, 1], (10)) * np.arange(2, 0., -0.1))

        df_fcast = (
            pd.DataFrame({'date': a_date, 'y': a_y,
                          'source': 's1', 'is_actuals': False, 'is_best_fit': True, 'model': 'linear'})
        )

        df1 = pd.concat([df_actuals, df_fcast], ignore_index=True, sort=False)
        df_result = get_pi(df1, n=100)

        a_date_actuals_withnull = pd.date_range('2014-01-01', periods=10, freq='W')
        a_y_actuals_withnull = np.concatenate([np.full(5, np.NaN),np.arange(0, 5.)])
        df_actuals_withnull = (
            pd.DataFrame({'date': a_date_actuals_withnull, 'y': a_y_actuals_withnull,
                          'source': 's1', 'is_actuals': True, 'is_best_fit': False, 'model': 'actuals'})
        )

        a_date_withnull = pd.date_range('2014-01-01', periods=10, freq='W')

        df1_withnull = pd.concat([df_actuals_withnull, df_fcast], ignore_index=True, sort=False)
        df_result_withnull = get_pi(df1_withnull, n=100)

        logger_info('df_actuals_withnull:', df_actuals_withnull.groupby(['source', 'model']).head(20))
        logger_info('df_result:', df_result.groupby(['source', 'model']).tail(3))
        logger_info('df_result with null:', df_result_withnull.groupby(['source', 'model']).tail(100))
        # todo - add proper expected value, uncomment assert
        # self.assert_frame_equal(df_result[['date', 'source', 'is_actuals', 'model', 'y']],
        #                         df_result_withnull[['date', 'source', 'is_actuals', 'model', 'y']])


        # Test 4D - Input has null values at the start of actuals series
        a_date_actuals = pd.date_range('2014-01-01', periods=10, freq='W')
        a_y_actuals = np.arange(0, 10.)
        df_actuals = (
            pd.DataFrame({'date': a_date_actuals, 'y': a_y_actuals,
                          'source': 's1', 'is_actuals': True, 'is_best_fit': False, 'model': 'actuals'})
        )

        a_date = pd.date_range('2014-01-01', periods=20, freq='W')
        a_y = np.arange(0, 20.) + (np.tile([-1, 1], (10)) * np.arange(2, 0., -0.1))
        a_y_withnull = np.concatenate([np.full(5,np.NaN),np.arange(0,15.),])

        df_fcast = (
            pd.DataFrame({'date': a_date, 'y': a_y,
                          'source': 's1', 'is_actuals': False, 'is_best_fit': True, 'model': 'linear'})
        )

        df_fcast_withnull = (
            pd.DataFrame({'date': a_date, 'y': a_y_withnull,
                          'source': 's1', 'is_actuals': False, 'is_best_fit': True, 'model': 'linear'})
        )

        df1 = pd.concat([df_actuals, df_fcast], ignore_index=True, sort=False)
        df_result = get_pi(df1, n=100)


        df1_withnull = pd.concat([df_actuals, df_fcast_withnull], ignore_index=True, sort=False)
        df_result_withnull = get_pi(df1_withnull, n=100)

        logger_info('df_fcast_withnull:', df_fcast_withnull.groupby(['source', 'model']).head(20))
        logger_info('df_result:', df_result.groupby(['source', 'model']).tail(100))
        logger_info('df_result with null:', df_result_withnull.groupby(['source', 'model']).tail(100))
        # Prediction intervals are random, so we need to exclude them from comparison
        # self.assert_frame_equal(df_result[['date', 'source', 'is_actuals', 'model', 'y']],
        #                         df_result_withnull[['date', 'source', 'is_actuals', 'model', 'y']])
        # TODO: ADD VALID CHECK -


    def test_get_pi_gap(self):
        def check_result(df_result):
            self.assertTrue('q5' in df_result.columns)

        # Test 1 - Input has gaps

        a_date_actuals = pd.date_range('2014-01-01', periods=10, freq='W')
        a_y_actuals = np.arange(0, 10.)
        df_actuals = (
            pd.DataFrame({'date': a_date_actuals, 'y': a_y_actuals,
                          'source': 's1', 'is_actuals': True, 'is_best_fit': False, 'model': 'actuals'})
        )

        a_date = pd.date_range('2014-01-01', periods=20, freq='W')
        a_y = np.arange(0, 20.) + (np.tile([-1, 1], (10)) * np.arange(2, 0., -0.1))

        df_fcast = (
            pd.DataFrame({'date': a_date, 'y': a_y,
                          'source': 's1', 'is_actuals': False, 'is_best_fit': True, 'model': 'linear'})
        )

        df_actuals_gap = pd.concat([df_actuals.head(3), df_actuals.tail(3)])

        df = pd.concat([df_actuals_gap, df_fcast], ignore_index=True, sort=False)

        df_result = get_pi(df, n=100)
        # logger_info('df_result1:', df_result1)
        logger_info('df_result1:', df_result.groupby(['source', 'model']).head(2))
        logger_info('df_result1:', df_result.groupby(['source', 'model']).tail(2))

        check_result(df_result)

        # Test 2 - Input has nulls

        df_actuals_null = df_actuals.copy()
        df_actuals_null.loc[5, 'y'] = np.NaN

        logger_info('df_actuals_null:', df_actuals_null)

        df = pd.concat([df_actuals_null, df_fcast], ignore_index=True, sort=False)

        df_result = get_pi(df, n=100)
        # logger_info('df_result1:', df_result1)
        logger_info('df_result2:', df_result.groupby(['source', 'model']).head(20))
        logger_info('df_result2:', df_result.groupby(['source', 'model']).tail(20))

        self.assertFalse(df_result.loc[df_result.date > df_actuals.date.max()].q5.isnull().any())

        check_result(df_result)

    def test_forecast_pi_missing(self):
        path_candy = os.path.join(base_folder, 'candy_production.csv')
        df_monthly_candy = pd.read_csv(path_candy)
        dict_result = run_forecast(df_monthly_candy,
                                   col_name_y='IPG3113N',
                                   col_name_date='observation_date', extrapolate_years=2,
                                   simplify_output=False)

        df_fcast = dict_result.get('forecast')
        logger_info('df_fcast: ', df_fcast.tail())

        self.assertIn('q5', df_fcast.columns)

    def test_run_forecast_yearly_model(self):
        df1 = pd.DataFrame({'y': np.arange(0, 10.), 'date': pd.date_range('2000-01-01', periods=10, freq='YS')})
        dict_result = run_forecast(simplify_output=False, df_y=df1, l_model_trend=[forecast_models.model_linear],
                                   extrapolate_years=10.)

        df_data = dict_result['data']
        df_metadata = dict_result['metadata']
        df_optimize_info = dict_result['optimize_info']

        logger_info('df_metadata:', df_metadata)
        logger_info('df_optimize_info:', df_optimize_info)
        logger_info('df_data:', df_data.groupby(['source', 'model']).tail(30))

        # Repeat test - 2 sources

        df1a = df1.copy()
        df1b = df1.copy()
        df1a['source'] = 'src1'
        df1b['source'] = 'src2'
        df2 = pd.concat([df1a, df1b], sort=False, ignore_index=True)

        logger_info('df input:', df2)

        dict_result = run_forecast(simplify_output=False, df_y=df2, l_model_trend=[forecast_models.model_linear],
                                   extrapolate_years=10.)

        df_data = dict_result['data']
        df_metadata = dict_result['metadata']
        df_optimize_info = dict_result['optimize_info']

        logger_info('df_metadata:', df_metadata)
        logger_info('df_optimize_info:', df_optimize_info)
        logger_info('df_data:', df_data.groupby(['source', 'model']).tail(60))

        # Same, with simplify_output=True

        df_result = run_forecast(simplify_output=True, df_y=df2, l_model_trend=[forecast_models.model_linear],
                                 extrapolate_years=10.)
        logger_info('df_result:', df_result)

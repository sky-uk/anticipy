"""

Author: Pedro Capelastegui
Created on 04/12/2015

"""

import os
import platform
import unittest

from anticipy.forecast import *
from anticipy.utils_test import PandasTest

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
        logger.info(log_first_line + '\r\nAIC_C:' +
                    str(fcast_driver_output.dict_aic_c))
        # logger_info('AIC_C:',fcast_driver_output[0])


# usage:
# compute_prof(l_dict_result2_d, scheduler = 'processes', num_workers=4, title='Test figure') # noqa


def compute_prof(*args, **kwargs):
    with Profiler() as prof, ResourceProfiler(dt=0.25) as rprof:
        out = compute(*args, **kwargs)
    visualize([prof, rprof,  # cprof
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
        a_x_in = np.arange(0, 10).astype(np.int64)
        a_x = a_x_in + 2
        a_x2_in = np.tile(np.arange(0, 5), 2).astype(np.int64)
        a_x2 = a_x2_in + 2
        a_x2_out = np.repeat(np.arange(0, 5), 2).astype(np.int64)
        a_x2_repeat = a_x2_out + 2
        a_source = ['s1'] * 5 + ['s2'] * 5
        a_weight = np.full(10, 1.0)
        a_date = pd.date_range('2014-01-01', periods=10, freq='D')
        a_date2 = np.tile(pd.date_range('2014-01-01', periods=5, freq='D'), 2)
        a_date2_out = np.repeat(
            pd.date_range(
                '2014-01-01',
                periods=5,
                freq='D'),
            2)

        logger.info('Test 0: Empty input')
        self.assertIsNone(normalize_df(pd.DataFrame))

        logger.info('Test 1: Output with x,y columns')
        df_expected = pd.DataFrame({'y': a_y, 'x': a_x_in, })[['x', 'y']]

        l_input = [
            [pd.DataFrame({'y': a_y}), {}],
            [pd.DataFrame({'y': a_y, 'x': a_x_in}), {}],
            [pd.DataFrame({'y_test': a_y, 'x_test': a_x_in}),
             {'col_name_y': 'y_test', 'col_name_x': 'x_test'}]
        ]
        for df, kwargs in l_input:
            run_test(df, df_expected, **kwargs)

        logger.info('Test 2: Output with x,y,weight columns')
        df_expected = \
        pd.DataFrame({'y': a_y, 'x': a_x_in, 'weight': a_weight})[
            ['x', 'y', 'weight']]

        l_input = [
            [pd.DataFrame({'y': a_y, 'weight': a_weight}), {}],
            [pd.DataFrame({'y': a_y, 'x': a_x_in, 'weight': a_weight}), {}],
            [pd.DataFrame({'y_test': a_y, 'x_test': a_x_in,
                           'weight_test': a_weight}),
             {'col_name_y': 'y_test', 'col_name_x': 'x_test',
              'col_name_weight': 'weight_test'}]
        ]
        for df, kwargs in l_input:
            run_test(df, df_expected, **kwargs)

        logger.info('Test 3: Output with x,y,weight,date columns')
        logger.info('Test 3a: Input includes x')
        # If x column is present, it is preserved - otherwise, we create it
        # from date column
        df_expected = pd.DataFrame({'y': a_y, 'x': a_x_in, 'weight': a_weight,
                                    'date': a_date})[
            ['date', 'x', 'y', 'weight']]

        l_input = [
            [pd.DataFrame({'y': a_y, 'x': a_x_in, 'weight': a_weight,
                           'date': a_date}), {}],
            [pd.DataFrame({'y_test': a_y, 'x_test': a_x_in,
                           'weight_test': a_weight, 'date_test': a_date}),
             {'col_name_y': 'y_test', 'col_name_x': 'x_test',
              'col_name_weight': 'weight_test',
              'col_name_date': 'date_test'}]
        ]
        for df, kwargs in l_input:
            run_test(df, df_expected, **kwargs)

        logger.info('Test 3b: Input has no x')
        df_expected = pd.DataFrame({'y': a_y, 'x': a_x, 'weight': a_weight,
                                    'date': a_date})[
            ['date', 'x', 'y', 'weight']]

        l_input = [
            [pd.DataFrame({'y': a_y, 'weight': a_weight, 'date': a_date}),
             {}],
            [pd.DataFrame({'y': a_y, 'weight': a_weight}, index=a_date), {}],
        ]
        for df, kwargs in l_input:
            run_test(df, df_expected, **kwargs)

        logger.info('Test 4: Input series')
        df_expected = pd.DataFrame({'y': a_y, 'x': a_x_in, })[['x', 'y']]

        l_input = [
            [pd.Series(a_y, name='y'), {}],
            [pd.Series(a_y, name='y', index=a_x_in), {}],
            [pd.Series(a_y, name='y_test'), {'col_name_y': 'y_test'}],
            # [pd.DataFrame({'y_test': a_y, 'x_test': a_x}),
            #   {'col_name_y':'y_test','col_name_x':'x_test'}]
        ]
        for df, kwargs in l_input:
            run_test(df, df_expected, **kwargs)

        logger.info('Test 5: Input series with datetimeindex')
        # If input is series with datetimeindex, create x from date
        df_expected = pd.DataFrame({'y': a_y, 'x': a_x, 'date': a_date})[
            ['date', 'x', 'y']]

        l_input = [
            [pd.Series(a_y, name='y', index=a_date), {}],
            [pd.Series(a_y, name='y_test', index=a_date),
             {'col_name_y': 'y_test'}],
            # [pd.DataFrame({'y_test': a_y, 'x_test': a_x}),
            #   {'col_name_y':'y_test','col_name_x':'x_test'}]
        ]
        for df, kwargs in l_input:
            run_test(df, df_expected, **kwargs)

        logger.info('Test 6: Input df, output with x, y,'
                    ' weight, date, source columns')
        df_expected = (
            pd.DataFrame({'y': a_y, 'x': a_x2, 'source': a_source,
                          'weight': a_weight, 'date': a_date2})
            [['date', 'source', 'x', 'y', 'weight']]
        )

        l_input = [
            [pd.DataFrame({'y': a_y, 'weight': a_weight,
                           'date': a_date2, 'source': a_source}), {}],
            # Datetime index not supported with source - could be
            # added back with multindex
            # [pd.DataFrame({'y': a_y, 'weight': a_weight},index = a_date),
            #   {}],
        ]
        for df, kwargs in l_input:
            run_test(df, df_expected, **kwargs)

        logger.info('Test 6b - input includes x column')
        df_expected = (
            pd.DataFrame({'y': a_y, 'x': a_x2_in, 'source': a_source,
                          'weight': a_weight, 'date': a_date2})
            [['date', 'source', 'x', 'y', 'weight']]
        )

        l_input = [
            [pd.DataFrame({'y': a_y, 'x': a_x2_in, 'weight': a_weight,
                           'source': a_source, 'date': a_date2}), {}],
            [pd.DataFrame({'y_test': a_y, 'x_test': a_x2_in,
                           'weight_test': a_weight, 'date_test': a_date2,
                           'source_test': a_source}),
             {'col_name_y': 'y_test', 'col_name_x': 'x_test',
              'col_name_weight': 'weight_test',
              'col_name_date': 'date_test',
              'col_name_source': 'source_test'}]
        ]
        for df, kwargs in l_input:
            run_test(df, df_expected, **kwargs)

        logger.info('Test 7: Input df has multiple values per date per source')
        df_expected = (
            pd.DataFrame({'y': a_y, 'x': a_x2_repeat, 'weight': a_weight,
                          'date': a_date2_out})
            [['date', 'x', 'y', 'weight']]
        )

        l_input = [
            [pd.DataFrame({'y': a_y, 'weight': a_weight, 'date': a_date2}),
             {}],
        ]
        for df, kwargs in l_input:
            run_test(df, df_expected, **kwargs)
        logger.info('Test 7b - Input includes x column')
        df_expected = (
            pd.DataFrame({'y': a_y, 'x': a_x2_out, 'weight': a_weight,
                          'date': a_date2_out})
            [['date', 'x', 'y', 'weight']]
        )

        l_input = [
            [pd.DataFrame(
                {'y': a_y, 'x': a_x2_in, 'weight': a_weight, 'date': a_date2}),
                {}],
            [pd.DataFrame({'y_test': a_y, 'x_test': a_x2_in,
                           'weight_test': a_weight, 'date_test': a_date2,
                           }),
             {'col_name_y': 'y_test', 'col_name_x': 'x_test',
              'col_name_weight': 'weight_test',
              'col_name_date': 'date_test',
              }]
        ]
        for df, kwargs in l_input:
            run_test(df, df_expected, **kwargs)

        logger.info('Test 8: input df has date column in string form')
        a_date_str = a_date2.astype(str)
        df_expected = (
            pd.DataFrame({'y': a_y, 'x': a_x2_in + 2, 'source': a_source,
                          'weight': a_weight, 'date': a_date2})
            [['date', 'source', 'x', 'y', 'weight']]
        )

        l_input = [
            [pd.DataFrame({'y': a_y, 'weight': a_weight, 'date': a_date_str,
                           'source': a_source}), {}],
        ]
        for df, kwargs in l_input:
            run_test(df, df_expected, **kwargs)

        logger.info('Test 8b - input has x column')
        a_date_str = a_date2.astype(str)
        df_expected = (
            pd.DataFrame({'y': a_y, 'x': a_x2_in, 'source': a_source,
                          'weight': a_weight, 'date': a_date2})
            [['date', 'source', 'x', 'y', 'weight']]
        )

        l_input = [
            [pd.DataFrame({'y': a_y, 'x': a_x2_in, 'weight': a_weight,
                           'source': a_source, 'date': a_date_str}), {}],
            [pd.DataFrame({'y_test': a_y, 'x_test': a_x2_in,
                           'weight_test': a_weight, 'date_test': a_date_str,
                           'source_test': a_source}),
             {'col_name_y': 'y_test', 'col_name_x': 'x_test',
              'col_name_weight': 'weight_test',
              'col_name_date': 'date_test',
              'col_name_source': 'source_test'}]
        ]
        for df, kwargs in l_input:
            run_test(df, df_expected, **kwargs)

        # Test 9: unordered input df

        df_expected = pd.DataFrame({'y': a_y, 'x': a_x_in, })[['x', 'y']]

        l_input = [
            [pd.DataFrame({'y': a_y[::-1]}), {}],
            [pd.DataFrame({'y': a_y[::-1], 'x': a_x_in[::-1]}), {}],
        ]
        for df, kwargs in l_input:
            run_test(df, df_expected, **kwargs)

        logger.info('Test 10: candy production dataset')
        path_candy = os.path.join(base_folder, 'candy_production.csv')
        df_candy_raw = pd.read_csv(path_candy)
        df_candy = df_candy_raw.pipe(
            normalize_df,
            col_name_y='IPG3113N',
            col_name_date='observation_date')
        logger_info('df_candy:', df_candy.tail())

        logger.info('Test 11: test_normalize.csv')
        path_file = os.path.join(base_folder, 'test_normalize.csv')
        df_test_raw = pd.read_csv(path_file)
        df_test = df_test_raw.pipe(normalize_df, )
        logger_info('df_test:', df_test.x.diff().loc[df_test.x.diff() > 1.0])
        self.assertFalse((df_test.x.diff() > 31.0).any())

        logger.info('Test 11b: test_normalize.csv, with gaps')
        path_file = os.path.join(base_folder, 'test_normalize.csv')
        df_test_raw = pd.read_csv(path_file)
        df_test_raw = pd.concat([df_test_raw.head(10), df_test_raw.tail(10)])
        df_test = df_test_raw.pipe(normalize_df, )
        logger_info('df_test:', df_test)
        logger_info('df_test:', df_test.x.diff().loc[df_test.x.diff() > 1.0])
        self.assertTrue((df_test.x.max() == 1311))

    def test_forecast_input(self):
        y_values1 = pd.DataFrame(
            {'a': np.full(100, 0.0),
             'b': np.round(np.arange(-0.5, 0.5, 0.01), 2), },
            index=pd.date_range('2014-01-01', periods=100, freq='D'))
        # Too few samples
        n = 4
        y_values1b = pd.DataFrame({'a': np.full(n, 0.0)}, index=pd.date_range(
            '2014-01-01', periods=n, freq='D'))

        y_values2 = pd.DataFrame({'a': np.full(100, 0.0)},
                                 index=pd.date_range(
            '2014-01-01', periods=100, freq='D'))

        # SolverConfig with trend
        conf1 = ForecastInput(
            source_id='source1',
            l_model_trend=[
                forecast_models.model_constant,
                forecast_models.model_linear],
            l_model_season=None,
            df_y=y_values1,
            weights_y_values=1.0,
            date_start_actuals=None)
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
        a_y = pd.DataFrame(
            {'a': np.arange(10.0), 'b': -np.arange(10.0)}).values
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
        self.assert_array_equal(
            residuals, [5., 4., 3., 2., 1., 0., 1., 2., 3., 4., 5.,
                        6., 7., 8., 9., 10., 11., 12., 13., 14.])
        # Test - Use a_weights to weight residuals based on time
        # Using parameter(0,0)
        a_y = np.arange(10.0)
        a_x = np.arange(10.0)
        a_weights = np.linspace(1., 2., 10)
        logger_info('a_y: ', a_y)
        logger_info('a_weights: ', a_weights)
        residuals = get_residuals(
            [0, 0], model, a_x, a_y, a_date, a_weights=a_weights)
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
            logger.info(
                'result cost: %s, shape: %s, x: %s, message: %s',
                result.cost,
                result.fun.shape,
                result.x,
                result.message)

        for model in l_model:
            logger.info('#### Model function: %s', model.name)

            df_result = optimize_least_squares(model, a_x, a_y, a_date)
            logger_info('result:', df_result)
            self.assertTrue(df_result.success.any())
            # logger_info('result.x:',res_trend.x)

            df_result = optimize_least_squares(
                model, a_x_long, a_y_long, a_date)
            logger_info('result:', df_result)
            self.assertTrue(df_result.success.any())

    def test_optimize_least_squares_cache(self):
        a_date = pd.date_range('2020-01-01', '2022-01-01')
        # logger_info('a_date', a_date)
        a_x = np.arange(0, a_date.size)
        # logger_info('a_x', a_x)
        a_y = a_date.month * 10 + a_date.weekday
        logger_info('a_y', a_y)

        l_models = [
            forecast_models.model_season_month,
            forecast_models.model_season_wday,
            forecast_models.model_season_month +
            forecast_models.model_season_wday,
            forecast_models.model_season_fourier_yearly,
            forecast_models.model_calendar_uk
        ]
        dict_t_summary = dict()
        l_df_result = []
        for model in l_models:
            time_start = datetime.now()
            df_result_cache = optimize_least_squares(
                model, a_x, a_y, a_date)
            fit_time_cache = (datetime.now() - time_start).total_seconds()
            time_start = datetime.now()
            df_result_no_cache = optimize_least_squares(
                model, a_x, a_y, a_date, use_cache=False)
            fit_time_no_cache = (datetime.now() - time_start).total_seconds()

            l_df_result += [
                df_result_cache.assign(model=str(model), is_cache=True),
                df_result_no_cache.assign(model=str(model), is_cache=False),
            ]
            dict_t_summary[str(model)] = [fit_time_cache, fit_time_no_cache]

        df_result = pd.concat(l_df_result, ignore_index=True)

        logger_info('result summary: ', df_result)

        df_t_summary = pd.DataFrame(dict_t_summary).T
        df_t_summary.columns = ['t_cache', 't_no_cache']
        logger_info('time summary: ', df_t_summary)

    def test_fit_model(self):
        # Input dataframes must have an y column, and may have columns x,date,
        # weight

        # Setup
        # TODO: Use pre-normalized input dfs, rather than callling
        # normalize_df()
        dict_df_y = {
            # Single ts
            'df_1ts_nodate': pd.DataFrame({'y': np.full(100, 0.0),
                                           'weight_test': np.full(100, 1.0)}),
            # 2 ts
            'df_2ts_nodate': pd.DataFrame(
                {'y': np.concatenate([np.full(100, 0.0),
                                      np.round(np.arange(
                                          -0.5, 0.5, 0.01), 2)]),
                 'weight_test': np.concatenate([np.full(100, 0.1),
                                                np.full(100, 1.0)]),
                 'x': np.tile(np.arange(0, 100), 2), }),
            # 1 ts with datetime index
            'df_1ts_w': pd.DataFrame(
                {'y': np.full(100, 0.0),
                 'weight_test': np.full(100, 1.0)
                 },
                index=pd.date_range('2014-01-01', periods=100, freq='W')),
            # 2 ts with datetime index
            'df_2ts_w': pd.DataFrame(
                {'y': np.concatenate([np.full(100, 0.0),
                                      np.round(
                                          np.arange(-0.5, 0.5, 0.01), 2)]),
                 'weight_test': np.concatenate([np.full(100, 0.1),
                                                np.full(100, 1.0)]),
                 },
                index=np.tile(pd.date_range('2014-01-01', periods=100,
                                            freq='W'), 2)),
            # Single ts, freq=D
            'df_1ts_d': pd.DataFrame(
                {'y': np.full(100, 0.0),
                 'weight_test': np.full(100, 1.0)},
                index=pd.date_range('2014-01-01', periods=100, freq='D')),
            # 2 ts with datetime index, freq=D
            'df_2ts_d': pd.DataFrame(
                {'y': np.concatenate([np.full(100, 0.0),
                                      np.round(np.arange(
                                          -0.5, 0.5, 0.01), 2)]),
                 'weight_test': np.concatenate([np.full(100, 0.1),
                                                np.full(100, 1.0)])},
                index=np.tile(pd.date_range('2014-01-01', periods=100,
                                            freq='D'), 2))
        }
        l_source1 = [
            'df_1ts_nodate',
            'df_2ts_nodate',
            'df_1ts_w',
            'df_1ts_w',
            'df_2ts_d',
            'df_2ts_d']
        l_source2 = ['df_1ts_d', 'df_2ts_d']

        # Naive trend models - cannot add seasonality
        l_model1a = [
            # model_naive never actually goes to fit_model
            forecast_models.model_naive,
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
        l_model1 = l_model1a + l_model1b + l_model1c

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
            logger.info(
                'Fitting src: %s , mod: %s, add_weight: %s',
                source,
                model,
                add_weight)
            dict_fit_model = fit_model(
                model, df_y, source=source, df_actuals=df_y
            )
            return dict_fit_model
            # logger_info('Result: ',result)

        # Test - single solver type, return best fit
        for (source, model, add_weight) in itertools.product(
                l_source1, l_model1a + l_model1c, l_add_weight):
            dict_fit_model = run_test_logic(source, model, add_weight)
            result_tmp = dict_fit_model['metadata']
            info_tmp = dict_fit_model['optimize_info']
            l_results += [result_tmp]
            l_optimize_info += [info_tmp]

        # Now for models that require datetimeindex
        for (source, model, add_weight) in itertools.product(
                l_source2, l_model1b + l_model2, l_add_weight):
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
        df_optimize_info = pd.concat(
            l_optimize_info, sort=False, ignore_index=True)

        self.assertFalse(df_result.cost.pipe(pd.isnull).any())

        logger_info('Result summary:', df_result)
        logger_info('Optimize info summary:', df_optimize_info)

    def test_fit_model_metadata(self):
        # Check that weight column in metadata is correct
        df_in = pd.DataFrame(
            dict(x=np.arange(20), y=10., weight=[0] + [1.] * 19))
        dict_result = fit_model(
            forecast_models.model_constant,
            df_in,
        )
        df_metadata = dict_result.get('metadata')
        logger_info('metadata:', df_metadata)
        str_weights = df_metadata.weights.iloc[0]
        self.assertEqual(str_weights, '0.0-1.0')

    @unittest.skip('Dask not supported yet')
    def test_fit_model_dask(self):
        # Input dataframes must have an y column, and may have columns x,date,
        # weight

        # Setup
        # TODO: Use pre-normalized input dfs, rather than callling
        # normalize_df()
        dict_df_y = {
            # Single ts
            'df_1ts_nodate': pd.DataFrame(
                {'y': np.full(100, 0.0),
                 'weight_test': np.full(100, 1.0)}),
            # 2 ts
            'df_2ts_nodate': pd.DataFrame(
                {'y': np.concatenate([np.full(100, 0.0),
                                      np.round(
                                          np.arange(-0.5, 0.5, 0.01), 2)]),
                 'weight_test': np.concatenate([np.full(100, 0.1),
                                                np.full(100, 1.0)]),
                 'x': np.tile(np.arange(0, 100), 2),
                 }),
            # 1 ts with datetime index
            'df_1ts_w': pd.DataFrame(
                {'y': np.full(100, 0.0),
                 'weight_test': np.full(100, 1.0)
                 },
                index=pd.date_range('2014-01-01', periods=100, freq='W')),
            # 2 ts with datetime index
            'df_2ts_w': pd.DataFrame(
                {'y': np.concatenate([np.full(100, 0.0), np.round(
                    np.arange(-0.5, 0.5, 0.01), 2)]),
                 'weight_test': np.concatenate([np.full(100, 0.1),
                                                np.full(100, 1.0)]),
                 },
                index=np.tile(pd.date_range('2014-01-01', periods=100,
                                            freq='W'), 2)),
            # Single ts, freq=D
            'df_1ts_d': pd.DataFrame(
                {'y': np.full(100, 0.0),
                 'weight_test': np.full(100, 1.0)},
                index=pd.date_range('2014-01-01', periods=100, freq='D')),
            # 2 ts with datetime index, freq=D
            'df_2ts_d': pd.DataFrame(
                {'y': np.concatenate([np.full(100, 0.0), np.round(
                    np.arange(-0.5, 0.5, 0.01), 2)]),
                 'weight_test': np.concatenate([np.full(100, 0.1),
                                                np.full(100, 1.0)])},
                index=np.tile(pd.date_range('2014-01-01', periods=100,
                                            freq='D'), 2))
        }
        l_source1 = [
            'df_1ts_nodate',
            'df_2ts_nodate',
            'df_1ts_w',
            'df_1ts_w',
            'df_2ts_d',
            'df_2ts_d']
        l_source2 = ['df_1ts_d', 'df_2ts_d']

        l_model1 = [
            # model_naive never actually goes to fit_model
            forecast_models.model_naive,
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
            # df_y = dict_df_y[source].copy()
            # if add_weight:  # Enable weight column
            #    df_y['weight']=df_y['weight_test']
            col_name_weight = 'weight' if add_weight else 'no-weight'
            df_y = df_y.pipe(normalize_df, col_name_weight=col_name_weight)
            # logger.info('Fitting src: %s , mod: %s, add_weight: %s', source, model, add_weight) # noqa
            # dict_fit_model = delayed(fit_model)(model, df_y,  source=source, df_actuals = df_y) # noqa
            dict_fit_model = fit_model(
                model, df_y, source=source, df_actuals=df_y)
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
            df_optimize_info = pd.concat(
                l_optimize_info, sort=False, ignore_index=True)
            return df_metadata, df_optimize_info

        l_dict_fit_model_d = []

        # Test - single solver type, return best fit
        for (source, model, add_weight) in itertools.product(
                l_source1, l_model1, l_add_weight):
            l_dict_fit_model_d += [
                delayed(run_test_logic)(
                    dict_df_y[source].copy(),
                    source,
                    model,
                    add_weight)]

        # Now for models that require datetimeindex
        for (source, model, add_weight) in itertools.product(
                l_source2, l_model2, l_add_weight):
            l_dict_fit_model_d += [
                delayed(run_test_logic)(
                    dict_df_y[source].copy(),
                    source,
                    model,
                    add_weight)]

        # Finally, we use trend+seasonality with all models
        for (source, model, add_weight) in itertools.product(
                l_source2, l_model3, l_add_weight):
            l_dict_fit_model_d += [
                delayed(run_test_logic)(
                    dict_df_y[source].copy(),
                    source,
                    model,
                    add_weight)]

        logger.info('generated delayed')

        # client = Client()
        # logger_info('client:',client)
        # l_dict_fit_model, = compute(l_dict_fit_model_d)
        l_dict_fit_model, = compute_prof(
            l_dict_fit_model_d, scheduler='processes', num_workers=4)
        # l_dict_fit_model, = compute(l_dict_fit_model_d, scheduler='processes', num_workers=4) # noqa
        # l_dict_fit_model, = compute(l_dict_fit_model_d, scheduler='distributed', num_workers=4) # noqa
        # l_dict_fit_model, = compute(l_dict_fit_model_d, scheduler='threads', num_workers=4) # noqa
        # l_dict_fit_model = l_dict_fit_model_d

        df_metadata, df_optimize_info = aggregate_dict_fit_model(
            l_dict_fit_model)
        # result_d = delayed(aggregate_dict_fit_model)(l_dict_fit_model_d)
        # result_d = delayed(aggregate_dict_fit_model)(l_dict_fit_model_d)
        # (df_metadata, df_optimize_info), = compute(result_d)
        # result, = compute(result_d)
        logger_info('Result summary:', df_metadata)
        logger_info('Optimize info summary:', df_optimize_info)
        # client.close()

    @unittest.skip('Dask not supported yet')
    def test_fit_model_dask2(self):
        # Input dataframes must have an y column, and may have columns x,date,
        # weight

        # Setup
        # TODO: Use pre-normalized input dfs, rather than callling
        # normalize_df()
        dict_df_y = {
            # Single ts
            'df_1ts_nodate': pd.DataFrame(
                {'y': np.full(100, 0.0),
                 'weight_test': np.full(100, 1.0)}),
            # 2 ts
            'df_2ts_nodate': pd.DataFrame(
                {'y': np.concatenate([np.full(100, 0.0),
                                      np.round(
                                          np.arange(-0.5, 0.5, 0.01), 2)]),
                 'weight_test': np.concatenate([np.full(100, 0.1),
                                                np.full(100, 1.0)]),
                 'x': np.tile(np.arange(0, 100), 2),
                 }),
            # 1 ts with datetime index
            'df_1ts_w': pd.DataFrame(
                {'y': np.full(100, 0.0),
                 'weight_test': np.full(100, 1.0)
                 },
                index=pd.date_range('2014-01-01', periods=100, freq='W')),
            # 2 ts with datetime index
            'df_2ts_w': pd.DataFrame(
                {'y': np.concatenate([np.full(100, 0.0), np.round(
                    np.arange(-0.5, 0.5, 0.01), 2)]),
                 'weight_test': np.concatenate([np.full(100, 0.1),
                                                np.full(100, 1.0)]),
                 },
                index=np.tile(pd.date_range('2014-01-01', periods=100,
                                            freq='W'), 2)),
            # Single ts, freq=D
            'df_1ts_d': pd.DataFrame(
                {'y': np.full(100, 0.0),
                 'weight_test': np.full(100, 1.0)},
                index=pd.date_range('2014-01-01', periods=100,
                                    freq='D')),
            # 2 ts with datetime index, freq=D
            'df_2ts_d': pd.DataFrame(
                {'y': np.concatenate([np.full(100, 0.0), np.round(
                    np.arange(-0.5, 0.5, 0.01), 2)]),
                 'weight_test': np.concatenate([np.full(100, 0.1),
                                                np.full(100, 1.0)])},
                index=np.tile(pd.date_range('2014-01-01', periods=100,
                                            freq='D'), 2))
        }
        l_source1 = [
            'df_1ts_nodate',
            'df_2ts_nodate',
            'df_1ts_w',
            'df_1ts_w',
            'df_2ts_d',
            'df_2ts_d']
        l_source2 = ['df_1ts_d', 'df_2ts_d']

        l_model1 = [
            # model_naive never actually goes to fit_model
            forecast_models.model_naive,
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
            # df_y = dict_df_y[source].copy()
            if add_weight:  # Enable weight column
                df_y['weight'] = df_y['weight_test']
            df_y = df_y.pipe(normalize_df)
            # logger.info('Fitting src: %s , mod: %s, add_weight: %s', source, model, add_weight) # noqa
            # dict_fit_model = delayed(fit_model)(model, df_y,  source=source, df_actuals = df_y) # noqa
            dict_fit_model = fit_model(
                model, df_y, source=source, df_actuals=df_y)
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
            df_metadata = delayed(
                pd.concat)(
                l_results,
                sort=False,
                ignore_index=False)
            df_optimize_info = delayed(
                pd.concat)(
                l_optimize_info,
                sort=False,
                ignore_index=False)
            return df_metadata, df_optimize_info

        l_dict_fit_model_d = []

        # Test - single solver type, return best fit

        l_dict_fit_model_d += [
            delayed(fit_model)(
                model,
                dict_df_y[source].pipe(
                    delayed(normalize_df),
                    col_name_weight=weight),
                source=source,
                df_actuals=dict_df_y[source].pipe(
                    delayed(normalize_df),
                    col_name_weight=weight)) for (
                source,
                model,
                weight) in itertools.product(
                l_source1,
                l_model1,
                l_weight)]

        # Now for models that require datetimeindex
        l_dict_fit_model_d += [
            delayed(fit_model)(
                model,
                dict_df_y[source].pipe(
                    delayed(normalize_df),
                    col_name_weight=weight),
                source=source,
                df_actuals=dict_df_y[source].pipe(
                    delayed(normalize_df),
                    col_name_weight=weight)) for (
                source,
                model,
                weight) in itertools.product(
                l_source2,
                l_model2,
                l_weight)]

        # Finally, we use trend+seasonality with all models
        l_dict_fit_model_d += [
            delayed(fit_model)(
                model,
                dict_df_y[source].pipe(
                    delayed(normalize_df),
                    col_name_weight=weight),
                source=source,
                df_actuals=dict_df_y[source].pipe(
                    delayed(normalize_df),
                    col_name_weight=weight)) for (
                source,
                model,
                weight) in itertools.product(
                l_source2,
                l_model3,
                l_weight)]

        # # Finally, we use trend+seasonality with all models
        # for (source, model, weight) in itertools.product(
        #         l_source2, l_model3, l_weight):
        #     df_y = dict_df_y[source].pipe(normalize_df, col_name_weight=weight)  # noqa
        #     l_dict_fit_model_d += [delayed(fit_model)(model, df_y, source=source, df_actuals=df_y)]  # noqa

        logger.info('generated delayed')

        # l_dict_fit_model, = compute(l_dict_fit_model_d)
        l_dict_fit_model = l_dict_fit_model_d

        # df_metadata, df_optimize_info = aggregate_dict_fit_model(l_dict_fit_model) # noqa
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
            return pd.concat(
                l_metadata, sort=False, ignore_index=False), pd.concat(
                l_opt, sort=False, ignore_index=False)

        model = forecast_models.model_linear
        df_y = pd.DataFrame(
            {'y': np.full(100, 0.0),
             'weight_test': np.full(100, 1.0)}).pipe(normalize_df)

        l_dict_result2_d = [
            delayed(fit_model)(
                model,
                df_y,
                source=i,
                df_actuals=df_y) for i in np.arange(
                0,
                20)]
        result_d = delayed(aggregate_result)(l_dict_result2_d)
        # result = compute(result_d,scheduler='processes',num_workers=2)
        result = compute(result_d)
        logger_info('result', result)

    def test_fit_model_date_gaps(self):
        # Setup
        # 2 ts with datetime index, freq=D
        df_2ts_d = pd.DataFrame(
            {'y': np.concatenate([np.full(100, 0.0), np.round(
                np.arange(-0.5, 0.5, 0.01), 2)]),
             'weight': np.concatenate([np.full(100, 0.1),
                                       np.full(100, 1.0)])},
            index=np.tile(pd.date_range('2014-01-01', periods=100,
                                        freq='D'), 2))

        df_y = pd.concat([df_2ts_d.head(), df_2ts_d.tail()])

        model = forecast_models.model_linear

        l_col_name_weight = [None, 'weight']

        l_results = []
        l_optimize_info = []

        def run_test_logic(col_name_weight):
            logger.info('Fitting col_w: %s', col_name_weight)
            df_y_tmp = df_y.pipe(normalize_df,
                                 col_name_weight=col_name_weight)
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
            # l1[0] * l2[1], # This is a duplicate: linear*null=linear+null =
            # linear
            l1[1] * l2[0],
            # l1[1] * l2[1], # This is a duplicate: constant*null =
            # constant+null = constant
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
            'df_1ts_nodate': pd.DataFrame(
                {'y': np.full(100, 0.0),
                 'weight': np.full(100, 1.0)}),
            # 2 ts
            'df_2ts_nodate': pd.DataFrame(
                {'y': np.concatenate([np.full(100, 0.0),
                                      np.round(np.arange(
                                          -0.5, 0.5, 0.01), 2)]),
                 'weight': np.concatenate([np.full(100, 0.1),
                                           np.full(100, 1.0)]),
                 'x': np.tile(np.arange(0, 100), 2),
                 }),
            # 1 ts with datetime index
            'df_1ts_w': pd.DataFrame(
                {'y': np.full(100, 0.0),
                 'weight': np.full(100, 1.0)
                 },
                index=pd.date_range('2014-01-01', periods=100, freq='W')),

            # 2 ts with datetime index
            'df_2ts_w': pd.DataFrame(
                {'y': np.concatenate([np.full(100, 0.0),
                                      np.round(np.arange(
                                          -0.5, 0.5, 0.01), 2)]),
                 'weight': np.concatenate([np.full(100, 0.1),
                                           np.full(100, 1.0)]),
                 },
                index=np.tile(pd.date_range('2014-01-01', periods=100,
                                            freq='W'), 2)),
            # Single ts, freq=D
            'df_1ts_d': pd.DataFrame(
                {'y': np.full(100, 0.0),
                 'weight': np.full(100, 1.0)},
                index=pd.date_range('2014-01-01', periods=100, freq='D')),

            # Single ts, freq=D , index named 'date
            'df_1ts_d2': pd.DataFrame(
                {'y': np.full(100, 0.0),
                 'weight': np.full(100, 1.0)},
                index=pd.date_range('2014-01-01', periods=100, freq='D',
                                    name='date')).reset_index(),

            # 2 ts with datetime index, freq=D
            'df_2ts_d': pd.DataFrame(
                {'y': np.concatenate([np.full(100, 0.0), np.round(
                    np.arange(-0.5, 0.5, 0.01), 2)]),
                 'weight': np.concatenate([np.full(100, 0.1),
                                           np.full(100, 1.0)])},
                index=np.tile(pd.date_range('2014-01-01', periods=100,
                                            freq='D'), 2))
        }

        l_source_d = ['df_1ts_d', 'df_2ts_d', 'df_1ts_d2']
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
        for (
                source,
                col_name_weight,
                model) in itertools.product(
            l_source_d,
            l_col_name_weight,
            get_list_model(
                l_model_trend,
                l_model_season,
                'both')):
            df_y = dict_df_y[source].copy().pipe(
                normalize_df, col_name_weight=col_name_weight)
            logger.info(
                'Fitting src: %s , mod: %s, col_w: %s',
                source,
                model,
                col_name_weight)
            for i in np.arange(0, n_iterations):
                dict_fit_model = fit_model(
                    model, df_y, source=source, freq='D')
                l_results += [dict_fit_model['metadata']]
                l_optimize_info += [dict_fit_model['optimize_info']]

        # Fit , run n iterations, freq='D' - test function composition in
        # different order
        for (
                source,
                col_name_weight,
                model) in itertools.product(
            l_source_d,
            l_col_name_weight,
            get_list_model(
                l_model_season,
                l_model_trend,
                'both')):
            df_y = dict_df_y[source].copy().pipe(
                normalize_df, col_name_weight=col_name_weight)
            logger.info(
                'Fitting src: %s , mod: %s, col_w: %s',
                source,
                model,
                col_name_weight)
            for i in np.arange(0, n_iterations):
                dict_fit_model = fit_model(
                    model, df_y, source=source, freq='D')
                l_results += [dict_fit_model['metadata']]
                l_optimize_info += [dict_fit_model['optimize_info']]

        df_result = pd.concat(l_results)
        df_optimize_info = pd.concat(l_optimize_info)
        logger_info('Result summary:', df_result)
        logger_info('Optimize info summary:', df_optimize_info)

    def test_extrapolate_model(self):
        # with freq=None, defaults to W
        df_y_forecast = extrapolate_model(
            forecast_models.model_constant,
            [1.0],
            '2017-01-01',
            '2017-01-01',
            freq=None,
            extrapolate_years=1.0)
        logger_info('df_y_forecast', df_y_forecast.tail(1))
        logger_info('Result length:', df_y_forecast.index.size)
        self.assertEqual(df_y_forecast.index.size, 53)

        df_y_forecast = extrapolate_model(
            forecast_models.model_constant,
            [1.0],
            '2017-01-01',
            '2017-12-31',
            freq='D',
            extrapolate_years=1.0)
        logger_info('df_y_forecast', df_y_forecast.tail(1))
        logger_info('Result length:', df_y_forecast.index.size)
        self.assertEqual(df_y_forecast.index.size, 365 * 2)

        df_y_forecast = extrapolate_model(
            forecast_models.model_constant,
            [1.0],
            '2017-01-01',
            '2017-12-31',
            freq='MS',
            extrapolate_years=1.0)
        logger_info('df_y_forecast', df_y_forecast.tail(1))
        logger_info('Result length:', df_y_forecast.index.size)
        self.assertEqual(df_y_forecast.index.size, 12 * 2)

        df_y_forecast = extrapolate_model(
            forecast_models.model_constant,
            [1.0],
            '2000-01-01',
            '2009-01-01',
            freq='YS',
            extrapolate_years=10.0)
        logger_info('df_y_forecast', df_y_forecast.tail(20))
        logger_info('Result length:', df_y_forecast.index.size)
        self.assertEqual(df_y_forecast.index.size, 20)

        # TODO: Test other time frequencies, e.g. Q, H, Y.

    def test_get_df_actuals_clean(self):
        dict_df_y = {
            # Single ts
            'df_1ts_nodate': pd.DataFrame(
                {'y': np.full(100, 0.0),
                 'weight': np.full(100, 1.0)}),
            # 2 ts
            'df_2ts_nodate': pd.DataFrame(
                {'y': np.concatenate([np.full(100, 0.0),
                                      np.round(np.arange(
                                          -0.5, 0.5, 0.01), 2)]),
                 'weight': np.concatenate([np.full(100, 0.1),
                                           np.full(100, 1.0)]),
                 'x': np.tile(np.arange(0, 100), 2),
                 }),
            # 1 ts with datetime index
            'df_1ts_w': pd.DataFrame(
                {'y': np.full(100, 0.0),
                 'weight': np.full(100, 1.0)
                 },
                index=pd.date_range('2014-01-01', periods=100,
                                    freq='W')),
            # 1 ts with datetime index named 'date
            'df_1ts_w-2': pd.DataFrame(
                {'y': np.full(100, 0.0),
                 'weight': np.full(100, 1.0)
                 },
                index=pd.date_range('2014-01-01', periods=100,
                                    freq='W', name='date')),
            # 1 ts with datetime column
            'df_1ts_w-3': pd.DataFrame(
                {'y': np.full(100, 0.0),
                 'weight': np.full(100, 1.0),
                 'date': pd.date_range('2014-01-01', periods=100,
                                       freq='W')
                 })
        }
        # Simple test - check for crashes

        for k in dict_df_y.keys():
            logger.info('Input: %s', k)
            df_in = dict_df_y.get(k).pipe(normalize_df)
            logger_info('DF_IN', df_in.tail(3))
            df_result = get_df_actuals_clean(df_in, 'test', 'test')
            logger_info('Result:', df_result.tail(3))
            unique_models = df_result.model.drop_duplicates().reset_index(
                drop=True)
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
        if not include_all_fits:
            # In this case, there should be only one model fitted
            # per data source
            self.assertTrue(df_metadata.is_best_fit.all())
            self.assertTrue((df_data.is_best_fit | df_data.is_actuals).all())
            # The following may not be true if a model doesn't converge
            self.assertEqual(df_metadata.index.size, n_sources)
            self.assertEqual(
                df_data.loc[~df_data.is_actuals].drop_duplicates(
                    'source_long').index.size, n_sources)

        # Check that actuals are included
        self.assertTrue((df_data.is_actuals.any()))

        # Check that dtype is not corrupted
        self.assertTrue(np.issubdtype(df_data.y.astype(float), np.float64))

    def _test_run_forecast_check_length_new_api(self, **kwargs):
        freq = kwargs.get('freq', 'D')

        freq = detect_freq(kwargs.get('df_y').pipe(normalize_df))

        # Changes e.g. W-MON to W
        freq_short = freq[0:1] if freq is not None else None
        # Todo: change to dict to support more frequencies
        freq_units_per_year = 52.0 if freq_short == 'W' else 365.0

        extrapolate_years = kwargs.get('extrapolate_years', 1.0)

        # Both additive and multiplicative
        dict_result = run_forecast(
            simplify_output=False,
            extrapolate_years=extrapolate_years,
            **kwargs)

        df_data = dict_result['data']
        df_metadata = dict_result['metadata']

        logger_info('df_metadata:', df_metadata)
        logger_info('df_data:', df_data.groupby(['source', 'model']).tail(1))

        df_data_size = df_data.groupby(
            ['source', 'model', 'is_actuals']).size().rename(
            'group_size').reset_index()
        df_data_size_unique = (
            df_data.drop_duplicates(
                ['source', 'model', 'is_actuals', 'date']) .groupby(
                ['source', 'model', 'is_actuals']).size().rename(
                'group_size').reset_index()
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
            logger.info('Testing source: %s', source)
            df_y_tmp = df_y.loc[df_y.source == source]

            size_actuals_unique_tmp = df_y_tmp.drop_duplicates('x').index.size
            size_actuals_tmp = df_y_tmp.index.size

            df_data_size_tmp = df_data_size.loc[
                df_data_size.source == source]
            df_data_size_actuals = df_data_size_tmp.loc[
                df_data_size_tmp.is_actuals]
            df_data_size_fcast = df_data_size_tmp.loc[
                ~df_data_size_tmp.is_actuals]

            # logger.info('DEBUG: group size: %s',100 +
            # extrapolate_years*freq_units_per_year)
            # This assert doesn't work for all years - some have 365 days,
            # some 366. Currently running with 365-day year

            logger.info(
                'DEBUG: df_data_size_fcast.group_size %s , size_actuals_tmp'
                ' %s, total %s',
                df_data_size_fcast.group_size.values,
                size_actuals_tmp,
                size_actuals_tmp +
                extrapolate_years *
                freq_units_per_year)

            self.assertTrue(
                (df_data_size_actuals.group_size == size_actuals_tmp).all())

            self.assert_array_equal(
                df_data_size_fcast.group_size,
                size_actuals_unique_tmp +
                extrapolate_years *
                freq_units_per_year)
            self.assertFalse(df_data_size_fcast.empty)

    def _test_run_forecast(self, freq='D'):
        logger.info('Testing run_forecast - freq: %s', freq)
        # freq_short = freq[0:1]  # Changes e.g. W-MON to W
        # freq_units_per_year = 52.0 if freq_short == 'W' else 365.0
        # Todo: change to dict to support more frequencies

        # Input dataframe without date column
        df_y0 = pd.DataFrame(
            {'y': np.concatenate([np.full(100, 0.0),
                                  np.round(np.arange(-0.5, 0.5, 0.01), 2)]),
             'weight': np.concatenate([np.full(100, 0.1), np.full(100, 1.0)]),
             },
        )

        df_y1 = pd.DataFrame(
            {'y': np.concatenate([np.full(100, 0.0), np.round(
                np.arange(-0.5, 0.5, 0.01), 2)]),
             'weight': np.concatenate([np.full(100, 0.1),
                                       np.full(100, 1.0)]),
             },
            index=np.tile(pd.date_range('2014-01-01', periods=100, freq=freq),
                          2))

        # Too few samples
        n = 4
        df_y1b = pd.DataFrame({'y': np.full(n, 0.0)}, index=pd.date_range(
            '2017-01-01', periods=n, freq=freq))

        df_y2 = pd.DataFrame({'y': np.full(100, 0.0)}, index=pd.date_range(
            '2017-01-01', periods=100, freq=freq))

        # Df with source column
        df_y3 = pd.DataFrame(
            {'y': np.concatenate([np.full(100, 0.0),
                                  np.round(np.arange(-0.5, 0.5, 0.01), 2)]),
             'weight': np.concatenate([np.full(100, 0.1), np.full(100, 1.0)]),
             'source': ['src1'] * 100 + ['src2'] * 100
             },
            index=np.tile(pd.date_range('2014-01-01', periods=100, freq=freq),
                          2))
        # As above, with renamed columns
        df_y3b = pd.DataFrame(
            {'y_test': np.concatenate([np.full(100, 0.0), np.round(
                np.arange(-0.5, 0.5, 0.01), 2)]),
             'weight_test': np.concatenate([np.full(100, 0.1),
                                            np.full(100, 1.0)]),
             'source_test': ['src1'] * 100 + ['src2'] * 100,
             'date_test': np.tile(pd.date_range('2014-01-01',
                                                periods=100, freq=freq), 2)
             })

        # # Model lists
        l_model_trend1 = [forecast_models.model_linear]
        l_model_trend1b = [
            forecast_models.model_linear,
            forecast_models.model_season_wday_2]
        l_model_trend2 = [
            forecast_models.model_linear,
            forecast_models.model_exp]

        l_model_season1 = [forecast_models.model_season_wday_2]
        l_model_season2 = [
            forecast_models.model_season_wday_2,
            forecast_models.model_null]

        # New test - forecast length
        logger.info('Testing Output Length')
        logger.info('Testing Output Length - df_y1')
        self._test_run_forecast_check_length_new_api(
            df_y=df_y1,
            include_all_fits=False,
            l_model_trend=l_model_trend1b,
            source_id='source1',
            l_model_naive=[]
        )
        logger.info('Testing Output Length - df_y2')
        self._test_run_forecast_check_length_new_api(
            df_y=df_y2,
            include_all_fits=False,
            l_model_trend=l_model_trend2,
            l_model_season=l_model_season2,
            source_id='source2',
            l_model_naive=[]
        )

    def test_runforecast(self):
        for freq in ['D',
                     'W']:
            self._test_run_forecast(freq=freq)

    def test_run_forecast_metadata(self):
        df1 = pd.DataFrame({'y': np.arange(0, 10.)}, index=pd.date_range(
            '2014-01-01', periods=10, freq='D'))
        dict_result = run_forecast(
            simplify_output=False, df_y=df1, l_model_trend=[
                forecast_models.model_linear])

        df_data = dict_result['data']
        df_metadata = dict_result['metadata']
        logger_info('metadata.fit_time dtype', df_metadata.fit_time.dtype)
        self.assertTrue(pd.api.types.is_numeric_dtype(df_metadata.fit_time))

    def test_run_forecast_simple_linear_model(self):
        df1 = pd.DataFrame({'y': np.arange(0, 10.)}, index=pd.date_range(
            '2014-01-01', periods=10, freq='D'))
        dict_result = run_forecast(
            simplify_output=False, df_y=df1, l_model_trend=[
                forecast_models.model_linear])

        df_data = dict_result['data']
        df_metadata = dict_result['metadata']
        df_optimize_info = dict_result['optimize_info']

        logger_info('df_metadata:', df_metadata)
        logger_info('df_optimize_info:', df_optimize_info)
        logger_info('df_data:', df_data.groupby(['source', 'model']).tail(30))

        df2 = pd.DataFrame(
            {'y': np.arange(0, 10.), 'source': ['src1'] * 5 + ['src2'] * 5},
            index=pd.date_range('2014-01-01', periods=10, freq='D'))
        dict_result = run_forecast(
            simplify_output=False, df_y=df2, l_model_trend=[
                forecast_models.model_linear])

        df_data = dict_result['data']
        df_metadata = dict_result['metadata']
        df_optimize_info = dict_result['optimize_info']

        logger_info('df_metadata:', df_metadata)
        logger_info('df_optimize_info:', df_optimize_info)
        logger_info('df_data:', df_data.groupby(['source', 'model']).tail(60))

    def test_run_forecast_naive(self):
        logger.info('Test 1 - linear series, 1 source')
        df1 = pd.DataFrame(
            {'y': np.arange(0, 10.)},
            index=pd.date_range('2014-01-01', periods=10, freq='D'))
        dict_result = run_forecast(
            simplify_output=False,
            df_y=df1,
            l_model_trend=[forecast_models.model_naive],
            extrapolate_years=10. / 365)

        df_data = dict_result['data']
        df_metadata = dict_result['metadata']
        df_optimize_info = dict_result['optimize_info']

        logger_info('df_metadata:', df_metadata)
        logger_info('df_optimize_info:', df_optimize_info)
        logger_info('df_data:', df_data.groupby(['source', 'model']).tail(40))

        logger.info('Test 2 - 2 sources')
        df2 = pd.DataFrame(
            {'y': np.arange(0, 10.), 'source': ['src1'] * 5 + ['src2'] * 5},
            index=pd.date_range('2014-01-01', periods=10, freq='D'))
        dict_result = run_forecast(
            simplify_output=False, df_y=df2,
            l_model_trend=[forecast_models.model_naive],
            extrapolate_years=10. / 365)

        df_data = dict_result['data']
        df_metadata = dict_result['metadata']
        df_optimize_info = dict_result['optimize_info']

        logger_info('df_metadata:', df_metadata)
        logger_info('df_optimize_info:', df_optimize_info)
        logger_info('df_data:', df_data.groupby(['source', 'model']).tail(60))

        logger.info('test 3: weight column')
        df1 = pd.DataFrame(
            {'y': np.arange(0, 10.),
             'weight': array_zeros_in_indices(10, [5, 6])},
            index=pd.date_range('2014-01-01', periods=10, freq='D'))
        dict_result = run_forecast(
            simplify_output=False, df_y=df1,
            l_model_trend=[forecast_models.model_naive],
            extrapolate_years=10. / 365)

        df_data = dict_result['data']
        df_metadata = dict_result['metadata']
        df_optimize_info = dict_result['optimize_info']

        logger_info('df_metadata:', df_metadata)
        logger_info('df_optimize_info:', df_optimize_info)
        logger_info('df_data:', df_data.groupby(['source', 'model']).tail(60))

        a_y_result = df_data.loc[df_data.model == 'naive'].y.values
        logger_info('a_y_result:', a_y_result)
        self.assert_array_equal(
            a_y_result,
            np.concatenate([np.array(
                [0., 0., 1., 2., 3., 4., 4., 4., 7., 8., 9., ]),
                np.full(9, 9.)]))

        df_forecast = dict_result['forecast']
        logger_info('df_forecast', df_forecast)

        logger.info('Test 3b - initial sample is 0-weight, '
                    'extrapolate_years=0')
        df1.weight[0:2] = 0.
        logger_info('df1:', df1)

        dict_result = run_forecast(
            simplify_output=False, df_y=df1,
            l_model_trend=[forecast_models.model_naive],
            extrapolate_years=0)

        df_data = dict_result['data']
        df_metadata = dict_result['metadata']
        df_optimize_info = dict_result['optimize_info']

        logger_info('df_metadata:', df_metadata)
        logger_info('df_optimize_info:', df_optimize_info)
        logger_info('df_data:', df_data.groupby(['source', 'model']).tail(60))

        logger.info('Test 3c: weight column, season_add_mult = \'both\'')

        df1 = pd.DataFrame(
            {'y': np.arange(0, 10.),
             'weight': array_zeros_in_indices(10, [5, 6])},  # noqa
            index=pd.date_range('2014-01-01', periods=10, freq='D'))  # noqa
        dict_result = run_forecast(
            simplify_output=False, df_y=df1,
            l_model_trend=[forecast_models.model_naive],  # noqa
            extrapolate_years=10. / 365,
            season_add_mult='both')

        df_data = dict_result['data']
        df_metadata = dict_result['metadata']
        df_optimize_info = dict_result['optimize_info']

        logger_info('df_metadata:', df_metadata)
        logger_info('df_optimize_info:', df_optimize_info)
        logger_info('df_data:', df_data.groupby(['source', 'model']).tail(60))

        a_y_result = df_data.loc[df_data.model == 'naive'].y.values
        logger_info('a_y_result:', a_y_result)
        self.assert_array_equal(
            a_y_result,
            np.concatenate([
                np.array([0., 0., 1., 2., 3., 4., 4., 4., 7., 8., 9., ]),
                np.full(9, 9.)
            ]))

        df_forecast = dict_result['forecast']
        logger_info('df_forecast', df_forecast)

        logger.info('Test 4: find_outliers')

        df1 = pd.DataFrame(
            {'y': np.arange(0, 10.) + 10 * array_ones_in_indices(10, [5, 6])},
            index=pd.date_range('2014-01-01', periods=10, freq='D'))
        dict_result = run_forecast(
            simplify_output=False, df_y=df1,
            l_model_trend=[forecast_models.model_naive],
            extrapolate_years=10. / 365, find_outliers=True)

        df_data = dict_result['data']
        df_metadata = dict_result['metadata']
        df_optimize_info = dict_result['optimize_info']

        logger_info('df_metadata:', df_metadata)
        logger_info('df_optimize_info:', df_optimize_info)
        logger_info('df_data:',
                    df_data.groupby(['source', 'model']).tail(60))  # noqa

        a_y_result = df_data.loc[df_data.model == 'naive'].y.values
        logger_info('a_y_result:', a_y_result)
        self.assert_array_equal(
            a_y_result,
            np.concatenate([
                np.array([0., 0., 1., 2., 3., 4., 4., 4., 7., 8., 9., ]),
                np.full(9, 9.)
            ]))

        df_forecast = dict_result['forecast']
        logger_info('df_forecast', df_forecast)

        logger.info('Test 4b: find_outliers, season_add_mult = \'both\'')

        df1 = pd.DataFrame(
            {'y': np.arange(0, 10.) + 10 * array_ones_in_indices(10, [5, 6])},
            index=pd.date_range('2014-01-01', periods=10, freq='D'))
        dict_result = run_forecast(
            simplify_output=False, df_y=df1,
            l_model_trend=[forecast_models.model_naive],
            extrapolate_years=10. / 365, find_outliers=True,
            season_add_mult='both')

        df_data = dict_result['data']
        df_metadata = dict_result['metadata']
        df_optimize_info = dict_result['optimize_info']

        logger_info('df_metadata:', df_metadata)
        logger_info('df_optimize_info:', df_optimize_info)
        logger_info('df_data:', df_data.groupby(['source', 'model']).tail(60))

        a_y_result = df_data.loc[df_data.model == 'naive'].y.values
        logger_info('a_y_result:', a_y_result)
        self.assert_array_equal(
            a_y_result,
            np.concatenate([
                np.array([0., 0., 1., 2., 3., 4., 4., 4., 7., 8., 9., ]),
                np.full(9, 9.)
            ]
            ))

        df_forecast = dict_result['forecast']
        logger_info('df_forecast', df_forecast)

        logger.info('Test 5: Series with gap')

        df1 = (
            pd.DataFrame(
                {'y': np.arange(0, 10.),
                 # 'weight': array_zeros_in_indices(10, [5, 6]),
                 'date': pd.date_range('2014-01-01', periods=10, freq='D')},
            )
        )

        df1 = pd.concat(
            [df1.head(5), df1.tail(3)],
            sort=False, ignore_index=False
        ).pipe(normalize_df)

        dict_result = run_forecast(
            simplify_output=False, df_y=df1,
            l_model_trend=[],
            l_model_naive=[forecast_models.model_naive,
                           forecast_models.model_snaive_wday],
            extrapolate_years=10. / 365,
            season_add_mult='both')

        df_data = dict_result['data']
        df_metadata = dict_result['metadata']
        df_optimize_info = dict_result['optimize_info']

        logger_info('df_metadata:', df_metadata)
        logger_info('df_optimize_info:', df_optimize_info)
        logger_info('df_data:', df_data.groupby(['source', 'model']).tail(60))

        a_y_result = df_data.loc[df_data.model == 'naive'].y.values
        logger_info('a_y_result:', a_y_result)
        self.assert_array_equal(
            a_y_result,
            np.concatenate([
                np.array([0., 0., 1., 2., 3., 4., 4., 4., 7., 8., 9., ]),
                np.full(9, 9.)
            ]))

        df_forecast = dict_result['forecast']
        logger_info('df_forecast', df_forecast)

        logger.info('Test 6: Series with spike, '
                    'find_outliers=True, use model_snaive_wday')
        df1 = (
            pd.DataFrame(
                {'y': np.arange(0, 21.) + 10 * array_ones_in_indices(21, 7),
                 # 'weight': array_zeros_in_indices(10, [5, 6]),
                 'date': pd.date_range('2014-01-01', periods=21, freq='D')},
            ))
        # array_ones_in_indices(n, l_indices)
        dict_result = run_forecast(
            simplify_output=False,
            df_y=df1,
            l_model_trend=[],
            l_model_season=[],
            l_model_naive=[
                forecast_models.model_snaive_wday],
            extrapolate_years=20. / 365,
            season_add_mult='both',
            find_outliers=True)

        df_data = dict_result['data']
        df_metadata = dict_result['metadata']
        df_optimize_info = dict_result['optimize_info']

        logger_info('df_metadata:', df_metadata)
        logger_info('df_optimize_info:', df_optimize_info)
        df_data['wday'] = df_data.date.dt.weekday
        logger_info('df_data:', df_data.groupby(['source', 'model']).tail(60))

        a_y_result = df_data.loc[df_data.model == 'snaive_wday'].y.values
        logger_info('a_y_result:', a_y_result)
        self.assert_array_equal(
            a_y_result,
            np.array([0., 1., 2., 3., 4., 5., 6., 0., 8., 9., 10., 11., 12.,
                      13., 14., 15., 16., 17., 18., 19., 20., 14., 15., 16.,
                      17., 18., 19., 20., 14., 15., 16., 17., 18., 19.])
        )

        df_forecast = dict_result['forecast']
        logger_info('df_forecast', df_forecast)

    def test_run_forecast_naive2(self):
        logger.info('Test 1 - run forecast with naive model, find outliers')
        # Test 1: run forecast with  naive model, find_outliers,
        # season_add_mult = 'add', weekly samples
        path_df_naive = os.path.join(base_folder, 'df_test_naive.csv')
        df_test_naive = pd.read_csv(path_df_naive)

        l_season_yearly = [
            forecast_models.model_season_month,
            # model_season_fourier_yearly,
            forecast_models.model_null]

        l_season_weekly = [  # forecast_models.model_season_wday_2,
            forecast_models.model_season_wday, forecast_models.model_null]

        dict_result = run_forecast(
            simplify_output=False, df_y=df_test_naive,
            # l_model_trend=[forecast_models.model_naive],
            l_model_naive=[forecast_models.model_naive],
            l_season_yearly=l_season_yearly,
            l_season_weekly=l_season_weekly,
            extrapolate_years=75. / 365, find_outliers=True,
            season_add_mult='add')

        df_data = dict_result['data']
        df_metadata = dict_result['metadata']
        df_optimize_info = dict_result['optimize_info']

        logger_info('df_metadata:', df_metadata)
        logger_info('df_optimize_info:', df_optimize_info)
        logger_info('df_data:', df_data.loc[
            (df_data.date > '2017-12-01') & (df_data.date < '2018-02-01')])

        a_y_result = df_data.loc[df_data.model == 'naive'].y.values
        # logger_info('a_y_result:', a_y_result)

        df_forecast = dict_result['forecast']
        logger_info('df_forecast', df_forecast.loc[
            (df_forecast.date > '2017-12-01') & (
                    df_forecast.date < '2018-02-01')])

        # After first spike, naive forecast and actuals start matching,
        # only if season_add_mult='both'
        self.assertNotEqual(
            df_data.loc[(df_data.date == '2018-01-07') &
                        (df_data.model == 'naive')].y.iloc[0],
            df_data.loc[(df_data.date == '2018-01-07') &
                        (df_data.model == 'actuals')].y.iloc[0])

        logger.info('Test 2 - run forecast with  naive model, find_outliers,'
                    'season_add_mult = \'both\', weekly samples')
        # path_df_naive = os.path.join(base_folder, 'df_test_naive.csv')
        # df_test_naive = pd.read_csv(path_df_naive)

        l_season_yearly = [
            forecast_models.model_season_month,
            # model_season_fourier_yearly,
            forecast_models.model_null]

        l_season_weekly = [  # forecast_models.model_season_wday_2,
            forecast_models.model_season_wday, forecast_models.model_null]

        dict_result = run_forecast(
            simplify_output=False, df_y=df_test_naive,
            # l_model_trend=[forecast_models.model_naive],
            l_model_naive=[forecast_models.model_naive],
            l_season_yearly=l_season_yearly,
            l_season_weekly=l_season_weekly,
            extrapolate_years=75. / 365, find_outliers=True,
            season_add_mult='both')

        df_data = dict_result['data']
        df_metadata = dict_result['metadata']
        df_optimize_info = dict_result['optimize_info']

        logger_info('df_metadata:', df_metadata)
        logger_info('df_optimize_info:', df_optimize_info)
        logger_info('df_data:', df_data.loc[
            (df_data.date > '2017-12-01') & (df_data.date < '2018-02-01')])

        a_y_result = df_data.loc[df_data.model == 'naive'].y.values
        # logger_info('a_y_result:', a_y_result)

        df_forecast = dict_result['forecast']
        logger_info('df_forecast', df_forecast.loc[
            (df_forecast.date > '2017-12-01') & (
                df_forecast.date < '2018-02-01')])

        # After first spike, naive forecast and actuals start matching, only if
        # season_add_mult='both'
        self.assertNotEqual(
            df_data.loc[(df_data.date == '2018-01-07') &
                        (df_data.model == 'naive')].y.iloc[0],
            df_data.loc[(df_data.date == '2018-01-07') &
                        (df_data.model == 'actuals')].y.iloc[0])

        logger.info('Test 3 - multiple model_naive runs')
        path_df_naive = os.path.join(base_folder, 'df_test_naive.csv')
        df_test_naive = pd.read_csv(path_df_naive)

        model_naive2 = forecast_models.ForecastModel(
            'naive2', 0, forecast_models._f_model_naive)

        l_model_naive = [forecast_models.model_naive, model_naive2]

        dict_result = run_forecast(
            simplify_output=False,
            df_y=df_test_naive,
            l_model_trend=[],
            l_season_yearly=l_season_yearly,
            l_season_weekly=l_season_weekly,
            l_model_naive=l_model_naive,
            extrapolate_years=75. / 365,
            find_outliers=True,
            season_add_mult='add',
        )

        df_data = dict_result['data']
        df_metadata = dict_result['metadata']
        df_optimize_info = dict_result['optimize_info']

        logger_info('df_metadata:', df_metadata)
        logger_info('df_optimize_info:', df_optimize_info)
        logger_info('df_data:', df_data.loc[
            (df_data.date > '2017-12-01') & (df_data.date < '2018-02-01')])

        a_y_result = df_data.loc[df_data.model == 'naive'].y.values
        # logger_info('a_y_result:', a_y_result)

        df_forecast = dict_result['forecast']
        logger_info('df_forecast', df_forecast.loc[
            (df_forecast.date > '2017-12-01') & (
                df_forecast.date < '2018-02-01')])

        # After first spike, naive forecast and actuals start matching, only if
        # season_add_mult='both'
        self.assertNotEqual(
            df_data.loc[(df_data.date == '2018-01-07') &
                        (df_data.model == 'naive')].y.iloc[0],
            df_data.loc[(df_data.date == '2018-01-07') &
                        (df_data.model == 'actuals')].y.iloc[0])

    def test_run_forecast_sparse_with_gaps(self):
        df_test = pd.DataFrame({'date': pd.to_datetime(
            ['2018-08-01', '2018-08-09']), 'y': [1., 2.]})
        df_out = run_forecast(df_test, extrapolate_years=1.0)
        logger_info('df_out', df_out)

    def test_run_forecast_output_options(self):
        freq = 'D'
        freq_short = freq[0:1]  # Changes e.g. W-MON to W
        # Todo: change to dict to support more frequencies
        freq_units_per_year = 52.0 if freq_short == 'W' else 365.0

        df_y = pd.DataFrame({'y': np.full(100, 0.0)}, index=pd.date_range(
            '2014-01-01', periods=100, freq=freq))

        # SolverConfig with trend
        conf1 = ForecastInput(
            source_id='source1',
            l_model_trend=[
                forecast_models.model_linear,
                forecast_models.model_constant],
            l_model_season=None,
            df_y=df_y,
            date_start_actuals=None)

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
            logger_info('df_data:', df_data.groupby(
                ['source', 'model']).tail(1))
            # TODO: ADD ASSERTS

    def test_run_forecast_step(self):
        # Setup
        freq = 'D'
        df_y1 = pd.DataFrame({'y': 5 * [10.0] + 5 * [20.0]},
                             index=pd.date_range('2014-01-01', periods=10,
                                                 freq=freq))

        # SolverConfig with trend
        conf1 = ForecastInput(
            source_id='source1',
            l_model_trend=[
                forecast_models.model_constant,
                forecast_models.model_constant +
                forecast_models.model_step],
            l_model_season=None,
            df_y=df_y1,
            weights_y_values=1.0,
            date_start_actuals=None)

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
                             index=pd.date_range('2014-01-01',
                                                 periods=10, freq=freq))

        # SolverConfig with trend
        conf1 = ForecastInput(
            source_id='source1',
            l_model_trend=[
                forecast_models.model_constant +
                forecast_models.model_two_steps],
            l_model_season=None,
            df_y=df_y1,
            weights_y_values=1.0,
            date_start_actuals=None)

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
        df_y1 = pd.DataFrame({'y': [10., 10.1, 10.2, 10.3, 10.4, 20.0, 20.1,
                                    20.2, 20.3, 20.4, 20.5, 20.6]},
                             index=pd.date_range('2014-01-01', periods=12,
                                                 freq=freq))

        # SolverConfig with trend
        conf1 = ForecastInput(
            source_id='source1',
            l_model_trend=[
                forecast_models.model_constant,
                forecast_models.model_sigmoid_step,
                forecast_models.model_constant +
                forecast_models.model_sigmoid_step,
                forecast_models.model_linear +
                forecast_models.model_sigmoid_step,
                forecast_models.model_linear *
                forecast_models.model_sigmoid_step],
            l_model_season=None,
            df_y=df_y1,
            weights_y_values=1.0,
            date_start_actuals=None)

        dict_result = run_l_forecast([conf1], include_all_fits=True)
        df_data = dict_result['data']
        df_metadata = dict_result['metadata']
        df_optimize_info = dict_result['optimize_info']

        logger_info('df_metadata:', df_metadata)
        logger_info('df_optimize_info:', df_optimize_info)
        logger_info('df_data:', df_data.groupby(['source', 'model']).tail(1))

        # Same with negative step
        df_y1 = pd.DataFrame({'y': [20.0, 20.1, 20.2, 20.3, 20.4, 20.5,
                                    20.6, 10., 10.1, 10.2, 10.3, 10.4]},
                             index=pd.date_range('2014-01-01', periods=12,
                                                 freq=freq))

        conf1 = ForecastInput(
            source_id='source1',
            l_model_trend=[
                forecast_models.model_constant,
                forecast_models.model_sigmoid_step,
                forecast_models.model_constant +
                forecast_models.model_sigmoid_step,
                forecast_models.model_linear +
                forecast_models.model_sigmoid_step,
                forecast_models.model_linear *
                forecast_models.model_sigmoid_step],
            l_model_season=None,
            df_y=df_y1,
            weights_y_values=1.0,
            date_start_actuals=None)

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
            l_model_season=[
                forecast_models.model_null],
            df_y=df_y,
            weights_y_values=1.0,
            date_start_actuals=None)
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
        if platform.system() != 'Darwin':
            # matplotlib tests don't work on mac
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
            l_model_season=[
                forecast_models.model_null],
            df_y=df_y,
            weights_y_values=1.0,
            date_start_actuals=None)
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
        if platform.system() != 'Darwin':
            # matplotlib tests don't work on mac
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
                model = forecast_models.model_linear * forecast_models.model_sigmoid  # noqa
                y_lin = a_lin * a_x + b_lin
                y_in = sigmoid(a_x, a_in, b_in, c_in, d_in) * y_lin
                input_params = [a_lin, b_lin]
                y_rand = np.random.uniform(
                    low=0.001, high=0.1 * b_in, size=len(a_x)) * y_lin
            else:
                a_in = 30  # the constant
                model = forecast_models.model_constant + forecast_models.model_sigmoid  # noqa
                y_in = sigmoid(a_x, a_in, b_in, c_in, d_in)
                input_params = [a_in]
                y_rand = np.random.uniform(
                    low=0.001, high=0.1 * b_in, size=len(a_x))

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
                    # forecast_models.model_linear + forecast_models.model_sigmoid, # noqa
                    # forecast_models.model_linear * forecast_models.model_sigmoid # noqa
                ],
                l_model_season=None, df_y=df_y, weights_y_values=1.0,
                date_start_actuals=None
            )

            dict_result = run_l_forecast([conf1],
                                         include_all_fits=True)
            df_data = dict_result['data']
            df_metadata = dict_result['metadata']
            # df_optimize_info = dict_result['optimize_info']

            df = df_data.loc[:, ['y', 'date', 'model']]

            df = df.pivot(values='y', columns='model', index='date')
            if platform.system() != 'Darwin':
                # matplotlib tests don't work on mac
                df.plot()
            output_params = df_metadata.loc[df_metadata.is_best_fit,
                                            'params_str']
            logger.info('Input parameters: %s, Output parameters: %s',
                        input_params, output_params.iloc[0])
            pass  # to see the plot

    def test_run_forecast_get_outliers(self):

        # Test 1 - no outliers
        a_y = [20.0, 20.1, 20.2, 20.3, 20.4, 20.5]
        a_date = pd.date_range(start='2018-01-01', periods=len(a_y), freq='D')
        df = pd.DataFrame({'y': a_y})

        dict_result = run_forecast(
            df,
            find_outliers=True,
            simplify_output=False,
            include_all_fits=True,
            season_add_mult='add')
        logger_info('Metadata', dict_result['metadata'])
        logger_info('data', dict_result['data'].tail(3))

        # Check that dtype of y is not corrupted by None values from weight
        # mask - this happens when no spikes found
        self.assertTrue(np.issubdtype(dict_result['data'].y, np.float64))

        # Test 2 - Single step
        a_y = [19.8, 19.9, 20.0, 20.1, 20.2, 20.3, 20.4, 20.5,
               20.6, 10., 10.1, 10.2, 10.3, 10.4,
               10.5, 10.6, 10.7, 10.8, 10.9]
        a_date = pd.date_range(start='2018-01-01', periods=len(a_y), freq='D')
        df = pd.DataFrame({'y': a_y})

        dict_result = run_forecast(
            df,
            find_outliers=True,
            simplify_output=False,
            include_all_fits=True,
            season_add_mult='add')
        logger_info('Metadata', dict_result['metadata'])
        logger_info('data', dict_result['data'].tail(3))

        # Check that dtype of y is not corrupted by None values from weight
        # mask - this happens when no spikes found
        self.assertTrue(np.issubdtype(dict_result['data'].y, np.float64))

        # Test 3 - Single spike

        a_y = [19.8, 19.9, 20.0, 20.1, 20.2, 20.3, 20.4, 20.5,
               20.6, 10., 20.7, 20.8, 20.9, 21.0,
               21.1, 21.2, 21.3, 21.4, 21.5]
        a_date = pd.date_range(start='2018-01-01', periods=len(a_y), freq='D')
        df_spike = pd.DataFrame({'y': a_y})

        dict_result = run_forecast(
            df_spike,
            find_outliers=True,
            simplify_output=False,
            include_all_fits=True,
            season_add_mult='add')
        df_data = dict_result['data']
        mask = df_data.loc[df_data.model == 'actuals'].weight
        self.assert_array_equal(
            mask, [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, ])

        # Test 5 - 2 spikes and 1 step
        a_y = [19.8, 19.9, 30.0, 30.1, 20.2, 20.3, 20.4, 20.5,
               20.6, 10., 10.1, 10.2, 10.3, 10.4,
               10.5, 10.6, 30.7, 10.8, 10.9]

        df = pd.DataFrame({'y': a_y}).pipe(normalize_df)

        dict_result = run_forecast(
            df,
            find_outliers=True,
            simplify_output=False,
            include_all_fits=True,
            season_add_mult='add')
        logger_info('Metadata', dict_result['metadata'])
        df_result = dict_result['data']
        logger_info('data', df_result.tail(3))
        mask = df_result.loc[df_result.model == 'actuals'].weight
        self.assert_array_equal(
            mask, [1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1])

    def test_run_forecast_auto_season(self):
        """Test run_forecast with automatic seasonality detection"""

        # Yearly sinusoidal function

        # With daily samples
        length = 2 * 365
        # size will be +-10 +- uniform error
        a_date = pd.date_range(start='2018-01-01', freq='D', periods=length)
        a_y = (10 + np.random.uniform(low=0, high=1, size=length) +
               10 * (np.sin(np.linspace(-4 * np.pi, 4 * np.pi, length))))
        df_y = pd.DataFrame({'y': a_y}, index=a_date)

        dict_result = run_forecast(
            df_y,
            season_add_mult='add',
            simplify_output=False,
            include_all_fits=True,
            l_model_trend=[forecast_models.model_linear],
            l_model_naive=[]
        )
        df_metadata = dict_result['metadata']

        l_model_expected = sorted(
            ['linear',
             '(linear+(season_wday+season_fourier_yearly))',
             '(linear+season_wday)',
             '(linear+season_fourier_yearly)'])
        self.assert_array_equal(df_metadata.model, l_model_expected)
        logger_info('df_metadata:', df_metadata)

        # As above, with additive and multiplicative seasonality

        dict_result = run_forecast(
            df_y,
            season_add_mult='both',
            simplify_output=False,
            include_all_fits=True,
            l_model_trend=[forecast_models.model_linear],
            l_model_naive=[]
        )
        df_metadata = dict_result['metadata']

        l_model_expected = np.array([
            'linear',
            '(linear+season_fourier_yearly)',
            '(linear+(season_wday+season_fourier_yearly))',
            '(linear+season_wday)',
            '(linear*season_fourier_yearly)',
            '(linear*season_wday)',
            '(linear*(season_wday*season_fourier_yearly))',
        ])
        l_model_expected.sort()

        self.assert_array_equal(df_metadata.model.values, l_model_expected)
        logger_info('df_metadata:', df_metadata)

    def test_run_forecast_auto_composition(self):
        """Test run_forecast with automatic model composition detection"""

        np.random.seed(1)  # Ensure predictable test results

        logger.info('Test 1 - detect additive model')

        # With daily samples
        length = 2 * 365
        # size will be 100 +-10 +- uniform error
        a_date = pd.date_range(start='2018-01-01', freq='D', periods=length)
        a_y = (100 + np.random.uniform(low=0, high=1, size=length) +
               10 * (np.sin(np.linspace(-4 * np.pi, 4 * np.pi, length))))
        df_y = pd.DataFrame({'y': a_y}, index=a_date)

        dict_result = run_forecast(
            df_y,
            season_add_mult='auto',
            simplify_output=False,
            include_all_fits=True,
            l_model_trend=[forecast_models.model_linear],
            l_model_naive=[]
        )
        df_metadata = dict_result['metadata']

        l_model_expected = sorted(
            ['linear',
             '(linear+(season_wday+season_fourier_yearly))',
             '(linear+season_wday)',
             '(linear+season_fourier_yearly)'])
        self.assert_array_equal(df_metadata.model, l_model_expected)
        logger_info('df_metadata:', df_metadata)

        logger.info('Test 2 - detect multiplicative model')

        # With daily samples
        length = 2 * 365
        # size will be +-10 +- uniform error
        a_date = pd.date_range(start='2018-01-01', freq='D', periods=length)
        a_y = (1000 +
               0.1 * np.arange(length) *
               (1 + (np.sin(np.linspace(-40 * np.pi, 40 * np.pi, length)))))
        df_y = pd.DataFrame({'y': a_y}, index=a_date)

        dict_result = run_forecast(
            df_y,
            season_add_mult='auto',
            simplify_output=False,
            include_all_fits=True,
            l_model_trend=[forecast_models.model_linear],
            l_model_naive=[]
        )
        df_metadata = dict_result['metadata']

        l_model_expected = sorted(
            ['linear',
             '(linear*(season_wday*season_fourier_yearly))',
             '(linear*season_wday)',
             '(linear*season_fourier_yearly)'])
        self.assert_array_equal(df_metadata.model, l_model_expected)
        logger_info('df_metadata:', df_metadata)

    def test_run_forecast_with_weight(self):
        df1 = pd.DataFrame({'y': np.arange(0, 10.), 'date': pd.date_range(
            '2014-01-01', periods=10, freq='D'), 'weight': 1.})
        dict_result = run_forecast(
            simplify_output=False,
            df_y=df1,
            l_model_trend=[
                forecast_models.model_linear],
            extrapolate_years=10. / 365,
            l_model_naive=[]
        )

        df_forecast = dict_result['forecast']
        df_data = dict_result['data']
        df_metadata = dict_result['metadata']
        df_optimize_info = dict_result['optimize_info']

        logger_info('df_forecast:', df_forecast.groupby(
            ['source', 'model']).tail(30))
        logger_info('df_metadata:', df_metadata)
        logger_info('df_optimize_info:', df_optimize_info)
        logger_info('df_data:', df_data.groupby(['source', 'model']).tail(30))

        df_forecast_filtered = df_forecast.loc[~df_forecast.is_actuals & (
            df_forecast.date > '2014-01-10')]
        self.assert_series_equal(
            df_forecast_filtered.y,
            df_forecast_filtered.q5)

        df1b = df1.copy()
        df1b.loc[0, 'weight'] = 0.

        dict_result = run_forecast(
            simplify_output=False,
            df_y=df1b,
            l_model_trend=[
                forecast_models.model_linear],
            extrapolate_years=10. / 365,
            l_model_naive=[]
        )

        df_forecast = dict_result['forecast']
        df_data = dict_result['data']
        df_metadata = dict_result['metadata']
        df_optimize_info = dict_result['optimize_info']

        logger_info('df_forecast:', df_forecast.groupby(
            ['source', 'model']).tail(30))
        logger_info('df_metadata:', df_metadata)
        logger_info('df_optimize_info:', df_optimize_info)
        logger_info('df_data:', df_data.groupby(['source', 'model']).tail(30))

        len_forecast = df_data.loc[~df_data.is_actuals].index.size
        # First sample shouldn't be included due to weight=0
        self.assertEqual(len_forecast, 19)

        # Since fit is perfect, prediction interval should be equal to point
        # forecast
        df_forecast_filtered = df_forecast.loc[~df_forecast.is_actuals & (
            df_forecast.date > '2014-01-10')]
        self.assert_series_equal(
            df_forecast_filtered.y,
            df_forecast_filtered.q5)

        # Test with model_ramp
        # Param A of model_ramp needs to be within the 15-85 percentile
        # of valid x values
        # Before a bugfix, we would get initial guesses of A=2,
        # with boundaries (5.6, 8.4)
        # Note: somehow validate bounds doesn't catch this!

        df1c = df1.copy()
        df1c.loc[0:4, 'weight'] = 0.

        dict_result = run_forecast(
            simplify_output=False,
            df_y=df1c,
            l_model_trend=[
                forecast_models.model_ramp],
            extrapolate_years=10. / 365,
            l_model_naive=[]
        )

        df_forecast = dict_result['forecast']
        df_data = dict_result['data']
        df_metadata = dict_result['metadata']
        df_optimize_info = dict_result['optimize_info']

        logger_info('df_forecast:', df_forecast.groupby(
            ['source', 'model']).tail(30))
        logger_info('df_metadata:', df_metadata)
        logger_info('df_optimize_info:', df_optimize_info)
        logger_info('df_data:', df_data.groupby(['source', 'model']).tail(30))

        len_forecast = df_data.loc[~df_data.is_actuals].index.size
        # First 5 samples shouldn't be included due to weight=0
        self.assertEqual(len_forecast, 15)

        # # Since fit is perfect, prediction interval should be equal to
        # point forecast
        # df_forecast_filtered = df_forecast.loc[~df_forecast.is_actuals &
        #   (df_forecast.date>'2014-01-10')]
        # self.assert_series_equal(df_forecast_filtered.y,
        #   df_forecast_filtered.q5)

    def test_detect_freq(self):

        # Initial test - what happens with single sample input?
        a_date = pd.a_date = pd.date_range('2014-01-01', periods=1, freq='H')
        result = detect_freq(a_date)
        # self.assertEqual(result, 'H')

        a_date = pd.a_date = pd.date_range(
            '2014-01-01', periods=24 * 7, freq='H')
        result = detect_freq(a_date)
        self.assertEqual(result, 'H')

        a_date = pd.a_date = pd.date_range(
            '2014-01-01', periods=4 * 365, freq='D')
        result = detect_freq(a_date)
        self.assertEqual(result, 'D')

        l_freq_wday = [
            'W-MON',
            'W-TUE',
            'W-WED',
            'W-THU',
            'W-FRI',
            'W-SAT',
            'W-SUN']
        for freq_wday in l_freq_wday:
            a_date = pd.a_date = pd.date_range(
                '2014-01-01', periods=4 * 52, freq=freq_wday)
            result = detect_freq(a_date)
            self.assertEqual(result, freq_wday)

        a_date = pd.a_date = pd.date_range(
            '2014-01-01', periods=4 * 12, freq='M')
        result = detect_freq(a_date)
        self.assertEqual(result, 'M')

        a_date = pd.a_date = pd.date_range(
            '2014-01-01', periods=4 * 12, freq='MS')
        result = detect_freq(a_date)
        self.assertEqual(result, 'MS')

        a_date = pd.a_date = pd.date_range(
            '2014-01-01', periods=4 * 12, freq='Q')
        result = detect_freq(a_date)
        self.assertEqual(result, 'Q')

        a_date = pd.a_date = pd.date_range(
            '2014-01-01', periods=4 * 12, freq='Y')
        result = detect_freq(a_date)
        self.assertEqual(result, 'Y')

        # Test with input dataframe

        a_date = pd.a_date = pd.date_range(
            '2014-01-01', periods=24 * 7, freq='H')
        df_y = pd.DataFrame({'date': a_date})
        result = detect_freq(df_y)
        self.assertEqual(result, 'H')

        a_date = pd.a_date = pd.date_range(
            '2014-01-01', periods=4 * 365, freq='D')
        df_y = pd.DataFrame({'date': a_date})
        result = detect_freq(df_y)
        self.assertEqual(result, 'D')

        a_date = pd.a_date = pd.date_range(
            '2014-01-01', periods=4 * 12, freq='M')
        df_y = pd.DataFrame({'date': a_date})
        result = detect_freq(df_y)
        self.assertEqual(result, 'M')

        a_date = pd.a_date = pd.date_range(
            '2014-01-01', periods=4 * 12, freq='Q')
        df_y = pd.DataFrame({'date': a_date})
        result = detect_freq(df_y)
        self.assertEqual(result, 'Q')

        a_date = pd.a_date = pd.date_range(
            '2014-01-01', periods=4 * 12, freq='Y')
        df_y = pd.DataFrame({'date': a_date})
        result = detect_freq(df_y)
        self.assertEqual(result, 'Y')

        a_date = pd.a_date = pd.date_range(
            '2014-01-01', periods=4 * 12, freq='YS')
        df_y = pd.DataFrame({'date': a_date})
        result = detect_freq(df_y)
        self.assertEqual(result, 'YS')

        # Test with sparse input series
        a_date = pd.to_datetime(['2018-08-01', '2018-08-09'])
        df_y = pd.DataFrame({'date': a_date})
        result = detect_freq(df_y)
        self.assertEqual(result, 'D')

    # TODO: ADD TEST WITH NULL VALUES, E.G. MODEL_NAIVE_WDAY
    def test_get_pi(self):

        def check_result(df_result):
            self.assertTrue('q5' in df_result.columns)
            df_result_actuals = df_result.loc[df_result.is_actuals]
            if 'weight' in df_result_actuals.columns:
                df_result_actuals = df_result_actuals.loc[
                    ~df_result_actuals.is_weight]
            date_max_actuals = df_result_actuals.date.max()
            logger_info('debug: date max actuals', date_max_actuals)

            df_result_forecast = df_result.loc[
                ~df_result.is_actuals & (df_result.date > date_max_actuals)]
            self.assertFalse(df_result_forecast.q5.isnull().any())

        # First test with single source
        # then test applied function on df grouped by source
        logger.info('Test 1a - Single source')
        a_date_actuals = pd.date_range('2014-01-01', periods=10, freq='W')
        a_y_actuals = np.arange(0, 10.)
        df_actuals = (
            pd.DataFrame({'date': a_date_actuals, 'y': a_y_actuals,
                          'source': 's1', 'is_actuals': True,
                          'is_best_fit': False, 'model': 'actuals'})
        )

        a_date = pd.date_range('2014-01-01', periods=20, freq='W')
        a_y = np.arange(0, 20.) + (
            np.tile([-1, 1], 10) * np.arange(2, 0., -0.1))

        df_fcast = (pd.DataFrame({'date': a_date,
                                  'y': a_y,
                                  'source': 's1',
                                  'is_actuals': False,
                                  'is_best_fit': True,
                                  'model': 'linear'}))

        df1 = pd.concat([df_actuals, df_fcast], ignore_index=True, sort=False)

        df_result = get_pi(df1, n_sims=100)
        logger_info('df_result1:', df_result.groupby(
            ['source', 'model']).head(2))
        logger_info('df_result1:', df_result.groupby(
            ['source', 'model']).tail(2))
        # TODO: Add checks
        check_result(df_result)

        logger.info('Test 1b - n_cum>1')
        df_result = get_pi(df1, n_sims=100, n_cum=7)
        logger_info('df_result1:', df_result.groupby(
            ['source', 'model']).head(2))
        logger_info('df_result1:', df_result.groupby(
            ['source', 'model']).tail(2))
        # TODO: Add checks
        check_result(df_result)

        # Test 1c - input dataframe without is_best_fit column, source column
        logger.info('Test 1c - Single source, no source column')
        df1c = df1[['date', 'is_actuals', 'model', 'y']]
        df_result = get_pi(df1c, n_sims=100)
        # logger_info('df_result1:', df_result1)
        logger_info('df_result1:', df_result.groupby(
            ['model']).head(2))
        logger_info('df_result1:', df_result.groupby(
            ['model']).tail(2))

        check_result(df_result)

        logger.info('Test 2 - 2 sources')
        df1b = df1.copy()
        df1b.source = 's2'
        df2 = pd.concat([df1, df1b], sort=False)

        df_result = get_pi(df2, n_sims=100)
        # logger_info('df_result2:', df_result2)
        logger_info('df_result:', df_result.groupby(
            ['source', 'model']).head(2))
        logger_info('df_result:', df_result.groupby(
            ['source', 'model']).tail(2))
        # TODO: Add checks
        check_result(df_result)

        logger.info('Test 2b - n_cum>1')
        df_result = get_pi(df2, n_sims=100, n_cum=5)
        # logger_info('df_result2:', df_result2)
        logger_info('df_result:', df_result.groupby(
            ['source', 'model']).head(2))
        logger_info('df_result:', df_result.groupby(
            ['source', 'model']).tail(2))
        # TODO: Add checks
        check_result(df_result)

        # Test 3 - Input has actuals but no forecast - can happen if fit not
        # possible
        logger.info('Test 3 - Input missing forecast')
        df3 = df_actuals
        df_result = get_pi(df3, n_sims=100)
        self.assertIsNotNone(df3)
        self.assertFalse('q5' in df_result.columns)
        # logger_info('df_result1:', df_result1)
        logger_info('df_result3:', df_result.groupby(
            ['source', 'model']).head(1))
        logger_info('df_result3:', df_result.groupby(
            ['source', 'model']).tail(1))

        logger.info('Test 4 - Input with nulls at end')
        a_date_actuals = pd.date_range('2014-01-01', periods=10, freq='W')
        a_y_actuals = np.arange(0, 10.)
        df_actuals = (
            pd.DataFrame({'date': a_date_actuals, 'y': a_y_actuals,
                          'source': 's1', 'is_actuals': True,
                          'is_best_fit': False, 'model': 'actuals'})
        )

        a_date = pd.date_range('2014-01-01', periods=20, freq='W')
        a_y = np.arange(0, 20.) + (
            np.tile([-1, 1], 10) * np.arange(2, 0., -0.1))

        df_fcast = (pd.DataFrame({'date': a_date,
                                  'y': a_y,
                                  'source': 's1',
                                  'is_actuals': False,
                                  'is_best_fit': True,
                                  'model': 'linear'}))

        df1 = pd.concat([df_actuals, df_fcast], ignore_index=True, sort=False)
        df_result = get_pi(df1, n_sims=100)

        a_date_actuals_withnull = pd.date_range(
            '2014-01-01', periods=20, freq='W')
        a_y_actuals_withnull = np.concatenate(
            [np.arange(0, 10.), np.full(10, np.NaN)])
        df_actuals_withnull = (
            pd.DataFrame({'date': a_date_actuals, 'y': a_y_actuals,
                          'source': 's1', 'is_actuals': True,
                          'is_best_fit': False, 'model': 'actuals'})
        )

        a_date_withnull = pd.date_range('2014-01-01', periods=20, freq='W')

        df1_withnull = pd.concat(
            [df_actuals_withnull, df_fcast], ignore_index=True, sort=False)
        df_result_withnull = get_pi(df1_withnull, n_sims=100)

        logger_info('df_result:', df_result.groupby(
            ['source', 'model']).tail(3))
        logger_info('df_result with null:', df_result_withnull.groupby(
            ['source', 'model']).tail(3))
        # Prediction intervals are random, so we need to exclude them from
        # comparison
        self.assert_frame_equal(df_result[['date',
                                           'source',
                                           'is_actuals',
                                           'model',
                                           'y']],
                                df_result_withnull[['date',
                                                    'source',
                                                    'is_actuals',
                                                    'model',
                                                    'y']])

        logger.info('Test 4b - Input with nulls at end, weight column')
        df_weight = (pd.DataFrame({'date': a_date,
                                   'y': 1,
                                   'source': 's1',
                                   'is_actuals': False,
                                   'is_best_fit': True,
                                   'model': 'linear',
                                   'weight': 1.0}))
        df_weight_withnull = (
            pd.DataFrame({'date': a_date_withnull, 'y': 1,
                          'source': 's1', 'is_actuals': False,
                          'is_best_fit': True, 'model': 'linear',
                          'weight': 1.0})
        )

        df1['weight'] = 1.
        df1_withnull['weight'] = 1.

        df1b = pd.concat([df1, df_weight], ignore_index=True, sort=False)
        df1b_withnull = pd.concat(
            [df1_withnull, df_weight_withnull], ignore_index=True, sort=False)

        df_result_b = get_pi(df1b, n_sims=100)
        df_result_b_withnull = get_pi(df1b_withnull, n_sims=100)

        logger_info('df_result b :', df_result_b.groupby(
            ['source', 'model']).tail(3))
        logger_info('df_result b with null:',
                    df_result_b_withnull.groupby(['source', 'model']).tail(3))
        # Prediction intervals are random, so we need to exclude them from
        # comparison
        self.assert_frame_equal(
            df_result_b[['date', 'source', 'is_actuals', 'model', 'y']],
            df_result_b_withnull[['date', 'source', 'is_actuals',
                                  'model', 'y']])

        check_result(df_result_b)
        check_result(df_result_b_withnull)

        logger.info('Test 4c - Input with nulls at start')
        a_date_actuals = pd.date_range('2014-01-01', periods=10, freq='W')
        a_y_actuals = np.arange(0, 10.)
        df_actuals = (
            pd.DataFrame({'date': a_date_actuals, 'y': a_y_actuals,
                          'source': 's1', 'is_actuals': True,
                          'is_best_fit': False, 'model': 'actuals'})
        )

        a_date = pd.date_range('2014-01-01', periods=20, freq='W')
        a_y = np.arange(0, 20.) + (
            np.tile([-1, 1], (10)) * np.arange(2, 0., -0.1))

        df_fcast = (pd.DataFrame({'date': a_date,
                                  'y': a_y,
                                  'source': 's1',
                                  'is_actuals': False,
                                  'is_best_fit': True,
                                  'model': 'linear'}))

        df1 = pd.concat([df_actuals, df_fcast], ignore_index=True, sort=False)
        df_result = get_pi(df1, n_sims=100)

        a_date_actuals_withnull = pd.date_range(
            '2014-01-01', periods=10, freq='W')
        a_y_actuals_withnull = np.concatenate(
            [np.full(5, np.NaN), np.arange(0, 5.)])
        df_actuals_withnull = (
            pd.DataFrame({'date': a_date_actuals_withnull,
                          'y': a_y_actuals_withnull,
                          'source': 's1', 'is_actuals': True,
                          'is_best_fit': False, 'model': 'actuals'})
        )

        a_date_withnull = pd.date_range('2014-01-01', periods=10, freq='W')

        df1_withnull = pd.concat(
            [df_actuals_withnull, df_fcast], ignore_index=True, sort=False)
        df_result_withnull = get_pi(df1_withnull, n_sims=100)

        logger_info('df_actuals_withnull:', df_actuals_withnull.groupby(
            ['source', 'model']).head(20))
        logger_info('df_result:', df_result.groupby(
            ['source', 'model']).tail(3))
        logger_info('df_result with null:', df_result_withnull.groupby(
            ['source', 'model']).tail(100))
        # todo - add proper expected value, uncomment assert
        # self.assert_frame_equal(df_result[['date', 'source', 'is_actuals',
        # 'model', 'y']],
        # df_result_withnull[['date', 'source', 'is_actuals', 'model', 'y']])

        logger.info('Test 4d - Input with nulls at start of forecast')
        a_date_actuals = pd.date_range('2014-01-01', periods=10, freq='W')
        a_y_actuals = np.arange(0, 10.)
        df_actuals = (
            pd.DataFrame({'date': a_date_actuals, 'y': a_y_actuals,
                          'source': 's1', 'is_actuals': True,
                          'is_best_fit': False, 'model': 'actuals'})
        )

        a_date = pd.date_range('2014-01-01', periods=20, freq='W')
        a_y = np.arange(0, 20.) + (
            np.tile([-1, 1], (10)) * np.arange(2, 0., -0.1))
        a_y_withnull = np.concatenate(
            [np.full(5, np.NaN), np.arange(0, 15.), ])

        df_fcast = (pd.DataFrame({'date': a_date,
                                  'y': a_y,
                                  'source': 's1',
                                  'is_actuals': False,
                                  'is_best_fit': True,
                                  'model': 'linear'}))

        df_fcast_withnull = (
            pd.DataFrame({'date': a_date, 'y': a_y_withnull,
                          'source': 's1', 'is_actuals': False,
                          'is_best_fit': True, 'model': 'linear'})
        )

        df1 = pd.concat([df_actuals, df_fcast], ignore_index=True, sort=False)
        df_result = get_pi(df1, n_sims=100)

        df1_withnull = pd.concat(
            [df_actuals, df_fcast_withnull], ignore_index=True, sort=False)
        df_result_withnull = get_pi(df1_withnull, n_sims=100)

        logger_info('df_fcast_withnull:', df_fcast_withnull.groupby(
            ['source', 'model']).head(20))
        logger_info('df_result:', df_result.groupby(
            ['source', 'model']).tail(100))
        logger_info('df_result with null:', df_result_withnull.groupby(
            ['source', 'model']).tail(100))
        # Prediction intervals are random,
        # so we need to exclude them from comparison
        # self.assert_frame_equal(
        # df_result[['date', 'source', 'is_actuals', 'model', 'y']],
        #  df_result_withnull[['date', 'source', 'is_actuals', 'model', 'y']])
        # TODO: ADD VALID CHECK -

    def test_get_pi_gap(self):
        def check_result(df_result):
            self.assertTrue('q5' in df_result.columns)

        logger.info('Test 1 - Input has gaps')
        a_date_actuals = pd.date_range('2014-01-01', periods=10, freq='W')
        a_y_actuals = np.arange(0, 10.)
        df_actuals = (
            pd.DataFrame({'date': a_date_actuals, 'y': a_y_actuals,
                          'source': 's1', 'is_actuals': True,
                          'is_best_fit': False, 'model': 'actuals'})
        )

        a_date = pd.date_range('2014-01-01', periods=20, freq='W')
        a_y = np.arange(0, 20.) + (
            np.tile([-1, 1], (10)) * np.arange(2, 0., -0.1))

        df_fcast = (pd.DataFrame({'date': a_date,
                                  'y': a_y,
                                  'source': 's1',
                                  'is_actuals': False,
                                  'is_best_fit': True,
                                  'model': 'linear'}))

        df_actuals_gap = pd.concat([df_actuals.head(3), df_actuals.tail(3)])

        df = pd.concat([df_actuals_gap, df_fcast],
                       ignore_index=True, sort=False)

        # df_result = get_pi(df, n_sims=100)
        # # logger_info('df_result1:', df_result1)
        # logger_info('df_result1:', df_result.groupby(
        #     ['source', 'model']).head(2))
        # logger_info('df_result1:', df_result.groupby(
        #     ['source', 'model']).tail(2))
        #
        # check_result(df_result)
        #
        # logger.info('Test 2 - Input has nulls')
        # df_actuals_null = df_actuals.copy()
        # df_actuals_null.loc[5, 'y'] = np.NaN
        #
        # logger_info('df_actuals_null:', df_actuals_null)
        #
        # df = pd.concat([df_actuals_null, df_fcast],
        #                ignore_index=True, sort=False)
        #
        # df_result = get_pi(df, n_sims=100)
        # # logger_info('df_result1:', df_result1)
        # logger_info('df_result2:', df_result.groupby(
        #     ['source', 'model']).head(20))
        # logger_info('df_result2:', df_result.groupby(
        #     ['source', 'model']).tail(20))
        #
        # self.assertFalse(
        #     df_result.loc[df_result.date >
        #                   df_actuals.date.max()].q5.isnull().any())
        #
        # check_result(df_result)

        logger.info('Test 3 - Input has weight 0')
        df_actuals_weight0 = df_actuals.copy()
        # df_actuals_null.loc[5, 'y'] = np.NaN
        df_actuals_weight0['weight'] = 1.
        df_actuals_weight0.loc[5, 'weight'] = 0.
        df_actuals_weight0.loc[5, 'y'] = -5000.

        logger_info('df_actuals_weight0:', df_actuals_weight0)

        df = pd.concat([df_actuals_weight0, df_fcast],
                       ignore_index=True, sort=False)

        df_result = get_pi(df, n_sims=100)
        # logger_info('df_result1:', df_result1)
        logger_info('df_result2:', df_result.groupby(
            ['source', 'model']).head(20))
        logger_info('df_result2:', df_result.groupby(
            ['source', 'model']).tail(20))

        self.assertFalse(
            df_result.loc[df_result.date >
                          df_actuals.date.max()].q5.isnull().any())

        check_result(df_result)

    def test_forecast_pi_weight0(self):
        logger.info('Test 1 - Input has gaps')
        a_date_actuals = pd.date_range('2014-01-01', periods=10, freq='W')
        a_y_actuals = np.arange(0, 10.)
        df_actuals = (
            pd.DataFrame({'date': a_date_actuals, 'y': a_y_actuals,
                          'source': 's1', 'is_actuals': True,
                          'is_best_fit': False, 'model': 'actuals'})
        )

        a_date = pd.date_range('2014-01-01', periods=20, freq='W')
        a_y = np.arange(0, 20.) + (
                np.tile([-1, 1], (10)) * np.arange(2, 0., -0.1))

        df_fcast = (pd.DataFrame({'date': a_date,
                                  'y': a_y,
                                  'source': 's1',
                                  'is_actuals': False,
                                  'is_best_fit': True,
                                  'model': 'linear'}))

        df_actuals_weight0 = df_actuals.copy()
        # df_actuals_null.loc[5, 'y'] = np.NaN
        df_actuals_weight0['weight'] = 1.
        df_actuals_weight0.loc[5, 'weight'] = 0.
        df_actuals_weight0.loc[5, 'y'] = -5000.

        logger_info('df_actuals_weight0:', df_actuals_weight0)

        df_result = run_forecast(
            df_actuals_weight0,
            extrapolate_years=0,
            l_model_season=[],
            simplify_output=True)

        logger_info('df_result', df_result)
        # Extremely low values mean weight filter is not working
        assert df_result.q5.min() > -4000

    def test_forecast_pi_missing(self):
        logger.info('Test1a - Generate forecast, check PI is added')
        path_candy = os.path.join(base_folder, 'candy_production.csv')
        df_monthly_candy = pd.read_csv(path_candy).head(100)
        dict_result = run_forecast(
            df_monthly_candy,
            col_name_y='IPG3113N',
            col_name_date='observation_date',
            extrapolate_years=2,
            l_model_season=[],
            simplify_output=False)
        df_fcast = dict_result.get('forecast')
        logger_info('df_fcast: ', df_fcast.tail())
        self.assertIn('q5', df_fcast.columns)

        logger.info('Test1b - Generate forecast with n_cum=30')
        dict_result = run_forecast(
            df_monthly_candy,
            col_name_y='IPG3113N',
            col_name_date='observation_date',
            extrapolate_years=2,
            l_model_season=[],
            simplify_output=False,
            n_cum=30
        )

        df_fcast = dict_result.get('forecast')
        logger_info('df_result2:', df_fcast.groupby(
            ['source', 'model']).tail(2))
        self.assertIn('q5', df_fcast.columns)

        logger.info('Test2a - Generate forecast, multiple models,'
                    ' check PI is added')

        df_monthly_candy2 = pd.concat(
            [df_monthly_candy.assign(source='s1'),
             df_monthly_candy.assign(source='s2')]
        )
        dict_result = run_forecast(
            df_monthly_candy2,
            col_name_y='IPG3113N',
            col_name_date='observation_date',
            extrapolate_years=2,
            l_model_season=[],
            simplify_output=False)

        df_fcast = dict_result.get('forecast')
        logger_info('df_result head:', df_fcast.groupby(
            ['source', 'model']).head(2))
        logger_info('df_result tail :', df_fcast.groupby(
            ['source', 'model']).tail(2))
        self.assertIn('q5', df_fcast.columns)

        logger.info('Test2b - Generate forecast with n_cum=30')
        dict_result = run_forecast(
            df_monthly_candy2,
            col_name_y='IPG3113N',
            col_name_date='observation_date',
            extrapolate_years=2,
            l_model_season=[],
            simplify_output=False,
            n_cum=300
        )
        df_fcast = dict_result.get('forecast')
        logger_info('df_result head:', df_fcast.groupby(
            ['source', 'model']).head(2))
        logger_info('df_result tail :', df_fcast.groupby(
            ['source', 'model']).tail(2))
        self.assertIn('q5', df_fcast.columns)

    def test_run_forecast_yearly_model(self):
        df1 = pd.DataFrame({'y': np.arange(0, 10.), 'date': pd.date_range(
            '2000-01-01', periods=10, freq='YS')})
        dict_result = run_forecast(
            simplify_output=False,
            df_y=df1,
            l_model_trend=[
                forecast_models.model_linear],
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

        dict_result = run_forecast(
            simplify_output=False,
            df_y=df2,
            l_model_trend=[
                forecast_models.model_linear],
            extrapolate_years=10.)

        df_data = dict_result['data']
        df_metadata = dict_result['metadata']
        df_optimize_info = dict_result['optimize_info']

        logger_info('df_metadata:', df_metadata)
        logger_info('df_optimize_info:', df_optimize_info)
        logger_info('df_data:', df_data.groupby(['source', 'model']).tail(60))

        # Same, with simplify_output=True

        df_result = run_forecast(
            simplify_output=True,
            df_y=df2,
            l_model_trend=[
                forecast_models.model_linear],
            extrapolate_years=10.)
        logger_info('df_result:', df_result)

    def test_run_forecast_validate_input(self):
        # -- TEST 1 - Validate model_season_wday
        # Test that seasonality models applied to time series with
        # period-aligned gaps (e.g. missing Fridays) raise exc.
        # Original time series, no gaps
        df1 = pd.DataFrame({'y': np.full(14, 10.0), 'source': 'src1',
                            'date': pd.date_range('2018-01-01',
                                                  periods=14, freq='D')})

        # Copy of df1 with gaps on same weekday
        df2 = df1.copy()
        df2['weight'] = array_zeros_in_indices(14, [5, 12])
        df2['source'] = 'src2'
        dict_forecast1 = run_forecast(
            df1,
            extrapolate_years=0.1,
            simplify_output=False,
            l_model_trend=(forecast_models.model_linear +
                           forecast_models.model_season_wday),
            l_model_season=[],
            l_model_naive=[],
            include_all_fits=True)
        df_forecast1 = dict_forecast1['data']  # This works fine

        # In this case, we don't get a fit
        dict_forecast2 = run_forecast(
            df2,
            extrapolate_years=0.1,
            simplify_output=False,
            l_model_trend=(forecast_models.model_linear +
                           forecast_models.model_season_wday),
            l_model_season=[],
            l_model_naive=[],
            include_all_fits=True)
        df_metadata = dict_forecast2.get('metadata')
        logger_info('df_metadata:', df_metadata)
        self.assert_series_equal(df_metadata.status, pd.Series('INPUT_ERR'))

        df_data = dict_forecast2.get('data')
        logger_info('df_data:', df_data)
        # Output only includes actuals due to no fit
        self.assertEqual(df_data.index.size, 14)

    def test_run_forecast_linalgerror(self):
        # Testing a dataset that raises a linalgerror from run_forecast()
        path_df_in = os.path.join(base_folder,
                                  'df_test_forecast_linalgerror.csv')
        df_in = pd.read_csv(path_df_in, parse_dates=['date'])
        dict_result = run_forecast(df_in, simplify_output=False,
                                   include_all_fits=True)
        df_metadata = dict_result['metadata']
        logger_info('df_metadata', df_metadata)

    def test_run_forecast_cache(self):
        a_date = pd.date_range('2020-01-01', '2022-01-01')
        a_x = np.arange(0, a_date.size)
        a_y = a_date.month * 10 + a_date.weekday
        df_in = pd.DataFrame(data=dict(date=a_date, x=a_x, y=a_y))

        l_models = [
            forecast_models.model_season_month,
            forecast_models.model_season_wday,
            forecast_models.model_season_month +
            forecast_models.model_season_wday,
            forecast_models.model_season_fourier_yearly,
            forecast_models.model_calendar_uk,
            forecast_models.model_season_month +
            forecast_models.model_calendar_uk,
        ]

        def run_fcast(use_cache=True):
            dict_result = run_forecast(
                simplify_output=False,
                include_all_fits=True,
                df_y=df_in,
                l_model_trend=l_models,
                l_model_season=[],
                l_model_naive=[],
                l_model_calendar=[],
                extrapolate_years=2.,
                use_cache=use_cache
            )
            return dict_result

        dict_result_cache = run_fcast(True)
        dict_result_no_cache = run_fcast(False)

        df_metadata = (
            pd.concat([
                dict_result_cache.get('metadata').assign(use_cache=True),
                dict_result_no_cache.get('metadata').assign(use_cache=False),
            ], ignore_index=True)
                .sort_values(['model', 'use_cache'])
        )
        df_optimize = (
            pd.concat([
                dict_result_cache.get('optimize_info').assign(use_cache=True),
                dict_result_no_cache.get('optimize_info').assign(
                    use_cache=False),
            ], ignore_index=True)
                .sort_values(['model', 'use_cache'])
        )
        logger_info('df_metadata:', df_metadata)
        logger_info('df_optimize_info:', df_optimize)
        df_fit_time = df_metadata[['model', 'use_cache', 'fit_time']]
        logger_info('df_fit_time:', df_fit_time)

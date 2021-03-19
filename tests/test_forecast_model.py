"""

Author: Pedro Capelastegui
Created on 04/12/2015
"""

import unittest
from argparse import Namespace
from datetime import datetime

from anticipy import forecast_models
from anticipy.forecast import normalize_df
from anticipy.forecast_models import *
from anticipy.forecast_models import _fillna_wday
from anticipy.model_utils import interpolate_df
from anticipy.utils_test import PandasTest

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def logger_info(msg, data):
    logger.info(msg + '\n%s', data)


pd.set_option('display.max_columns', 40)
pd.set_option('display.max_rows', 200)
pd.set_option('display.width', 1000)


def get_initial_guess(f_model, t_values):
    return f_model(t_values, None, None, get_aic_k=False)


def array_ones_in_indices(n, l_indices):
    return np.isin(np.arange(0, n), l_indices).astype(float)


def array_zeros_in_indices(n, l_indices):
    return (~np.isin(np.arange(0, n), l_indices)).astype(float)


def array_true_in_indices(n, l_indices):
    return np.isin(np.arange(0, n), l_indices)


class TestForecastModel(PandasTest):
    def setUp(self):
        pass

    def test_model_naive(self):
        a_x = np.arange(0, 10)
        a_date = pd.date_range('2014-01-01', periods=10, freq='D')
        a_y = 10 * a_x
        df_actuals = pd.DataFrame({'date': a_date, 'x': a_x, 'y': a_y}).head()

        a_y_result = model_naive(a_x, a_date, None, df_actuals=df_actuals)
        logger_info('a_y result: ', a_y_result)
        a_y_expected = np.array(
            [0., 0., 10, 20., 30., 40, 40., 40., 40., 40., ])
        self.assert_array_equal(a_y_result, a_y_expected)

        # TODO: model composition disabled, check that exception is thrown
        # # Model composition
        # a_params = np.array([1.,0.,])
        # a_y_result = (model_naive + model_linear) (a_x, a_date, a_params,
        #   df_actuals=df_actuals)
        # logger_info('a_y result: ', a_y_result)

    def test_fillna_wday(self):
        a_x = np.arange(0, 70)
        a_date = pd.date_range('2014-01-01', periods=70, freq='D')
        a_y = 1. * a_x

        a_y_gap = np.where(np.isin(a_x, [5, 10, 15, 20]), np.NaN, a_y)

        df_actuals = pd.DataFrame({'date': a_date, 'x': a_x,
                                   'y': a_y_gap})

        time_start = datetime.now()
        df_result = _fillna_wday(df_actuals)
        runtime = datetime.now() - time_start
        logger_info('run time: ', runtime)
        logger_info('df_actuals:', df_actuals)
        logger_info('df_result:', df_result)

    def test_model_snaive_wday(self):
        logger.info('Test 1 - default settings')
        a_x = np.arange(0, 21)
        a_date = pd.date_range('2014-01-01', periods=21, freq='D')
        a_y = 10. * a_x
        df_actuals = pd.DataFrame({'date': a_date, 'x': a_x,
                                   'y': a_y}).head(7)

        a_y_result = model_snaive_wday(
            a_x, a_date, None, df_actuals=df_actuals)
        logger_info('a_y result: ', a_y_result)
        a_y_expected = np.array(
            [np.NaN] * 7 + np.arange(0, 70., 10.).tolist() * 2)
        self.assert_array_equal(a_y_result, a_y_expected)

        logger.info('Test 2 - null values on last week')
        df_actuals = pd.DataFrame({'date': a_date, 'x': a_x,
                                   'y': a_y}).head(14)
        df_actuals.y.iloc[-1] = np.NaN
        logger_info('actuals: ', df_actuals)
        a_y_result = model_snaive_wday(
            a_x, a_date, None, df_actuals=df_actuals)
        logger_info('a_y result: ', a_y_result)

        # TODO: model composition disabled, check that exception is thrown
        # # Model composition
        # a_params = np.array([1.,0.,])
        # a_y_result = (model_naive + model_linear) (a_x, a_date, a_params,
        #   df_actuals=df_actuals)
        # logger_info('a_y result: ', a_y_result)

    def test_forecast_model_simple_models(self):
        # TODO: test all models with is_mult True and False

        a_x = np.arange(0, 10)
        a_date = pd.date_range('2014-01-01', periods=10, freq='D')

        def test_model(
                name,
                model,
                params,
                a_expected,
                l_is_mult=None,
                a_date=a_date,
                a_x=a_x):
            if l_is_mult is None:
                l_is_mult = [True, False]
            for is_mult in l_is_mult:
                params = np.array(params)
                a = model(a_x, a_date, params, is_mult)
                logger_info('a {}, is_mult={} :'.format(name, is_mult), a)
                self.assert_array_equal(a, a_expected)
            # Test init params
            params = model.f_init_params(None, None, None)
            self.assertIsInstance(params, np.ndarray)
            bounds = model.f_bounds(None, None, None)
            logger.info('params: %s', params)
            self.assertTrue(validate_initial_guess(params, bounds))
            params = model.f_init_params(a_x, None, a_x)
            self.assertIsInstance(params, np.ndarray)
            bounds = model.f_bounds(a_x, None, a_x)
            logger.info('a_x: %s', a_x)
            logger.info('params: %s', params)
            logger.info('bounds: %s', bounds)
            self.assertTrue(validate_initial_guess(params, bounds))
            params = model.f_init_params(None, a_x, a_date)
            self.assertIsInstance(params, np.ndarray)
            bounds = model.f_bounds(None, a_x, a_x)
            logger.info('a_x: %s', a_x)
            logger.info('params: %s', params)
            logger.info('bounds: %s', bounds)
            self.assertTrue(validate_initial_guess(params, bounds))
            params = model.f_init_params(a_x, a_x, a_date)
            self.assertIsInstance(params, np.ndarray)
            bounds = model.f_bounds(a_x, a_x, a_x)
            logger.info('params: %s', params)
            self.assertTrue(validate_initial_guess(params, bounds))

            # Test cache
            dict_cache_vars = model.init_cache(a_x, a_date)
            logger.info('cache_vars: %s', dict_cache_vars)

        test_model('constant', model_constant, [42],
                   np.full(10, 42.))

        test_model('linear', model_linear, [-1., 10],
                   np.arange(10., 0, -1))

        test_model('ramp', model_ramp, [5., 1.],
                   np.concatenate([np.full(5, 0.), np.arange(0, 5.)]), [False])

        test_model('ramp', model_ramp, [5., 1.],
                   np.concatenate([np.full(5, 1.), np.arange(1, 6.)]), [True])

        test_model('exp', model_exp, [10., 2], np.array(
            [10., 20., 40., 80., 160., 320., 640., 1280., 2560., 5120.]))

        test_model('step', model_step, [5., 100.],
                   np.array(5 * [0.] + 5 * [100.]), [False])

        test_model('step', model_step, [5., 100.],
                   np.array(5 * [1.] + 5 * [100.]), [True])

        test_model('step_date', get_model_step_date('2014-01-06'), [100.],
                   np.array(5 * [0.] + 5 * [100.]), [False])

        test_model('step_date', get_model_step_date('2014-01-06'), [100.],
                   np.array(5 * [1.] + 5 * [100.]), [True])

        test_model('spike', model_spike, [10., 4., 6.],
                   np.array(4 * [0.] + 2 * [10.] + 4 * [0.]), [False])

        test_model('spike', model_spike, [10., 4., 6.],
                   np.array(4 * [1.] + 2 * [10.] + 4 * [1.]), [True])

        test_model('decay', model_decay, [10., -1000., 0.],
                   np.array([10.] + 9 * [0.]))

        test_model(
            'spike_date',
            get_model_spike_date(
                '2014-01-05',
                '2014-01-07'),
            [10.],
            np.array(
                4 *
                [0.] +
                2 *
                [10.] +
                4 *
                [0.]),
            [False])

        test_model(
            'spike_date',
            get_model_spike_date(
                '2014-01-05',
                '2014-01-07'),
            [10.],
            np.array(
                4 *
                [1.] +
                2 *
                [10.] +
                4 *
                [1.]),
            [True])

        test_model('2 steps', model_two_steps, [5., 100., 7, 200.],
                   np.array(5 * [0.] + 2 * [100.] + 3 * [300.]), [False])

        test_model('2 steps', model_two_steps, [5., 100., 7, 3.],
                   np.array(5 * [1.] + 2 * [100.] + 3 * [300.]), [True])

        test_model('season_wday', model_season_wday, 10 * np.arange(1., 7.),
                   np.array([20., 30., 40., 50., 60., 0, 10., 20., 30., 40.]),
                   [False])

        test_model('season_wday', model_season_wday, 10 * np.arange(1., 7.),
                   np.array([20., 30., 40., 50., 60., 1, 10., 20., 30., 40.]),
                   [True])

        a_x2 = np.arange(0, 12)
        a_date2 = pd.date_range('2014-01-01', periods=12, freq='D')

        test_model('season_month', model_season_month, 10 * np.arange(2., 13.),
                   np.array([60., 70., 80., 90., 100, 110., 120.,
                             0., 20., 30., 40., 50., ]), [False],
                   a_date=pd.date_range('2014-06-01', periods=12, freq='M'),
                   a_x=a_x2)

        test_model('season_month', model_season_month,
                   10 * np.arange(2., 13.),
                   np.array([60., 70., 80., 90., 100, 110., 120.,
                             1., 20., 30., 40., 50., ]), [True],
                   a_date=pd.date_range('2014-06-01', periods=12, freq='M'),
                   a_x=a_x2)

        test_model('season_fourier_yearly', model_season_month,
                   10 * np.arange(2., 13.),
                   np.array([60., 70., 80., 90., 100, 110., 120., 1.,
                             20., 30., 40., 50., ]), [True],
                   a_date=pd.date_range('2014-06-01', periods=12, freq='M'),
                   a_x=a_x2)

        # test fourier model
        from anticipy.forecast_models import _f_init_params_fourier

        for is_mult in [False, True]:
            a_x = 10 * np.arange(2., 13.)
            a_date = pd.date_range('2014-06-01', periods=10, freq='M')
            params = _f_init_params_fourier()
            a = model_season_fourier_yearly(a_x, a_date, params, is_mult)
            logger_info(
                'a {}, is_mult={} :'.format(
                    'model_season_fourier_yearly',
                    is_mult),
                a)

        for is_mult in [False, True]:
            a_x = 10 * np.arange(2., 13.)
            a_date = pd.date_range('2014-06-01', periods=10, freq='M')
            params = np.full(20, 1.)
            a = model_season_fourier_yearly(a_x, a_date, params, is_mult)
            logger_info(
                'a {}, is_mult={} :'.format(
                    'model_season_fourier_yearly',
                    is_mult),
                a)

        test_model('ukcalendar', model_calendar_uk,
                   [1, 1],
                   # First parameter changes value of New Year
                   array_ones_in_indices(10, 0) + np.ones(10),
                   l_is_mult=[True])
        test_model('ukcalendar', model_calendar_uk,
                   [1, 1],
                   # First parameter changes value of New Year
                   array_ones_in_indices(10, 0) + np.zeros(10),
                   l_is_mult=[False])

        test_model('uscalendar', model_calendar_us,
                   [1],
                   # First parameter changes value of New Year
                   array_ones_in_indices(10, 0) + np.ones(10),
                   l_is_mult=[True])
        test_model('uscalendar', model_calendar_us,
                   [1],
                   # First parameter changes value of New Year
                   array_ones_in_indices(10, 0) + np.zeros(10),
                   l_is_mult=[False])

    def test_forecast_model_composite(self):
        a_x = np.arange(1, 11.)
        a_y = np.arange(1, 11.)
        a_date = pd.date_range('2014-01-01', periods=10, freq='D')
        a_date_month = pd.date_range('2014-01-01', periods=10, freq='M')

        dict_model = {
            'constant': model_constant,
            'linear': model_linear,
            'ramp': model_ramp,
            'exp': model_exp,
            'season_wday': model_season_wday,
            # TODO: ADD season_wday_2
            'season_month': model_season_month,
            'step': model_step,
            'two_steps': model_two_steps,
        }
        dict_params = {
            'constant': np.array([1.]),
            'linear': np.array([1., 0.]),
            'ramp': np.array([6., 1.]),
            'exp': np.array([1., 2.]),
            'season_wday': np.arange(1., 7.),
            'season_month': np.arange(2., 13.),
            'step': np.array([6., 100.]),
            'two_steps': np.array([6., 100., 8, 200.]),
        }
        dict_expected_add = {
            'constant': np.full(10, 1.),
            'linear': np.arange(1., 11.),
            'ramp': np.concatenate([np.full(5, 0.), np.arange(0, 5.)]),
            'exp': 2 ** np.arange(1., 11.),
            'season_wday': np.arange(2., 12., ) % 7,
            'season_month': np.full(10, 0.),
            'step': np.array(5 * [0.] + 5 * [100.]),
            'two_steps': np.array(5 * [0.] + 2 * [100.] + 3 * [300.]),
        }
        dict_expected_mult = {
            'constant': np.full(10, 1.),
            'linear': np.arange(1., 11.),
            'ramp': np.concatenate([np.full(5, 1.), np.arange(1, 6.)]),
            'exp': 2 ** np.arange(1., 11.),
            'season_wday': np.array([2., 3., 4., 5., 6., 1.,
                                     1., 2., 3., 4., ]),
            'season_month': np.full(10, 1.),
            'step': np.array(5 * [1.] + 5 * [100.]),
            'two_steps': np.array(5 * [1.] + 2 * [100.] + 3 * [20000.]),
        }

        def test_model_1(key):
            model = dict_model[key]
            initial_guess = model.f_init_params(a_x, a_y)
            logger.info('Testing model %s - name: %s', key, model.name)
            self.assert_array_equal(
                model(
                    a_x,
                    a_date,
                    dict_params[key]),
                dict_expected_add[key])
            logger.info('Initial guess: %s', model.f_init_params(a_x, a_y))
            self.assertEqual(len(initial_guess), model.n_params)
            # Test cache
            dict_cache_vars = model.init_cache(a_x, a_date)
            logger.info('cache_vars: %s', dict_cache_vars)

        for key in dict_model.keys():
            test_model_1(key)

        def test_model_2_add(key1, key2):
            model = dict_model[key1] + dict_model[key2]
            initial_guess = model.f_init_params(a_x, a_y)
            logger.info(
                'Testing model %s, %s - name: %s',
                key1,
                key2,
                model.name)
            logger.info(
                'Parameters: %s , %s',
                dict_params[key1],
                dict_params[key2])
            logger.info('Initial guess: %s', initial_guess)
            self.assertEqual(len(initial_guess), model.n_params)
            model_output = model(a_x, a_date, np.concatenate(
                [dict_params[key1], dict_params[key2]]))
            logger.info('Model output: %s', model_output)
            self.assert_array_equal(
                model_output,
                dict_expected_add[key1] +
                dict_expected_add[key2])
            # Test cache
            dict_cache_vars = model.init_cache(a_x, a_date)
            logger.info('cache_vars: %s', dict_cache_vars)

        for key1, key2 in itertools.product(
                dict_model.keys(), dict_model.keys()):
            logger.info('Keys: %s , %s', key1, key2)
            test_model_2_add(key1, key2)

        def test_model_2_mult(key1, key2):
            model = dict_model[key1] * dict_model[key2]
            initial_guess = model.f_init_params(a_x, a_y)
            logger.info(
                'Testing model %s, %s - name: %s',
                key1,
                key2,
                model.name)
            logger.info(
                'Parameters: %s , %s',
                dict_params[key1],
                dict_params[key2])
            logger.info('Initial guess: %s', initial_guess)
            self.assertEqual(len(initial_guess), model.n_params)
            model_output = model(a_x, a_date, np.concatenate(
                [dict_params[key1], dict_params[key2]]))
            logger.info('Model output: %s', model_output)
            self.assert_array_equal(
                model_output,
                dict_expected_mult[key1] *
                dict_expected_mult[key2])
            # Test cache
            dict_cache_vars = model.init_cache(a_x, a_date)
            logger.info('cache_vars: %s', dict_cache_vars)

        for key1, key2 in itertools.product(
                dict_model.keys(), dict_model.keys()):
            logger.info('Keys: %s , %s', key1, key2)
            test_model_2_mult(key1, key2)

    def test_forecast_model_composite_null(self):
        a_x = np.arange(0, 10.)
        a_y = np.arange(0, 10.)
        a_date = pd.date_range('2014-01-01', periods=10, freq='D')
        a_date_month = pd.date_range('2014-01-01', periods=10, freq='M')

        dict_model = {
            'constant': model_constant,
            'linear': model_linear,
            'exp': model_exp,
            'season_wday': model_season_wday,
            'season_month': model_season_month,
        }

        dict_params = {
            'constant': np.array([1.]),
            'linear': np.array([1., 0.]),
            'exp': np.array([1., 2.]),
            'season_wday': np.arange(1., 7.),
            'season_month': np.arange(1., 13.)
        }
        dict_expected = {
            'constant': np.full(10, 1.),
            'linear': np.arange(0., 10.),
            'exp': 2 ** np.arange(0., 10.),
            'season_wday': np.arange(2., 12., ) % 7,
            'season_month': np.full(10, 0.),
        }

        def test_model_2_add_null(key1):
            model = dict_model[key1] + model_null
            initial_guess = model.f_init_params(a_x, a_y)
            logger.info('Testing model %s, - name: %s', key1, model.name)
            logger.info('Parameters: %s', dict_params[key1])
            logger.info('Initial guess: %s', initial_guess)
            self.assertEqual(len(initial_guess), model.n_params)
            self.assert_array_equal(model(a_x, a_date,
                                          dict_params[key1]),
                                    dict_expected[key1])

        for key in dict_model.keys():
            test_model_2_add_null(key)

        def test_model_2_mult_null(key1):
            model_original = dict_model[key1]
            model = model_original * model_null
            initial_guess = model.f_init_params(a_x, a_y)
            logger.info('Testing model %s, - name: %s', key1, model.name)
            logger.info('Parameters: %s', dict_params[key1])
            logger.info('Initial guess: %s', initial_guess)
            self.assertEqual(model, model_original)

        for key in dict_model.keys():
            test_model_2_mult_null(key)

    def test_forecast_model_composite_3(self):
        # Test composition of 3+ models
        a_x = np.arange(0, 10.)
        a_y = np.arange(0, 10.)
        a_date = pd.date_range('2014-01-01', periods=10, freq='D')
        a_date_month = pd.date_range('2014-01-01', periods=10, freq='M')

        dict_model = {
            'constant': model_constant,
            'linear': model_linear,
            'ramp': model_ramp,
            'exp': model_exp,
            'season_wday': model_season_wday,
            'season_month': model_season_month,
        }

        dict_params = {
            'constant': np.array([1.]),
            'linear': np.array([1., 0.]),
            'ramp': np.array([6., 1.]),
            'exp': np.array([1., 2.]),
            'season_wday': np.arange(1., 7.),
            'season_month': np.arange(1., 13.)
        }
        dict_expected = {
            'constant': np.full(10, 1.),
            'linear': np.arange(0., 10.),
            'ramp': np.concatenate([np.full(5, 0.), np.arange(0, 5.)]),
            'exp': 2 ** np.arange(0., 10.),
            'season_wday':
            # np.arange(2., 12., ) % 7,
                np.array([2., 3., 4., 5., 6., 1., 1., 2., 3., 4.]),
            'season_month': np.full(10, 1.),
        }

        def test_model_3(model, params, expected):
            initial_guess = model.f_init_params(a_x, a_y)
            logger.info('Testing model: %s', model.name)
            logger.info('Parameters: %s', params)
            logger.info('Initial guess: %s', initial_guess)
            self.assertEqual(len(initial_guess), model.n_params)
            self.assert_array_equal(model(a_x, a_date, params),
                                    expected)

        test_model_3(
            (model_linear * model_linear) + model_constant,
            np.concatenate([dict_params['linear'], dict_params['linear'],
                            dict_params['constant']]),
            (dict_expected['linear'] * dict_expected['linear']) +
            dict_expected['constant']
        )

        test_model_3(
            model_linear * (model_linear + model_constant),
            np.concatenate([dict_params['linear'], dict_params['linear'],
                            dict_params['constant']]),
            dict_expected['linear'] * (dict_expected['linear'] +
                                       dict_expected['constant'])
        )

        test_model_3(
            (model_linear * model_season_wday) + model_constant,
            np.concatenate([dict_params['linear'], dict_params['season_wday'],
                            dict_params['constant']]),
            (dict_expected['linear'] * dict_expected['season_wday']) +
            dict_expected['constant']
        )

    def test_forecast_model_bounds(self):

        dict_model = {
            'constant': model_constant,
            'linear': model_linear,
            'exp': model_exp,
            'season_wday': model_season_wday,
            'season_month': model_season_month,
            'step': model_step,
            'two_steps': model_two_steps,
            'sigmoid_step': model_sigmoid_step,
            'ramp': model_ramp
        }
        dict_expected = dict()
        for model_name, model_obj in dict_model.items():
            n_params = model_obj.n_params
            exp = n_params * [-np.inf], n_params * [np.inf]
            dict_expected[model_name] = exp

        # Manually set the boundaries here
        dict_expected['sigmoid_step'] = (
            [-np.inf, -np.inf, 0.0], [np.inf, np.inf, np.inf])

        def test_model_bounds(key, model, expected):
            bounds = model.f_bounds()
            params = model.n_params
            logger.info('Testing model: %s', model.name)
            logger.info('Bounds: %s', bounds)
            logger.info('Expected: %s', expected)
            self.assertEqual(params, len(bounds[0]))
            self.assertTupleEqual(bounds, expected)

        for model_name, model_obj in dict_model.items():
            test_model_bounds(model_name, model_obj, dict_expected[model_name])

    def test_get_model_outliers(self):
        # TODO: change input dfs to normalized form, rather than call
        # normalize_df

        # Test 1 - no outliers
        a_y = [20.0, 20.1, 20.2, 20.3, 20.4, 20.5]
        a_date = pd.date_range(start='2018-01-01', periods=len(a_y), freq='D')
        df = pd.DataFrame({'y': a_y}).pipe(normalize_df)

        mask_step, mask_spike = get_model_outliers(df)
        logger.info('Model 1a: Step: %s, Spike: %s ', mask_step, mask_spike)
        self.assert_array_equal(mask_step, np.full(len(a_y), False))
        self.assert_array_equal(mask_spike, np.full(len(a_y), False))

        # 1b - with datetime index
        df = pd.DataFrame({'y': a_y}, index=a_date).pipe(normalize_df)

        mask_step, mask_spike = get_model_outliers(df)
        logger.info('Model 1b: Step: %s, Spike: %s ', mask_step, mask_spike)
        self.assert_array_equal(mask_step, np.full(len(a_y), False))
        self.assert_array_equal(mask_spike, np.full(len(a_y), False))

        # Test 2 - Single step
        a_y = np.array([19.8, 19.9, 20.0, 20.1, 20.2, 20.3, 20.4, 20.5,
                        20.6, 10., 10.1, 10.2, 10.3, 10.4,
                        10.5, 10.6, 10.7, 10.8, 10.9])
        a_date = pd.date_range(start='2018-01-01', periods=len(a_y), freq='D')
        df = pd.DataFrame({'y': a_y}).pipe(normalize_df)

        mask_step, mask_spike = get_model_outliers(df)
        logger.info('Model 2a: Step: %s, Spike: %s ', mask_step, mask_spike)
        self.assert_array_equal(mask_step,
                                array_true_in_indices(a_y.size, 9))
        self.assert_array_equal(mask_spike, np.full(len(a_y), False))

        # 2b - with date column
        df = pd.DataFrame({'y': a_y}, index=a_date).pipe(normalize_df)

        mask_step, mask_spike = get_model_outliers(df)
        logger.info('Model 2b: Step: %s, Spike: %s ', mask_step, mask_spike)
        self.assert_array_equal(mask_step,
                                array_true_in_indices(a_y.size, 9))
        self.assert_array_equal(mask_spike, np.full(len(a_y), False))

        # Test 3 - Two step changes
        a_y = np.array([-1, 0, 1, 2, 3, 5, 6, 8, 10, 15, 16, 18,
                        20.1, 20.2, 20.3, 20.4, 20.5, 20.6,
                        10., 10.1, 10.2, 10.3, 10.4])

        df = pd.DataFrame({'y': a_y}).pipe(normalize_df)

        mask_step, mask_spike = get_model_outliers(df)
        logger.info('Model 3: Step: %s, Spike: %s ', mask_step, mask_spike)
        self.assert_array_equal(mask_step,
                                array_true_in_indices(a_y.size, [9, 18]))
        self.assert_array_equal(mask_spike, np.full(len(a_y), False))

        # Test 4 - Consecutive changes
        a_y = np.array([-1, 0, 1, 2, 3, 5, 6, 8, 15, 16, 21, 20.1,
                        20.2, 20.3, 20.4, 20.5, 20.6, 20.7, 20.8,
                        10., 10.1, 10.2, 10.3, 10.4])
        df = pd.DataFrame({'y': a_y}).pipe(normalize_df)

        mask_step, mask_spike = get_model_outliers(df)
        logger.info('Model 4: Step: %s, Spike: %s ', mask_step, mask_spike)
        self.assert_array_equal(
            mask_step, array_true_in_indices(
                a_y.size, [
                    8, 9, 10, 19]))
        self.assert_array_equal(mask_spike, np.full(len(a_y), False))

        # spikes

        # Test 5 - 2 spikes and 1 step
        a_y = np.array([19.8, 19.9, 30.0, 30.1, 20.2, 20.3, 20.4, 20.5,
                        20.6, 10., 10.1, 10.2, 10.3, 10.4,
                        10.5, 10.6, 30.7, 10.8, 10.9])

        df = pd.DataFrame({'y': a_y}).pipe(normalize_df)

        mask_step, mask_spike = get_model_outliers(df)
        logger.info('Model 5a: Step: %s, Spike: %s ', mask_step, mask_spike)
        self.assert_array_equal(mask_step,
                                array_true_in_indices(a_y.size, [9]))
        self.assert_array_equal(mask_spike,
                                array_true_in_indices(a_y.size, [2, 3, 16]))

        # 5b - with datetime index
        a_date = pd.date_range(start='2018-01-01', periods=len(a_y), freq='D')
        df = pd.DataFrame({'y': a_y}, index=a_date).pipe(normalize_df)

        mask_step, mask_spike = get_model_outliers(df)
        logger.info('Model 5b: Step: %s, Spike: %s ', mask_step, mask_spike)
        self.assert_array_equal(mask_step,
                                array_ones_in_indices(a_y.size, [9]))
        self.assert_array_equal(mask_spike,
                                array_true_in_indices(a_y.size, [2, 3, 16]))

        # Test 6a - single spike
        a_y = np.array([19.8, 19.9, 20.0, 20.1, 20.2, 20.3, 20.4, 20.5,
                        20.6, 10., 20.8, 20.9, 21.0,
                        21.1, 21.2, 21.3, 21.4, 21.5, 21.6])

        df = pd.DataFrame({'y': a_y}).pipe(normalize_df)

        mask_step, mask_spike = get_model_outliers(df)
        logger.info('Model 6a: Step: %s, Spike: %s ', mask_step, mask_spike)
        self.assert_array_equal(mask_step, np.full(len(a_y), False))
        self.assert_array_equal(mask_spike,
                                array_true_in_indices(a_y.size, [9]))

        # Test 6b - single spike co-located with step
        a_y = np.array([19.8, 19.9, 20.0, 20.1, 20.2, 20.3, 20.4, 20.5,
                        20.6, 10., 30.7, 30.8, 30.9, 31.0,
                        31.1, 31.2, 31.3, 31.4, 31.5])

        df = pd.DataFrame({'y': a_y}).pipe(normalize_df)

        mask_step, mask_spike = get_model_outliers(df)
        logger.info('Model 6b: Step: %s, Spike: %s ', mask_step, mask_spike)
        self.assert_array_equal(mask_step,
                                array_true_in_indices(a_y.size, [9, 10]))
        self.assert_array_equal(mask_spike,
                                array_true_in_indices(a_y.size, [9]))

    # TODO: Work in progress
    @unittest.skip('outliers with gap not implemented yet')
    def test_get_model_outliers_withgap(self):

        # # Test 1 - short series with null value - nulls cause no outliers
        a_y = [0., 1., np.NaN, 3., 4., 5., 6., 7., ]
        a_date = pd.date_range(start='2018-01-01', periods=len(a_y), freq='D')
        df = pd.DataFrame({'y': a_y, 'date': a_date}).pipe(normalize_df)

        mask_step, mask_spike = get_model_outliers(df)
        logger_info('Model 1:', mask_step)
        self.assertIsNone(mask_step)
        self.assertIsNone(mask_spike)

        # Test 1b -  series with multiple values per x -- raises ValueError
        a_y = np.arange(0, 10.)
        a_date = pd.date_range(start='2018-01-01', periods=len(a_y), freq='D')
        df = pd.DataFrame({'y': a_y, 'date': a_date})
        df = pd.concat([df.head(5), df.head(6).tail(2)]).pipe(normalize_df)

        with self.assertRaises(ValueError):
            mask_step, mask_spike = get_model_outliers(df)

        # Test 2 - short series with gap value - no real outliers
        a_y = np.arange(0, 10.)
        a_date = pd.date_range(start='2018-01-01', periods=len(a_y), freq='D')
        df = pd.DataFrame({'y': a_y, 'date': a_date})
        df = pd.concat([df.head(5), df.tail(-6)]).pipe(normalize_df)

        mask_step, mask_spike = get_model_outliers(df)
        logger_info('Model 1:', mask_step)
        self.assertIsNotNone(mask_step)  # Incorrectly finds a step
        self.assertIsNone(mask_spike)  # No spikes

        # Test 2b - after interpolating, can get outliers - finds none

        df_nogap = df.pipe(interpolate_df, include_mask=True)
        mask_step, mask_spike = get_model_outliers(df_nogap)
        logger_info('df 1 - no gap:', df_nogap)
        self.assertIsNone(mask_step)  # No steps
        self.assertIsNone(mask_spike)  # No spikes

        # # Test 3 - short series with gap value - with outliers
        a_y = np.arange(0, 10.)
        a_y2 = np.arange(1, 11.)
        a_date = pd.date_range(start='2018-01-01', periods=len(a_y), freq='D')
        df = pd.DataFrame({'y': a_y, 'date': a_date})
        df2 = pd.DataFrame({'y': a_y2, 'date': a_date})
        df = pd.concat([df.head(5), df2.tail(-6)]).pipe(normalize_df)

        mask_step, mask_spike = get_model_outliers(df)
        logger_info('Model 1:', mask_step)
        self.assertIsNotNone(mask_step)  # Incorrectly finds a step
        self.assertIsNone(mask_spike)  # No spikes

        # Test 3b - after interpolating with interpolate_df() - TODO: REMOVE
        # THIS

        df_nogap = df.pipe(interpolate_df, include_mask=True)
        mask_step, mask_spike = get_model_outliers(df_nogap)

        df_nogap['mask_step'] = mask_step
        df_nogap['step_in_filled_gap'] = df_nogap.mask_step * \
            df_nogap.is_gap_filled

        df_nogap['mask_step_patch'] = df_nogap.step_in_filled_gap.shift(
            -1).fillna(0)
        df_nogap = df_nogap.loc[~df_nogap.is_gap_filled]
        df_nogap['mask_step_patch'] = df_nogap.mask_step_patch.shift(
            1).fillna(0)
        df_nogap['mask_step'] = df_nogap.mask_step + df_nogap.mask_step_patch
        df_nogap = df_nogap[['date', 'x', 'y', 'mask_step']]
        logger_info('df 1 - no gap:', df_nogap)

        self.assert_array_equal(
            df_nogap.mask_step, array_ones_in_indices(
                df_nogap.index.size, [5]))

        self.assertIsNone(mask_spike)  # No spikes

        # TODO: we need to
        # - filter out filled gaps
        # - get list of steps
        # - if a step is in a filled gap, move to next sample

    def test_get_fixed_model(self):
        logger.info('Test 1 - default settings')
        a_x = np.arange(0, 10)
        a_date = pd.date_range('2014-01-01', periods=10, freq='D')

        a1 = model_constant(a_x, a_date, np.array([42]))
        model_constant_fixed = get_fixed_model(model_constant, np.array([42]))
        print(model_constant_fixed)
        a2 = model_constant_fixed(a_x, a_date, None)
        self.assert_array_equal(a1, a2)

        logger.info('Test 2 - 0-param model')
        model_out = get_fixed_model(model_naive, [])
        self.assertEqual(model_out.name, 'naive')

    def test_fix_params_fmodel(self):
        a_x = np.arange(0, 10)
        a_date = pd.date_range('2014-01-01', periods=10, freq='D')

        a1 = model_linear(a_x, a_date, np.array([10., -1.]))
        model_linear_fixed = fix_params_fmodel(model_linear, [10., np.NaN])
        logger_info('model_linear_fixed:', model_linear_fixed)
        self.assertEqual(model_linear_fixed.n_params, 1)
        a2 = model_linear_fixed(a_x, a_date, params=[-1.])
        self.assert_array_equal(a1, a2)

    # TODO: Implement test
    def test_validate_model_bounds(self):
        pass

    def test_get_l_model_auto_season(self):

        logger.info('Test 0 - Test for series with single sample')
        a_date = pd.a_date = pd.date_range('2014-01-01', periods=1, freq='D')
        l_expected = [model_null]
        l_result = get_l_model_auto_season(a_date)
        self.assert_array_equal(l_result, l_expected)

        logger.info('Test 1 - Tests for series with daily samples')
        # Test 1.1 - not enough samples for weekly seasonality
        a_date = pd.a_date = pd.date_range('2014-01-01', periods=10, freq='D')
        l_expected = [model_null]
        l_result = get_l_model_auto_season(a_date)
        self.assert_array_equal(l_result, l_expected)

        # Test 1.2 - enough samples for weekly seasonality
        a_date = pd.a_date = pd.date_range('2014-01-01', periods=12, freq='D')
        l_expected = [model_null, model_season_wday]
        l_result = get_l_model_auto_season(a_date, min_periods=1.5)
        self.assert_array_equal(l_result, l_expected)

        # Test 1.2b - l_season_yearly is empty list
        l_result = get_l_model_auto_season(
            a_date, l_season_yearly=[], min_periods=1.5)
        self.assert_array_equal(l_result, l_expected)

        # Test 1.3 - Weekly and yearly seasonality
        a_date = pd.a_date = pd.date_range('2014-01-01', periods=549, freq='D')
        l_expected = sorted([model_null,
                             model_season_wday * model_season_fourier_yearly,
                             model_season_wday,
                             model_season_fourier_yearly])
        l_result = get_l_model_auto_season(
            a_date, min_periods=1.5, season_add_mult='mult')
        self.assert_array_equal(l_result, l_expected)

        l_expected = [
            model_null,
            model_season_wday +
            model_season_fourier_yearly,
            model_season_wday,
            model_season_fourier_yearly]
        l_expected.sort()
        l_result = get_l_model_auto_season(
            a_date, min_periods=1.5, season_add_mult='add')
        self.assert_array_equal(l_result, l_expected)

        # Test 1.3b - Weekly and yearly seasonality, empty l_season_yearly
        l_result = get_l_model_auto_season(
            a_date, l_season_yearly=[], min_periods=1.5, season_add_mult='add')
        l_expected = [model_null, model_season_wday]
        self.assert_array_equal(l_result, l_expected)

        # Test 1.3c - Weekly and yearly seasonality, empty l_season_weekly
        l_result = get_l_model_auto_season(
            a_date, l_season_weekly=[], min_periods=1.5, season_add_mult='add')
        l_expected = [model_null, model_season_fourier_yearly]
        self.assert_array_equal(l_result, l_expected)

        logger.info('Test 2 - Tests for series with weekly samples')
        # Test 2.2 - not enough samples for yearly seasonality
        a_date = pd.a_date = pd.date_range('2014-01-01', periods=12, freq='W')
        l_expected = [model_null]
        l_result = get_l_model_auto_season(a_date, min_periods=1.5)
        self.assert_array_equal(l_result, l_expected)

        # Test 2.3 - Weekly and yearly seasonality
        a_date = pd.a_date = pd.date_range('2014-01-01', periods=80, freq='W')
        l_expected = [model_null, model_season_fourier_yearly]
        l_result = get_l_model_auto_season(a_date, min_periods=1.5)
        self.assert_array_equal(l_result, l_expected)

        logger.info('Test 3 - Tests for series with monthly samples')
        # Test 3.2 - not enough samples for yearly seasonality
        a_date = pd.a_date = pd.date_range('2014-01-01', periods=12, freq='M')
        l_expected = [model_null]
        l_result = get_l_model_auto_season(a_date, min_periods=1.5)
        self.assert_array_equal(l_result, l_expected)

        # Test 3.3 - Weekly and yearly seasonality
        a_date = pd.a_date = pd.date_range('2014-01-01', periods=20, freq='M')
        l_expected = [model_null, model_season_fourier_yearly]
        l_result = get_l_model_auto_season(a_date, min_periods=1.5)
        self.assert_array_equal(l_result, l_expected)

        logger.info('Test 4 - Tests for series with quarterly samples')
        # Test 4.2 - not enough samples for yearly seasonality
        a_date = pd.a_date = pd.date_range('2014-01-01', periods=5, freq='Q')
        l_expected = [model_null]
        l_result = get_l_model_auto_season(a_date, min_periods=1.5)
        self.assert_array_equal(l_result, l_expected)

        # Test 4.3 - Weekly and yearly seasonality
        a_date = pd.a_date = pd.date_range('2014-01-01', periods=7, freq='Q')
        l_expected = [model_null, model_season_fourier_yearly]
        l_result = get_l_model_auto_season(a_date, min_periods=1.5)
        self.assert_array_equal(l_result, l_expected)

    def test_simplify_model(self):
        # Test 1: normal bounds
        model_dummy = Namespace()
        model_dummy.f_bounds = lambda a_x, a_y, a_date: (
            np.array([3.]), np.array([7.]))
        model_dummy.n_params = 1
        model_dummy.name = 'dummy'

        model_result = simplify_model(model_dummy)
        logger_info('model_dummy', model_dummy)
        logger_info('result:', model_result)
        self.assertEqual(model_dummy, model_result)

        # Test 2: min and max bounds match - model transformed into fixed model
        model_dummy = Namespace()
        model_dummy.f_bounds = lambda a_x, a_y, a_date: (
            np.array([5.]), np.array([5.]))
        model_dummy.n_params = 1
        model_dummy.name = 'dummy'

        model_result = simplify_model(model_dummy)
        logger_info('model_dummy', model_dummy)
        logger_info('result:', model_result)
        self.assertEqual(model_result.n_params, 0)

    def test_validate_initial_guess(self):
        result = validate_initial_guess(
            np.array([5., 5.]), (np.array([0., 0.]), np.array([10., 10.])))
        self.assertTrue(result)

        result = validate_initial_guess(
            np.array([0., 10.]), (np.array([0., 0.]), np.array([10., 10.])))
        self.assertTrue(result)

        result = validate_initial_guess(
            np.array([-1., 11.]), (np.array([0., 0.]), np.array([10., 10.])))
        self.assertFalse(result)

    def test_validate_input(self):
        # Test1: default f_validate_input
        model1 = ForecastModel('model1', 0, forecast_models._f_model_null,
                               l_f_validate_input=None)
        model2 = ForecastModel(
            'model2',
            0,
            forecast_models._f_model_null,
            l_f_validate_input=forecast_models._f_validate_input_default)
        model3 = ForecastModel(
            'model3', 0, forecast_models._f_model_null, l_f_validate_input=[
                forecast_models._f_validate_input_default])

        l_expected = [forecast_models._f_validate_input_default]
        self.assertListEqual(model1.l_f_validate_input, l_expected)
        self.assertListEqual(model2.l_f_validate_input, l_expected)
        self.assertListEqual(model3.l_f_validate_input, l_expected)
        # Check composition
        self.assertListEqual(
            (model1 + model2 + model3).l_f_validate_input,
            l_expected)
        self.assertListEqual(
            (model1 * model2 * model3).l_f_validate_input,
            l_expected)

        # Test2 : test non-default input functions
        def f1(a_x, a_y, a_date):
            assert False
        model4 = ForecastModel('model3', 0, forecast_models._f_model_null,
                               l_f_validate_input=[f1])
        self.assertListEqual(model4.l_f_validate_input, [f1])
        # Check composition
        l_expected = [forecast_models._f_validate_input_default, f1]
        l_result1 = (model1 + model4).l_f_validate_input
        l_result2 = (model1 * model4).l_f_validate_input

        def assert_list_func_equal(l_result, l_expected):
            # can't sort lists of functions, so we need to brute force the
            # equality test
            self.assertEqual(len(l_result), len(l_expected))
            for result in l_result:
                self.assertIn(result, l_expected)

        assert_list_func_equal(l_result1, l_expected)
        assert_list_func_equal(l_result2, l_expected)

        # Test3: model.validate_input()
        self.assertTrue(model1.validate_input(None, None, None))
        self.assertTrue(model2.validate_input(None, None, None))
        self.assertTrue(model3.validate_input(None, None, None))
        self.assertFalse(model4.validate_input(None, None, None))

        self.assertTrue((model1 + model2).validate_input(None, None, None))
        self.assertFalse((model1 + model4).validate_input(None, None, None))

        # Test 4: model_season_wday.validate_input():

        # True if input date series includes all 7 weekdays
        a_date_incomplete = pd.date_range('2018-01-01', periods=5, freq='D')
        a_date_complete = pd.date_range('2018-01-01', periods=50, freq='D')

        self.assertFalse(model_season_wday.validate_input(None, None, None))
        self.assertFalse(
            model_season_wday.validate_input(
                None, None, a_date_incomplete))
        self.assertTrue(
            model_season_wday.validate_input(
                None, None, a_date_complete))

    def test_model_l_cache_vars(self):
        a_date = pd.date_range('2020-01-01', '2020-06-01', freq='M')
        a_x = np.arange(0, a_date.size)

        # table: model - expected cache vars
        df_models = pd.DataFrame(
            columns=['model', 'l_cache_vars_expected'],
            data=[
                [model_linear, []],
                [model_season_wday, 'a_weekday'],
                [model_season_month, 'a_month'],
                [model_season_fourier_yearly, 'a_t_fourier'],
                [model_linear + model_season_wday, 'a_weekday'],
                [model_season_wday * model_season_month,
                 ['a_month', 'a_weekday']]
            ]
        )
        for i, row in df_models.iterrows():
            model = row.model
            l_cache_vars = model.l_cache_vars
            expected = forecast_models._as_list(row.l_cache_vars_expected)
            logger.info('Model: %s, l_cache: %s, expected: %s',
                        model, l_cache_vars, expected)
            self.assertSetEqual(set(l_cache_vars), set(expected))
            dict_cache_vars = model.init_cache(a_x, a_date)
            logger_info('dict cache vars: ', dict_cache_vars)

    def test_model_dict_f_cache(self):
        a_date = pd.date_range('2020-01-01', '2020-06-01', freq='M')
        a_x = np.arange(0, a_date.size)

        model_datelist = get_model_from_datelist(
            'datelist',
            ['2018-01-01', '2018-01-02'],
            ['2018-12-25', '2019-12-25']
        )

        # table: model - expected cache functions
        df_models = pd.DataFrame(
            columns=['model', 'l_cache_vars_expected'],
            data=[
                [model_linear, []],
                [model_calendar_uk, []],
                [model_calendar_us, []],
                [model_calendar_ita, []],
                [model_datelist, []],
                [model_season_wday_2, []]
            ]
        )
        for i, row in df_models.iterrows():
            model = row.model
            dict_f_cache = model.dict_f_cache
            expected = forecast_models._as_list(row.l_cache_vars_expected)
            logger.info('Model: %s, l_cache: %s, expected: %s',
                        model, dict_f_cache, expected)
            # self.assertListEqual(dict_f_cache, expected)
            dict_cache_vars = model.init_cache(a_x, a_date)
            logger_info('dict cache vars: ', dict_cache_vars)

            ## todo: doesn't work

    def test_get_model_from_calendars(self):
        model_calendar = get_model_from_calendars(CalendarChristmasUK())
        logger_info('model_calendar:', model_calendar)
        self.assertEqual(model_calendar.n_params, 1)
        logger_info('parameters:', model_calendar.n_params)

        model_calendar = get_model_from_calendars(
            [CalendarChristmasUK(), CalendarBankHolUK()],
            'calendar2'
        )
        logger_info('model_calendar:', model_calendar)
        self.assertEqual(model_calendar.n_params, 2)
        logger_info('parameters:', model_calendar.n_params)

    def test_get_model_from_date_list(self):
        model_datelist = get_model_from_datelist(
            'datelist',
            ['2018-01-01', '2018-01-02'],
            ['2018-12-25', '2019-12-25']
        )
        logger_info('model_datelist:', model_datelist)
        logger_info('parameters:', model_datelist.n_params)
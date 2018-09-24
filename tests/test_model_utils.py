"""

Author: Pedro Capelastegui
Created on 04/12/2015 
"""

import logging
import unittest
from itertools import chain, repeat

import numpy as np
import os
import itertools
import pandas as pd
from unittest import TestCase
# This line fixes import errors
from anticipy.utils_test import PandasTest
from anticipy.model_utils import *
from anticipy import forecast_models

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def logger_info(msg, data):
    logger.info(msg + '\n%s', data)


# class TestModelUtils(TestCase):
class TestModelUtils(PandasTest):
    def setUp(self):
        pass

    def test_array_transpose(self):
        a = np.arange(10)
        self.assertEqual(a.shape, (10,))
        at = array_transpose(a)
        self.assertEqual(at.shape, (10, 1))

    def test_apply_a_x_scaling(self):
        a_x = np.arange(10)
        l_models = [
            # No model - default config
            None,
            # Model requires omega n
            forecast_models.model_linear,
            # Model requires scaling
            forecast_models.model_sigmoid,
            # aperiodic_models.get_model_logistic_4_param # Todo: Implement and test model function with a_x scaling
        ]
        for model in l_models:
            # No asserts - just check that the function runs without crashing, and manually check results in logs
            a_x = apply_a_x_scaling(a_x, model)
            logger.info('f_model: %s', model)
            logger_info('a_x', a_x)

    def test_get_a_x_date_extrapolate(self):
        # TODO: TEST Output size, scenarios with different frequencies
        l_df_y = [
            # Single ts
            pd.Series(index=pd.date_range('2016-01-01', periods=10, freq='W'),
                      data=np.arange(10)),
            # Multiple ts
            pd.DataFrame(index=pd.date_range('2016-01-01', periods=10, freq='W'),
                         data={'a': np.arange(10), 'b': np.arange(10)})
        ]
        l_models = [
            # No model - default config
            None,
            # Model requires omega n
            forecast_models.model_linear,
            # Model requires scaling
            forecast_models.model_sigmoid,
            # aperiodic_models.get_model_logistic_4_param # Todo: Implement and test model function with a_x scaling
        ]
        l_time_resolutions = [
            # Default config
            'W-SUN',
            'W',
            'W-MON',
            'D',
            'MS',
            'YS'
        ]
        # logger_info('list_ts',l_df_y)
        for (df_type, model, time_resolution) in itertools.product(['single', 'multi'], l_models, l_time_resolutions):
            dict_df = {
                # Single ts
                'single': pd.Series(index=pd.date_range('2016-01-01', periods=10, freq=time_resolution),
                                    data=np.arange(10)),
                # Multiple ts
                'multi': pd.DataFrame(index=pd.date_range('2016-01-01', periods=10, freq=time_resolution),
                                      data={'a': np.arange(10), 'b': np.arange(10)})
            }
            ts = dict_df.get(df_type)

            # No asserts - just check that the function runs without crashing, and manually check results in logs
            s_x = get_s_x_extrapolate(ts.index.min(), ts.index.max(), model=model, freq=time_resolution,
                                      extrapolate_years=1.0)
            logger.info('type of df: %s, f_model: %s , time_resolution: %s', df_type, model, time_resolution)
            logger_info('s_x', s_x.tail(3))
            logger_info('a_dates', s_x.tail(3).index)
            self.assertIsInstance(s_x.index, pd.DatetimeIndex)
            self.assertLessEqual(s_x.index.max(), ts.index.max() + 1.1 * pd.Timedelta(1, 'Y'))
            self.assertGreaterEqual(s_x.index.max(), ts.index.max() + 0.9 * pd.Timedelta(1, 'Y'))

            # Check that all actuals values are in extrapolated series
            self.assertEquals(np.setdiff1d(ts.index, s_x.index).size, 0)

        ts = l_df_y[0]
        model = l_models[0]
        time_resolution = l_time_resolutions[0]
        s_x = get_s_x_extrapolate(ts.index.min(), ts.index.max(), model=model, freq=time_resolution,
                                  extrapolate_years=3.0)
        logger.info('# of ts: %s, f_model: %s , time_resolution: %s', ts.shape, model, time_resolution)
        logger_info('a_x', s_x.head(3))
        logger_info('a_x index', s_x.head(3).index)
        self.assertIsInstance(s_x.index, pd.DatetimeIndex)
        logger_info('t_values len', len(s_x))
        self.assertEquals(len(s_x), 10 + 3.0 * 52)

        # Test with freq='D'
        l_df_y = [
            # Single ts
            pd.Series(index=pd.date_range('2016-01-01', periods=10, freq='D'),
                      data=np.arange(10)),
            # Multiple ts
            pd.DataFrame(index=pd.date_range('2016-01-01', periods=10, freq='D'),
                         data={'a': np.arange(10), 'b': np.arange(10)})
        ]
        l_models = [
            # No model - default config
            None,
            # Model requires omega n
            forecast_models.model_linear,
            # Model requires scaling
            # aperiodic_models.get_model_logistic_4_param # Todo: Implement and test model function with a_x scaling
        ]
        l_time_resolutions = [
            # Default config
            'D'
        ]
        logger_info('list_ts', l_df_y)
        for (ts, model, time_resolution) in itertools.product(l_df_y, l_models, l_time_resolutions):
            # No asserts - just check that the function runs without crashing, and manually check results in logs
            s_x = get_s_x_extrapolate(ts.index.min(), ts.index.max(), model=model, freq=time_resolution)
            logger.info('# of ts: %s, f_model: %s , time_resolution: %s', ts.shape, model, time_resolution)
            logger_info('s_x', s_x.tail(3))
            logger_info('a_dates', s_x.tail(3).index)
            self.assertIsInstance(s_x.index, pd.DatetimeIndex)

        ts = l_df_y[0]
        model = l_models[0]
        time_resolution = l_time_resolutions[0]
        s_x = get_s_x_extrapolate(ts.index.min(), ts.index.max(), model=model, freq=time_resolution,
                                  extrapolate_years=3.0)
        logger.info('# of ts: %s, f_model: %s , time_resolution: %s', ts.shape, model, time_resolution)
        logger_info('t_values', s_x.tail(3))
        logger_info('t_values_index', s_x.index)
        self.assertIsInstance(s_x.index, pd.DatetimeIndex)
        logger_info('t_values len', len(s_x))
        self.assertEquals(len(s_x), 10 + 3.0 * 365)

    def test_get_aic_c(self):

        # Known error scenario: 0 error, 1 parameters - should return  -inf
        aic_c1 = get_aic_c(0, 10, 1)
        logger_info('AIC_C:', aic_c1)
        self.assertTrue(np.isneginf(aic_c1))

        def print_aic_c(fit_error, n, n_params):
            aic_c1 = get_aic_c(fit_error, n, n_params)
            logger.info('AIC_C (%s, %s, %s): %s', fit_error, n, n_params, aic_c1)

        print_aic_c(0.1, 10, 1)
        print_aic_c(0.1, 10, 2)
        print_aic_c(0.1, 10, 3)
        print_aic_c(0.001, 10, 1)
        print_aic_c(0.001, 10, 2)
        print_aic_c(0.001, 10, 3)
        print_aic_c(0.1, 100, 1)
        print_aic_c(0.1, 100, 2)
        print_aic_c(0.1, 100, 3)
        print_aic_c(0, 10, 1)
        print_aic_c(0, 10, 2)
        print_aic_c(0, 10, 3)

    def test_get_s_aic_c_best_result_key(self):
        s_tmp = pd.DataFrame({'c1': [1], 'c2': [2], 'c3': [-np.inf]}).set_index(['c1', 'c2'])['c3']
        result1 = get_s_aic_c_best_result_key(s_tmp)
        logger_info('DEBUG: ', result1)
        self.assertTupleEqual(get_s_aic_c_best_result_key(s_tmp), (1, 2))

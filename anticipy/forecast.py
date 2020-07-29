# -*- coding: utf-8 -*-
#
# License:          This module is released under the terms of the LICENSE file
#                   contained within this applications INSTALL directory

"""
Functions to run forecast
"""

# -- Coding Conventions
#    http://www.python.org/dev/peps/pep-0008/   -   Use the Python style guide
# http://sphinx.pocoo.org/rest.html          -   Use Restructured Text for
# docstrings


# -- Public Imports
from __future__ import division  # Required to avoid python 2 divison bugs
import itertools
import logging
import numpy as np
import pandas as pd
import scipy
from numpy.linalg import LinAlgError
from scipy import optimize
from datetime import datetime

# -- Private Imports
from anticipy import forecast_models, model_utils
# -- Globals
from anticipy.model_utils import detect_freq

logger = logging.getLogger(__name__)


# -- Exception classes

# -- Functions
def logger_info(msg, data):
    # Convenience function for easier log typing
    logger.info(msg + '\n%s', data)


# Utility functions
def to_str_function_list(l_function):
    # Get string representation for a list of functions or ForecastModel
    # instances
    if l_function is None:
        return None
    return [f.name if f is not None else None for f in l_function]


def is_null_model(f_model):
    # Check if model is null model
    return f_model.name == 'null'


def get_residuals(
        params,
        model,
        a_x,
        a_y,
        a_date,
        a_weights=None,
        df_actuals=None):
    """
    Given a time series, a model function and a set of parameters,
    get the residuals

    :param params: parameters for model function
    :type params: numpy array of floats
    :param model: model function. Usage: model(a_x, a_date, params)
    :type model: function or ForecastModel instance
    :param a_x: X axis for model function.
    :type a_x: numpy array of floats
    :param a_y: Input time series values, to compare to the model function
    :type a_y: numpy array of floats
    :param a_date: Dates for the input time series
    :type a_date: numpy array of datetimes
    :param a_weights: weights for each individual sample
    :type a_weights: numpy array of floats
    :param df_actuals: The original dataframe with actuals data.
                       Not required for regression but used by naive models
    :type df_actuals: pandas DataFrame
    :return: array with residuals, same length as a_x, a_y
    :rtype: numpy array of floats
    """
    # Note: remove this assert for performance
    assert a_y.ndim == 1
    # Note: none of the input arrays should include NaN values
    # We do not check this with asserts due to performance - this function is
    # in the optimization loop

    y_predicted = model(a_x, a_date, params, df_actuals=df_actuals)
    residuals = (a_y - y_predicted)
    if a_weights is not None:  # Do this only if different residual weights
        residuals = residuals * a_weights
    result = np.abs(residuals)
    return result


def optimize_least_squares(
        model,
        a_x,
        a_y,
        a_date,
        a_weights=None,
        df_actuals=None):
    """
    Given a time series and a model function, find the set of
    parameters that minimises residuals

    :param model: model function, to be fitted against the actuals
    :type model: function
    :param a_x: X axis for model function.
    :type a_x: numpy array of floats
    :param a_y: Input time series values, to compare to the model function
    :type a_y: numpy array of floats
    :param a_date: Dates for the input time series
    :type a_date: numpy array of datetimes
    :param a_weights: weights for each individual sample
    :type a_weights: numpy array of floats
    :param df_actuals: The original dataframe with actuals data.
                       Not required for regression but used by naive models
    :type df_actuals: pandas DataFrame
    :return:
        | table(success, params, cost, optimality,
        |       iterations, status, jac_evals, message):
        |
        | - success (bool): True if successful fit
        | - params (list): Parameters of fitted model
        | - cost (float): Value of cost function
        | - optimality(float)
        | - iterations (int) : Number of function evaluations
        | - status (int) : Status code
        | - jac_evals(int) : Number of Jacobian evaluations
        | - message (str) : Output message
    :rtype: pandas.DataFrame
    """
    assert a_y.ndim == 1

    # Ask the model to provide an initial guess
    initial_guess = model.f_init_params(a_x, a_y)

    bounds = model.f_bounds(a_x, a_y)

    assert forecast_models.validate_initial_guess(initial_guess, bounds), \
        'Initial guess outside of bounds: {} - {}, {}'.format(
            model, initial_guess, bounds)

    # In multi-ts scenarios, we apply this filter to ignore residuals for null
    # y_values
    filter_null_residuals = ~np.isnan(a_y)
    if np.all(filter_null_residuals):
        filter_null_residuals = None

    # Set up arguments for get_residuals
    f_model_args = (model, a_x, a_y, a_date)
    try:
        result = scipy.optimize.least_squares(
            get_residuals, initial_guess,
            args=f_model_args,
            kwargs={
                'a_weights': a_weights,
                'df_actuals': df_actuals},
            # method='lm',
            method='trf',
            x_scale='jac',
            # verbose=1,
            bounds=bounds
        )
        dict_result_df = {
            'optimality': result['optimality'],
            'success': result['success'],
            'cost': result['cost'],
            'iterations': result['nfev'],
            'jac_evals': result['njev'],
            'status': result['status'],
            'message': result['message'],
            'params': [result['x']]
        }
    except LinAlgError as e:
        logger.info('LinAlgError: Model did not converge - %s', model)
        dict_result_df = {
            'optimality': 0.,
            'success': False,
            'cost': np.NaN,
            'iterations': 0.,
            'jac_evals': 0.,
            'status': 0,
            'message': 'LinAlgError',
            'params': None
        }

    df_result = pd.DataFrame(data=dict_result_df, index=pd.Index([0]))
    df_result = df_result[['success',
                           'params',
                           'cost',
                           'optimality',
                           'iterations',
                           'status',
                           'jac_evals',
                           'message']]
    return df_result


def _get_df_fit_model(source, model, weights, actuals_x_range, freq,
                      is_fit, cost, aic_c, params, status,
                      fit_time=0):
    # Generate a metadata dataframe for the output of fit_model()
    if params is None:
        params = np.array([])
    df_result = (
        pd.DataFrame(
            columns=[
                'source',
                'model',
                'weights',
                'actuals_x_range',
                'freq',
                'is_fit',
                'cost',
                'aic_c',
                'fit_time',
                'params_str',
                'status',
                'source_long',
                'params',
                'model_obj'
            ],
            data=[[
                source,
                model.name,
                weights,
                actuals_x_range,
                freq,
                is_fit,
                cost,
                aic_c,
                fit_time,
                np.array_str(params, precision=1),
                status,
                '{}:{}:{}:{}'.format(
                    source, weights, freq, actuals_x_range),
                params,
                model
            ]]))
    return df_result


def _get_empty_df_result_optimize(
        source,
        model,
        status,
        weights,
        freq,
        actuals_x_range):
    # Generate an optimize_info dataframe for fit_model(), for an empty result
    source_long = '{}:{}:{}:{}'.format(source, weights, freq, actuals_x_range)
    return pd.DataFrame(columns=['source',
                                 'model',
                                 'success',
                                 'params_str',
                                 'cost',
                                 'optimality',
                                 'iterations',
                                 'status',
                                 'jac_evals',
                                 'message',
                                 'source_long',
                                 'params'],
                        data=[[source,
                               model,
                               False,
                               '[]',
                               np.NaN,
                               np.NaN,
                               np.NaN,
                               status,
                               np.NaN,
                               status,
                               source_long,
                               []]])


def normalize_df(df_y,
                 col_name_y='y',
                 col_name_weight='weight',
                 col_name_x='x',
                 col_name_date='date',
                 col_name_source='source'):
    """
    Converts an input dataframe for run_forecast() into a normalized format
    suitable for fit_model()

    :param df_y: unformatted input dataframe, for use by run_forecast()
    :type df_y: pandas.DataFrame
    :param col_name_y: name for column with time series values
    :type col_name_y: basestring
    :param col_name_weight: name for column with time series weights
    :type col_name_weight: basestring
    :param col_name_x: name for column with time series indices
    :type col_name_x: basestring
    :param col_name_date: name for column with time series dates
    :type col_name_date: basestring
    :param col_name_source: name for column with time series
                            source identifiers
    :type col_name_source: basestring
    :return: formatted input dataframe, for use by run_forecast()
    :rtype: pandas.DataFrame
     """

    assert df_y is not None
    if df_y.empty:
        return None

    if isinstance(df_y, pd.Series):
        df_y = df_y.to_frame()
    assert isinstance(df_y, pd.DataFrame)
    assert col_name_y in df_y.columns, \
        'Dataframe needs to have a column named "{}"'.format(
            col_name_y)
    df_y = df_y.copy()

    # Rename columns to normalized values
    rename_col_dict = {
        col_name_y: 'y',
        col_name_weight: 'weight',
        col_name_x: 'x',
        col_name_date: 'date',
        col_name_source: 'source'
    }
    df_y = df_y.copy().rename(rename_col_dict, axis=1)

    # Placeholder - need to replace all references to col_name_z with z
    col_name_y = 'y'
    col_name_weight = 'weight'
    col_name_x = 'x'
    col_name_date = 'date'
    col_name_source = 'source'

    # Ensure y column is float
    df_y[col_name_y] = df_y[col_name_y].astype(float)

    multiple_sources = col_name_source in df_y.columns
    l_sources = df_y[col_name_source].drop_duplicates()\
        if multiple_sources else ['test_source']

    l_df_results = []
    for source in l_sources:
        df_y_tmp = df_y.loc[df_y[col_name_source] ==
                            source].copy() if multiple_sources else df_y
        # Setup date, x columns
        if col_name_date not in df_y.columns and isinstance(
                df_y.index, pd.DatetimeIndex):  # use index as i_date
            df_y_tmp[col_name_date] = df_y_tmp.index
        elif col_name_date in df_y.columns:
            # Ensure that date column is timestamp dtype
            df_y_tmp[col_name_date] = df_y_tmp[col_name_date].pipe(
                pd.to_datetime)

        # if isinstance(df_y_tmp.index, pd.DatetimeIndex):
        # We don't need a date index after this point
        df_y_tmp = df_y_tmp.reset_index(drop=True)

        if col_name_x not in df_y_tmp.columns:
            if col_name_date in df_y_tmp.columns:
                # Need to extract numeric index from a_date
                df_date_interp = (
                    df_y_tmp[[col_name_date]]
                        .drop_duplicates()
                        .pipe(model_utils.interpolate_df, interpolate_y=False)
                        .rename_axis(col_name_x)
                        .reset_index())
                df_y_tmp = (
                    df_date_interp.merge(df_y_tmp)
                )
            else:  # With no date, extract column x from a numeric index
                df_y_tmp[col_name_x] = df_y_tmp.index

        l_df_results += [df_y_tmp]

    # Rename columns to normalized values
    rename_col_dict = {
        col_name_y: 'y',
        col_name_weight: 'weight',
        col_name_x: 'x',
        col_name_date: 'date',
        col_name_source: 'source'
    }

    df_result = pd.concat(l_df_results, sort=False, ignore_index=True)

    # Sort columns, filter unused columns
    df_result = df_result[[c for c in [
        'date', 'source', 'x', 'y', 'weight'] if c in df_result.columns]]
    sort_columns = ['source', 'x'] if 'source' in df_result.columns else ['x']
    df_result = df_result.sort_values(sort_columns).reset_index(drop=True)
    return df_result


def fit_model(model, df_y, freq='W', source='test', df_actuals=None):
    """
    Given a time series and a model, optimize model parameters and return

    :param model: model function. Usage: model(a_x, a_date, params)
    :type model: function or ForecastModel instance
    :param df_y:
        | Dataframe with the following columns:
        | - y:
        | - date: (optional)
        | - weight: (optional)
        | - x: (optional)
    :type df_y: pandas.DataFrame
    :param source: source identifier for this time series
    :type source: basestring
    :param freq: 'W' or 'D' . Used only for metadata
    :type freq: basestring
    :param df_actuals: The original dataframe with actuals data.
        Not required for regression but used by naive models
    :type df_actuals: pandas DataFrame
    :return: table (source, model_name, y_weights , freq, is_fit, aic_c,
        params)
    :rtype: pandas.DataFrame

    This function calls optimize_least_squares() to perform the optimization
    loop. It performs some cleaning up of input
    and output parameters.
    """
    col_name_y = 'y'
    col_name_weight = 'weight'
    col_name_x = 'x'
    col_name_date = 'date'

    assert df_y is not None and isinstance(
        df_y, pd.DataFrame) and col_name_y in df_y.columns

    # Setup
    f_model_name = model.name
    n_params = model.n_params

    df_y = df_y.copy()
    # Filter out any sample where df_y is null
    df_y = df_y.loc[~df_y[col_name_y].pipe(pd.isna)]

    # Metadata
    if col_name_weight not in df_y.columns:
        weights = '1'
    else:
        weights = '{}-{}'.format(df_y[col_name_weight].min(),
                                 df_y[col_name_weight].max())

    # Filter out any sample where a_weights is 0
    if col_name_weight in df_y.columns:
        df_y[col_name_weight] = df_y[col_name_weight].fillna(0)
        df_y = df_y.loc[df_y[col_name_weight] > 0]

    # Residual normalization
    if df_y[col_name_x].duplicated().any():
        df_k = df_y.groupby(col_name_x).size().rename(
            'k_weight_normalize').reset_index()
        df_y = df_y.merge(df_k)
        if col_name_weight not in df_y:
            df_y[col_name_weight] = 1.0
        # Adjust residual weight based on number of values per sample
        # E.g. a sample with 2 values in the input series will multiply
        # residuals by 0.5
        df_y[col_name_weight] = df_y[col_name_weight] /\
            df_y['k_weight_normalize']

    # Get input arrays
    a_y = df_y[col_name_y].values
    a_x = model_utils.apply_a_x_scaling(df_y[col_name_x].values, model)
    a_weights = df_y[col_name_weight].values \
        if col_name_weight in df_y.columns else None
    # Need to convert series to DatetimeIndex
    i_date = pd.DatetimeIndex(
        df_y[col_name_date]) if col_name_date in df_y.columns else None

    # Metadata
    cost = np.NaN
    is_fit = False
    params = []
    # Get first and last actuals date, for metadata. If no a_date, use a_x
    # instead.
    date_start_actuals = i_date.min().date()\
        if i_date is not None else a_x.min()
    date_end_actuals = i_date.max().date()\
        if i_date is not None else a_x.max()
    actuals_x_range = '{}::{}'.format(date_start_actuals, date_end_actuals)

    if df_y.empty:
        logger.info('Cannot fit - empty df_y: %s', source)
        status = 'EMPTY_TS'
        df_result = _get_df_fit_model(
            source,
            model,
            weights,
            actuals_x_range,
            freq,
            is_fit,
            cost,
            np.NaN,
            None,
            status)
        df_result_optimize = _get_empty_df_result_optimize(
            source, model, status, weights, freq, actuals_x_range)

    elif a_x.size < n_params + 2:
        logger.info('Not enough samples in source %s for %s: %s (needs %s)',
                    source, f_model_name, a_x.size, n_params + 2)
        status = 'TS_TOO_SHORT'
        df_result = _get_df_fit_model(
            source,
            model,
            weights,
            actuals_x_range,
            freq,
            is_fit,
            cost,
            np.NaN,
            None,
            status)
        df_result_optimize = _get_empty_df_result_optimize(
            source, model, status, weights, freq, actuals_x_range)
    elif not model.validate_input(a_x, a_y, i_date):
        logger.info('Invalid input for %s for %s:', source, f_model_name)
        status = 'INPUT_ERR'
        df_result = _get_df_fit_model(
            source,
            model,
            weights,
            actuals_x_range,
            freq,
            is_fit,
            cost,
            np.NaN,
            None,
            status)
        df_result_optimize = _get_empty_df_result_optimize(
            source, model, status, weights, freq, actuals_x_range)
    else:  # Get results
        model = forecast_models.simplify_model(model, a_x, a_y, i_date)

        if model.n_params == 0:
            # 0-parameter model, no fit required
            time_start = datetime.now()
            a_residuals = get_residuals(
                None,
                model,
                a_x,
                a_y,
                i_date,
                a_weights,
                df_actuals=df_actuals)
            runtime_res = datetime.now() - time_start
            cost = 0.5 * np.nansum(a_residuals ** 2)
            is_fit = True
            params = np.array([])
            status = 'FIT'

            # Process results
            aic_c = model_utils.get_aic_c(cost, len(df_y), n_params)

            df_result = _get_df_fit_model(
                source,
                model,
                weights,
                actuals_x_range,
                freq,
                is_fit,
                cost,
                aic_c,
                params,
                status,
                fit_time=runtime_res
            )

            dict_result_df = {
                'optimality': 0.,
                'success': True,
                'cost': cost,
                'iterations': 0.,
                'jac_evals': 0.,
                'status': 0,
                'message': 'Naive model fitted',
                'params': '-'
            }
            df_result_optimize = pd.DataFrame(
                data=dict_result_df, index=pd.Index([0]))
            df_result_optimize = df_result_optimize[[
                'success', 'params', 'cost', 'optimality', 'iterations',
                'status', 'jac_evals', 'message']]
            df_result_optimize['source'] = source
            df_result_optimize['source_long'] = df_result.source_long.iloc[0]
            df_result_optimize['model'] = model
            df_result_optimize['params_str'] = df_result.params_str.iloc[0]
            df_result_optimize = df_result_optimize[['source',
                                                     'model',
                                                     'success',
                                                     'params_str',
                                                     'cost',
                                                     'optimality',
                                                     'iterations',
                                                     'status',
                                                     'jac_evals',
                                                     'message',
                                                     'source_long',
                                                     'params']]
        else:
            time_start = datetime.now()
            df_result_optimize = optimize_least_squares(
                model, a_x, a_y, i_date, a_weights, df_actuals=df_actuals)
            runtime = datetime.now() - time_start
            cost = df_result_optimize.cost.iloc[0]
            is_fit = df_result_optimize.success.iloc[0]
            params = df_result_optimize.params.iloc[0]
            status = 'FIT' if is_fit else 'NO-FIT'

            # Process results
            if status in ['FIT', 'NO-FIT']:
                aic_c = model_utils.get_aic_c(cost, len(df_y), n_params)
            else:
                aic_c = np.NaN

            df_result = _get_df_fit_model(
                source,
                model,
                weights,
                actuals_x_range,
                freq,
                is_fit,
                cost,
                aic_c,
                params,
                status,
                runtime
            )

            df_result_optimize['source'] = source
            df_result_optimize['source_long'] = df_result.source_long.iloc[0]
            df_result_optimize['model'] = model
            df_result_optimize['params_str'] = df_result.params_str.iloc[0]
            df_result_optimize = df_result_optimize[['source',
                                                     'model',
                                                     'success',
                                                     'params_str',
                                                     'cost',
                                                     'optimality',
                                                     'iterations',
                                                     'status',
                                                     'jac_evals',
                                                     'message',
                                                     'source_long',
                                                     'params']]

    dict_result = {'metadata': df_result, 'optimize_info': df_result_optimize}
    return dict_result


def extrapolate_model(
        model,
        params,
        date_start_actuals,
        date_end_actuals,
        freq='W',
        extrapolate_years=2.0,
        x_start_actuals=0.,
        df_actuals=None):
    """
    Given a model and a set of parameters, generate model output for a date
    range plus a number of additional years.

    :param model: model function. Usage: model(a_x, a_date, params)
    :type model: function or ForecastModel instance
    :param params: parameters for model function
    :type params: numpy array of floats
    :param date_start_actuals: date or numeric index for first actuals sample
    :type date_start_actuals: str, datetime, int or float
    :param date_end_actuals: date or numeric index for last actuals sample
    :type date_end_actuals: str, datetime, int or float
    :param freq: Time unit between samples. Supported units are 'W' for
        weekly samples, or 'D' for daily samples.
        (untested) Any date unit or time unit accepted by numpy should also
        work, see
        https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.datetime.html#arrays-dtypes-dateunits # noqa
    :type freq: basestring
    :param extrapolate_years: Number of years (or fraction of year) covered by
        the generated time series, after the
        end of the actuals
    :type extrapolate_years: float
    :param x_start_actuals:
    :type x_start_actuals:
    :param df_actuals: The original dataframe with actuals data. Not required
        for regression but used by naive models
    :type df_actuals: pandas DataFrame
    :return: dataframe with a time series extrapolated from the model function
    :rtype: pandas.DataFrame, with an 'y' column of floats
    """
    s_x = model_utils.get_s_x_extrapolate(
        date_start_actuals,
        date_end_actuals,
        model=model,
        freq=freq,
        extrapolate_years=extrapolate_years,
        x_start_actuals=x_start_actuals)
    a_y_forecast = model(s_x.values, s_x.index, params, df_actuals=df_actuals)
    s_y_forecast = pd.Series(data=a_y_forecast, index=s_x.index, name='y')
    df_y_forecast = pd.DataFrame(s_y_forecast)
    return df_y_forecast


def get_list_model(l_model_trend, l_model_season, season_add_mult='both'):
    """
    Generate a list of composite models from lists of trend and seasonality
    models

    :param l_model_trend: list of trend models
    :type l_model_trend: list of ForecastModel
    :param l_model_season: list of seasonality models
    :type l_model_season: list of ForecastModel
    :param season_add_mult: 'mult', 'add' or 'both', for
        multiplicative/additive composition (or both types)
    :type season_add_mult: basestring
    :return:
    :rtype: list of ForecastModel

    All combinations of possible composite models are included
    """
    if l_model_season is None or len(l_model_season) < 1:
        l_model_tmp = l_model_trend
    elif l_model_trend is None or len(l_model_trend) < 1:
        l_model_tmp = l_model_season
    else:
        l_model_tmp = []
        if season_add_mult != 'mult':  # 'add' or 'both'
            l_model_tmp += [model_trend + model_season for
                            model_trend, model_season in
                            itertools.product(l_model_trend, l_model_season)]
        if season_add_mult != 'add':  # 'mult' or 'both'
            l_model_tmp += [model_trend * model_season for
                            model_trend, model_season in
                            itertools.product(l_model_trend, l_model_season)]

    l_model_tmp = pd.Series(l_model_tmp).drop_duplicates().tolist()
    return l_model_tmp


def get_df_actuals_clean(df_actuals, source, source_long):
    """
    Convert an actuals dataframe to a clean format

    :param df_actuals: dataframe in normalized format, with columns y and
        optionally x, date, weight
    :type df_actuals: pandas.DataFrame
    :param source: source identifier for this time series
    :type source: basestring
    :param source_long: long-format source identifier for this time series
    :type source_long: basestring
    :return: clean actuals dataframe
    :rtype: pandas.DataFrame
    """
    # Add actuals as entries in result dicts
    df_actuals = df_actuals.copy()  # .rename_axis('date')
    if 'date' not in df_actuals.columns:
        df_actuals = df_actuals.rename({'x': 'date'}, axis=1)
    df_actuals = df_actuals[[c for c in [
        'date', 'weight', 'y'] if c in df_actuals.columns]]

    df_actuals['model'] = 'actuals'
    df_actuals['source'] = source
    df_actuals['source_long'] = source_long
    df_actuals['is_actuals'] = True
    if 'weight' not in df_actuals.columns:
        df_actuals['weight'] = 1.0
    return df_actuals


def _get_df_fcast_clean(df_fcast, source, source_long, model):
    # This removes any forecast samples with null values, e.g. from naive
    # models
    df_fcast = df_fcast.loc[~df_fcast.y.pipe(pd.isnull)]
    df_fcast = df_fcast.copy().rename_axis('date').reset_index()
    df_fcast['source'] = source
    df_fcast['source_long'] = source_long
    df_fcast['model'] = model
    df_fcast['is_actuals'] = False
    df_fcast['weight'] = 1.0
    return df_fcast


def run_forecast(df_y, l_model_trend=None, l_model_season=None,
                 date_start_actuals=None, source_id='src',
                 col_name_y='y', col_name_weight='weight',
                 col_name_x='x', col_name_date='date',
                 col_name_source='source',
                 extrapolate_years=0, season_add_mult='add',
                 include_all_fits=False,
                 simplify_output=True,
                 find_outliers=False,
                 l_season_yearly=None,
                 l_season_weekly=None,
                 verbose=None,
                 l_model_naive=None,
                 l_model_calendar=None,
                 n_cum=1,
                 pi_q1=5,
                 pi_q2=20
                 ):
    """
    Generate forecast for one or more input time series

    :param df_y:
        | input dataframe with the following columns:
        | - Mandatory: a value column, with the time series values
        | - Optional: weight column, source ID column, index column, date
        |             column
    :type df_y: pandas.DataFrame
    :param l_model_trend: list of trend models
    :type l_model_trend: list of ForecastModel
    :param l_model_season: list of seasonality models
    :type l_model_season: list of ForecastModel
    :param date_start_actuals: date or numeric index for first actuals sample
        to be used for forecast. Previous samples are ignored
    :type date_start_actuals: str, datetime, int or float
    :param source_id: source identifier for time series, if source column
        is missing
    :type source_id: basestring
    :param col_name_y: name for column with time series values
    :type col_name_y: basestring
    :param col_name_weight: name for column with time series weights
    :type col_name_weight: basestring
    :param col_name_x: name for column with time series indices
    :type col_name_x: basestring
    :param col_name_date: name for column with time series dates
    :type col_name_date: basestring
    :param col_name_source: name for column with time series source
        identifiers
    :type col_name_source: basestring
    :param extrapolate_years: Number of years (or fraction of year) covered
        by the forecast, after the end of the actuals
    :type extrapolate_years: float
    :param season_add_mult: 'add', 'mult', or 'both'. Whether forecast
        seasonality will be additive, multiplicative, or the best fit
        of the two.
    :type season_add_mult: str
    :param find_outliers: If True, find outliers in input data, ignore
        outlier samples in forecast
    :type find_outliers: bool
    :param include_all_fits: If True, also include non-optimal models in
        output
    :type include_all_fits: bool
    :param simplify_output: If False, return dict with forecast and metadata.
        Otherwise, return only forecast.
    :type simplify_output: bool
    :param l_season_yearly: yearly seasonality models to consider in automatic
        seasonality detection
    :type l_season_yearly: list of ForecastModel
    :param l_season_weekly: yearly seasonality models to consider in automatic
        seasonality detection
    :type l_season_weekly: list of ForecastModel
    :param verbose: If True, enable verbose logging
    :type verbose: bool
    :param l_model_naive: list of naive models to consider for forecast.
        Naive models are not fitted with regression, they are based on
        the last actuals samples
    :type l_model_naive: list of ForecastModel
    :param l_model_calendar: list of calendar models to consider for forecast,
        to handle holidays and calendar-based events
    :type l_model_calendar: list of ForecastModel
    :param n_cum: Used for widening prediction interval. Interval widens every
        n_sims samples.
    :type n_cum: int
    :param pi_q1: Percentile for outer prediction interval (defaults to 5%-95%)
    :type pi_q1: int
    :param pi_q2: Percentile for inner prediction interval (defaults to 20%-80%)
    :type pi_q2: int
    :return:
        | With simplify_output=False, returns a dictionary with 4 dataframes:
        | - forecast: output time series with prediction interval
        | - data: output time series. If include_all_fits, includes all fitting
        |         models
        | - metadata: forecast metadata table
        | - optimize_info: debugging metadata from scipy.optimize
        |
        | With simplify_output=True, returns the 'forecast' dataframe,
        | as described above
    :rtype: pandas.DataFrame or dict of pandas.DataFrames

    """
    # TODO: Add check for non-duplicate source ids
    l_dict_result = []

    df_y = normalize_df(
        df_y,
        col_name_y,
        col_name_weight,
        col_name_x,
        col_name_date,
        col_name_source)
    if df_y is None:  # Empty input
        return None

    if 'source' not in df_y.columns:
        return run_forecast_single(df_y,
                                   l_model_trend,
                                   l_model_season,
                                   date_start_actuals,
                                   source_id,
                                   extrapolate_years,
                                   season_add_mult,
                                   include_all_fits,
                                   simplify_output,
                                   find_outliers,
                                   l_season_yearly,
                                   l_season_weekly,
                                   l_model_naive=l_model_naive,
                                   l_model_calendar=l_model_calendar,
                                   n_cum=n_cum,
                                   pi_q1=pi_q1,
                                   pi_q2=pi_q2
                                   )
    else:
        for src_tmp in df_y.source.drop_duplicates():
            if verbose:
                logger.info('Running forecast for source: %s', src_tmp)
            df_y_tmp = df_y.loc[df_y.source == src_tmp].reset_index(drop=True)
            dict_result_tmp = run_forecast_single(
                df_y_tmp,
                l_model_trend,
                l_model_season,
                date_start_actuals,
                src_tmp,
                extrapolate_years,
                season_add_mult,
                include_all_fits,
                False,  # Simplify output
                find_outliers,
                l_season_yearly,
                l_season_weekly,
                l_model_naive=l_model_naive,
                l_model_calendar=l_model_calendar,
                n_cum=n_cum,
                pi_q1=pi_q1,
                pi_q2=pi_q2
            )
            l_dict_result += [dict_result_tmp]
    # Generate output
    dict_result = aggregate_forecast_dict_results(l_dict_result)
    if simplify_output:
        return dict_result.get('forecast')
    else:
        return dict_result


def aggregate_forecast_dict_results(l_dict_result):
    """
    Aggregates a list of dictionaries with forecast outputs into a single
    dictionary

    :param l_dict_result: list with outputs dictionaries from
        run_forecast_single
    :type l_dict_result: list of dictionaries
    :return: aggregated dictionary
    :rtype: dict
    """
    l_df_data = []
    l_df_metadata = []
    l_df_optimize_info = []
    # Forecast with prediction interval
    l_df_forecast = []

    for dict_result in l_dict_result:
        l_df_data += [dict_result['data']]
        l_df_metadata += [dict_result['metadata']]
        l_df_optimize_info += [dict_result['optimize_info']]
        l_df_forecast += [dict_result['forecast']]

    # Generate output
    df_data = pd.concat(l_df_data, sort=False, ignore_index=True)
    df_metadata = pd.concat(l_df_metadata, sort=False, ignore_index=True)
    df_optimize_info = pd.concat(
        l_df_optimize_info,
        sort=False,
        ignore_index=True)
    df_forecast = pd.concat(l_df_forecast, sort=False, ignore_index=True)

    return {
        'forecast': df_forecast,
        'data': df_data,
        'metadata': df_metadata,
        'optimize_info': df_optimize_info}


def _get_use_ramp(df_y):
    # Return True if piecewise linear should be used
    # By default,  piecewise linear requires > 2y
    if 'date' in df_y.columns:
        # Add calendar models
        s_date_tmp = df_y.date
        if 'weight' in df_y.columns:
            s_date_tmp = s_date_tmp.loc[df_y.weight > 0]

        s_date = s_date_tmp.sort_values().drop_duplicates()
        max_date_delta = s_date.max() - s_date.min()

        if pd.isna(max_date_delta):
            use_ramp = False
        else:
            use_ramp = (
                # Need more than 2 full years
                (max_date_delta > pd.Timedelta(2 * 365, unit='d'))
            )
    else:
        use_ramp = False
    return use_ramp


def _get_use_calendar(df_y):
    # Return True if calendar models should be used
    # use when x
    if 'date' in df_y.columns:
        # Add calendar models
        s_date_tmp = df_y.date
        if 'weight' in df_y.columns:
            s_date_tmp = s_date_tmp.loc[df_y.weight > 0]

        s_date = s_date_tmp.sort_values().drop_duplicates()
        min_date_delta = s_date.diff().min()
        max_date_delta = s_date.max() - s_date.min()

        if pd.isna(min_date_delta) or pd.isna(max_date_delta):
            use_calendar = False
        else:
            use_calendar = (
                # Need more than a full year
                    (max_date_delta > pd.Timedelta(365, unit='d')) &
                    # Need at least daily samples
                    (min_date_delta <= pd.Timedelta(1, unit='d'))
            )
    else:
        use_calendar = False
    return use_calendar


def _get_model_is_add_mult(model, add_mult_in, l_model_add):
    """Return add_mult string for a model"""
    # If add_mult_in is add or mult, return that
    if add_mult_in in ('add', 'mult'):
        return add_mult_in
    # Otherwise, need to check if model is in list of additive models
    if model in l_model_add:
        return 'add'
    else:
        return 'mult'


def run_forecast_single(df_y,
                        l_model_trend=None,
                        l_model_season=None,
                        date_start_actuals=None,
                        source_id='src',
                        extrapolate_years=0,
                        season_add_mult='add',
                        include_all_fits=False,
                        simplify_output=True,
                        find_outliers=False,
                        l_season_yearly=None,
                        l_season_weekly=None,
                        l_model_naive=None,
                        l_model_calendar=None,
                        n_cum=1,
                        pi_q1=5,
                        pi_q2=20
                        ):
    """
    Generate forecast for one input time series

    :param df_y:
        | input dataframe with the following columns:
        | - y: time series values
        | - x: time series indices
        | - weight: time series weights (optional)
        | - date: time series dates (optional)
    :type df_y: pandas.DataFrame
    :param l_model_trend: list of trend models
    :type l_model_trend: list of ForecastModel
    :param l_model_season: list of seasonality models
    :type l_model_season: list of ForecastModel
    :param date_start_actuals: date or numeric index for first actuals sample
        to be used for forecast. Previous samples are ignored
    :type date_start_actuals: str, datetime, int or float
    :param source_id: source identifier for time series
    :type source_id: basestring
    :param extrapolate_years:
    :type extrapolate_years: float
    :param season_add_mult: 'add', 'mult', or 'both'. Whether forecast
        seasonality will be additive, multiplicative, or the best fit
        of the two.
    :type season_add_mult: str
    :param include_all_fits: If True, also include non-optimal models in
        output
    :type include_all_fits: bool
    :param simplify_output: If False, return dict with forecast and metadata.
        Otherwise, return only forecast.
    :type simplify_output: bool
    :param find_outliers: If True, find outliers in input data, ignore outlier
        samples in forecast
    :type find_outliers: bool
    :param l_season_yearly: yearly seasonality models to consider in automatic
        seasonality detection
    :type l_season_yearly: list of ForecastModel
    :param l_season_weekly: yearly seasonality models to consider in automatic
        seasonality detection
    :type l_season_weekly: list of ForecastModel
    :param l_model_naive: list of naive models to consider for forecast.
        Naive models are not fitted with regression, they are based on
        the last actuals samples
    :type l_model_naive: list of ForecastModel
    :param l_model_calendar: list of calendar models to consider for forecast,
        to handle holidays and calendar-based events
    :type l_model_calendar: list of ForecastModel
    :param n_cum: Used for widening prediction interval. Interval widens every
        n_sims samples.
    :type n_cum: int
    :param pi_q1: Percentile for outer prediction interval (defaults to 5%-95%)
    :type pi_q1: int
    :param pi_q2: Percentile for inner prediction interval (defaults to 20%-80%)
    :type pi_q2: int
    :return:
        | With simplify_output=False, returns a dictionary with 4 dataframes:
        | - forecast: output time series with prediction interval
        | - data: output time series. If include_all_fits, includes all fitting
        |         models
        | - metadata: forecast metadata table
        | - optimize_info: debugging metadata from scipy.optimize
        |
        | With simplify_output=True, returns the 'forecast' dataframe,
        | as described above
    :rtype: pandas.DataFrame or dict of pandas.DataFrames
    """
    l_df_data = []
    l_df_metadata = []
    l_df_optimize_info = []

    # Each element in l_fcast_input describes all model configurations for a
    # source time series
    source = source_id

    if 'date' in df_y.columns:
        freq = detect_freq(df_y.date)
    else:
        freq = None

    df_y = df_y.copy()
    df_y_unfiltered = df_y.copy()

    # Filter: only actuals after date_start_actuals
    if date_start_actuals is not None and 'date' in df_y.columns:
        df_y = df_y.loc[df_y.date >= date_start_actuals]

    date_start_actuals = df_y.date.min() \
        if 'date' in df_y.columns else df_y.x.min()
    date_end_actuals = df_y.date.max() \
        if 'date' in df_y.columns else df_y.x.max()

    # If we find outliers, we add a model with dummy variables for the outliers
    if find_outliers:
        mask_step, mask_spike = forecast_models.get_model_outliers(df_y)
        # Make weight = 0 to ignore spike outliers
        if 'weight' in df_y.columns:
            df_y['weight'] = df_y['weight'] * (~mask_spike).astype(float)
        else:
            df_y['weight'] = (~mask_spike).astype(float)
        assert np.issubdtype(df_y.weight.astype(float), np.float64)
        # TODO add models for steps

    # Add actuals to output
    # Get weight for metadata
    if 'weight' not in df_y.columns:
        df_y['weight'] = 1
        weights = '1'
    else:
        weights = '{}-{}'.format(df_y['weight'].min(), df_y['weight'].max())

    # Get long source_id
    if isinstance(date_start_actuals, datetime):
        date_start_actuals_short = date_start_actuals.date()
        date_end_actuals_short = date_end_actuals.date()
    else:
        date_start_actuals_short = date_start_actuals
        date_end_actuals_short = date_end_actuals
    actuals_x_range = '{}::{}'.format(
        date_start_actuals_short,
        date_end_actuals_short)
    source_long = '{}:{}:{}:{}'.format(source, weights, freq, actuals_x_range)
    df_actuals = get_df_actuals_clean(df_y, source, source_long)
    l_df_data += [df_actuals]

    if l_model_trend is None:
        # By default use linear trend, and piecewise linear for series > 2y
        if _get_use_ramp(df_y):
            l_model_trend = [
                forecast_models.model_linear,
                forecast_models.model_linear + forecast_models.model_ramp]
        else:
            l_model_trend = [forecast_models.model_linear]

    l_model_season_add = None
    l_model_season_mult = None
    if l_model_season is None:
        if 'date' in df_y.columns:
            s_date_tmp = df_y.date
            if 'weight' in df_y.columns:
                s_date_tmp = s_date_tmp.loc[df_y.weight > 0]

            l_model_season_add = forecast_models.get_l_model_auto_season(
                s_date_tmp,
                season_add_mult='add',
                l_season_yearly=l_season_yearly,
                l_season_weekly=l_season_weekly,
            )
            l_model_season_mult = forecast_models.get_l_model_auto_season(
                s_date_tmp,
                season_add_mult='mult',
                l_season_yearly=l_season_yearly,
                l_season_weekly=l_season_weekly,
            )
    else:
        l_model_season_add = l_model_season
        l_model_season_mult = l_model_season

    l_model_add = get_list_model(l_model_trend, l_model_season_add, 'add')
    l_model_mult = get_list_model(l_model_trend, l_model_season_mult, 'mult')

    if l_model_calendar is not None and 'date' in df_y.columns:
        # Add calendar models

        s_date_tmp = df_y.date
        if 'weight' in df_y.columns:
            s_date_tmp = s_date_tmp.loc[df_y.weight > 0]

        # Requires daily samples, at least one year worth of data
        # TODO: Should also check for missing calendar dates
        s_date = s_date_tmp.sort_values().drop_duplicates()
        min_date_delta = s_date.diff().min()
        max_date_delta = s_date.max() - s_date.min()

        if pd.isna(min_date_delta) or pd.isna(max_date_delta):
            use_calendar = False
        else:
            use_calendar = (
                # Need more than a full year
                    (max_date_delta > pd.Timedelta(365, unit='d')) &
                    # Need at least daily samples
                    (min_date_delta <= pd.Timedelta(1, unit='d'))
            )
        if use_calendar:
            l_model_add = get_list_model(l_model_add, l_model_calendar, 'add')
            l_model_mult = get_list_model(
                l_model_mult, l_model_calendar, 'mult')

    if season_add_mult == 'auto':  # detect
        is_mult = model_utils.is_multiplicative(df_y)
        season_add_mult = 'mult' if is_mult else 'add'

    if season_add_mult == 'add':
        l_model = l_model_add
    elif season_add_mult == 'mult':
        l_model = l_model_mult
    else:  # both
        l_model = list(set(l_model_add + l_model_mult))
    # logger_info('debug l_Model',l_model)
    if l_model_naive is not None:
        l_model = l_model_naive + l_model

    # exclude samples with weight = 0
    df_y = df_y.loc[df_y.weight > 0]
    date_start_actuals = df_y.date.min() \
        if 'date' in df_y.columns else df_y.x.min()
    x_start_actuals = df_y.x.min()

    # Actuals table with filled x-axis gaps,
    #  used for extrapolation, naive model fitting
    df_actuals_interpolated = (
        df_y_unfiltered
            .pipe(model_utils.interpolate_df, interpolate_y=False)
            .sort_values(['x'])
            .reset_index(drop=True)
            # Update weight column in df_actuals_interpolated
            .drop(columns=['weight'], errors='ignore')
            .merge(df_y[['x', 'weight']], how='left')
    )
    df_actuals_interpolated['weight'] = \
        df_actuals_interpolated.weight.fillna(0)
    # Set all 0-weight samples to null
    df_actuals_interpolated['y'] = \
        df_actuals_interpolated['y'].where(
            df_actuals_interpolated.weight > 0., np.NaN)

    # Note - In the above steps, we first remove any samples with weight = 0
    # from df_y, the data used for fitting
    # then we fill gaps in dates from
    # df_actuals_interpolated, the table used for extrapolating.
    # The filled gaps have NaN values in the y column, 0 weight

    l_model.sort()
    for model in l_model:
        dict_fit_model = fit_model(
            model, df_y, freq, source, df_actuals=df_actuals_interpolated)
        df_metadata_tmp = dict_fit_model['metadata']
        df_optimize_info = dict_fit_model['optimize_info']

        # Add add_mult to metadata
        metadata_add_mult = _get_model_is_add_mult(
            model, season_add_mult, l_model_add)
        df_metadata_tmp['add_mult'] = metadata_add_mult

        l_df_metadata += [df_metadata_tmp]
        l_df_optimize_info += [df_optimize_info]
        source_long = df_metadata_tmp.source_long.iloc[0]
        params = df_metadata_tmp.params.iloc[0]

        if df_metadata_tmp.is_fit.iloc[0]:  # If model is fit

            # date_start_actuals = df_y.date.min()
            # date_end_actuals = df_y.date.max()

            df_data_tmp = extrapolate_model(
                model,
                params,
                date_start_actuals,
                date_end_actuals,
                freq,
                extrapolate_years,
                x_start_actuals=x_start_actuals,
                df_actuals=df_actuals_interpolated)

            df_data_tmp = _get_df_fcast_clean(
                df_data_tmp, source_id, source_long, model.name)

            l_df_data += [df_data_tmp]

    # Generate output
    df_data = pd.concat(l_df_data, sort=False, ignore_index=True)
    df_metadata = pd.concat(l_df_metadata, sort=False, ignore_index=True)
    df_optimize_info = pd.concat(
        l_df_optimize_info,
        sort=False,
        ignore_index=True)
    # Determine best fits
    df_best_fit = (
        df_metadata.loc[df_metadata.is_fit]
            .sort_values('aic_c')
        [['source', 'source_long', 'model']]
            .groupby('source', as_index=False)
            .first()
    )
    df_best_fit['is_best_fit'] = True

    df_metadata = df_metadata.merge(df_best_fit, how='left')
    df_metadata['is_best_fit'] = df_metadata['is_best_fit'].fillna(
        False).astype(bool)
    # Adjust weight column - fit_model may be missing filtered 0-weight rows
    df_metadata['weights'] = weights
    df_data = df_data.merge(df_best_fit, how='left').reset_index(drop=True)
    df_data['is_best_fit'] = df_data['is_best_fit'].fillna(False).astype(bool)

    if not include_all_fits:
        df_metadata = df_metadata.loc[df_metadata.is_best_fit].reset_index(
            drop=True)
        df_data = df_data.loc[df_data.is_best_fit |
                              df_data.is_actuals].reset_index(drop=True)

    df_forecast = df_data.pipe(get_pi, n_sims=100, n_cum=n_cum,
                               pi_q1=pi_q1, pi_q2=pi_q2)
    dict_result = {
        'forecast': df_forecast,
        'data': df_data,
        'metadata': df_metadata,
        'optimize_info': df_optimize_info}

    if simplify_output:
        return df_forecast
    else:
        return dict_result


# TODO: REMOVE THIS FUNCTION
def run_l_forecast(l_fcast_input,
                   col_name_y='y', col_name_weight='weight',
                   col_name_x='x', col_name_date='date',
                   col_name_source='source',
                   extrapolate_years=0, season_add_mult='add',
                   include_all_fits=False,
                   find_outliers=False):
    """
    Generate forecasts for a list of SolverConfig objects, each including
    a time series, model functions, and other configuration parameters.

    :param l_fcast_input: List of forecast input configurations.
        Each element includes a time series,
        candidate forecast models for trend and seasonality,
        and other configuration parameters. For each input
        configuration, a forecast time series will be generated.
    :type l_fcast_input: list of ForecastInput
    :param return_all_models:
        | If True, result includes non-fitting models, with null AIC and an
        | empty forecast df. Otherwise, result includes only fitting models,
        | and for time series where no fitting model is available, a
        | 'no-best-model' entry with null AIC and an empty forecast
        | df is added.
    :type return_all_models: bool
    :param return_all_fits: If True, result includes all models for each
        nput time series. Otherwise, only the best model is included.
    :type return_all_fits: bool
    :param extrapolate_years:
    :type extrapolate_years: float
    :param season_add_mult: 'add', 'mult', or 'both'. Whether forecast
        seasonality will be additive, multiplicative,
        or the best fit of the two.
    :type season_add_mult: str
    :param fill_gaps_y_values: If True, gaps in time series will be
        filled with NaN values
    :type fill_gaps_y_values: bool
    :param freq: 'W' or 'D' . Sampling frequency of the output forecast:
        weekly or daily.
    :type freq: str
    :return:
        | dict(data,metadata)
        | data: dataframe(date, source, model, y)
        | metadata: dataframe('source', 'model', 'res_weights', 'freq',
        | 'is_fit', 'cost', 'aic_c', 'params', 'status')
    :rtype: dict

    """
    # TODO: Add check for non-duplicate source ids
    # We can take solver_config_list that are a list or a single
    # forecast_input
    if not isinstance(l_fcast_input, list):
        l_fcast_input = [l_fcast_input]

    l_dict_result = []
    for fcast_input in l_fcast_input:
        dict_result = run_forecast(
            fcast_input.df_y,
            fcast_input.l_model_trend,
            fcast_input.l_model_season,
            fcast_input.date_start_actuals,
            fcast_input.source_id,
            col_name_y,
            col_name_weight,
            col_name_x,
            col_name_date,
            col_name_source,
            extrapolate_years,
            season_add_mult,
            include_all_fits,
            simplify_output=False,
            find_outliers=find_outliers)
        l_dict_result += [dict_result]

    # Generate output
    return aggregate_forecast_dict_results(l_dict_result)


# Forecast configuration

# TODO: REMOVE THIS CLASS
class ForecastInput:
    """
    Class that encapsulates input variables for forecast.run_forecast()
    """

    def __init__(
            self,
            source_id,
            df_y,
            l_model_trend=None,
            l_model_season=None,
            weights_y_values=1.0,
            date_start_actuals=None):
        self.source_id = source_id
        self.df_y = df_y
        self.l_model_trend = l_model_trend if l_model_trend is not None else [
            forecast_models.model_linear]
        self.l_model_season = l_model_season
        self.weights_y_values = weights_y_values
        self.date_start_actuals = date_start_actuals

    def __str__(self):
        str_result = (
            'SolverConfig: {source_id} ; {df_y_shape} ; {weights_y_values};'
            ' {l_model_trend}; {l_model_season} ; {date_start_actuals}').\
            format(
            source_id=self.source_id,
            df_y_shape=self.df_y.shape,
            l_model_trend=to_str_function_list(
                self.l_model_trend),
            l_model_season=to_str_function_list(
                self.l_model_season),
            weights_y_values=self.weights_y_values,
            date_start_actuals=self.date_start_actuals)
        return str_result

    def __repr__(self):
        return self.__str__()

    # TODO: REMOVE
    @classmethod
    def create(cls, source_id, df_y, l_model_trend, l_model_season=None,
               weights_y_values=1.0, date_start_actuals=None):
        return cls(source_id, df_y, pd.Series(l_model_trend), l_model_season,
                   weights_y_values, date_start_actuals)


def get_pi(df_forecast, n_sims=100, n_cum=1,
           pi_q1=5, pi_q2=20):
    """
    Generate prediction intervals for a table with multiple forecasts,
    using bootstrapped residuals.

    :param df_forecast: forecasted time series
    :type df_forecast: pandas.DataFrame
    :param n_sims: Number of bootstrapped samples for prediction interval
    :type n_sims: int
    :param n_cum: Used for widening prediction interval. Interval widens every
        n_sims samples.
    :type n_cum: int
    :param pi_q1: Percentile for outer prediction interval (defaults to 5%-95%)
    :type pi_q1: int
    :param pi_q2: Percentile for inner prediction interval (defaults to 20%-80%)
    :type pi_q2: int
    :return:
        | Forecast time series table with added columns:
        | - q5: 5% percentile of prediction interval
        | - q5: 20% percentile of prediction interval
        | - q5: 80 percentile of prediction interval
        | - q5: 95% percentile of prediction interval
    :rtype: pandas.DataFrame

    Based on https://otexts.org/fpp2/prediction-intervals.html
    """
    if 'source' in df_forecast.columns and df_forecast.source.nunique() > 1:
        df_result = (
            df_forecast.groupby('source', as_index=False)
            .apply(_get_pi_single_source,
                   n_sims=n_sims, n_cum=n_cum, pi_q1=pi_q1, pi_q2=pi_q2)
            .sort_values(['source', 'is_actuals', 'date'])
            .reset_index(drop=True)
        )
    else:
        df_result = _get_pi_single_source(
            df_forecast, n_sims, n_cum, pi_q1, pi_q2)
    return df_result


def _df_add_pi_percentiles(df_fcast, a_sample,
                           pi_q1=5, pi_q2=20):
    """
    Given a forecast dataframe and an array of bootstrapped residual samples,
    add quantile columns (defining prediction interval) to forecast dataframe

    :param df_fcast: Forecast dataframe with column 'y' for forecasted values
    :type df_fcast: pandas.DataFrame
    :param a_sample: 2D array with shape (n_sims, df_fcast.index.size)
    :type a_sample: numpy.array
    :param pi_q1: Percentile for outer prediction interval (defaults to 5%-95%)
    :type pi_q1: int
    :param pi_q2: Percentile for inner prediction interval
        (defaults to 20%-80%)
    :type pi_q2: int
    """
    assert 0 < pi_q1 < 100 and 0 < pi_q2 < 100
    l_percentiles = pi_q1, 100 - pi_q1, pi_q2, 100 - pi_q2
    # dict of arrays with percentile values
    dict_q = {'q{}'.format(q): np.percentile(a_sample, q, axis=0)
              for q in l_percentiles}

    df_fcast = df_fcast.copy()
    for q in dict_q.keys():
        df_fcast[q] = df_fcast.y + dict_q.get(q)
    return df_fcast


def _get_pi_single_source(df_forecast, n_sims=100, n_cum=1,
                          pi_q1=5, pi_q2=20):
    """
    Generate prediction interval for a single forecast

    :param df_forecast:
    :type df_forecast: pandas.DataFrame
    :param n_sims: Number of bootstrapped samples for prediction interval
    :type n_sims: int
    :param n_cum: Used for widening prediction interval. Interval widens every
        n_sims samples.
    :type n_cum: int
    :param pi_q1: Percentile for outer prediction interval (defaults to 5%-95%)
    :type pi_q1: int
    :param pi_q2: Percentile for inner prediction interval (defaults to 20%-80%)
    :type pi_q2: int
    :return:
    :rtype: pandas.DataFrame
    """

    if 'is_best_fit' in df_forecast.columns:
        df_forecast = df_forecast.loc[df_forecast.is_actuals |
                                      df_forecast.is_best_fit].copy()
    else:
        df_forecast = df_forecast.copy()

    if 'source' in df_forecast.columns:
        l_cols = ['date', 'source']
    else:
        l_cols = ['date']

    if 'weight' in df_forecast.columns and (df_forecast.weight == 0).any():

        # Filter out dates for outliers with weight=0
        df_filtered_dates = (
            df_forecast.loc[(df_forecast.weight > 0) &
                            (df_forecast.is_actuals)]
            [['date', 'source']]
        )

        # Take filtered actuals
        df_actuals_unfiltered = df_forecast.loc[df_forecast.is_actuals &
                                                ~df_forecast.y.isnull()]
        df_actuals = (df_actuals_unfiltered[['date', 'y']]
                      .merge(df_filtered_dates, how='inner')
                      .rename({'y': 'actuals'}, axis=1)
                      )
        date_last_actuals = df_actuals.date.max()
    else:  # No weight data - use all actuals rows
        df_actuals_unfiltered = df_forecast.loc[df_forecast.is_actuals &
                                                ~df_forecast.y.isnull()]
        df_actuals = (df_actuals_unfiltered[['date', 'y']]
                      .rename({'y': 'actuals'}, axis=1)
                      )
        date_last_actuals = df_actuals.date.max()
    # Compute residuals for filtered actuals
    df_residuals_tmp = df_forecast.loc[
        ~df_forecast.is_actuals & ~df_forecast.y.pipe(
            pd.isnull)][l_cols + ['model', 'y']]

    df_residuals = df_residuals_tmp.merge(df_actuals, how='inner')
    df_residuals['res'] = df_residuals['actuals'] - df_residuals['y']

    # Filter out null values, e.g. due to null actuals
    df_residuals = df_residuals.loc[~df_residuals.res.isnull()]

    if df_residuals.empty:  # May happen if no forecast could be generated
        logger.warning(
            'No forecast data for source %s',
            df_forecast.source.head(1).iloc[0])
        return df_actuals_unfiltered[l_cols + ['is_actuals', 'model', 'y']]

    # Generate table with prediction interval
    df_forecast_pi = (
        df_forecast.loc[~df_forecast.is_actuals &
                        (df_forecast.date > date_last_actuals)]
        [l_cols + ['model', 'y']]
    )

    s_residuals = df_residuals.res
    # Residuals series with non-null mean lead to skewed PIs
    s_residuals = s_residuals - np.mean(s_residuals)

    a_forecast_point = df_forecast_pi.y.values

    n_samples_fcast = a_forecast_point.size
    # Number of samples in cum series, used to widen interval
    n_samples_short = int(np.ceil(n_samples_fcast / n_cum))

    # Series to widen prediction interval every n_cum samples
    a_sample_pi_cum = (
        s_residuals
        .sample(n_samples_short * n_sims, replace=True)
        .values.reshape(n_sims, n_samples_short)
    )
    a_sample_pi_cum = np.cumsum(a_sample_pi_cum, axis=1)
    # TODO: REPLACE NP.REPEAT WITH INTERPOLATE
    a_sample_pi_cum_expanded = (
        np.repeat(a_sample_pi_cum, n_cum, axis=1)
        [:, :n_samples_fcast]
    )
    # Series with residuals samples - base width of prediction interval
    a_sample_base = (
        s_residuals
            .sample(n_samples_fcast * n_sims, replace=True)
            .values.reshape(n_sims, n_samples_fcast)
    )
    a_sample = a_sample_base + a_sample_pi_cum_expanded

    df_forecast_pi = _df_add_pi_percentiles(
        df_forecast_pi, a_sample, pi_q1, pi_q2)

    df_forecast_pi['is_actuals'] = False

    # Past forecast samples, constant width prediction interval
    df_forecast_past = (
        df_forecast.loc[~df_forecast.is_actuals &
                        (df_forecast.date <= date_last_actuals)]
        [l_cols + ['model', 'is_actuals', 'y']]
    )
    n_samples_fcast_past = df_forecast_past.index.size
    a_sample_past = (
        s_residuals
        .sample(n_samples_fcast_past * n_sims, replace=True)
        .values.reshape(n_sims, n_samples_fcast_past)
    )
    df_forecast_past = _df_add_pi_percentiles(
        df_forecast_past, a_sample_past, pi_q1, pi_q2)

    df_actuals_unfiltered = df_actuals_unfiltered[
        l_cols + ['is_actuals', 'model', 'y']]
    df_pi_result = pd.concat([df_actuals_unfiltered,
                              df_forecast_past,
                              df_forecast_pi,
                              ],
                             sort=False,
                             ignore_index=True)

    return df_pi_result

# -*- coding: utf-8 -*-
#
# License:          This module is released under the terms of the LICENSE file 
#                   contained within this applications INSTALL directory

"""
Functions to run forecast
"""

# -- Coding Conventions
#    http://www.python.org/dev/peps/pep-0008/   -   Use the Python style guide
#    http://sphinx.pocoo.org/rest.html          -   Use Restructured Text for docstrings

# -- Public Imports
import logging
import numpy as np
import pandas as pd
import scipy
from scipy import optimize
import itertools
from functools import reduce


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
    if l_function is None:
        return None
    return [f.name if f is not None else None for f in l_function]

def is_null_model(f_model):
    return f_model.name == 'null'


def _is_multi_ts(a):
    return a.ndim > 1 and a.shape[1] > 1


def _has_different_res_weights(res_weights):
    # Check if a residuals parameter is a vector with length > 1
    return res_weights is not None and hasattr(res_weights, "__getitem__") and len(res_weights) > 1


# TODO: replace get_residuals with this
def get_residuals(params, model, a_x, a_y, a_date, a_weights=None, filter_null_residuals=None, df_actuals=None):
    """
    Given a time series, a model function and a set of parameters, get the residuals

    :param params: parameters for model function
    :type params: numpy array
    :param model: model function. Usage: model(a_x, a_date, params)
    :type model: function
    :param a_x: X axis for model function.
    :type a_x: float array
    :param a_y: Input time series values, to compare to the model function
    :type a_y: float array
    :param a_date: Dates for the input time series
    :type a_date: datetime array
    :param a_weights: weights for each individual sample
    :type a_weights: numpy array

    :return: array with residuals, same length as a_x, a_y
    :rtype: float array
    """
    # Note: remove this assert for performance
    assert a_y.ndim == 1
    # Note: none of the input arrays should include NaN values
    # We do not check this with asserts due to performance - this function is in the optimization loop

    y_predicted = model(a_x, a_date, params, df_actuals=df_actuals)
    residuals = (a_y - y_predicted)
    if a_weights is not None:     # Do this only if different residual weights
        residuals = residuals * a_weights
    result = np.abs(residuals)
    return result


def optimize_least_squares(model, a_x, a_y, a_date, a_weights=None, f_t_scaling=None, df_actuals=None):
    """
    Given a time series and a model function, find the set of parameters that minimises residuals

    :param model: model function, to be fitted against the actuals
    :type model: function
    :param a_x:
    :type a_x: float array
    :param a_y:
    :type a_y: float array
    :param a_date:
    :type a_date: datetime array
    :param res_weights:
    :type res_weights:
    :param use_t_scaling:
    :type use_t_scaling:
    :param bounds:
    :type bounds: 2-tuple of array_like
    :return:
        | table(success, params, cost, optimality, iterations, status, jac_evals, message):
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

    # Check that input is sorted - not required - taken care by normalize_df()
    # assert np.all(np.diff(a_x) >= 0), 'Input not sorted on x axis'

    # Ask the model to provide an initial guess
    initial_guess = model.f_init_params(a_x, a_y)

    bounds = model.f_bounds(a_x, a_y)

    assert forecast_models.validate_initial_guess(initial_guess, bounds), \
        'Initial guess outside of bounds: {} - {}, {}'.format(model, initial_guess, bounds)

    # In multi-ts scenarios, we apply this filter to ignore residuals for null y_values
    filter_null_residuals = ~np.isnan(a_y)
    if np.all(filter_null_residuals):
        filter_null_residuals = None

    # t_scaling: we use this to assign different weight to residuals based on date # TODO: Implement scaling functions
    if f_t_scaling:
        a_weights_tmp = f_t_scaling(a_x)
        a_weights = a_weights_tmp if a_weights is None else a_weights*a_weights_tmp

    # Set up arguments for get_residuals
    f_model_args = (model, a_x, a_y, a_date)

    result = scipy.optimize.least_squares(get_residuals, initial_guess,
                                          args=f_model_args,
                                          kwargs={'a_weights': a_weights, 'df_actuals':df_actuals},
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
    df_result = pd.DataFrame(data=dict_result_df, index=pd.Index([0]))
    df_result = df_result[['success', 'params', 'cost', 'optimality', 'iterations', 'status', 'jac_evals', 'message']]
    return df_result


def _get_df_fit_model(source, model, weights, actuals_x_range, freq,
                      is_fit, cost, aic_c, params, status):
    if params is None:
        params = np.array([])
    df_result = (
        pd.DataFrame(columns=['source', 'model', 'weights', 'actuals_x_range', 'freq',
                              'is_fit', 'cost', 'aic_c', 'params_str', 'status', 'source_long', 'params'],
                     data=[[source, model, weights, actuals_x_range, freq,
                            is_fit, cost, aic_c, np.array_str(params, precision=1), status,
                            '{}:{}:{}:{}'.format(source, weights, freq, actuals_x_range),
                            params
                            ]])
    )
    return df_result


def _get_empty_df_result_optimize(source, model, status, weights, freq, actuals_x_range):
    source_long = '{}:{}:{}:{}'.format(source, weights, freq, actuals_x_range)
    return pd.DataFrame(columns=['source', 'model', 'success', 'params_str', 'cost', 'optimality', 'iterations',
                                 'status', 'jac_evals', 'message', 'source_long', 'params'],
                        data=[[source, model, False, '[]', np.NaN, np.NaN, np.NaN, status, np.NaN, status,
                               source_long, []]])


def normalize_df(df_y,
                 col_name_y='y',
                 col_name_weight='weight',
                 col_name_x='x',
                 col_name_date='date',
                 col_name_source='source'):
    """
    Converts an input dataframe for run_forecast() into a normalized format suitable for fit_model()

    :param df_y:
    :type df_y: pandas.DataFrame
    :param col_name_y:
    :type col_name_y: str
    :param col_name_weight:
    :type col_name_weight: str
    :param col_name_x:
    :type col_name_x: str
    :param col_name_date:
    :type col_name_date: str
    """

    assert df_y is not None
    if df_y.empty:
        return None

    if isinstance(df_y, pd.Series):
        df_y = df_y.to_frame()
    assert isinstance(df_y, pd.DataFrame)
    assert col_name_y in df_y.columns, 'Dataframe needs to have a column named "{}"'.format(col_name_y)
    df_y = df_y.copy()

    # Rename columns to normalized values
    rename_col_dict = {
        col_name_y:'y',
        col_name_weight:'weight',
        col_name_x:'x',
        col_name_date:'date',
        col_name_source:'source'
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
    l_sources = df_y[col_name_source].drop_duplicates() if multiple_sources else ['test_source']

    l_df_results = []
    for source in l_sources:
        df_y_tmp = df_y.loc[df_y[col_name_source] == source].copy() if multiple_sources else df_y
        # Setup date, x columns
        if col_name_date not in df_y.columns and isinstance(df_y.index, pd.DatetimeIndex):  # use index as i_date
            df_y_tmp[col_name_date] = df_y_tmp.index
        elif col_name_date in df_y.columns:  # Ensure that date column is timestamp dtype
            df_y_tmp[col_name_date] = df_y_tmp[col_name_date].pipe(pd.to_datetime)

        #if isinstance(df_y_tmp.index, pd.DatetimeIndex):
        # We don't need a date index after this point
        df_y_tmp = df_y_tmp.reset_index(drop=True)

        if col_name_x not in df_y_tmp.columns:
            if col_name_date in df_y_tmp.columns:
                # Need to extract numeric index from a_date
                df_date_interp = (
                    df_y_tmp[[col_name_date]].drop_duplicates().pipe(model_utils.interpolate_df).rename_axis(col_name_x).reset_index()
                )
                df_y_tmp = (
                    df_date_interp.merge(df_y_tmp)
                )
            else:       # With no date, extract column x from a numeric index
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

    df_result = pd.concat(l_df_results, sort=False, ignore_index=True) #.rename(rename_col_dict, axis=1)

    # Sort columns, filter unused columns
    df_result = df_result[[c for c in ['date', 'source', 'x', 'y', 'weight'] if c in df_result.columns]]
    sort_columns = ['source','x'] if 'source' in df_result.columns else ['x']
    df_result = df_result.sort_values(sort_columns).reset_index(drop=True)
    return df_result


def fit_model(model, df_y, freq='W', source='test', df_actuals=None):
    """
    Given a time series and a model, optimize model parameters and return

    :param model:
    :type model: function
    :param df_y:
        | Dataframe with the following columns:
        | - y:
        | - date: (optional)
        | - weight: (optional)
        | - x: (optional)
    :type df_y: pandas.DataFrame
    :param source:
    :type source:
    :param freq: 'W' or 'D' . Used only for metadata
    :type freq: str
    :return: table (source, model_name, y_weights , freq, is_fit, aic_c, params)
    :rtype: pandas.DataFrame

    This function calls optimize_least_squares() to perform the optimization loop. It performs some cleaning up of input
    and output parameters.
    """
    col_name_y = 'y'
    col_name_weight = 'weight'
    col_name_x = 'x'
    col_name_date = 'date'

    assert df_y is not None and isinstance(df_y, pd.DataFrame) and col_name_y in df_y.columns

    # Setup
    f_model_name = model.name
    n_params = model.n_params

    df_y = df_y.copy()
    # Filter out any sample where df_y is null
    df_y = df_y.loc[~df_y[col_name_y].pipe(pd.isna)]

    # Filter out any sample where a_weights is 0
    if col_name_weight in df_y.columns:
        df_y[col_name_weight] = df_y[col_name_weight].fillna(0)
        df_y = df_y.loc[df_y[col_name_weight] >= 0]

    # Metadata
    if col_name_weight not in df_y.columns:
        weights = '1'
    else:
        weights = '{}-{}'.format(df_y[col_name_weight].min(), df_y[col_name_weight].max())

    # Residual normalization
    if df_y[col_name_x].duplicated().any():
        df_k = df_y.groupby(col_name_x).size().rename('k_weight_normalize').reset_index()
        df_y = df_y.merge(df_k)
        if col_name_weight not in df_y:
            df_y[col_name_weight] = 1.0
        # Adjust residual weight based on number of values per sample
        # E.g. a sample with 2 values in the input series will multiply residuals by 0.5
        df_y[col_name_weight] = df_y[col_name_weight]/df_y['k_weight_normalize']

    # Get input arrays
    a_y = df_y[col_name_y].values
    a_x = model_utils.apply_a_x_scaling(df_y[col_name_x].values, model)
    a_weights = df_y[col_name_weight].values if col_name_weight in df_y.columns else None
    # Need to convert series to DatetimeIndex
    i_date = pd.DatetimeIndex(df_y[col_name_date]) if col_name_date in df_y.columns else None

    # Metadata
    cost = np.NaN
    is_fit = False
    params = []
    # Get first and last actuals date, for metadata. If no a_date, use a_x instead.
    date_start_actuals = i_date.min().date() if i_date is not None else a_x.min()
    date_end_actuals = i_date.max().date() if i_date is not None else a_x.max()
    actuals_x_range = '{}::{}'.format(date_start_actuals, date_end_actuals)

    if df_y.empty:
        logger.info('Cannot fit - empty df_y: %s', source)
        status = 'EMPTY_TS'
        df_result = _get_df_fit_model(source, model.name, weights, actuals_x_range, freq,
                                      is_fit, cost, np.NaN, None, status)
        df_result_optimize = _get_empty_df_result_optimize(source, model, status, weights, freq, actuals_x_range)

    elif a_x.size < n_params + 2:
        logger.info('Not enough samples in source %s for %s: %s (needs %s)',
                    source, f_model_name, a_x.size, n_params + 2)
        status = 'TS_TOO_SHORT'
        df_result = _get_df_fit_model(source, model.name, weights, actuals_x_range, freq,
                                      is_fit, cost, np.NaN, None, status)
        df_result_optimize = _get_empty_df_result_optimize(source, model, status, weights, freq, actuals_x_range)
    else:       # Get results
        model = forecast_models.simplify_model(model, a_x, a_y, i_date)

        if model.n_params==0:
            # 0-parameter model, cannot be fit
            #logger.info('Model has 0 parameters - no fitting required')

            a_residuals = get_residuals(None, model, a_x, a_y, i_date, a_weights, df_actuals=df_actuals)
            cost = 0.5*np.nansum(a_residuals**2)
            is_fit = True
            params = np.array([])
            status = 'FIT'

            # Process results

            aic_c = model_utils.get_aic_c(cost, len(df_y), n_params)

            df_result = _get_df_fit_model(source, model.name, weights, actuals_x_range, freq,
                                          is_fit, cost, aic_c, params, status)

            dict_result_df = {
                'optimality': 0.,
                'success': True,
                'cost':cost,
                'iterations': 0.,
                'jac_evals': 0.,
                'status': 0,
                'message': 'Naive model fitted',
                'params': '-'
            }
            df_result_optimize = pd.DataFrame(data=dict_result_df, index=pd.Index([0]))
            df_result_optimize = df_result_optimize[
                ['success', 'params', 'cost', 'optimality', 'iterations', 'status', 'jac_evals', 'message']]
            df_result_optimize['source'] = source
            df_result_optimize['source_long'] = df_result.source_long.iloc[0]
            df_result_optimize['model'] = model
            df_result_optimize['params_str'] = df_result.params_str.iloc[0]
            df_result_optimize = df_result_optimize[
                ['source', 'model', 'success', 'params_str', 'cost', 'optimality', 'iterations',
                 'status', 'jac_evals', 'message', 'source_long', 'params']]
        else:
            df_result_optimize = optimize_least_squares(model, a_x, a_y, i_date, a_weights, df_actuals=df_actuals)
            cost = df_result_optimize.cost.iloc[0]
            is_fit = df_result_optimize.success.iloc[0]
            params = df_result_optimize.params.iloc[0]
            status = 'FIT' if is_fit else 'NO-FIT'

            # Process results
            if status in ['FIT','NO-FIT']:
                aic_c = model_utils.get_aic_c(cost, len(df_y), n_params)
            else:
                aic_c = np.NaN

            df_result = _get_df_fit_model(source, model.name, weights, actuals_x_range, freq,
                                          is_fit, cost, aic_c, params, status)

            df_result_optimize['source'] = source
            df_result_optimize['source_long'] = df_result.source_long.iloc[0]
            df_result_optimize['model'] = model
            df_result_optimize['params_str'] = df_result.params_str.iloc[0]
            df_result_optimize = df_result_optimize [['source','model','success','params_str','cost','optimality','iterations',
                                                      'status','jac_evals','message','source_long','params']]

    dict_result = {'metadata':df_result, 'optimize_info':df_result_optimize}
    return dict_result


def extrapolate_model(model, params, date_start_actuals, date_end_actuals, freq='W', extrapolate_years=2.0,
                      x_start_actuals=0., df_actuals=None):
    """
    Given a model and a set of parameters, generate model output for a date range plus a number of additional years.

    :param model:
    :type model:
    :param params:
    :type params:
    :param date_start_actuals:
    :type date_start_actuals:
    :param date_end_actuals:
    :type date_end_actuals:
    :param freq:
    :type freq:
    :param extrapolate_years:
    :type extrapolate_years:
    :return:
    :rtype:
    """
    s_x = model_utils.get_s_x_extrapolate(date_start_actuals, date_end_actuals, model=model, freq=freq,
                                          extrapolate_years=extrapolate_years, x_start_actuals=x_start_actuals)
    a_y_forecast = model(s_x.values, s_x.index, params, df_actuals=df_actuals)
    s_y_forecast = pd.Series(data=a_y_forecast, index=s_x.index, name='y')
    df_y_forecast = pd.DataFrame(s_y_forecast)
    return df_y_forecast


def get_list_model(l_model_trend, l_model_season, season_add_mult='both'):
    if l_model_season is None or len(l_model_season) < 1:
        l_model_tmp = l_model_trend
    elif l_model_trend is None or len(l_model_trend) < 1:
        l_model_tmp = l_model_season
    else:
        l_model_tmp = []
        if season_add_mult != 'mult':   # 'add' or 'both'
            l_model_tmp += [model_trend+model_season for model_trend, model_season in
                            itertools.product(l_model_trend, l_model_season)]
        if season_add_mult != 'add':    # 'mult' or 'both'
            l_model_tmp += [model_trend*model_season for model_trend, model_season in
                            itertools.product(l_model_trend, l_model_season)]

    l_model_tmp = pd.Series(l_model_tmp).drop_duplicates().tolist()
    return l_model_tmp


def get_df_actuals_clean(df_actuals, source, source_long):
    """

    :param df_actuals: dataframe in normalized format, with columns y and optionally x, date, weight
    :type df_actuals:
    :param source:
    :type source:
    :param source_long:
    :type source_long:
    :return:
    :rtype:
    """
    # Add actuals as entries in result dicts
    df_actuals = df_actuals.copy()  # .rename_axis('date')
    if 'date' not in df_actuals.columns:
        df_actuals = df_actuals.rename({'x': 'date'}, axis=1)
    df_actuals = df_actuals[[c for c in ['date', 'weight', 'y'] if c in df_actuals.columns]]

    df_actuals['model']='actuals'
    df_actuals['source'] = source
    df_actuals['source_long'] = source_long
    df_actuals['is_actuals'] = True
    if not 'weight' in df_actuals.columns:
        df_actuals['weight'] = 1.0
    return df_actuals


def _get_df_fcast_clean(df_fcast, source, source_long,model):
    # TODO: cleanup
    # This removes any forecast samples with null values, e.g. from naive models
    df_fcast = df_fcast.loc[ ~df_fcast.y.pipe(pd.isnull)]
    df_fcast = df_fcast.copy().rename_axis('date').reset_index()
    df_fcast['source'] = source
    df_fcast['source_long'] = source_long
    df_fcast['model'] = model
    df_fcast['is_actuals'] = False
    df_fcast['weight'] = 1.0
    return df_fcast


"""
# TODO: api improvements:
- change default df format to have columns: x,y, date, weight
- currently, we assume a datetimeindex

"""


def run_forecast_from_input_list(l_dict_input):
    # Run forecasts from a list of dictionaries with keyword arguments

    # Handle both scalars and list-likes
    s_input = pd.Series(l_dict_input)

    l_dict_result = []
    for dict_input in l_dict_input:
        dict_result_tmp = run_forecast(**dict_input)
        l_dict_result += [dict_result_tmp]

    # Generate output
    return aggregate_forecast_dict_results(l_dict_result)


def run_forecast(df_y, l_model_trend=None, l_model_season=None,
                 date_start_actuals=None, source_id='src',
                 col_name_y='y', col_name_weight='weight',
                 col_name_x='x', col_name_date='date',
                 col_name_source='source',
                 extrapolate_years=0, season_add_mult='add',
                 include_all_fits=False,
                 simplify_output=True,
                 do_find_steps_and_spikes=False,
                 find_outliers=False,
                 l_season_yearly=None,
                 l_season_weekly=None,
                 verbose=None,
                 l_model_naive=None
                 ):
    """
    Generate forecast for one or more input time series

    :return:
    :rtype:
    :param df_y:
    :type df_y:
    :param l_model_trend:
    :type l_model_trend:
    :param l_model_season:
    :type l_model_season:
    :param date_start_actuals:
    :type date_start_actuals:
    :param source_id:
    :type source_id:
    :param col_name_y:
    :type col_name_y:
    :param col_name_weight:
    :type col_name_weight:
    :param col_name_x:
    :type col_name_x:
    :param col_name_date:
    :type col_name_date:
    :param col_name_source:
    :type col_name_source:
    :param return_all_models:
        | If True, result includes non-fitting models, with null AIC and an empty forecast df.
        | Otherwise, result includes only fitting models, and for time series where no fitting model is available,
        | a 'no-best-model' entry with null AIC and an empty forecast df is added.
    :type return_all_models: bool
    :param return_all_fits: If True, result includes all models for each input time series. Otherwise, only the
        best model is included.
    :type return_all_fits: bool
    :param extrapolate_years:
    :type extrapolate_years: float
    :param season_add_mult: 'add', 'mult', or 'both'. Whether forecast seasonality will be additive, multiplicative,
        or the best fit of the two.
    :type season_add_mult: str
    :param fill_gaps_y_values: If True, gaps in time series will be filled with NaN values
    :type fill_gaps_y_values: bool
    :param freq: 'W' or 'D' . Sampling frequency of the output forecast: weekly or daily.
    :type freq: str
    :param do_find_steps_and_spikes: if True, find steps and spikes, create fixed models and add them
        to the list of models
    :type do_find_steps_and_spikes: bool
    :param find_outliers:
    :type find_outliers:
    :param include_all_fits:
    :type include_all_fits:
    :param simplify_output: If False, return dict with forecast and metadata. Otherwise, return only forecast.
    :type simplify_output: bool
    :return:
    :rtype:
    """
    # TODO: Add check for non-duplicate source ids
    l_dict_result = []

    df_y = normalize_df(df_y, col_name_y, col_name_weight, col_name_x, col_name_date, col_name_source)
    if df_y is None: # Empty input
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
                                   do_find_steps_and_spikes,
                                   find_outliers,
                                   l_season_yearly,
                                   l_season_weekly,
                                   l_model_naive=l_model_naive
                                   )
    else:
        for src_tmp in df_y.source.drop_duplicates():
            if verbose:
                logger.info('Running forecast for source: %s', src_tmp)
            df_y_tmp = df_y.loc[df_y.source==src_tmp].reset_index(drop=True)
            dict_result_tmp = run_forecast_single(df_y_tmp,
                                                  l_model_trend,
                                                  l_model_season,
                                                  date_start_actuals,
                                                  src_tmp,
                                                  extrapolate_years,
                                                  season_add_mult,
                                                  include_all_fits,
                                                  False,  # Simplify output
                                                  do_find_steps_and_spikes,
                                                  find_outliers,
                                                  l_season_yearly,
                                                  l_season_weekly,
                                                  l_model_naive=l_model_naive
                                                  )
            l_dict_result += [dict_result_tmp]
    # Generate output
    dict_result = aggregate_forecast_dict_results(l_dict_result)
    if simplify_output:
        return dict_result.get('forecast')
    else:
        return dict_result


def aggregate_forecast_dict_results(l_dict_result):
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
    df_optimize_info = pd.concat(l_df_optimize_info, sort=False, ignore_index=True)
    df_forecast = pd.concat(l_df_forecast, sort=False, ignore_index=True)

    return {'forecast': df_forecast, 'data': df_data, 'metadata': df_metadata, 'optimize_info': df_optimize_info}

def run_forecast_single(df_y,
                        l_model_trend=None,
                        l_model_season=None,
                        date_start_actuals=None,
                        source_id='src',
                        extrapolate_years=0,
                        season_add_mult='add',
                        include_all_fits=False,
                        simplify_output=True,
                        do_find_steps_and_spikes=False,
                        find_outliers=False,
                        l_season_yearly=None,
                        l_season_weekly=None,
                        l_model_naive=None
                        ):
    """

    :param df_y:
    :type df_y:
    :param l_model_trend:
    :type l_model_trend:
    :param l_model_season:
    :type l_model_season:
    :param date_start_actuals:
    :type date_start_actuals:
    :param source_id:
    :type source_id:
    :param col_name_y:
    :type col_name_y:
    :param col_name_weight:
    :type col_name_weight:
    :param col_name_x:
    :type col_name_x:
    :param col_name_date:
    :type col_name_date:
    :param return_all_models:
        | If True, result includes non-fitting models, with null AIC and an empty forecast df.
        | Otherwise, result includes only fitting models, and for time series where no fitting model is available,
        | a 'no-best-model' entry with null AIC and an empty forecast df is added.
    :type return_all_models: bool
    :param return_all_fits: If True, result includes all models for each input time series. Otherwise, only the
        best model is included.
    :type return_all_fits: bool
    :param extrapolate_years:
    :type extrapolate_years: float
    :param season_add_mult: 'add', 'mult', or 'both'. Whether forecast seasonality will be additive, multiplicative,
        or the best fit of the two.
    :type season_add_mult: str
    :param fill_gaps_y_values: If True, gaps in time series will be filled with NaN values
    :type fill_gaps_y_values: bool
    :param freq: 'W' or 'D' . Sampling frequency of the output forecast: weekly or daily.
    :type freq: str
    :param do_find_steps_and_spikes: if True, find steps and spikes, create fixed models and add them
        to the list of models
    :type do_find_steps_and_spikes: bool
    :return:
    :rtype:
    """
    l_df_data = []
    l_df_metadata = []
    l_df_optimize_info = []

    # Each element in l_fcast_input describes all model configurations for a source time series
    source = source_id

    if 'date' in df_y.columns:
        freq = detect_freq(df_y.date)
    else:
        freq = None

    df_y=df_y.copy()
    df_y_unfiltered = df_y.copy()

    if date_start_actuals is not None and 'date' in df_y.columns:  # Filter: only actuals after date_start_actuals
        df_y = df_y.loc[df_y.date >= date_start_actuals]

    date_start_actuals = df_y.date.min() if 'date' in df_y.columns else df_y.x.min()
    date_end_actuals = df_y.date.max() if 'date' in df_y.columns else df_y.x.max()

    # If we find outliers, we add a model with dummy variables for the outliers
    if find_outliers:
        model_outliers, outlier_mask = forecast_models.get_model_outliers(df_y)
        if outlier_mask is not None:
            if 'weight' in df_y.columns:
                df_y['weight'] = df_y['weight'] * outlier_mask
            else:
                df_y['weight'] = outlier_mask
            assert np.issubdtype(df_y.weight.astype(float), np.float64)
    else:
        model_outliers = None

    # Add actuals to output
    # Get weight for metadata
    if 'weight' not in df_y.columns:
        df_y['weight']=1
        weights = '1'
    else:
        weights = '{}-{}'.format(df_y['weight'].min(), df_y['weight'].max())

    # Get long source_id
    if isinstance(date_start_actuals, pd.datetime):
        date_start_actuals_short = date_start_actuals.date()
        date_end_actuals_short = date_end_actuals.date()
    else:
        date_start_actuals_short = date_start_actuals
        date_end_actuals_short = date_end_actuals
    actuals_x_range = '{}::{}'.format(date_start_actuals_short, date_end_actuals_short)
    source_long = '{}:{}:{}:{}'.format(source, weights, freq, actuals_x_range)
    df_actuals = get_df_actuals_clean(df_y, source, source_long)
    l_df_data+=[df_actuals]

    if l_model_trend is None:
        # By default, try linear and piecewise linear
        l_model_trend = [
            #forecast_models.model_naive,
                         forecast_models.model_linear,
                         forecast_models.model_linear+forecast_models.model_ramp]
    l_model_season_add = None
    l_model_season_mult = None
    if l_model_season is None:
        if 'date' in df_y.columns:
            s_date_tmp = df_y.date
            if 'weight' in df_y.columns:
                s_date_tmp = s_date_tmp.loc[df_y.weight>0]

            l_model_season_add = forecast_models.get_l_model_auto_season(s_date_tmp,season_add_mult='add',
                                                                     l_season_yearly=l_season_yearly,
                                                                     l_season_weekly=l_season_weekly,
                                                                     )
            l_model_season_mult = forecast_models.get_l_model_auto_season(s_date_tmp,season_add_mult='mult',
                                                                     l_season_yearly=l_season_yearly,
                                                                     l_season_weekly=l_season_weekly,
                                                                     )
    else:
        l_model_season_add = l_model_season
        l_model_season_mult = l_model_season


    l_model_add = get_list_model(l_model_trend, l_model_season_add, 'add')
    l_model_mult = get_list_model(l_model_trend, l_model_season_mult, 'mult')

    if season_add_mult == 'add':
        l_model = l_model_add
    elif season_add_mult == 'mult':
        l_model = l_model_mult
    else: # both
        l_model = np.unique([l_model_add+l_model_mult]).tolist()
    # logger_info('debug l_Model',l_model)
    if l_model_naive is not None:
        l_model = l_model_naive+l_model

    # if model_outliers is not None:
    #     l_model_outlier = [forecast_models.model_null, model_outliers]
    #     l_model = get_list_model(l_model, l_model_outlier, 'add')

    if do_find_steps_and_spikes:
        a_y = df_y.y.values
        a_x = df_y.y

        a_date = df_y.date if 'date' in df_y.columns else None

        steps, spikes = forecast_models.find_steps_and_spikes(a_x, a_y, a_date)
        if steps:
            steps_summed = reduce(lambda x, y: x + y, steps)
            steps_summed.name = '{}_fixed_steps'.format(len(steps))
            l_model = [model + steps_summed for model in l_model]
        if spikes:
            spikes_mult = reduce(lambda x, y: x * y, spikes)
            spikes_mult.name = '{}_fixed_spikes'.format(len(spikes))
            # filter values during the spike
            a_y_filt = spikes_mult(a_x, a_date, [])
            df_y[a_y_filt == 0] = np.nan

    # exclude samples with weight = 0
    df_y = df_y.loc[df_y.weight > 0]
    date_start_actuals = df_y.date.min() if 'date' in df_y.columns else df_y.x.min()
    x_start_actuals = df_y.x.min()

    df_actuals_cols = [c for c in ['date','x'] if c in df_y.columns]

    df_actuals_interpolated = (  # Fills gaps, used for extrapolation
        df_y_unfiltered
        .merge(df_y_unfiltered[df_actuals_cols].drop_duplicates('x').pipe(model_utils.interpolate_df), how='right')
        .sort_values(['x']).reset_index(drop=True)
    )
    # Update weight column in df_actuals_interpolated
    df_actuals_interpolated = df_actuals_interpolated.drop(columns=['weight'],errors='ignore')
    df_actuals_interpolated = df_actuals_interpolated.merge(df_y[['x','weight']],how='left')
    df_actuals_interpolated['weight']=df_actuals_interpolated.weight.fillna(0)

    # Note - In the above steps, we first remove any samples with weight = 0
    # from the data used for fitting
    # then we fill gaps in dates from the table used for extrapolating.
    # The filled gaps have NaN values in the y column, 0 weight

    for model in l_model:

        dict_fit_model = fit_model(model, df_y, freq, source, df_actuals=df_y_unfiltered)
        df_metadata_tmp = dict_fit_model['metadata']
        df_optimize_info = dict_fit_model['optimize_info']

        l_df_metadata += [df_metadata_tmp]
        l_df_optimize_info += [df_optimize_info]
        source_long = df_metadata_tmp.source_long.iloc[0]
        params = df_metadata_tmp.params.iloc[0]

        if df_metadata_tmp.is_fit.iloc[0]:  # If model is fit

            # date_start_actuals = df_y.date.min()
            # date_end_actuals = df_y.date.max()

            df_data_tmp = extrapolate_model(model, params,
                                            date_start_actuals,date_end_actuals,
                                            freq, extrapolate_years, x_start_actuals = x_start_actuals,
                                            df_actuals=df_actuals_interpolated)

            df_data_tmp = _get_df_fcast_clean(df_data_tmp, source_id, source_long, model.name)

            l_df_data += [df_data_tmp]

    # Generate output
    df_data = pd.concat(l_df_data, sort=False, ignore_index=True)
    df_metadata = pd.concat(l_df_metadata, sort=False, ignore_index=True)
    df_optimize_info = pd.concat(l_df_optimize_info, sort=False, ignore_index=True)

    # Determine best fits
    df_best_fit = (
        df_metadata.loc[df_metadata.is_fit]
            .sort_values('aic_c')
            .groupby('source', as_index=False).first()
        [['source_long', 'model']]
    )
    df_best_fit['is_best_fit'] = True

    df_metadata = df_metadata.merge(df_best_fit, how='left')
    df_metadata['is_best_fit'] = df_metadata['is_best_fit'].fillna(False).astype(bool)
    df_data = df_data.merge(df_best_fit, how='left').reset_index(drop=True)
    df_data['is_best_fit'] = df_data['is_best_fit'].fillna(False).astype(bool)

    if not include_all_fits:
        df_metadata = df_metadata.loc[df_metadata.is_best_fit].reset_index(drop=True)
        df_data = df_data.loc[df_data.is_best_fit | df_data.is_actuals].reset_index(drop=True)

    df_forecast = df_data.pipe(get_pi, n=100)
    dict_result = {'forecast': df_forecast, 'data': df_data, 'metadata': df_metadata, 'optimize_info': df_optimize_info}

    if simplify_output:
        return df_forecast
    else:
        return dict_result


# TODO: Better define return_all_fits, return_all_models. Document and provide clear use cases
# TODO: Improve test, make shorter
def run_l_forecast(l_fcast_input,
                   col_name_y='y', col_name_weight='weight',
                   col_name_x='x', col_name_date='date',
                   col_name_source='source',
                   extrapolate_years=0, season_add_mult='add',
                   include_all_fits=False,
                   do_find_steps_and_spikes=False,
                   find_outliers=False):
    """
    Generate forecasts for a list of SolverConfig objects, each including a time series, model functions, and other
    configuration parameters.

    :param l_fcast_input: List of forecast input configurations. Each element includes a time series,
        candidate forecast models for trend and seasonality, and other configuration parameters. For each input
        configuration, a forecast time series will be generated.
    :type l_fcast_input: list of ForecastInput
    :param return_all_models:
        | If True, result includes non-fitting models, with null AIC and an empty forecast df.
        | Otherwise, result includes only fitting models, and for time series where no fitting model is available,
        | a 'no-best-model' entry with null AIC and an empty forecast df is added.
    :type return_all_models: bool
    :param return_all_fits: If True, result includes all models for each input time series. Otherwise, only the
        best model is included.
    :type return_all_fits: bool
    :param extrapolate_years:
    :type extrapolate_years: float
    :param season_add_mult: 'add', 'mult', or 'both'. Whether forecast seasonality will be additive, multiplicative,
        or the best fit of the two.
    :type season_add_mult: str
    :param fill_gaps_y_values: If True, gaps in time series will be filled with NaN values
    :type fill_gaps_y_values: bool
    :param freq: 'W' or 'D' . Sampling frequency of the output forecast: weekly or daily.
    :type freq: str
    :return:
        | dict(data,metadata)
        | data: dataframe(date, source, model, y)
        | metadata: dataframe('source', 'model', 'res_weights', 'freq', 'is_fit', 'cost', 'aic_c', 'params', 'status')
    :rtype: dict

    """
    # TODO: Add check for non-duplicate source ids
    l_df_data = []
    l_df_metadata = []
    l_df_optimize_info = []

    # We can take solver_config_list that are a list or a single forecast_input
    if type(l_fcast_input) is not list:
        l_fcast_input = [l_fcast_input]

    l_dict_result = []
    for fcast_input in l_fcast_input:
        dict_result = run_forecast(fcast_input.df_y, fcast_input.l_model_trend, fcast_input.l_model_season,
                                   fcast_input.date_start_actuals, fcast_input.source_id,
                                   col_name_y, col_name_weight,
                                   col_name_x, col_name_date,
                                   col_name_source,
                                   extrapolate_years, season_add_mult,
                                   include_all_fits, simplify_output=False,
                                   do_find_steps_and_spikes=do_find_steps_and_spikes,
                                   find_outliers=find_outliers)
        l_dict_result += [dict_result]

    # Generate output
    return aggregate_forecast_dict_results(l_dict_result)


# Forecast configuration

# TODO: Rename to ForecastInput
class ForecastInput:
    """
    Class that encapsulates input variables for forecast.run_forecast()
    """

    def __init__(self, source_id, df_y, l_model_trend=None, l_model_season=None,
                 weights_y_values=1.0, date_start_actuals=None):
        self.source_id = source_id
        self.df_y = df_y
        self.l_model_trend = l_model_trend if l_model_trend is not None else [forecast_models.model_linear]
        self.l_model_season = l_model_season
        self.weights_y_values = weights_y_values
        self.date_start_actuals = date_start_actuals

    def __str__(self):
        str_result = (
            'SolverConfig: {source_id} ; {df_y_shape} ; {weights_y_values};'
            ' {l_model_trend}; {l_model_season} ; {date_start_actuals}'
        ).format(source_id=self.source_id, df_y_shape=self.df_y.shape,
                 l_model_trend=to_str_function_list(self.l_model_trend),
                 l_model_season=to_str_function_list(self.l_model_season),
                 weights_y_values=self.weights_y_values, date_start_actuals=self.date_start_actuals)
        return str_result

    def __repr__(self):
        return self.__str__()

    # TODO: REMOVE
    @classmethod
    def create(cls, source_id, df_y, l_model_trend, l_model_season=None,
               weights_y_values=1.0, date_start_actuals=None):
        return cls(source_id, df_y, pd.Series(l_model_trend), l_model_season,
                   weights_y_values, date_start_actuals)


"""
Draft for a parallel computing version:
run_forecast_parallel(n)
- take solver_config_list, split into n parts
- open n processes for run_forecast, each with 1/n of solver_config_list
- merge outputs: a dict with a pd.concat() of each output dataframe
- challenge: pickling objects: solver_config_list, pandas dataframe
- potential solution: have solver_config_list replace dataframes with file paths 

"""


def get_pi(df_forecast, n=100):
    if 'source' in df_forecast.columns and df_forecast.source.nunique() > 1:
        df_result = (
            df_forecast
		.groupby('source', as_index=False)
		.apply(_get_pi_single_source, n)
		.sort_values(['source', 'is_actuals', 'date'])
		.reset_index(drop=True)
        )
    else:
        df_result = _get_pi_single_source(df_forecast, n)
    return df_result


# TODO: Test
def _get_pi_single_source(df_forecast, n=100):
    # n: Number of bootstrapped samples for prediction interval

    if 'is_best_fit' in df_forecast.columns:
        df_forecast = df_forecast.loc[df_forecast.is_actuals | df_forecast.is_best_fit].copy()
    else:
        df_forecast = df_forecast.copy()

    if 'source' in df_forecast.columns:
        l_cols = ['date', 'source']
    else:
        l_cols = ['date']

    # logger_info('DEBUG - df_forecast', df_forecast.head(1))
    if 'is_weight' in df_forecast.columns and df_forecast.is_weight.any():

        # Filter out dates for outliers with weight=0
        df_filtered_dates = (
            df_forecast.loc[df_forecast.is_weight & df_forecast.y > 0]
            [['date', 'source']]
        )

        # Take filtered actuals
        df_actuals_unfiltered = df_forecast.loc[df_forecast.is_actuals & ~df_forecast.is_weight &
                                                ~df_forecast.y.isnull()]
        df_actuals = (df_actuals_unfiltered[['date', 'y']]
                      .merge(df_filtered_dates, how='inner')
                      .rename({'y': 'actuals'}, axis=1)
                      )
        date_last_actuals = df_actuals.date.max()
    else:  # No weight data - use all actuals rows
        df_actuals_unfiltered = df_forecast.loc[df_forecast.is_actuals & ~df_forecast.y.isnull()]
        df_actuals = (df_actuals_unfiltered[['date', 'y']]
                      .rename({'y': 'actuals'}, axis=1)
                      )
        date_last_actuals = df_actuals.date.max()
    # Compute residuals for filtered actuals
    df_residuals_tmp = df_forecast.loc[~df_forecast.is_actuals & ~df_forecast.y.pipe(pd.isnull)][l_cols+['model', 'y']]

    df_residuals = df_residuals_tmp.merge(df_actuals, how='inner')
    df_residuals['res'] = df_residuals['actuals'] - df_residuals['y']

    # Filter out null values, e.g. due to null actuals
    df_residuals = df_residuals.loc[~df_residuals.res.isnull()]

    if df_residuals.empty:  # May happen if no forecast could be generated
        logger.warning('No forecast data for source %s', df_forecast.source.head(1).iloc[0])
        return df_actuals_unfiltered[l_cols + ['is_actuals', 'model', 'y']]

    # Generate table with prediction interval
    df_forecast_pi = (
        df_forecast
            .loc[~df_forecast.is_actuals & (df_forecast.date > date_last_actuals)]
        [l_cols+['model','y']]
    )

    s_residuals_tmp = df_residuals.res
    a_forecast_point = df_forecast_pi.y.values

    length = a_forecast_point.size

    a_sample = s_residuals_tmp.sample(length * n, replace=True).values.reshape(n, length)
    a_sample = np.cumsum(a_sample, axis=1)

    a_q5 = np.percentile(a_sample, 5, axis=0)
    a_q95 = np.percentile(a_sample, 95, axis=0)
    a_q80 = np.percentile(a_sample, 80, axis=0)
    a_q20 = np.percentile(a_sample, 20, axis=0)

    df_forecast_pi['q5'] = a_q5 + df_forecast_pi.y
    df_forecast_pi['q20'] = a_q20 + df_forecast_pi.y
    df_forecast_pi['q80'] = a_q80 + df_forecast_pi.y
    df_forecast_pi['q95'] = a_q95 + df_forecast_pi.y
    df_forecast_pi['is_actuals'] = False

    # Past forecast samples, no prediction interval
    df_forecast_past = (
        df_forecast
            .loc[~df_forecast.is_actuals & (df_forecast.date <= date_last_actuals)]
        [l_cols+['model', 'is_actuals', 'y']]
    )

    df_actuals_unfiltered = df_actuals_unfiltered[l_cols + ['is_actuals', 'model', 'y']]
    df_pi_result = pd.concat([df_actuals_unfiltered, df_forecast_past, df_forecast_pi, ], sort=False, ignore_index=True)

    return df_pi_result

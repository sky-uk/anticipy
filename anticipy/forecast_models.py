# -*- coding: utf-8 -*-
#
# License:          This module is released under the terms of the LICENSE file 
#                   contained within this applications INSTALL directory

"""
Defines the ForecastModel class, which encapsulates model functions used in forecast model fitting, as well as
their number of parameters and initialisation parameters.
"""

# TODO: Check parameter initialisation

# TODO: It may be convenient to have different model functions for addition and subtraction,
# e.g. to return 1 or 0 by default

# -- Coding Conventions
#    http://www.python.org/dev/peps/pep-0008/   -   Use the Python style guide
#    http://sphinx.pocoo.org/rest.html          -   Use Restructured Text for docstrings

# -- Public Imports
import logging
import numpy as np
import pandas as pd
import itertools

# -- Private Imports
from anticipy import model_utils

# -- Globals
logger = logging.getLogger(__name__)
dict_fourier = {
    'period': 365.25,  # days in year
    'harmonics': 10  # TODO: evaluate different harmonics values
}


# -- Exception classes

# -- Functions
def logger_info(msg, data):
    # Convenience function for easier log typing
    logger.info(msg + '\n%s', data)


def _is_multi_ts(a):
    return a.ndim > 1 and a.shape[1] > 1


def _get_f_init_params_default(n_params):
    # Generate a default function for initialising model parameters: use random values between 0 and 1
    return lambda a_x=None, a_y=None, a_date=None, is_mult=False: np.random.uniform(low=0.001, high=1, size=n_params)


def _get_f_bounds_default(n_params):
    # Generate a default function for model parameter boundaries. Default boundaries are (-inf, inf)
    return lambda a_x=None, a_y=None, a_date=None: (n_params * [-np.inf], n_params * [np.inf])


def _get_f_add_2_f_models(forecast_model1, forecast_model2):
    def f_add_2_f_models(a_x, a_date, params, is_mult=False, **kwargs):
        params1 = params[0:forecast_model1.n_params]
        params2 = params[forecast_model1.n_params:]
        return (
                forecast_model1.f_model(a_x, a_date, params1, is_mult=False, **kwargs) +
                forecast_model2.f_model(a_x, a_date, params2, is_mult=False, **kwargs)
        )

    return f_add_2_f_models


def _get_f_mult_2_f_models(forecast_model1, forecast_model2):
    def f_mult_2_f_models(a_x, a_date, params, is_mult=False, **kwargs):
        params1 = params[0:forecast_model1.n_params]
        params2 = params[forecast_model1.n_params:]
        return (
                forecast_model1.f_model(a_x, a_date, params1, is_mult=True, **kwargs) *
                forecast_model2.f_model(a_x, a_date, params2, is_mult=True, **kwargs)
        )

    return f_mult_2_f_models


def _get_f_add_2_f_init_params(f_init_params1, f_init_params2):
    def f_add_2_f_init_params(a_x, a_y, a_date=None, is_mult=False):
        return np.concatenate([f_init_params1(a_x, a_y, a_date, is_mult=False),
                               f_init_params2(a_x, a_y, a_date, is_mult=False)])
    return f_add_2_f_init_params

def _get_f_mult_2_f_init_params(f_init_params1, f_init_params2):
    def f_mult_2_f_init_params(a_x, a_y, a_date=None, is_mult=False):
        return np.concatenate([f_init_params1(a_x, a_y, a_date, is_mult=True),
                               f_init_params2(a_x, a_y, a_date, is_mult=True)])
    return f_mult_2_f_init_params


def _get_f_concat_2_bounds(forecast_model1, forecast_model2):
    def f_add_2_f_bounds(a_x, a_y, a_date=None):
        return np.concatenate((forecast_model1.f_bounds(a_x, a_y, a_date),
                               forecast_model2.f_bounds(a_x, a_y, a_date)), axis=1)

    return f_add_2_f_bounds


# def _get_add_2_bounds(f_bounds1, f_bounds2):
#     return np.concatenate((f_bounds1, f_bounds2), axis=1)


# -- Classes

class ForecastModel:
    """
    Class that encapsulates model functions for use in forecasting, as well as
    their number of parameters and functions for parameter initialisation.

    A ForecastModel instance is initialized with a model name, a number of model parameters, and a model function.
    Class instances are callable - when called as a function, their internal model function is used. The main purpose
    of ForecastModel objects is to generate predicted values for a time series, given a set of parameters.
    These values can be compared to the original series to get an array of residuals::

        y_predicted = model(a_x, a_date, params)
        residuals = (a_y - y_predicted)

    This is used in an optimization loop to obtain the optimal parameters for the model.

    The reason for using this class instead of raw model functions is that ForecastModel supports function composition::

        model_sum = fcast_model1 + fcast_model2  # fcast_model 1 and 2 are ForecastModel instances, and so is model_sum
        a_y1 = fcast_model1(a_x, a_date, params1) + fcast_model2(a_x, a_date, params2)
        params = np.concatenate([params1, params2])
        a_y2 = model_sum(a_x, a_date, params)
        a_y1 == a_y2  # True

    Forecast models can be added or multiplied, with the + and * operators. Multiple levels of composition are
    supported::

        model = (model1 + model2) * model3

    Model composition is used to aggregate trend and seasonality model components, among other uses.

    Model functions have the following signature:

    - f(a_x, a_date, params, is_mult)
    - a_x : array of floats
    - a_date: array of dates, same length as a_x. Only required for date-aware models, e.g. for weekly seasonality.
    - params: array of floats - model parameters - the optimisation loop updates this to fit our actual values. Each
      model function uses a fixed number of parameters.
    - is_mult: boolean. True if the model is being used with multiplicative composition. Required because
      some model functions (e.g. steps) have different behaviour
      when added to other models than when multiplying them.
    - returns an array of floats - with same length as a_x - output of the model defined by this object's
      modelling function f_model and the current set of parameters

    By default, model parameters are initialized as random values between 0 and 1.
    It is possible to define a parameter initialization function that picks initial values
    based on the original time series.
    This is passed during ForecastModel creation with the argument f_init_params.
    Parameter initialization is compatible with model composition:
    the initialization function of each component will be used for that component's parameters.

    Parameter initialisation functions have the following signature:

    - f_init_params(a_x, a_y, is_mult)
    - a_x: array of floats - same length as time series
    - a_y: array of floats - time series values
    - returns an array of floats - with length equal to this object's n_params value

    By default, model parameters have no boundaries.
    However, it is possible to define a boundary function for a model,
    that sets boundaries for each model parameter, based on the input time series.
    This is passed during ForecastModel creation with the argument f_bounds.
    Boundary definition is compatible with model composition:
    the boundary function of each component will be used for that component's parameters.

    Boundary functions have the following signature:

    - f_bounds(a_x, a_y, a_date)
    - a_x: array of floats - same length as time series
    - a_y: array of floats - time series values
    - a_date: array of dates, same length as a_x. Only required for date-aware models, e.g. for weekly seasonality.
    - returns a tuple of 2 arrays of floats. The first defines minimum parameter boundaries, and the second
      the maximum parameter boundaries.

    Our input time series should meet the following constraints:

    - Minimum required samples depends on number of model parameters
    - May include null values
    - May include multiple values per sample
    - A date array is only required if the model is date-aware

    Class Usage::

        model_x = ForecastModel(name, n_params, f_model, f_init_params)
        model_name = model_x.name                           # Get model name
        n_params = model_x.n_params                         # Get number of model parameters
        f_init_params = model_x.f_init_params               # Get parameter initialisation function
        init_params = f_init_params(t_values, y_values)     # Get initial parameters
        f_model = model_x.f_model                           # Get model fitting function
        y = f_model(a_x, a_date, parameters)                # Get model output

    The following pre-generated models are available. They are available as attributes from this module:

    .. csv-table:: Forecast models
       :header: "name", "params", "formula","notes"
       :widths: 20, 10, 20, 40

       "model_null",0, "y=0", "Does nothing. Used to disable components (e.g. seasonality)"
       "model_constant",1, "y=A", "Constant model"
       "model_linear",2, "y=Ax + B", "Linear model"
       "model_linear_nondec",2, "y=Ax + B", "Non decreasing linear model. With boundaries to ensure model slope >=0"
       "model_quasilinear",3, "y=A*(x^B) + C", "Quasilinear model"
       "model_exp",2, "y=A * B^x", "Exponential model"
       "model_step",2, "y=0 if x<A, y=B if x>=A", "Step model"
       "model_two_steps",4, "see model_step", "2 step models. Parameter initialization is aware of # of steps."
       "model_sigmoid_step",3, "y = A + (B - A) / (1 + np.exp(- D * (x - C)))", "Sigmoid step model"
       "model_sigmoid",3, "y = A + (B - A) / (1 + np.exp(- D * (x - C)))", "Sigmoid model"
       "model_season_wday",7, "see desc.", "Weekday seasonality model. Assigns a constant value to each weekday"
       "model_season_wday",6, "see desc.", "6-param weekday seasonality model. As above, with one constant set to 0."
       "model_season_wday_2",2, "see desc.", "Weekend seasonality model. Assigns a constant to each of weekday/weekend"
       "model_season_month",12, "see desc.", "Month seasonality model. Assigns a constant value to each month"
       "model_season_fourier_yearly",10, "see desc", "Fourier yearly seasonality model"

    """

    def __init__(self, name, n_params, f_model, f_init_params=None, f_bounds=None):
        """
        Create ForecastModel

        :param name:
        :type name:
        :param n_params:
        :type n_params:
        :param f_model:
        :type f_model:
        :param f_init_params:
        :type f_init_params:
        :param f_bounds:
        :type f_bounds:
        """
        self.name = name
        self.n_params = n_params
        self.f_model = f_model
        if f_init_params is not None:
            self.f_init_params = f_init_params
        else:
            # Default initial parameters: random values between 0 and 1
            self.f_init_params = _get_f_init_params_default(n_params)

        if f_bounds is not None:
            self.f_bounds = f_bounds
        else:
            self.f_bounds = _get_f_bounds_default(n_params)

        # TODO - REMOVE THIS - ASSUME NORMALIZED INPUT
        def _get_f_init_params_validated(f_init_params):
            # Adds argument validation to a parameter initialisation function
            def f_init_params_validated(a_x=None, a_y=None, a_date=None, is_mult=False):
                if a_x is not None and pd.isnull(a_x).any():
                    raise ValueError('a_x cannot have null values')
                return f_init_params(a_x, a_y, a_date, is_mult)

            return f_init_params_validated

        # Add logic to f_init_params that validates input
        self.f_init_params = _get_f_init_params_validated(self.f_init_params)

    def __call__(self, a_x, a_date, params, is_mult=False, **kwargs):
        # assert len(params)==self.n_params
        return self.f_model(a_x, a_date, params, is_mult, **kwargs)

    def __str__(self):
        return self.name

    def __repr__(self):
        return 'ForecastModel:{}'.format(self.name)

    def __add__(self, forecast_model):
        # Check for nulls
        if self.name == 'null':
            return forecast_model
        if forecast_model.name == 'null':
            return self
        name = '({}+{})'.format(self.name, forecast_model.name)
        n_params = self.n_params + forecast_model.n_params
        f_model = _get_f_add_2_f_models(self, forecast_model)
        f_init_params = _get_f_add_2_f_init_params(self.f_init_params, forecast_model.f_init_params)
        f_bounds = _get_f_concat_2_bounds(self, forecast_model)
        return ForecastModel(name, n_params, f_model, f_init_params,
                             f_bounds=f_bounds)

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, forecast_model):
        if self.name == 'null':
            return forecast_model
        if forecast_model.name == 'null':
            return self
        name = '({}*{})'.format(self.name, forecast_model.name)
        n_params = self.n_params + forecast_model.n_params
        f_model = _get_f_mult_2_f_models(self, forecast_model)
        f_init_params = _get_f_mult_2_f_init_params(self.f_init_params, forecast_model.f_init_params)
        f_bounds = _get_f_concat_2_bounds(self, forecast_model)
        return ForecastModel(name, n_params, f_model, f_init_params, f_bounds)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __eq__(self, other):
        if isinstance(self, other.__class__):
            return self.name == other.name
        return NotImplemented

    def __ne__(self, other):
        x = self.__eq__(other)
        if x is not NotImplemented:
            return not x
        return NotImplemented

    def __hash__(self):
        return hash(self.name)

    def __lt__(self, other):
        return self.name < other.name

# - Null model: 0

def _f_model_null(a_x, a_date, params, is_mult=False, **kwargs):
    # This model does nothing - used to disable model components (e.g. seasonality) when adding/multiplying
    # multiple functions
    return float(is_mult)  # Returns 1 if multiplying, 0 if adding


model_null = ForecastModel('null', 0, _f_model_null)


# - Constant model: :math:`Y = A`

def _f_model_constant(a_x, a_date, params, is_mult=False, **kwargs):
    [A] = params
    y = np.full(len(a_x), A)
    return y


def _f_init_params_constant(a_x=None, a_y=None, a_date=None, is_mult=False):
    if a_y is None:
        return np.random.uniform(0, 1, 1)
    else:
        return np.nanmean(a_y) + np.random.uniform(0, 1, 1)


model_constant = ForecastModel('constant', 1, _f_model_constant, _f_init_params_constant)


# - Naive model: Y = Y(x-1)
# Note: This model requires passing the actuals data - it is not fitted by regression
# We still pass it to forecast.fit_model() for consistency with the rest of the library

def _f_model_naive(a_x, a_date, params, is_mult=False, df_actuals=None):
    if df_actuals is None:
        raise ValueError('model_naive requires a df_actuals argument')
    df_out_tmp = pd.DataFrame({'date':a_date,'x':a_x})
    df_out = (
        df_actuals.drop_duplicates('x') # This is not really intended to work with multiple values per sample
        .merge(df_out_tmp, how='outer')
    )
    df_out['y'] = df_out.y.shift(1).fillna(method='ffill').fillna(method='bfill')
    df_out = df_out.loc[df_out.x.isin(a_x)]
    #df_out = df_out_tmp.merge(df_out, how='left') # TODO: CHECK THAT X,DATE order is preserved
    # TODO: df_out = df_out.merge(df_out_tmp, how='right')
    return df_out.y.values

model_naive = ForecastModel('naive',0, _f_model_naive)

# - Seasonal naive model
# Note: This model requires passing the actuals data - it is not fitted by regression
# We still pass it to forecast.fit_model() for consistency with the rest of the library

def _fillna_wday(df):
    df = df.copy()
    df['wday'] = df.date.dt.weekday
    df_tmp = df[['date', 'x']].copy()
    for wday in np.arange(0, 7):
        wday_name = 'wday_{}'.format(wday)
        df_tmp[wday_name] = (
            df.y.where((df.wday) == wday, np.NaN)  # for each wday column, set to null all values from other weekdays
                .fillna(method='ffill')  # fill nulls with last weekly sample
                .shift(1)  # shift so that model for each sample is last non-null weekly sample
                .where(df.wday == wday, np.NaN)  # set values for other weekdays to null, so we can aggregate with sum
        )
    # logger_info('debug: df_tmp: ', df_tmp)

    # Aggregate: add all weekly columns together, keep null only if all columns are null
    def aggregate_wday(s_tmp):
        if np.all(np.isnan(s_tmp)):
            return np.NaN
        else:
            return np.nansum(s_tmp)

    df['y_out'] = df_tmp.loc[:, df_tmp.columns.str.startswith('wday_')].apply(aggregate_wday, axis=1)
    return df


def _f_model_snaive_wday(a_x, a_date, params, is_mult=False, df_actuals=None):
    if df_actuals is None:
        raise ValueError('model_snaive_wday requires a df_actuals argument')

    df_actuals_model = _fillna_wday(df_actuals.drop_duplicates('x'))

    df_last_week = df_actuals_model.drop_duplicates('wday', keep='last')[['wday', 'y']]
    df_last_week['y_out'] = df_last_week['y']
    df_last_week = df_last_week[['wday', 'y_out']]

    # logger_info('df_actuals_model:', df_actuals_model)
    # logger_info('df_last_week:', df_last_week)

    df_out_tmp = pd.DataFrame({'date': a_date, 'x': a_x})
    df_out_tmp['wday'] = df_out_tmp.date.dt.weekday

    # logger_info('df_out_tmp:', df_out_tmp)

    df_out_actuals = (
        df_actuals_model.merge(df_out_tmp, how='left')
    )
    df_out_extrapolated = (
        df_out_tmp.loc[~df_out_tmp.date.isin(df_actuals_model.date)]
            .merge(df_last_week)
            .sort_values('date').reset_index(drop=True)
    )

    df_out = pd.concat([df_out_actuals, df_out_extrapolated], sort=False)

    # logger_info('df_out_actuals:', df_out_actuals)
    # logger_info('df_out_extrapolated:', df_out_extrapolated)

    # Note: the line below causes trouble when samples are filtered from a_x, a_date due to find_outliers
    df_out = df_out.loc[df_out.x.isin(a_x)]

    # logger_info('df_out:', df_out)

    return df_out.y_out.values

model_snaive_wday = ForecastModel('snaive_wday', 0, _f_model_snaive_wday)


# - Spike model: :math:`Y = A`, when x_min <= X < x_max
def _f_model_spike(a_x, a_date, params, is_mult=False, **kwargs):
    [A, x_min, x_max] = params
    if is_mult:
        c = 1
    else:
        c = 0
    y = np.concatenate((
        np.full(int(x_min), c),
        np.full(int(x_max - x_min), A),
        np.full(len(a_x) - int(x_max), c)
    ))
    return y


# TODO: test f_init_params for all models
def _f_init_params_spike(a_x=None, a_y=None, a_date=None, is_mult=False):
    """ params are spike height, x start, x end """
    # if not a_y.any():
    if a_y is None:
        return [1] + np.random.uniform(0, 1, 1) + [2]
    else:
        diffs = np.diff(a_y)
        # if diffs:
        if True:
            diff = max(diffs)
            x_start = np.argmax(diffs)
            x_end = x_start + 1
            return np.array([diff, x_start, x_end])
            # else:
            #     rand = np.random.randint(1, len(a_y) - 1)

            return np.array([1, rand, rand + 1])


model_spike = ForecastModel('spike', 3, _f_model_spike, _f_init_params_spike)


# - Spike model for dates - dates are fixed for each model

def _f_model_spike_date(a_x, a_date, params, date_start, date_end, is_mult=False):
    [A] = params
    mask_spike = (a_date >= date_start) * (a_date < date_end)
    if is_mult:
        y = mask_spike * A + ~mask_spike
    else:
        y = mask_spike * A

    return y


def _f_init_params_spike(a_x=None, a_y=None, a_date=None, is_mult=False):
    """ params are spike height, x start, x end """
    if a_y is None:
        return np.concatenate([np.array([1]) + np.random.uniform(0, 1, 1)])
    else:
        diffs = np.diff(a_y)
        # if diffs:
        if True:
            diff = max(diffs)
            return np.array([diff])
        # else:
        #     rand = np.random.randint(1, len(a_y) - 1)
        #     return [1]


def get_model_spike_date(date_start, date_end):
    f_model = (
        lambda a_x, a_date, params, is_mult=False, **kwargs:
        _f_model_spike_date(a_x, a_date, params, date_start, date_end, is_mult)
    )
    model_spike_date = ForecastModel('spike_date[{},{}]'.format(pd.to_datetime(date_start).date(),
                                                                pd.to_datetime(date_end).date()),
                                     1, f_model, _f_init_params_spike)
    return model_spike_date


# - Linear model: :math:`Y = A*x + B`

def _f_model_linear(a_x, a_date, params, is_mult=False, **kwargs):
    (A, B) = params
    y = A * a_x + B
    return y


def _f_init_params_linear(a_x=None, a_y=None, a_date=None, is_mult=False):
    if a_y is None:
        return np.random.uniform(low=0, high=1, size=2)
    else:  # TODO: Improve this
        if a_x is not None:
            a_x_size = np.unique(a_x).size-1
        else:
            a_x_size = a_y.size-1
        A = (a_y[-1]-a_y[0])/a_x_size
        B = a_y[0]
        # Uniform low= 0*m, high = 1*m
        return np.array([A, B])


model_linear = ForecastModel('linear', 2, _f_model_linear, _f_init_params_linear)


def f_init_params_linear_nondec(a_x=None, a_y=None, a_date=None, is_mult=False):
    params = _f_init_params_linear(a_x, a_y, a_date)
    if params[0] < 0:
        params[0] = 0
    return params


def f_bounds_linear_nondec(a_x=None, a_y=None, a_date=None):
    # first param should be between 0 and inf
    return [0, -np.inf], [np.inf, np.inf]


model_linear_nondec = ForecastModel('linear', 2, _f_model_linear,
                                    f_init_params=f_init_params_linear_nondec,
                                    f_bounds=f_bounds_linear_nondec)


# - QuasiLinear model: :math:`Y = A t^{B} + C`

def _f_model_quasilinear(a_x, a_date, params, is_mult=False, **kwargs):
    (A, B, C) = params
    y = A * np.power(a_x, B) + C
    return y


model_quasilinear = ForecastModel('quasilinear', 3, _f_model_quasilinear)


# - Exponential model: math::  Y = A * B^t
def _f_model_exp(a_x, a_date, params, is_mult=False, **kwargs):
    (A, B) = params
    y = A * np.power(B, a_x)
    return y


model_exp = ForecastModel('exponential', 2, _f_model_exp)


def f_init_params_exp_dec(a_x=None, a_y=None, a_date=None, is_mult=False):
    """ B param must be <= 1 to have exponential decreasing """
    params = _get_f_init_params_default(2)(a_x, a_y, a_date)
    return params


def f_bounds_exp_dec(a_x=None, a_y=None, a_date=None):
    # first param should be between 0 and inf
    return [-np.inf, -1], [np.inf, 1]


model_exp_dec = ForecastModel('exponential_dec', 2, _f_model_exp,
                              f_init_params=f_init_params_exp_dec,
                              f_bounds=f_bounds_exp_dec)


# - Step function: :math:`Y = {0, if x < A | B, if x >= A}`
# A is the time of step, and B is the step
def _f_step(a_x, a_date, params, is_mult=False, **kwargs):
    (A, B) = params
    if is_mult:
        y = 1 + (B - 1) * np.heaviside(a_x - A, 1)
    else:
        y = B * np.heaviside(a_x - A, 1)
    return y

# TODO: Implement initialisation for multiplicative composition
def _f_init_params_step(a_x=None, a_y=None, a_date=None, is_mult=False):
    if a_y is None:
        return np.random.uniform(0, 1, 2)
    else:
        if a_y.ndim > 1:
            a_y = a_y[:, 0]
        df = pd.DataFrame({'b': a_y})
        # max difference between consecutive values
        df['diff'] = df.diff().abs()
        # if is_mult, replace above line with something like np.concatenate([[np.NaN],a_y[:-1]/a_y[1:]])
        a = df.nlargest(1, 'diff').index[0]
        b = df['diff'].iloc[a]
        return np.array([a, b * 2])


model_step = ForecastModel('step', 2, _f_step, _f_init_params_step)


# - Spike model for dates - dates are fixed for each model

def _f_model_step_date(a_x, a_date, params, date_start, is_mult=False):
    [A] = params
    mask_step = (a_date>=date_start).astype(float)
    if is_mult:
        # y = mask_step*A + ~mask_step
        y = mask_step * (A - 1) + 1
    else:
        y = mask_step * A

    return y


# TODO: Implement initialisation for multiplicative composition
def _f_init_params_step_date(a_x=None, a_y=None, a_date=None, is_mult=False):
    if a_y is None:
        return np.random.uniform(0, 1, 1)
    else:
        if a_y.ndim > 1:
            a_y = a_y[:, 0]
        df = pd.DataFrame({'b': a_y})
        # max difference between consecutive values
        df['diff'] = df.diff().abs()
        # if is_mult, replace above line with something like np.concatenate([[np.NaN],a_y[:-1]/a_y[1:]])
        a = df.nlargest(1, 'diff').index[0]
        b = df['diff'].iloc[a]
        return np.array([b * 2])


def get_model_step_date(date_start):
    date_start = pd.to_datetime(date_start)
    f_model = (
        lambda a_x, a_date, params, is_mult=False, **kwargs:
        _f_model_step_date(a_x, a_date, params, date_start, is_mult)
    )
    model_step_date = ForecastModel('step_date[{}]'.format(date_start.date()),
                                    1, f_model, _f_init_params_step_date)
    return model_step_date


# Two step functions
def _f_n_steps(n, a_x, a_date, params, is_mult=False):
    if is_mult:
        y = 1
    else:
        y = 0

    for i in range(0, n + 1, 2):
        A, B = params[i: i + 2]
        if is_mult:
            y = y * _f_step(a_x, a_date, (A, B), is_mult)
        else:
            y = y + _f_step(a_x, a_date, (A, B), is_mult)
    return y


def _f_two_steps(a_x, a_date, params, is_mult=False, **kwargs):
    return _f_n_steps(n=2, a_x=a_x, a_date=a_date, params=params, is_mult=is_mult)


def _f_init_params_n_steps(n=2, a_x=None, a_y=None, a_date=None, is_mult=False):
    if a_y is None:
        return np.random.uniform(0, 1, n * 2)
    else:
        # max difference between consecutive values
        if a_y.ndim > 1:
            a_y = a_y[:, 0]
        df = pd.DataFrame({'b': a_y})
        df['diff'] = df.diff().abs()
        # if is_mult, replace above line with something like np.concatenate([[np.NaN],a_y[:-1]/a_y[1:]])
        a = df.nlargest(n, 'diff').index[0:n].values
        b = df['diff'].iloc[a].values
        params = []
        for i in range(0, n):
            params += [a[i], b[i]]
        return np.array(params)


def _f_init_params_two_steps(a_x=None, a_y=None, a_date=None, is_mult=False):
    return _f_init_params_n_steps(n=2, a_x=a_x, a_y=a_y, a_date=a_date, is_mult=is_mult)


model_two_steps = ForecastModel('two_steps', 2 * 2, _f_two_steps, _f_init_params_two_steps)


# - Sigmoid step function: :math:`Y = {A + (B - A) / (1 + np.exp(- D * (a_x - C)))}`
# Spans from A to B, C is the position of the step in x axis
# and D is how steep the increase is
def _f_sigmoid(a_x, a_date, params, is_mult=False, **kwargs):
    (B, C, D) = params
    if is_mult:
        A = 1
    else:
        A = 0
    # TODO check if a_x is negative
    y = A + (B - A) / (1 + np.exp(- D * (a_x - C)))
    return y


def _f_init_params_sigmoid_step(a_x=None, a_y=None, a_date=None, is_mult=False):
    if a_y is None:
        return np.random.uniform(0, 1, 3)
    else:
        if a_y.ndim > 1:
            a_y = a_y[:, 0]
        df = pd.DataFrame({'y': a_y})
        # max difference between consecutive values
        df['diff'] = df.diff().abs()
        c = df.nlargest(1, 'diff').index[0]
        b = df.loc[c, 'y']
        d = b * b
        return b, c, d


def _f_init_bounds_sigmoid_step(a_x=None, a_y=None, a_date=None):
    if a_y is None:
        return [-np.inf, -np.inf, 0.], 3 * [np.inf]

    if a_y.ndim > 1:
        a_y = a_y[:, 0]
    if a_x.ndim > 1:
        a_x = a_x[:, 0]
    diff = max(a_y) - min(a_y)
    b_min = -2 * diff
    b_max = 2 * diff
    c_min = min(a_x)
    c_max = max(a_x)
    d_min = 0.
    d_max = np.inf
    return [b_min, c_min, d_min], [b_max, c_max, d_max]


# In this model, parameter initialization is aware of number of steps
model_sigmoid_step = ForecastModel('sigmoid_step', 3, _f_sigmoid, _f_init_params_sigmoid_step,
                                   f_bounds=_f_init_bounds_sigmoid_step)

model_sigmoid = ForecastModel('sigmoid', 3, _f_sigmoid)


# Ramp functions - used for piecewise linear models

# example : model_linear_pw2 = model_linear + model_ramp
# example 2: model_linear_p23 = model_linear + model_ramp + model_ramp

# - Ramp function: :math:`Y = {0, if x < A | B, if x >= A}`
# A is the time of step, and B is the step
def _f_ramp(a_x, a_date, params, is_mult=False, **kwargs):
    (A, B) = params
    if is_mult:
        y = 1 + (a_x - A) * (B) * np.heaviside(a_x - A, 1)
    else:
        y = (a_x - A) * B * np.heaviside(a_x - A, 1)
    return y


def _f_init_params_ramp(a_x=None, a_y=None, a_date=None, is_mult=False):
    # TODO: set boundaries: a_x (0.2, 0.8)
    if a_y is None:
        if a_x is not None:
            nfirst_last = int(np.ceil(0.15 * a_x.size))
            a = np.random.uniform(a_x[nfirst_last],a_x[-nfirst_last-1],1)
        else:
            a = np.random.uniform(0, 1, 1)
        b = np.random.uniform(0, 1, 1)

        return np.concatenate([a,
                               b])
    else:
        df = pd.DataFrame({'b': a_y})  # TODO: FILTER A_Y BY 20-80 PERCENTILE IN A_X
        if a_x is not None:
            #
            df['x']=a_x
            # Required because we support input with multiple samples per x value
            df = df.drop_duplicates('x')
            df=df.set_index('x')
        # max difference between consecutive values -- this assumes no null values in series
        df['diff2'] = df.diff().diff().abs()

        # We ignore the last 15% of the time series
        skip_samples = int(np.ceil(df.index.size * 0.15))

        a = (df
            .head(-skip_samples)
            .tail(-skip_samples)
            .nlargest(1, 'diff2').index[0]
            )
        b = df['diff2'].loc[a]
        # TODO: replace b with estimation of slope in segment 2 minus slope in segment 1 - see init_params_linear
        # logger.info('DEBUG: init params ramp2: %s - %s ', a.tolist(),b.tolist())
        return np.array([a, b])


def _f_init_bounds_ramp(a_x=None, a_y=None, a_date=None):
    if a_x is None:
        a_min = -np.inf
        a_max = np.inf
    else:
        #a_min = np.min(a_x)
        nfirst_last = int(np.ceil(0.15 * a_x.size))
        a_min = a_x[nfirst_last]
        a_max = a_x[-nfirst_last]
        #a_min = np.percentile(a_x, 15)
        #a_max = np.percentile(a_x,85)
    if a_y is None:
        b_min = -np.inf
        b_max = np.inf
    else:
        # df = pd.DataFrame({'b': a_y})  # TODO: FILTER A_Y BY 20-80 PERCENTILE IN A_X
        # #max_diff2 = np.max(df.diff().diff().abs())
        # max_diff2 = np.max(np.abs(np.diff(np.diff(a_y))))
        #
        # b_min = -2*max_diff2
        # b_max = 2*max_diff2

        b_min = -np.inf
        b_max = np.inf
    # logger_info('DEBUG: BOUNDS:',(a_min, b_min,a_max, b_max))
    return ([a_min, b_min], [a_max, b_max])


model_ramp = ForecastModel('ramp', 2, _f_ramp, _f_init_params_ramp, _f_init_bounds_ramp)


# - Weekday seasonality

def _f_model_season_wday(a_x, a_date, params, is_mult=False, **kwargs):
    # Weekday seasonality model, 6 params
    params_long = np.concatenate([[float(is_mult)], params])  # params_long[0] is default series value,
    return params_long[a_date.weekday]


model_season_wday = ForecastModel('season_wday', 6, _f_model_season_wday)


# - Month seasonality
def _f_init_params_season_month(a_x=None, a_y=None, a_date=None, is_mult=False):
    if a_y is None or a_date is None:
        return np.random.uniform(low=-1, high=1, size=11)
    else:                   # TODO: Improve this
        l_params_long = [np.mean(a_y[a_date.month==i]) for i in np.arange(1,13)]
        l_baseline = l_params_long[-1]
        l_params = l_params_long[:-1]
        if not is_mult:
            l_params_add = l_params-l_baseline
            return l_params_add
        else:
            l_params_mult = l_params/l_baseline
            return l_params_mult

def _f_model_season_month(a_x, a_date, params, is_mult=False, **kwargs):
    # Month of December is taken as default level, has no parameter
    params_long = np.concatenate([[float(is_mult)], params])   # params_long[0] is default series value,
    return params_long[a_date.month-1]

model_season_month = ForecastModel('season_month', 11, _f_model_season_month, _f_init_params_season_month)

model_season_month_old = ForecastModel('season_month_old', 11, _f_model_season_month)


def _f_model_yearly_season_fourier(a_x, a_date, params, is_mult=False, **kwargs):
    # Infer the time series frequency to calculate the Fourier parameters

    period = dict_fourier['period']
    harmonics = dict_fourier['harmonics']

    return _f_model_season_fourier(a_date, params, period, harmonics, is_mult)


date_origin = pd.datetime(1970, 1, 1)


def _f_model_season_fourier(a_date, params, period, harmonics, is_mult=False):
    # convert to days since epoch
    t = (a_date - date_origin).days.values
    i = np.arange(1,harmonics+1)
    a_tmp = i.reshape(i.size,1)* t
    k = (2.0 * np.pi / period)
    y = np.concatenate([np.sin(k*a_tmp), np.cos(k*a_tmp)])

    # now multiply by the params
    y = np.matmul(params, y)
    return y


def _f_init_params_fourier_n_params(n_params, a_x=None, a_y=None, a_date=None, is_mult=False):
    if a_y is None:
        params = np.random.uniform(0.001, 1, n_params)
    else:
        # max difference in time series
        diff = a_y.max() - a_y.min()
        params = diff * np.random.uniform(0.001, 1, n_params)
    return params


def _f_init_params_fourier(a_x=None, a_y=None, a_date=None, is_mult=False):
    n_params = 2 * dict_fourier['harmonics']
    return _f_init_params_fourier_n_params(
        n_params, a_x=a_x, a_y=a_y, a_date=a_date, is_mult=is_mult)


def _f_init_bounds_fourier_nparams(n_params, a_x=None, a_y=None, a_date=None):
    if a_y is None:
        return n_params * [-np.inf], n_params * [np.inf]
    if a_y.ndim > 1:
        a_y = a_y[:, 0]

    diff = a_y.max() - a_y.min()
    return n_params * [-2 * diff], n_params * [2 * diff]


def _f_init_bounds_fourier_yearly(a_x=None, a_y=None, a_date=None):
    n_params = 2 * dict_fourier['harmonics']
    return _f_init_bounds_fourier_nparams(n_params, a_x, a_y, a_date)


model_season_fourier_yearly = ForecastModel(
    name='season_fourier_yearly',
    n_params=2 * dict_fourier['harmonics'],
    f_model=_f_model_yearly_season_fourier,
    f_init_params=_f_init_params_fourier,
    f_bounds=_f_init_bounds_fourier_yearly)


def get_fixed_model(forecast_model, params_fixed, is_mult=False):
    if len(params_fixed) != forecast_model.n_params:
        err = 'Wrong number of fixed parameters'
        raise ValueError(err)
    return ForecastModel(forecast_model.name + '_fixed', 0,
                         f_model=lambda a_x, a_date, params, is_mult=is_mult, **kwargs:
                         forecast_model.f_model(
                             a_x=a_x, a_date=a_date, params=params_fixed, is_mult=is_mult))


def get_iqr_thresholds(s_diff, low=0.25, high=0.75):
    # Get thresholds based on inter quantile range
    q1 = s_diff.quantile(low)
    q3 = s_diff.quantile(high)
    iqr = q3 - q1
    thr_low = q1 - 1.5 * iqr
    thr_hi = q3 + 1.5 * iqr
    return thr_low, thr_hi

def get_model_outliers_withgap(df, window=3):
    # TODO: ADD CHECK, TO PREVENT REDUNDANT OPS IN DF WITHOUT GAPS

    df_nogap = df.pipe(model_utils.interpolate_df, include_mask=True)
    mask_step, mask_spike = get_model_outliers(df_nogap)

    ## TODO: FOR EACH OF MASK STEP, MASK SPIKE, IF IT IS NONE, RETURN NONE
    if mask_spike is None and mask_step is None:
        return None,None
    if mask_spike is not None:
        df_nogap['mask_spike'] = mask_spike
    if mask_step is not None:
        df_nogap['mask_step'] = mask_step
        df_nogap['step_in_filled_gap'] = df_nogap.mask_step * df_nogap.is_gap_filled
        df_nogap['mask_step_patch'] = df_nogap.step_in_filled_gap.shift(-1).fillna(0)

    df_nogap = df_nogap.loc[~df_nogap.is_gap_filled]

    if mask_step is not None:
        df_nogap['mask_step_patch'] = df_nogap.mask_step_patch.shift(1).fillna(0)
        df_nogap['mask_step'] = df_nogap.mask_step + df_nogap.mask_step_patch

    logger_info('df 1 - no gap:', df_nogap)

    if mask_step is not None:
        mask_step = df_nogap.mask_step.values
    if mask_spike is not None:
        mask_spike = df_nogap.mask_spike.values
    return mask_step, mask_spike

    # todo - clean up, return



# TODO: Add option - estimate_outl_size
# TODO: Add option - sigmoid steps
# TODO: ADD option - gaussian spikes
def get_model_outliers(df, window=3):
    """

    :param df:
    :type df:
    :param window:
    :type window:
    :return:
    :rtype:
    Note: due to the way the thresholds are defined, we require 6+ samples in series to find a spike.
    """
    is_mult = False

    dfo = df.copy()  # dfo - df for outliers
    with_dates = 'date' in df.columns  # If df has datetime index, use date logic in steps/spikes
    x_col = 'date' if with_dates else 'x'

    if df[x_col].duplicated().any():
        raise ValueError('Input cannot have multiple values per sample')

    # logger_info('debug 0 :', dfo)

    dfo['dif'] = dfo.y.diff()  # .fillna(0)

    # TODO: If df has weight column, use only samples with weight=1 for IQR

    thr_low, thr_hi = get_iqr_thresholds(dfo.dif)
    # Identify changes of state when diff value exceeds thresholds
    dfo['ischange'] = ((dfo.dif < thr_low) | (dfo.dif > thr_hi)).astype(int)

    dfo['ischange_group'] = (
        (dfo.ischange)
            .rolling(window, win_type=None, center=True).max()
            .fillna(0).astype(int)
    )

    dfo['dif_filt'] = (dfo.dif * dfo.ischange).fillna(0)
    dfo['dif_filt_abs'] = dfo.dif_filt.abs()

    dfo['ischange_cumsum'] = dfo.ischange.cumsum()
    dfo['change_group'] = dfo.ischange_group.diff().abs().fillna(0).astype(int).cumsum()

    df_mean_gdiff = (
        dfo.loc[dfo.ischange.astype(bool)].groupby('change_group')['dif_filt'].mean()
            .rename('mean_group_diff').reset_index()
    )

    df_mean_gdiff_abs = (
        dfo.loc[dfo.ischange.astype(bool)].groupby('change_group')['dif_filt_abs'].mean()
            .rename('mean_group_diff_abs').reset_index()
    )

    dfo = dfo.merge(df_mean_gdiff, how='left').merge(df_mean_gdiff_abs, how='left')
    dfo.mean_group_diff = dfo.mean_group_diff.fillna(0)
    dfo.mean_group_diff_abs = dfo.mean_group_diff_abs.fillna(0)

    dfo['is_step'] = (dfo.mean_group_diff < thr_low) | (dfo.mean_group_diff > thr_hi)
    dfo['is_spike'] = (dfo.mean_group_diff_abs - dfo.mean_group_diff) > (thr_hi - thr_low) / 2
    dfo['ischange_cumsum'] = dfo.ischange.cumsum()

    # logger_info('DF_OUTL: ',dfo)

    df_outl = (
        dfo.loc[dfo.ischange.astype(bool)].groupby('change_group')
        .apply(lambda x:pd.Series({'outl_start':x.head(1)[x_col].iloc[0],'outl_end':x.tail(1)[x_col].iloc[0]}))
        .reset_index()
    )

    if df_outl.empty:  # No outliers - nothing to do
        return None, None

    df_outl = df_outl.merge(dfo[['change_group', 'is_spike', 'is_step']].drop_duplicates())

    dfo = dfo.merge(df_outl, how='left')
    dfo['outl_start'] = dfo.outl_start.fillna(0).astype(int)
    dfo['outl_end'] = dfo.outl_end.fillna(0).astype(int)

    dfo = dfo  # .reset_index()

    df_spikes = df_outl.loc[df_outl.is_spike]
    df_steps = df_outl.loc[df_outl.is_step]

    l_model_outl = []
    l_mask_step = []
    l_mask_spike = []

    for g in df_spikes.change_group:
        s_spike = df_spikes.loc[df_spikes.change_group == g].iloc[0]
        if with_dates:
            mask_spike_tmp = ~((dfo.date>=pd.to_datetime(s_spike.outl_start)) &
                              (dfo.date<pd.to_datetime(s_spike.outl_end)))
        else:
            mask_spike_tmp = ~((dfo.x.values>=s_spike.outl_start) &
                             (dfo.x.values<s_spike.outl_end))
        l_mask_spike += [mask_spike_tmp.astype(float)]

    for g in df_steps.change_group:
        s_step = df_steps.loc[df_steps.change_group == g].iloc[0]
        if with_dates:
            model_step_tmp = get_model_step_date(pd.to_datetime(s_step.outl_start))
        else:
            model_step_tmp = (
                fix_params_fmodel(
                    model_step, [s_step.outl_start, np.NaN])
            )
        l_model_outl += [model_step_tmp]
        if with_dates:
            mask_step_tmp = (dfo.date==pd.to_datetime(s_step.outl_start))
        else:
            mask_step_tmp = (dfo.x.values==s_step.outl_start)
        l_mask_step += [mask_step_tmp.astype(float)]

    if len(l_model_outl) == 0:
        model_outliers = None
    else:
        model_outliers = np.prod(l_model_outl) if is_mult else np.sum(l_model_outl)
    if len(l_mask_spike) == 0:
        mask_spike = None
    else:
        mask_spike = np.prod(l_mask_spike, axis=0)
    if len(l_mask_step)==0:
        mask_step = None
    else:
        mask_step = np.sum(l_mask_step, axis=0)
    return mask_step, mask_spike
    #return model_outliers, mask_spike
    # return model_outliers, l_mask_spike


def find_steps_and_spikes(a_x, a_y, a_date, window=3, max_changes=None):
    """
        Automatically locate steps and spikes for a_y time series.
        A rule of thumb for the window is two weeks.

    :param a_x: The values of the x axis
    :param a_y: The values of the y axis
    :param a_date: The values of the x axis, if they are dates
    :param window: The x-axis window to aggregate multiple steps/spikes
    :type window: int
    :param max_changes: Manual input of the number of changes that the
                        input time series is expected to have.
                        If max_changes = 0 or None, the function returns
                        all changes (steps / spikes) found.
    :type max_changes: int
    :return: changes_list
    :rtype: list of dictionaries
            Each dictionary includes information on a single change,
            and is of the format:
            {'change_type': change_type,  # ('spike' or 'step')
            'duration': duration,  # int - the x-axis duration of the change
            'diff': diff,  # the difference within the change
            'x': x,  # the x-axis at the middle of the change
            'date': date}  # the x-axis date at the middle of the change
    """
    # find NaNs in a_y
    df = pd.DataFrame({'y': a_y}).interpolate('slinear')

    # Find peak changes in diff
    df['diff'] = df.y.diff()

    # Outliers are < Q1 - 3 * IQR, > Q3 + 3 * IQR
    # Q1 and Q3 are the 25th (1st) and 75th (3rd) quartiles, and
    # IQR is the inter-quartile range
    q1 = df['diff'].quantile(0.25)
    q3 = df['diff'].quantile(0.75)
    iqr = q3 - q1
    low_thresh = q1 - 1.5 * iqr
    high_thresh = q3 + 1.5 * iqr

    # df['is_change'] = 0
    step_filt = (df['diff'] < low_thresh) | (df['diff'] > high_thresh)
    df['is_change'] = step_filt.astype(int)

    if not any(step_filt):
        return [], []

    # df.loc[step_filt, 'is_change'] = 1

    # Now that we have found the outliers in differences,
    # group consecutive steps together

    # get only the diffs that correspond to changes
    df['diff'] = df['diff'] * df['is_change']
    df[['diff_sum', 'change_sum']] = df[['diff', 'is_change']].rolling(
        window, win_type=None, center=True).sum()

    # we have steps, we may need to aggregate
    # We split the array with zeros.
    # This means that we treat all nearby changes as one,
    # within `window` values
    split = np.split(df['change_sum'],
                     np.where(df['change_sum'] == 0.)[0])
    # get rid of zero only series
    split = [i for i in split if i.any()]

    # Now we have a list of series with the changes
    changes_list = []
    for s in split:
        change_s = df.iloc[s.index]
        change_max_occur = change_s[change_s.is_change == 1].index.max()
        change_min_occur = change_s[change_s.is_change == 1].index.min()
        diff = change_s['diff'].sum()
        duration = change_max_occur - change_min_occur

        if low_thresh <= diff <= high_thresh:  # Change is a spike
            change_type = 'spike'
            # we keep the starting point as x
            x = change_min_occur
            # get the average change for the values of the change that
            # are in the changing threshold
            diff = change_s.loc[(change_s.is_change == 1) &
                                ((change_s['diff'] < low_thresh) |
                                 (change_s['diff'] > high_thresh)), 'diff'].abs().mean()

        else:  # Change is a step
            # here we have a different starting point
            x = (change_max_occur + change_min_occur - 1) / 2.0
            change_type = 'step'

        d = {'change_type': change_type,
             'duration': duration,
             'diff': diff,
             'x': x}
        changes_list += [d]

    # Sort by absolute difference, in descending order
    sorted_changes_list = sorted(changes_list, key=lambda ch: abs(ch['diff']),
                                 reverse=True)

    # Rule of thumb: the maximum number of changes
    # is the square root of the time series length
    max_max_changes = int(np.floor(np.sqrt(len(a_y))))
    # If we have a max_changes input value, select the ones with higher diff
    if (not max_changes) or (max_changes > max_max_changes):
        max_changes = max_max_changes

    changes_list = sorted_changes_list[:max_changes]

    steps = []
    spikes = []
    for c in changes_list:
        # get models
        if c['change_type'] == 'spike':
            # spike = create_fixed_spike(c['diff'], x=c['x'],
            #                            duration=c['duration'])
            spike = create_fixed_spike_ignored(x=c['x'],
                                               duration=c['duration'])
            spikes += [spike]
        elif c['change_type'] == 'step':
            step = create_fixed_step(diff=c['diff'], x=c['x'])
            steps += [step]
        else:
            raise ValueError('Invalid change type: ' + c['change_type'])

    return steps, spikes


def create_fixed_step(diff, x):
    fixed_params = [x, diff]
    return get_fixed_model(model_step, fixed_params)


def create_fixed_spike(diff, x, duration):
    fixed_params = [diff, x, x + duration]
    return get_fixed_model(model_spike, fixed_params)


def create_fixed_spike_ignored(x, duration):
    fixed_params = [0, x, x + duration]
    return get_fixed_model(model_spike, fixed_params, is_mult=True)


# Dummy variable models

def get_model_dummy(name, dummy, **kwargs):
    """
    Generate a model based on a dummy variable.

    :param name:
    :type name:
    :param dummy:
      | Can be a function or a list-like.
      | If a function, it must be of the form f_dummy(a_x, a_date), and return a numpy array of floats
      | with the same length as a_x and values that are either 0 or 1.
      | If a list-like of numerics, it will be converted to a f_dummy function as described above, which will
      | have values of 1 when a_x has one of the values in the list, and 0 otherwise.
      | If a list-like of date-likes, it will be converted to a f_dummy function as described above, which will
      | have values of 1 when a_date has one of the values in the list, and 0 otherwise.
    :type dummy: function, or list-like of numerics or datetime-likes
    :param kwargs:
    :type kwargs:
    :return:
      | A model that returns A when dummy is 1, and 0 (or 1 if is_mult==True) otherwise.
    :rtype: ForecastModel


    """
    return ForecastModel(name, 1, get_f_model_dummy(dummy), **kwargs)


def _validate_f_dummy(f_dummy):
    # Ensures that behaviour of f_dummy matches specs
    # Must return array of floats, same length as a_x, with values either 0. or 1.
    def validate_for_dummy(a_dummy):
        assert isinstance(a_dummy, np.ndarray)
        assert (np.setdiff1d(a_dummy, np.array([0., 1.])).size) == 0

    # validate_for_dummy(f_dummy(np.arange(0, 10), None)) # Crashes with f_dummy 's that require dates
    validate_for_dummy(f_dummy(np.arange(0, 10), pd.date_range('2018-01-01', '2018-01-10')))


def get_f_model_dummy(dummy):
    """
    Generate a model function for a dummy variable defined by f_dummy

    :param dummy:
    :type dummy: function or list-like of numerics or dates
    :return: model function based on dummy variable, to use on a ForecastModel
    :rtype: function
    """

    if callable(dummy):  # If dummy is a function, use it
        f_dummy = dummy
    else:
        f_dummy = get_f_dummy_from_list(dummy)  # If dummy is a list, convert to function

    _validate_f_dummy(f_dummy)

    def f_model_check(a_x, a_date, params, is_mult=False, **kwargs):
        # Uses internal f_check to assign 0 or 1 to each sample
        # If f_dummy(x)==1, return A
        # If f_dummy(x)==0, return 0 (or 1 if is_mult)
        [A] = params
        mask = f_dummy(a_x, a_date)
        if not is_mult:
            a_result = A * mask
        else:
            a_result = (A - 1.) * mask + 1
        return a_result

    return f_model_check


def get_f_dummy_from_list(list_check):
    """
    Generate a f_dummy function that defines a dummy variable, can be used for dummy models

    :param list_check: Input list
    :type list_check: list-like of numerics or datetime-likes
    :return: f_dummy
    :rtype: function
    """
    # Generate a f_dummy function that defines a dummy variable, can be used for dummy models
    s_check = pd.Series(list_check)
    if pd.api.types.is_numeric_dtype(s_check):
        list_check_numeric = s_check

        def f_dummy_list_numeric(a_x, a_date):
            # return a_x in check_numeric
            return np.isin(a_x, list_check_numeric).astype(float)

        return f_dummy_list_numeric
    else:
        try:
            list_check_date = pd.to_datetime(s_check)

            def f_dummy_list_date(a_x, a_date):
                # return a_x in check_numeric
                return np.isin(a_date, list_check_date).astype(float)

            return f_dummy_list_date
        except:
            raise ValueError('list_dummy must be a list-like with numeric or date-like values: %s', list_check)


model_season_wday_2 = get_model_dummy('season_wday_2', lambda a_x, a_date, **kwargs: (a_date.weekday < 5).astype(float))

# Example dummy model - checks if it is Christmas
model_dummy_christmas = get_model_dummy('dummy_christmas',
                                        lambda a_x, a_date, **kwargs: ((a_date.month == 12) & (a_date.day == 25)).astype(float))

# Example dummy model - checks if it is first day of month
model_dummy_month_start = get_model_dummy('dummy_month_start',
                                          lambda a_x, a_date, **kwargs: (a_date.day == 1).astype(float))


# Utility functions

def fix_params_fmodel(forecast_model, l_params_fixed):
    """
    Given a forecast model and a list of floats, modify the model so that some of its parameters become fixed

    :param forecast_model:
    :type forecast_model:
    :param l_params_fixed: List of floats with same length as number of parameters in model. For each element, a
        non-null value means that the parameter in that position is fixed to that value. A null value means that
        the parameter in that position is not fixed.
    :type l_params_fixed: list
    :return: A forecast model with a number of parameters equal to the number of null values in l_params_fixed,
        with f_model modified so that some of its parameters gain fixed values equal to the non-null values in l_params
    :rtype:
    """
    assert len(l_params_fixed) == forecast_model.n_params

    l_params_fixed = np.array(l_params_fixed)

    a_null = np.isnan(l_params_fixed)
    i_null = np.nonzero(a_null)

    name = '{}_fixed_{}'.format(forecast_model.name, str(l_params_fixed).replace('nan', ':')
                                )
    n_params = len(i_null[0])

    def f_model_fixed(a_x, a_date, params, is_mult=False, **kwargs):
        params_long = l_params_fixed
        params_long[i_null] = params
        return forecast_model.f_model(a_x, a_date, params_long, is_mult)

    def f_init_params_fixed(a_x=None, a_y=None, a_date=None, is_mult=False):
        # return params short
        params_init = forecast_model.f_init_params(a_x, a_y, a_date, is_mult)
        params_init_short = np.array(params_init)[i_null]
        return params_init_short

    def f_bounds_fixed(a_x=None, a_y=None, a_date=None):
        # return f_bounds short
        bounds_min, bounds_max = forecast_model.f_bounds(a_x, a_y, a_date)
        bounds_min_short = np.array(bounds_min)[i_null]
        bounds_max_short = np.array(bounds_max)[i_null]
        return bounds_min_short, bounds_max_short

    model_result = ForecastModel(name, n_params, f_model_fixed, f_init_params_fixed, f_bounds_fixed)
    return model_result


def simplify_model(f_model, a_x=None, a_y=None, a_date=None):
    """
    Check a model's bounds, and update model to make parameters fixed if their min and max bounds are equal

    :param f_model:
    :type f_model:
    :param a_x:
    :type a_x:
    :param a_y:
    :type a_y:
    :param a_date:
    :type a_date:
    :return:
    :rtype:
    """
    bounds_min, bounds_max = f_model.f_bounds(a_x, a_y, a_date)
    bounds_diff = np.array(bounds_max) - np.array(bounds_min)
    i_diff_zero = np.nonzero(bounds_diff == 0)
    # For any parameter, if bounds_min == bounds_max, that parameter becomes fixed

    if i_diff_zero[0].size == 0:
        return f_model
    else:  # We make parameters fixed if their min and max bounds are equal
        params_fixed = np.full(f_model.n_params, np.NaN)
        params_fixed[i_diff_zero, ] = bounds_max[i_diff_zero, ]
        f_model = fix_params_fmodel(f_model, params_fixed)
        logger.info('Some min and max bounds are equal - generating fixed model: %s', f_model.name)
        return f_model


def validate_initial_guess(initial_guess, bounds):
    initial_guess = np.array(initial_guess)
    bounds_min, bounds_max = bounds
    return np.all((initial_guess >= bounds_min) & (initial_guess <= bounds_max))


def get_l_model_auto_season(a_date, min_periods=1.5, season_add_mult='add',
                            l_season_yearly=None, l_season_weekly=None):
    """
    Generates a list of candidate seasonality models for an series of timestamps

    :param a_date:
    :type a_date:
    :param min_periods:
    :type min_periods:
    :param is_mult:
    :type is_mult:
    :return:
    :rtype:
    """
    s_date = pd.Series(a_date).sort_values().drop_duplicates()
    min_date_delta = s_date.diff().min()
    max_date_delta = s_date.max() - s_date.min()

    if pd.isna(min_date_delta) or pd.isna(max_date_delta):
        return [model_null]

    use_season_yearly = (
            (max_date_delta > pd.Timedelta(min_periods * 365, unit='d')) &  # Need more than a full year
            (min_date_delta <= pd.Timedelta(92, unit='d'))  # Need at least quarterly samples
    )

    use_season_weekly = (
            (max_date_delta > pd.Timedelta(min_periods * 7, unit='d')) &  # Need more than a full week
            (min_date_delta <= pd.Timedelta(1, unit='d'))  # Need at least daily samples
    )

    l_season_yearly_default = [
        # model_season_month,
        model_season_fourier_yearly,
        model_null] if l_season_yearly is None else l_season_yearly
    l_season_weekly_default = [model_season_wday, model_null] if l_season_weekly is None else l_season_weekly

    if use_season_weekly:
        l_season_weekly = l_season_weekly_default
    else:
        l_season_weekly = [model_null]

    if use_season_yearly:
        l_season_yearly = l_season_yearly_default
        # TODO: add season_yearly_fourier
        # TODO: add holiday list
    else:
        l_season_yearly = [model_null]

    l_result = [model_null]
    for s_w, s_y in itertools.product(l_season_weekly, l_season_yearly):

        model_season_add = s_w + s_y
        model_season_mult = s_w * s_y

        if season_add_mult in ['add']  and model_season_add != model_null:
            l_result += [model_season_add]
        if season_add_mult in ['mult'] and model_season_mult != model_null and \
                model_season_mult not in l_result:
            l_result += [model_season_mult]

    return l_result

    """
    todo: RENAME SEASON MODELS
    - season_yearly_month, season_yearly_fourier, season_yearly_quarter
    - season_weekly_wday, season_weekly_wkd
    
    """

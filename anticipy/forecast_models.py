# -*- coding: utf-8 -*-
#
# License:          This module is released under the terms of the LICENSE file
#                   contained within this applications INSTALL directory

"""
Defines the ForecastModel class, which encapsulates model functions used in
forecast model fitting, as well as their number of parameters and
initialisation parameters.
"""

# -- Coding Conventions
#    http://www.python.org/dev/peps/pep-0008/   -   Use the Python style guide
# http://sphinx.pocoo.org/rest.html          -   Use Restructured Text for
# docstrings


# -- Public Imports
import itertools
import logging
import numpy as np
import pandas as pd
from pandas.tseries.holiday import Holiday, AbstractHolidayCalendar, \
    MO, nearest_workday, next_monday, next_monday_or_tuesday, \
    GoodFriday, EasterMonday, USFederalHolidayCalendar
from pandas.tseries.offsets import DateOffset
from datetime import datetime

# -- Private Imports
from anticipy import model_utils

# -- Globals
logger = logging.getLogger(__name__)

dict_fourier = {  # Default configuration for fourier-based models
    'period': 365.25,  # days in year
    'harmonics': 10  # TODO: evaluate different harmonics values
}


# -- Functions

# ---- Utility functions


def logger_info(msg, data):
    # Convenience function for easier log typing
    logger.info(msg + '\n%s', data)


def _get_f_init_params_default(n_params):
    # Generate a default function for initialising model parameters: use
    # random values between 0 and 1
    return lambda a_x=None, a_y=None, a_date=None, is_mult=False:\
        np.random.uniform(low=0.001, high=1, size=n_params)


def _get_f_bounds_default(n_params):
    # Generate a default function for model parameter boundaries. Default
    # boundaries are (-inf, inf)
    return lambda a_x=None, a_y=None, a_date=None: (
        n_params * [-np.inf], n_params * [np.inf])


def _get_f_add_2_f_models(forecast_model1, forecast_model2):
    # Add model functions of 2 ForecastModels
    def f_add_2_f_models(a_x, a_date, params, is_mult=False, **kwargs):
        params1 = params[0:forecast_model1.n_params]
        params2 = params[forecast_model1.n_params:]
        return (
            forecast_model1.f_model(
                a_x,
                a_date,
                params1,
                is_mult=False,
                **kwargs) +
            forecast_model2.f_model(
                a_x,
                a_date,
                params2,
                is_mult=False,
                **kwargs))

    return f_add_2_f_models


def _get_f_mult_2_f_models(forecast_model1, forecast_model2):
    # Multiply model functions of 2 ForecastModels
    def f_mult_2_f_models(a_x, a_date, params, is_mult=False, **kwargs):
        params1 = params[0:forecast_model1.n_params]
        params2 = params[forecast_model1.n_params:]
        return (
            forecast_model1.f_model(
                a_x,
                a_date,
                params1,
                is_mult=True,
                **kwargs) *
            forecast_model2.f_model(
                a_x,
                a_date,
                params2,
                is_mult=True,
                **kwargs))

    return f_mult_2_f_models


def _get_f_add_2_f_init_params(f_init_params1, f_init_params2):
    # Compose parameter initialisation functions of 2 ForecastModels, using
    # addition
    def f_add_2_f_init_params(a_x, a_y, a_date=None, is_mult=False):
        return np.concatenate(
            [f_init_params1(a_x, a_y, a_date, is_mult=False),
             f_init_params2(a_x, a_y, a_date, is_mult=False)])

    return f_add_2_f_init_params


def _get_f_mult_2_f_init_params(f_init_params1, f_init_params2):
    # Compose parameter initialisation functions of 2 ForecastModels, using
    # multiplication
    def f_mult_2_f_init_params(a_x, a_y, a_date=None, is_mult=False):
        return np.concatenate(
            [f_init_params1(a_x, a_y, a_date, is_mult=True),
             f_init_params2(a_x, a_y, a_date, is_mult=True)])

    return f_mult_2_f_init_params


def _get_f_concat_2_bounds(forecast_model1, forecast_model2):
    # Compose parameter boundary functions of 2 ForecastModels
    def f_add_2_f_bounds(a_x, a_y, a_date=None):
        return np.concatenate(
            (forecast_model1.f_bounds(
                a_x, a_y, a_date), forecast_model2.f_bounds(
                a_x, a_y, a_date)), axis=1)

    return f_add_2_f_bounds


def _f_validate_input_default(a_x, a_y, a_date):
    # Default input validation funciton for a ForecastModel. Always returns
    # True
    return True


# -- Classes

class ForecastModel:
    """
    Class that encapsulates model functions for use in forecasting, as well as
    their number of parameters and functions for parameter initialisation.

    A ForecastModel instance is initialized with a model name, a number of
    model parameters, and a model function. Class instances are
    callable - when called as a function, their internal model function is
    used. The main purpose of ForecastModel objects is to generate predicted
    values for a time series, given a set of parameters. These values can be
    compared to the original series to get an array of residuals::

        y_predicted = model(a_x, a_date, params)
        residuals = (a_y - y_predicted)

    This is used in an optimization loop to obtain the optimal parameters for
    the model.

    The reason for using this class instead of raw model functions is that
    ForecastModel supports function composition::

        model_sum = fcast_model1 + fcast_model2
        # fcast_model 1 and 2 are ForecastModel instances, and so is model_sum
        a_y1 = fcast_model1(
            a_x, a_date, params1) + fcast_model2(a_x, a_date, params2)
        params = np.concatenate([params1, params2])
        a_y2 = model_sum(a_x, a_date, params)
        a_y1 == a_y2  # True

    Forecast models can be added or multiplied, with the + and * operators.
    Multiple levels of composition are supported::

        model = (model1 + model2) * model3

    Model composition is used to aggregate trend and seasonality model
    components, among other uses.

    Model functions have the following signature:

    - f(a_x, a_date, params, is_mult)
    - a_x : array of floats
    - a_date: array of dates, same length as a_x. Only required for date-aware
      models, e.g. for weekly seasonality.
    - params: array of floats - model parameters - the optimisation loop
      updates this to fit our actual values. Each
      model function uses a fixed number of parameters.
    - is_mult: boolean. True if the model is being used with multiplicative
      composition. Required because
      some model functions (e.g. steps) have different behaviour
      when added to other models than when multiplying them.
    - returns an array of floats - with same length as a_x - output of the
      model defined by this object's modelling function f_model and the
      current set of parameters

    By default, model parameters are initialized as random values between
    0 and 1. It is possible to define a parameter initialization function
    that picks initial values based on the original time series.
    This is passed during ForecastModel creation with the argument
    f_init_params. Parameter initialization is compatible with model
    composition: the initialization function of each component will be used
    for that component's parameters.

    Parameter initialisation functions have the following signature:

    - f_init_params(a_x, a_y, is_mult)
    - a_x: array of floats - same length as time series
    - a_y: array of floats - time series values
    - returns an array of floats - with length equal to this object's n_params
      value

    By default, model parameters have no boundaries. However, it is possible
    to define a boundary function for a model, that sets boundaries for each
    model parameter, based on the input time series. This is passed during
    ForecastModel creation with the argument f_bounds.
    Boundary definition is compatible with model composition:
    the boundary function of each component will be used for that component's
    parameters.

    Boundary functions have the following signature:

    - f_bounds(a_x, a_y, a_date)
    - a_x: array of floats - same length as time series
    - a_y: array of floats - time series values
    - a_date: array of dates, same length as a_x. Only required for date-aware
      models, e.g. for weekly seasonality.
    - returns a tuple of 2 arrays of floats. The first defines minimum
      parameter boundaries, and the second the maximum parameter boundaries.

    As an option, we can assign a list of input validation functions to a
    model. These functions analyse the inputs that will be used for fitting a
    model, returning True if valid, and False otherwise. The forecast logic
    will skip a model from fitting if any of the validation functions for that
    model returns False.

    Input validation functions have the following signature:

    - f_validate_input(a_x, a_y, a_date)
    - See the description of model functions above for more details on these
      parameters.

    Our input time series should meet the following constraints:

    - Minimum required samples depends on number of model parameters
    - May include null values
    - May include multiple values per sample
    - A date array is only required if the model is date-aware

    Class Usage::

        model_x = ForecastModel(name, n_params, f_model, f_init_params,
        l_f_validate_input)
        # Get model name
        model_name = model_x.name
        # Get number of model parameters
        n_params = model_x.n_params
        # Get parameter initialisation function
        f_init_params = model_x.f_init_params
        # Get initial parameters
        init_params = f_init_params(t_values, y_values)
        # Get model fitting function
        f_model = model_x.f_model
        # Get model output
        y = f_model(a_x, a_date, parameters)

    The following pre-generated models are available. They are available as attributes from this module: # noqa

    .. csv-table:: Forecast models
       :header: "name", "params", "formula","notes"
       :widths: 20, 10, 20, 40

       "model_null",0, "y=0", "Does nothing.
       Used to disable components (e.g. seasonality)"
       "model_constant",1, "y=A", "Constant model"
       "model_linear",2, "y=Ax + B", "Linear model"
       "model_linear_nondec",2, "y=Ax + B", "Non decreasing linear model.
       With boundaries to ensure model slope >=0"
       "model_quasilinear",3, "y=A*(x^B) + C", "Quasilinear model"
       "model_exp",2, "y=A * B^x", "Exponential model"
       "model_decay",4, "Y = A * e^(B*(x-C)) + D", "Exponential decay model"
       "model_step",2, "y=0 if x<A, y=B if x>=A", "Step model"
       "model_two_steps",4, "see model_step", "2 step models.
       Parameter initialization is aware of # of steps."
       "model_sigmoid_step",3, "y = A + (B - A) / (1 + np.exp(- D * (x - C)))
       ", "Sigmoid step model"
       "model_sigmoid",3, "y = A + (B - A) / (1 + np.exp(- D * (x - C)))", "
       Sigmoid model"
       "model_season_wday",7, "see desc.", "Weekday seasonality model.
       Assigns a constant value to each weekday"
       "model_season_wday",6, "see desc.", "6-param weekday seasonality model.
       As above, with one constant set to 0."
       "model_season_wday_2",2, "see desc.", "Weekend seasonality model.
       Assigns a constant to each of weekday/weekend"
       "model_season_month",12, "see desc.", "Month seasonality model.
       Assigns a constant value to each month"
       "model_season_fourier_yearly",10, "see desc", "Fourier
       yearly seasonality model"

    """

    def __init__(
            self,
            name,
            n_params,
            f_model,
            f_init_params=None,
            f_bounds=None,
            l_f_validate_input=None):
        """
        Create ForecastModel

        :param name: Model name
        :type name: basestring
        :param n_params: Number of parameters for model function
        :type n_params: int
        :param f_model: Model function
        :type f_model: function
        :param f_init_params: Parameter initialisation function
        :type f_init_params: function
        :param f_bounds: Boundary function
        :type f_bounds: function
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

        if l_f_validate_input is None:
            self.l_f_validate_input = [_f_validate_input_default]
        else:
            if not isinstance(l_f_validate_input, (list,)):
                self.l_f_validate_input = [l_f_validate_input]
            else:
                self.l_f_validate_input = l_f_validate_input

        # TODO - REMOVE THIS - ASSUME NORMALIZED INPUT
        def _get_f_init_params_validated(f_init_params):
            # Adds argument validation to a parameter initialisation function
            def f_init_params_validated(
                    a_x=None, a_y=None, a_date=None, is_mult=False):
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
        f_init_params = _get_f_add_2_f_init_params(
            self.f_init_params, forecast_model.f_init_params)
        f_bounds = _get_f_concat_2_bounds(self, forecast_model)
        l_f_validate_input = list(
            set(self.l_f_validate_input + forecast_model.l_f_validate_input))
        return ForecastModel(
            name,
            n_params,
            f_model,
            f_init_params,
            f_bounds=f_bounds,
            l_f_validate_input=l_f_validate_input)

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
        f_init_params = _get_f_mult_2_f_init_params(
            self.f_init_params, forecast_model.f_init_params)
        f_bounds = _get_f_concat_2_bounds(self, forecast_model)
        l_f_validate_input = list(
            set(self.l_f_validate_input + forecast_model.l_f_validate_input))
        return ForecastModel(
            name,
            n_params,
            f_model,
            f_init_params,
            f_bounds=f_bounds,
            l_f_validate_input=l_f_validate_input)

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

    def validate_input(self, a_x, a_y, a_date):
        try:
            l_result = [f_validate_input(a_x, a_y, a_date)
                        for f_validate_input in self.l_f_validate_input]
        except AssertionError:
            return False
        return True


# - Null model: 0


def _f_model_null(a_x, a_date, params, is_mult=False, **kwargs):
    # This model does nothing - used to disable model components
    # (e.g. seasonality) when adding/multiplying multiple functions
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


model_constant = ForecastModel(
    'constant',
    1,
    _f_model_constant,
    _f_init_params_constant)


# - Naive model: Y = Y(x-1)
# Note: This model requires passing the actuals data - it is not fitted by
# regression. We still pass it to forecast.fit_model() for consistency
# with the rest of the library

def _f_model_naive(a_x, a_date, params, is_mult=False, df_actuals=None):
    if df_actuals is None:
        raise ValueError('model_naive requires a df_actuals argument')
    df_out_tmp = pd.DataFrame({'date': a_date, 'x': a_x})
    df_out = (
        # This is not really intended to work with multiple values per sample
        df_actuals.drop_duplicates('x')
        .merge(df_out_tmp, how='outer')
        .sort_values('x')
    )
    df_out['y'] = (
        df_out.y.shift(1)
        .fillna(method='ffill')
        .fillna(method='bfill')
    )
    df_out = df_out.loc[df_out.x.isin(a_x)]
    # df_out = df_out_tmp.merge(df_out, how='left')
    # TODO: CHECK THAT X,DATE order is preserved
    # TODO: df_out = df_out.merge(df_out_tmp, how='right')
    return df_out.y.values


model_naive = ForecastModel('naive', 0, _f_model_naive)


# - Seasonal naive model
# Note: This model requires passing the actuals data - it is not fitted by
# regression. We still pass it to forecast.fit_model() for consistency
# with the rest of the library

def _fillna_wday(df):
    """
    In a time series, shift samples by 1 week
    and fill gaps with data from same weekday
    """

    def add_col_y_out(df):
        df = df.assign(y_out=df.y.shift(1).fillna(method='ffill'))
        return df

    df_out = (
        df
        .assign(wday=df.date.dt.weekday)
        .groupby('wday', as_index=False).apply(add_col_y_out)
        .sort_values(['x'])
        .reset_index(drop=True)
    )
    return df_out


def _f_model_snaive_wday(a_x, a_date, params, is_mult=False, df_actuals=None):
    """Naive model - takes last valid weekly sample"""
    if df_actuals is None:
        raise ValueError('model_snaive_wday requires a df_actuals argument')

    # df_actuals_model - table with actuals samples,
    #  adding y_out column with naive model values
    df_actuals_model = _fillna_wday(df_actuals.drop_duplicates('x'))

    # df_last_week - table with naive model values from last actuals week,
    #  to use in extrapolation
    df_last_week = (
        df_actuals_model
        # Fill null actual values with data from previous weeks
        .assign(y=df_actuals_model.y.fillna(df_actuals_model.y_out))
        .drop_duplicates('wday', keep='last')
        [['wday', 'y']]
        .rename(columns=dict(y='y_out'))
    )

    # Generate table with extrapolated samples
    df_out_tmp = pd.DataFrame({'date': a_date, 'x': a_x})
    df_out_tmp['wday'] = df_out_tmp.date.dt.weekday
    df_out_extrapolated = (
        df_out_tmp
        .loc[~df_out_tmp.date.isin(df_actuals_model.date)]
        .merge(df_last_week, how='left')
        .sort_values('x')
    )
    # Filter actuals table - only samples in a_x, a_date
    df_out_actuals_filtered = (
        # df_actuals_model.loc[df_actuals_model.x.isin(a_x)]
        # Using merge rather than simple filtering to account for
        #  dates with multiple samples
        df_actuals_model.merge(df_out_tmp, how='inner')
        .sort_values('x')
    )
    df_out = (
        pd.concat(
            [df_out_actuals_filtered, df_out_extrapolated],
            sort=False, ignore_index=True)
    )
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


model_spike = ForecastModel('spike', 3, _f_model_spike, _f_init_params_spike)


# - Spike model for dates - dates are fixed for each model

def _f_model_spike_date(
        a_x,
        a_date,
        params,
        date_start,
        date_end,
        is_mult=False):
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
    model_spike_date = ForecastModel(
        'spike_date[{},{}]'.format(
            pd.to_datetime(date_start).date(),
            pd.to_datetime(date_end).date()),
        1,
        f_model,
        _f_init_params_spike)
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
            a_x_size = np.unique(a_x).size - 1
        else:
            a_x_size = a_y.size - 1
        A = (a_y[-1] - a_y[0]) / a_x_size
        B = a_y[0]
        # Uniform low= 0*m, high = 1*m
        return np.array([A, B])


model_linear = ForecastModel(
    'linear',
    2,
    _f_model_linear,
    _f_init_params_linear)


def f_init_params_linear_nondec(
        a_x=None,
        a_y=None,
        a_date=None,
        is_mult=False):
    params = _f_init_params_linear(a_x, a_y, a_date)
    if params[0] < 0:
        params[0] = 0
    return params


def f_bounds_linear_nondec(a_x=None, a_y=None, a_date=None):
    # first param should be between 0 and inf
    return [0, -np.inf], [np.inf, np.inf]


model_linear_nondec = ForecastModel('linear_nondec', 2, _f_model_linear,
                                    f_init_params=f_init_params_linear_nondec,
                                    f_bounds=f_bounds_linear_nondec)


# - QuasiLinear model: :math:`Y = A t^{B} + C`

def _f_model_quasilinear(a_x, a_date, params, is_mult=False, **kwargs):
    (A, B, C) = params
    y = A * np.power(a_x, B) + C
    return y


model_quasilinear = ForecastModel('quasilinear', 3, _f_model_quasilinear)


# - Exponential model: math::  Y = A * B^t
# TODO: Deprecate - not safe to use
def _f_model_exp(a_x, a_date, params, is_mult=False, **kwargs):
    (A, B) = params
    y = A * np.power(B, a_x)
    return y


model_exp = ForecastModel('exponential', 2, _f_model_exp)


# - Exponential decay model: math::  Y = A * e^(B*(x-C)) + D
def _f_model_decay(a_x, a_date, params, is_mult=False, **kwargs):
    (A, B, D) = params
    y = A * np.exp(B * (a_x)) + D
    return y


def f_init_params_decay(a_x=None, a_y=None, a_date=None, is_mult=False):
    if a_y is None:
        return np.array([0, 0, 0])
    A = a_y[0] - a_y[-1]
    B = np.log(np.min(a_y) / np.max(a_y)) / (len(a_y) - 1)
    if B > 0 or B == -np.inf:
        B = -0.5
    C = a_y[-1]
    return np.array([A, B, C])


def f_bounds_decay(a_x=None, a_y=None, a_date=None):
    return [-np.inf, -np.inf, -np.inf], [np.inf, 0, np.inf]


model_decay = ForecastModel('decay', 3, _f_model_decay,
                            f_init_params=f_init_params_decay,
                            f_bounds=f_bounds_decay)


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
        # if is_mult, replace above line with something like
        # np.concatenate([[np.NaN],a_y[:-1]/a_y[1:]])
        a = df.nlargest(1, 'diff').index[0]
        b = df['diff'].iloc[a]
        return np.array([a, b * 2])


# TODO: Add boundaries for X axis
model_step = ForecastModel('step', 2, _f_step, _f_init_params_step)


# - Spike model for dates - dates are fixed for each model

def _f_model_step_date(a_x, a_date, params, date_start, is_mult=False):
    [A] = params
    mask_step = (a_date >= date_start).astype(float)
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
        # if is_mult, replace above line with something like
        # np.concatenate([[np.NaN],a_y[:-1]/a_y[1:]])
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
    return _f_n_steps(
        n=2,
        a_x=a_x,
        a_date=a_date,
        params=params,
        is_mult=is_mult)


def _f_init_params_n_steps(
        n=2,
        a_x=None,
        a_y=None,
        a_date=None,
        is_mult=False):
    if a_y is None:
        return np.random.uniform(0, 1, n * 2)
    else:
        # max difference between consecutive values
        if a_y.ndim > 1:
            a_y = a_y[:, 0]
        df = pd.DataFrame({'b': a_y})
        df['diff'] = df.diff().abs()
        # if is_mult, replace above line with something like
        # np.concatenate([[np.NaN],a_y[:-1]/a_y[1:]])
        a = df.nlargest(n, 'diff').index[0:n].values
        b = df['diff'].iloc[a].values
        params = []
        for i in range(0, n):
            params += [a[i], b[i]]
        return np.array(params)


def _f_init_params_two_steps(a_x=None, a_y=None, a_date=None, is_mult=False):
    return _f_init_params_n_steps(
        n=2,
        a_x=a_x,
        a_y=a_y,
        a_date=a_date,
        is_mult=is_mult)


model_two_steps = ForecastModel(
    'two_steps',
    2 * 2,
    _f_two_steps,
    _f_init_params_two_steps)


# - Sigmoid step function: `Y = {A + (B - A) / (1 + np.exp(- D * (a_x - C)))}`
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


def _f_init_params_sigmoid_step(
        a_x=None,
        a_y=None,
        a_date=None,
        is_mult=False):
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
model_sigmoid_step = ForecastModel(
    'sigmoid_step',
    3,
    _f_sigmoid,
    _f_init_params_sigmoid_step,
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
            a = np.random.uniform(a_x[nfirst_last], a_x[-nfirst_last - 1], 1)
        else:
            a = np.random.uniform(0, 1, 1)
        b = np.random.uniform(0, 1, 1)

        return np.concatenate([a,
                               b])
    else:
        # TODO: FILTER A_Y BY 20-80 PERCENTILE IN A_X
        df = pd.DataFrame({'b': a_y})
        if a_x is not None:
            #
            df['x'] = a_x
            # Required because we support input with multiple samples per x
            # value
            df = df.drop_duplicates('x')
            df = df.set_index('x')
        # max difference between consecutive values -- this assumes no null
        # values in series
        df['diff2'] = df.diff().diff().abs()

        # We ignore the last 15% of the time series
        skip_samples = int(np.ceil(df.index.size * 0.15))

        a = (df.head(-skip_samples).tail(
            -skip_samples).nlargest(1, 'diff2').index[0]
        )
        b = df['diff2'].loc[a]
        # TODO: replace b with estimation of slope in segment 2
        #   minus slope in segment 1 - see init_params_linear
        return np.array([a, b])


def _f_init_bounds_ramp(a_x=None, a_y=None, a_date=None):
    if a_x is None:
        a_min = -np.inf
        a_max = np.inf
    else:
        # a_min = np.min(a_x)
        nfirst_last = int(np.ceil(0.15 * a_x.size))
        a_min = a_x[nfirst_last]
        a_max = a_x[-nfirst_last]
        # a_min = np.percentile(a_x, 15)
        # a_max = np.percentile(a_x,85)
    if a_y is None:
        b_min = -np.inf
        b_max = np.inf
    else:
        # TODO: FILTER A_Y BY 20-80 PERCENTILE IN A_X
        # df = pd.DataFrame({'b': a_y})
        # #max_diff2 = np.max(df.diff().diff().abs())
        # max_diff2 = np.max(np.abs(np.diff(np.diff(a_y))))
        #
        # b_min = -2*max_diff2
        # b_max = 2*max_diff2

        b_min = -np.inf
        b_max = np.inf
    # logger_info('DEBUG: BOUNDS:',(a_min, b_min,a_max, b_max))
    return ([a_min, b_min], [a_max, b_max])


model_ramp = ForecastModel(
    'ramp',
    2,
    _f_ramp,
    _f_init_params_ramp,
    _f_init_bounds_ramp)


# - Weekday seasonality

def _f_model_season_wday(a_x, a_date, params, is_mult=False, **kwargs):
    # Weekday seasonality model, 6 params
    # params_long[0] is default series value,
    params_long = np.concatenate([[float(is_mult)], params])
    return params_long[a_date.weekday]


def _f_validate_input_season_wday(a_x, a_y, a_date):
    assert a_date is not None
    assert a_date.weekday.drop_duplicates().size == 7


model_season_wday = ForecastModel(
    'season_wday',
    6,
    _f_model_season_wday,
    l_f_validate_input=_f_validate_input_season_wday)


# - Month seasonality
def _f_init_params_season_month(
        a_x=None,
        a_y=None,
        a_date=None,
        is_mult=False):
    if a_y is None or a_date is None:
        return np.random.uniform(low=-1, high=1, size=11)
    else:  # TODO: Improve this
        l_params_long = [np.mean(a_y[a_date.month == i])
                         for i in np.arange(1, 13)]
        l_baseline = l_params_long[-1]
        l_params = l_params_long[:-1]
        if not is_mult:
            l_params_add = l_params - l_baseline
            return l_params_add
        else:
            l_params_mult = l_params / l_baseline
            return l_params_mult


def _f_model_season_month(a_x, a_date, params, is_mult=False, **kwargs):
    # Month of December is taken as default level, has no parameter
    # params_long[0] is default series value
    params_long = np.concatenate([[float(is_mult)], params])
    return params_long[a_date.month - 1]


model_season_month = ForecastModel(
    'season_month',
    11,
    _f_model_season_month,
    _f_init_params_season_month)

model_season_month_old = ForecastModel(
    'season_month_old', 11, _f_model_season_month)


def _f_model_yearly_season_fourier(
        a_x,
        a_date,
        params,
        is_mult=False,
        **kwargs):
    # Infer the time series frequency to calculate the Fourier parameters

    period = dict_fourier['period']
    harmonics = dict_fourier['harmonics']

    return _f_model_season_fourier(a_date, params, period, harmonics, is_mult)


date_origin = datetime(1970, 1, 1)


def _f_model_season_fourier(a_date, params, period, harmonics, is_mult=False):
    # convert to days since epoch
    t = (a_date - date_origin).days.values
    i = np.arange(1, harmonics + 1)
    a_tmp = i.reshape(i.size, 1) * t
    k = (2.0 * np.pi / period)
    y = np.concatenate([np.sin(k * a_tmp), np.cos(k * a_tmp)])

    # now multiply by the params
    y = np.matmul(params, y)
    return y


def _f_init_params_fourier_n_params(
        n_params,
        a_x=None,
        a_y=None,
        a_date=None,
        is_mult=False):
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
    return n_params * [-np.inf], n_params * [np.inf]


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
    # Generate model with some fixed parameters
    if forecast_model.n_params == 0:  # Nothing to do
        return forecast_model
    if len(params_fixed) != forecast_model.n_params:
        err = 'Wrong number of fixed parameters'
        raise ValueError(err)
    return ForecastModel(
        forecast_model.name + '_fixed', 0,
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


# TODO: Add option - estimate_outl_size
# TODO: Add option - sigmoid steps
# TODO: ADD option - gaussian spikes
def get_model_outliers(df, window=3):
    """
    Identify outlier samples in a time series

    :param df: Input time series
    :type df: pandas.DataFrame
    :param window: The x-axis window to aggregate multiple steps/spikes
    :type window: int
    :return:
        | tuple (mask_step, mask_spike)
        | mask_step: True if sample contains a step
        | mask_spike: True if sample contains a spike
    :rtype: tuple of 2 numpy arrays of booleans

    TODO: require minimum number of samples to find an outlier
    """

    dfo = df.copy()  # dfo - df for outliers
    # If df has datetime index, use date logic in steps/spikes
    with_dates = 'date' in df.columns
    x_col = 'date' if with_dates else 'x'

    if df[x_col].duplicated().any():
        raise ValueError('Input cannot have multiple values per sample')

    # Get the differences
    dfo['dif'] = dfo.y.diff()

    # We consider as outliers the values that are
    # 1.5 * IQR (interquartile range) beyond the quartiles.
    # These thresholds are obtained here
    thr_low, thr_hi = get_iqr_thresholds(dfo.dif)
    # Now identify the changes
    dfo['ischange'] = ((dfo.dif < thr_low) | (dfo.dif > thr_hi)).astype(int)

    # Whenever there are two or more consecutive changes
    # (that is, within `window` samples), we group them together
    dfo['ischange_group'] = (
        dfo.ischange.rolling(window, win_type=None, center=True).max().fillna(
            0).astype(int)
    )

    # We now have to calculate the difference within the
    # same group in order to identify if the consecutive changes
    # result in a step, a spike, or both.

    # We get the filtered difference
    dfo['dif_filt'] = (dfo.dif * dfo.ischange).fillna(0)
    # And the absolute value of that
    dfo['dif_filt_abs'] = dfo.dif_filt.abs()

    dfo['change_group'] = dfo.ischange_group.diff(
    ).abs().fillna(0).astype(int).cumsum()

    # this gets us the average difference of the outliers within each change
    # group
    df_mean_gdiff = (
        dfo.loc[dfo.ischange.astype(bool)].groupby('change_group')[
            'dif_filt'].mean().rename('mean_group_diff').reset_index())
    # this gets us the average absolute difference of the outliers within each
    # change group
    df_mean_gdiff_abs = (
        dfo.loc[dfo.ischange.astype(bool)].groupby('change_group')[
            'dif_filt_abs'].mean().rename(
            'mean_group_diff_abs').reset_index()
    )

    # Merge the differences with the original dfo
    dfo = dfo.merge(
        df_mean_gdiff,
        how='left').merge(
        df_mean_gdiff_abs,
        how='left')
    # Fill missing values with zero -> no change
    dfo.mean_group_diff = dfo.mean_group_diff.fillna(0)
    dfo.mean_group_diff_abs = dfo.mean_group_diff_abs.fillna(0)

    # the change group is a step if the mean_group_diff exceeds the thresholds
    dfo['is_step'] = dfo['ischange_group'] & (
        ((dfo.mean_group_diff < thr_low) | (dfo.mean_group_diff > thr_hi)))

    # the change group is a spike if the difference between the
    # mean_group_diff_abs and the average mean_group_diff exceeds
    # the average threshold value
    dfo['is_spike'] = (dfo.mean_group_diff_abs -
                       dfo.mean_group_diff.abs()) > (thr_hi - thr_low) / 2

    # Get the outlier start and end points for each group
    df_outl = (
        dfo.loc[dfo.ischange.astype(bool)].groupby('change_group').apply(
            lambda x: pd.Series(
                {'outl_start': x[x_col].iloc[0],
                 'outl_end': x[x_col].iloc[-1]})).reset_index()
    )

    if df_outl.empty:  # No outliers - nothing to do
        return np.full(dfo.index.size, False), np.full(dfo.index.size, False)

    dfo = dfo.merge(df_outl, how='left')
    # Get the start and end points in dfo
    if with_dates:
        # Convert to datetime, if we are using dates
        dfo['outl_start'] = pd.to_datetime(dfo.outl_start)
        dfo['outl_end'] = pd.to_datetime(dfo.outl_end)
        # Create the mask for spikes and steps
        dfo['mask_spike'] = (dfo['is_spike'] &
                             (dfo.date >= pd.to_datetime(dfo.outl_start)) &
                             (dfo.date < pd.to_datetime(dfo.outl_end)))
        dfo['mask_step'] = (dfo['is_step'] &
                            (dfo.date >= pd.to_datetime(dfo.outl_start)) &
                            (dfo.date <= pd.to_datetime(dfo.outl_end)))
    else:
        # For non-date x values, we fill na's and convert to int
        dfo['outl_start'] = dfo.outl_start.fillna(0).astype(int)
        dfo['outl_end'] = dfo.outl_end.fillna(0).astype(int)
        # Create the mask for spikes and steps
        dfo['mask_spike'] = (dfo['is_spike'] &
                             (dfo.x >= dfo.outl_start) &
                             (dfo.x < dfo.outl_end))
        dfo['mask_step'] = (dfo['is_step'] &
                            (dfo.x >= dfo.outl_start) &
                            (dfo.x <= dfo.outl_end))

    return dfo.mask_step.values, dfo.mask_spike.values


def create_fixed_step(diff, x):
    # Generate a fixed step model
    fixed_params = [x, diff]
    return get_fixed_model(model_step, fixed_params)


def create_fixed_spike(diff, x, duration):
    # Generate a fixed spike model
    fixed_params = [diff, x, x + duration]
    return get_fixed_model(model_spike, fixed_params)


def create_fixed_spike_ignored(x, duration):
    # Generate a fixed spike ignored model
    fixed_params = [0, x, x + duration]
    return get_fixed_model(model_spike, fixed_params, is_mult=True)


# Dummy variable models

def get_model_dummy(name, dummy, **kwargs):
    """
    Generate a model based on a dummy variable.

    :param name: Name of the model
    :type name: basestring
    :param dummy:
      | Can be a function or a list-like.
      | If a function, it must be of the form f_dummy(a_x, a_date),
      | and return a numpy array of floats
      | with the same length as a_x and values that are either 0 or 1.
      | If a list-like of numerics, it will be converted to a f_dummy function
      | as described above, which will have values of 1 when a_x has one of
      | the values in the list, and 0 otherwise. If a list-like of date-likes,
      | it will be converted to a f_dummy function as described above, which
      | will have values of 1 when a_date has one of the values in the list,
      | and 0 otherwise.
    :type dummy: function, or list-like of numerics or datetime-likes
    :param kwargs:
    :type kwargs:
    :return:
      | A model that returns A when dummy is 1, and 0 (or 1 if is_mult==True)
      | otherwise.
    :rtype: ForecastModel


    """
    return ForecastModel(name, 1, get_f_model_dummy(dummy), **kwargs)


def _validate_f_dummy(f_dummy):
    # Ensures that behaviour of f_dummy matches specs
    # Must return array of floats, same length as a_x, with values either 0.
    # or 1.
    def validate_for_dummy(a_dummy):
        assert isinstance(a_dummy, np.ndarray)
        assert np.setdiff1d(a_dummy, np.array([0., 1.])).size == 0

    # validate_for_dummy(f_dummy(np.arange(0, 10), None)) # Crashes with
    # f_dummy 's that require dates
    validate_for_dummy(
        f_dummy(
            np.arange(
                0, 10), pd.date_range(
                '2018-01-01', '2018-01-10')))


def get_f_model_dummy(dummy):
    """
    Generate a model function for a dummy variable defined by f_dummy

    :param dummy: dummy variable
    :type dummy: function or list-like of numerics or dates
    :return: model function based on dummy variable, to use on a ForecastModel
    :rtype: function
    """

    if callable(dummy):  # If dummy is a function, use it
        f_dummy = dummy
    elif isinstance(dummy, Holiday):
        f_dummy = get_f_dummy_from_holiday(dummy)
    elif isinstance(dummy, AbstractHolidayCalendar):
        f_dummy = get_f_dummy_from_calendar(dummy)
    else:
        # If dummy is a list, convert to function
        f_dummy = get_f_dummy_from_list(dummy)

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
            a_result = (A) * mask + 1
        return a_result

    return f_model_check


def get_f_dummy_from_list(list_check):
    """
    Generate a f_dummy function that defines a dummy variable, can be used
    for dummy models

    :param list_check: Input list
    :type list_check: list-like of numerics or datetime-likes
    :return: f_dummy
    :rtype: function
    """
    # Generate a f_dummy function that defines a dummy variable, can be used
    # for dummy models
    s_check = pd.Series(list_check)
    assert s_check.size, 'Input list cannot be empty'
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
        except BaseException:
            raise ValueError(
                'list_dummy must be a list-like with numeric or'
                'date-like values: %s', list_check)


def get_f_dummy_from_calendar(calendar):
    # Generate dummy model function from a pandas HolidayCalendar

    def f_dummy_calendar(a_x, a_date, **kwargs):
        # TODO: If we can pass dict_cal as an argument,
        #       use pre-loaded list of dates for performance

        # TODO: If we can guarantee sorted dates,
        #       change this to a_date[0], a_date[-1] for performance
        list_check_date = calendar.holidays(a_date.min(), a_date.max())
        return np.isin(a_date, list_check_date).astype(float)

    return f_dummy_calendar


def get_f_dummy_from_holiday(holiday):
    def f_dummy_holiday(a_x, a_date, **kwargs):
        # TODO: If we can pass dict_cal as an argument,
        #       use pre-loaded list of dates for performance
        # if dict_cal in kwargs.keys():
        #    list_check_date = dict_cal.get(holiday.name)
        # else:

        # TODO: If we can guarantee sorted dates,
        #       change this to a_date[0], a_date[-1] for performance
        list_check_date = holiday.dates(a_date.min(), a_date.max())
        return np.isin(a_date, list_check_date).astype(float)

    return f_dummy_holiday


model_season_wday_2 = get_model_dummy(
    'season_wday_2', lambda a_x, a_date, **kwargs:
    (a_date.weekday < 5).astype(float))

# Example dummy model - checks if it is Christmas
model_dummy_christmas = get_model_dummy(
    'dummy_christmas', lambda a_x, a_date, **kwargs:
    ((a_date.month == 12) & (a_date.day == 25)).astype(float))

# Example dummy model - checks if it is first day of month
model_dummy_month_start = get_model_dummy(
    'dummy_month_start', lambda a_x, a_date, **kwargs:
    (a_date.day == 1).astype(float))


class CalendarBankHolUK(AbstractHolidayCalendar):
    rules = [
        GoodFriday,
        EasterMonday,
        # Early May Bank Holiday - first Monday in May
        Holiday('Early May Bank Holiday', month=5, day=1,
                offset=DateOffset(weekday=MO(1))
                ),
        # Spring Bank Holiday - Last Monday in May
        Holiday('Spring Bank Holiday', month=5, day=31,
                offset=DateOffset(weekday=MO(-1))
                ),
        # August Bank holiday - Last Monday in August
        Holiday('August Bank Holiday', month=8, day=30,
                offset=DateOffset(weekday=MO(-1))
                )
    ]


class CalendarChristmasUK(AbstractHolidayCalendar):
    rules = [
        Holiday('New Year\'s Day', month=1, day=1, observance=next_monday),
        Holiday('Christmas', month=12, day=25, observance=next_monday),
        Holiday('Boxing Day', month=12, day=26,
                observance=next_monday_or_tuesday),
    ]


# Bank Holidays for Italy
class CalendarBankHolIta(AbstractHolidayCalendar):
    rules = [
        EasterMonday,
        Holiday('Festa della Liberazione', month=4, day=25),
        Holiday('Festa del lavoro', month=5, day=1),
        Holiday('Festa della Repubblica', month=6, day=2),
        Holiday('Ferragosto', month=8, day=15),
        Holiday('Tutti i Santi', month=11, day=1),
        Holiday('Immacolata Concezione', month=12, day=8),
    ]


class CalendarChristmasIta(AbstractHolidayCalendar):
    rules = [
        Holiday('New Year\'s Day', month=1, day=1, observance=next_monday),
        Holiday('Christmas', month=12, day=25, observance=next_monday),
        Holiday('Santo Stefano', month=12, day=26,
                observance=next_monday_or_tuesday),
        Holiday('Epiphany', month=1, day=6, observance=next_monday),
    ]


def get_model_from_calendars(l_calendar, name=None):
    """
    Create a ForecastModel based on a list of pandas Calendars.

    :param calendar:
    :type calendar: pandas.tseries.AbstractHolidayCalendar
    :return: model based on the input calendar
    :rtype: ForecastModel

    In pandas, Holidays and calendars provide a simple way to define
    holiday rules, to be used in any analysis that requires a predefined
    set of holidays. This function converts a Calendar object into a
    ForecastModel that assigns a parameter to each calendar rule.

    As an example, a Calendar with 1 rule defining Christmas dates
    generates a model with a single parameter, which
    determines the amount added/multiplied to samples falling on Christmas.
    A calendar with 2 rules for Christmas and New Year will have two parameters
    - the first one applying to samples in Christmas, and the second
    one applying to samples in New Year.

    Usage::

        from pandas.tseries.holiday import USFederalHolidayCalendar
        model_calendar = get_model_from_calendar(USFederalHolidayCalendar())

    """

    if isinstance(l_calendar, AbstractHolidayCalendar):
        l_calendar = [l_calendar]

    # Filter out calendars without rules
    l_calendar = [calendar for calendar in l_calendar if calendar.rules]

    assert len(l_calendar), 'Need 1+ valid calendars'

    if name is None:
        name = l_calendar[0].name

    l_model_dummy = [get_model_dummy(calendar.name, calendar)
                     for calendar in l_calendar]
    f_model_prod = np.prod(l_model_dummy)
    f_model_sum = np.sum(l_model_dummy)

    def _f_init_params_calendar(
            a_x=None, a_y=None, a_date=None, is_mult=False):
        if is_mult:
            return np.ones(len(l_model_dummy))
        else:
            return np.zeros(len(l_model_dummy))

    def _f_model_calendar(a_x, a_date, params, is_mult=False, **kwargs):
        f_all_dummies = f_model_prod if is_mult else f_model_sum
        return f_all_dummies(a_x, a_date, params, is_mult, **kwargs)

    model_calendar = ForecastModel(
        name,
        len(l_model_dummy),
        _f_model_calendar,
        _f_init_params_calendar
    )
    return model_calendar


model_calendar_uk = get_model_from_calendars(
    [CalendarChristmasUK(), CalendarBankHolUK()], 'calendar_uk')
model_calendar_us = get_model_from_calendars(USFederalHolidayCalendar(),
                                             'calendar_us')
# Calendar for Italy
model_calendar_ita = get_model_from_calendars(
    [CalendarChristmasIta(), CalendarBankHolIta()], 'calendar_uk')


def get_model_from_datelist(name=None, *args):
    """
    Create a ForecastModel based on one or more lists of dates.

    :param name: Model name
    :type name: str
    :param args: Each element in args is a list of dates.
    :type args:
    :return: model based on the input lists of dates
    :rtype: ForecastModel

    Usage::

        model_datelist1=get_model_from_date_list('datelist1',
                                                 [date1, date2, date3])
        model_datelists23 = get_model_from_date_list('datelists23',
                                                [date1, date2], [date3, date4])

    In the example above, model_datelist1 will have one parameter, which
    determines the amount added/multiplied to samples with dates matching
    either date1, date2 or date3. model_datelists23 will have two parameters
    - the first one applying to samples in date1 and date2, and the second
    one applying to samples in date 3 and date4

    """
    l_model_dummy = [get_model_dummy('model_dummy', pd.to_datetime(l_date))
                     for l_date in args]
    assert (len(l_model_dummy)), 'Need 1+ lists of dates'
    f_model_prod = np.prod(l_model_dummy)
    f_model_sum = np.sum(l_model_dummy)

    def _f_init_params_date_list(
            a_x=None, a_y=None, a_date=None, is_mult=False):
        if is_mult:
            return np.ones(len(l_model_dummy))
        else:
            return np.zeros(len(l_model_dummy))

    def _f_model_date_list(a_x, a_date, params, is_mult=False, **kwargs):
        f_all_dummies = f_model_prod if is_mult else f_model_sum
        return f_all_dummies(a_x, a_date, params, is_mult, **kwargs)

    model_date_list = ForecastModel(
        name,
        len(l_model_dummy),
        _f_model_date_list,
        _f_init_params_date_list
    )
    return model_date_list


# Utility functions

def fix_params_fmodel(forecast_model, l_params_fixed):
    """
    Given a forecast model and a list of floats, modify the model so that some
    of its parameters become fixed

    :param forecast_model: Input model
    :type forecast_model: ForecastModel
    :param l_params_fixed: List of floats with same length as number of
        parameters in model. For each element, a non-null value means
        that the parameter in that position is fixed to that value.
        A null value means that the parameter in that position is not fixed.
    :type l_params_fixed: list
    :return: A forecast model with a number of parameters equal to the number
        of null values in l_params_fixed, with f_model modified so that some
        of its parameters gain fixed values equal to the non-null values
        in l_params
    :rtype: ForecastModel
    """
    assert len(l_params_fixed) == forecast_model.n_params

    l_params_fixed = np.array(l_params_fixed)

    a_null = np.isnan(l_params_fixed)
    i_null = np.nonzero(a_null)

    name = '{}_fixed_{}'.format(
        forecast_model.name,
        str(l_params_fixed).replace(
            'nan',
            ':'))
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

    model_result = ForecastModel(
        name,
        n_params,
        f_model_fixed,
        f_init_params_fixed,
        f_bounds_fixed)
    return model_result


def simplify_model(f_model, a_x=None, a_y=None, a_date=None):
    """
    Check a model's bounds, and update model to make parameters fixed if their
    min and max bounds are equal

    :param f_model: Input model
    :type f_model: ForecastModel
    :param a_x: X axis for model function.
    :type a_x: numpy array of floats
    :param a_y: Input time series values, to compare to the model function
    :type a_y: numpy array of floats
    :param a_date: Dates for the input time series
    :type a_date: numpy array of datetimes
    :return: Model with simplified parameters based on bounds
    :rtype: ForecastModel
    """
    bounds_min, bounds_max = f_model.f_bounds(a_x, a_y, a_date)
    bounds_diff = np.array(bounds_max) - np.array(bounds_min)
    i_diff_zero = np.nonzero(bounds_diff == 0)
    # For any parameter, if bounds_min == bounds_max, that parameter becomes
    # fixed

    if i_diff_zero[0].size == 0:
        return f_model
    else:  # We make parameters fixed if their min and max bounds are equal
        params_fixed = np.full(f_model.n_params, np.NaN)
        params_fixed[i_diff_zero, ] = bounds_max[i_diff_zero, ]
        f_model = fix_params_fmodel(f_model, params_fixed)
        logger.info(
            'Some min and max bounds are equal - generating fixed model: %s',
            f_model.name)
        return f_model


def validate_initial_guess(initial_guess, bounds):
    # Check that initial parameter values fall within model bounds
    initial_guess = np.array(initial_guess)
    bounds_min, bounds_max = bounds
    return np.all(
        (initial_guess >= bounds_min) & (
            initial_guess <= bounds_max))


def get_l_model_auto_season(a_date, min_periods=1.5, season_add_mult='add',
                            l_season_yearly=None, l_season_weekly=None):
    """
    Generates a list of candidate seasonality models for an series of
    timestamps

    :param a_date: date array of a time series
    :type a_date: numpy array of timestamps
    :param min_periods: Minimum number of periods required to apply
        seasonality
    :type min_periods: float
    :param season_add_mult: 'add' or 'mult'
    :type is_mult: basestring
    :return: list of candidate seasonality models
    :rtype: list of ForecastModel
    """
    s_date = pd.Series(a_date).sort_values().drop_duplicates()
    min_date_delta = s_date.diff().min()
    max_date_delta = s_date.max() - s_date.min()

    if pd.isna(min_date_delta) or pd.isna(max_date_delta):
        return [model_null]

    use_season_yearly = (
        # Need more than a full year
        (max_date_delta > pd.Timedelta(min_periods * 365, unit='d')) &
        # Need at least quarterly samples
        (min_date_delta <= pd.Timedelta(92, unit='d'))
    )

    use_season_weekly = (
        # Need more than a full week
        (max_date_delta > pd.Timedelta(min_periods * 7, unit='d')) &
        # Need at least daily samples
        (min_date_delta <= pd.Timedelta(1, unit='d'))
    )

    l_season_yearly_default = [
        # model_season_month,
        model_season_fourier_yearly,
        model_null]
    l_season_weekly_default = [
        model_season_wday,
        model_null]

    if use_season_weekly:
        if l_season_weekly is None:
            l_season_weekly = l_season_weekly_default
        elif not len(l_season_weekly):  # Empty list
            l_season_weekly = [model_null]
            # Otherwise, use input l_season_weekly
    else:
        l_season_weekly = [model_null]

    if use_season_yearly:
        if l_season_yearly is None:
            l_season_yearly = l_season_yearly_default
        elif not len(l_season_yearly):  # Empty list
            l_season_yearly = [model_null]
        # Otherwise, use input l_season_yearly
    else:
        l_season_yearly = [model_null]

    l_result = [model_null]
    for s_w, s_y in itertools.product(l_season_weekly, l_season_yearly):

        model_season_add = s_w + s_y
        model_season_mult = s_w * s_y

        if season_add_mult in ['add'] and model_season_add != model_null:
            l_result += [model_season_add]
        if season_add_mult in ['mult'] and \
                model_season_mult != model_null and \
                model_season_mult not in l_result:
            l_result += [model_season_mult]
    # Sort values to make results more predictable, testable
    l_result.sort()
    return l_result

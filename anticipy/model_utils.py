# -*- coding: utf-8 -*-
#
# License:          This module is released under the terms of the LICENSE file 
#                   contained within this applications INSTALL directory

"""
Utility functions for model generation
"""

# -- Coding Conventions
#    http://www.python.org/dev/peps/pep-0008/   -   Use the Python style guide
#    http://sphinx.pocoo.org/rest.html          -   Use Restructured Text for docstrings

# -- Public Imports
import logging
import math
import numpy as np
import pandas as pd

# -- Private Imports

# -- Globals


logger = logging.getLogger(__name__)

dict_wday_name = {
    0: 'W-MON',
    1: 'W-TUE',
    2: 'W-WED',
    3: 'W-THU',
    4: 'W-FRI',
    5: 'W-SAT',
    6: 'W-SUN',
}


# -- Exception classes

# -- Functions
def logger_info(msg, data):
    # Convenience function for easier log typing
    logger.info(msg + '\n%s', data)


def array_transpose(a):
    """
    Transpose a 1-D numpy array

    :param a: An array with shape (n,)
    :type a: numpy.Array
    :return: The original array, with shape (n,1)
    :rtype: numpy.Array
    """
    return a[np.newaxis, :].T


# TODO: rework to support model composition
def model_requires_scaling(model):
    """
    Given a :py:class:`nsa.forecast.forecast_models.ForecastModel`  return True if the function requires
    scaling a_x

    :param model: A get_model_<modeltype> function from :py:mod:`nsa.forecast.model.periodic_models` or
        :py:mod:`nsa.forecast.model.aperiodic_models`
    :type model: function
    :return: True if function is logistic or sigmoidal
    :rtype: bool
    """
    requires_scaling = model is not None and model.name in [
        'logistic',
        'sigmoid'
    ]
    return requires_scaling


def apply_a_x_scaling(a_x, model=None, scaling_factor=100.0):
    """
    Modify a_x for forecast_models that require it

    :param a_x: x axis of time series
    :type a_x: numpy array
    :param model: a :py:class:`nsa.forecast.forecast_models.ForecastModel`
    :type model: function or None
    :param scaling_factor: Value used for scaling t_values for logistic models
    :type scaling_factor: float
    :return: a_x with scaling applied, if required
    :rtype: numpy array
    """
    if model_requires_scaling(model):  # todo: check that this is still useful
        a_x = a_x / scaling_factor
    return a_x


dict_freq_units_per_year = {'A': 1.0, 'Y': 1.0, 'D': 365.0, 'W': 52.0, 'M': 12, 'Q': 4, 'H': 24 * 365.0}


def get_s_x_extrapolate(date_start_actuals, date_end_actuals, model=None, freq='W',  extrapolate_years=2.5,
                        shifted_origin=0, scaling_factor=100.0, x_start_actuals=0.):
    """
    Return t_values series with DateTimeIndex, covering the date range for the actuals, plus a forecast period.


    :param date_start_actuals: date or numeric index for first actuals sample
    :type date_start_actuals: str, datetime, int or float
    :param date_end_actuals: date or numeric index for last actuals sample
    :type date_end_actuals: str, datetime, int or float
    :param extrapolate_years:
    :type extrapolate_years: float
    :param model:
    :type model: function
    :param freq: Time unit between samples. Supported units are 'W' for weekly samples, or 'D' for daily samples.
        (untested) Any date unit or time unit accepted by numpy should also work, see
        https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.datetime.html#arrays-dtypes-dateunits
    :type freq: str or int
    :param shifted_origin: Offset to apply to a_x
    :type shifted_origin: int
    :param scaling_factor: Value used for scaling a_x for certain model functions
    :type scaling_factor: float
    :return: Series of floats with DateTimeIndex. To be used as (a_date, a_x) input for a model function.
    :rtype: pandas.Series

    The returned series covers the actuals time domain plus a forecast period lasting extrapolate_years, in years.
    The number of additional samples for the forecast period is time_resolution * extrapolate_years, rounded down
    """
    if isinstance(date_start_actuals, str) or isinstance(date_start_actuals, pd.datetime):  # Use dates if available
        date_start_actuals = pd.to_datetime(date_start_actuals)
        date_end_actuals = pd.to_datetime(date_end_actuals)

        if freq is None: # Default frequency
            freq='W'

        freq_short = freq[0:1]      # Changes e.g. W-MON to W
        # freq_units_per_year = 52.0 if freq_short=='W' else 365.0   # Todo: change to dict to support more frequencies
        freq_units_per_year = dict_freq_units_per_year.get(freq_short, 365.0)
        extrapolate_units = extrapolate_years*freq_units_per_year
        date_end_forecast = date_end_actuals+pd.to_timedelta(extrapolate_units, unit=freq_short)

        index = pd.date_range(date_start_actuals, date_end_forecast, freq=freq, name='date')
    else:  # Otherwise, use numeric index - we extrapolate future samples equal to 100*extrapolate_years
        index = pd.Index(np.arange(date_start_actuals, date_end_actuals+100*extrapolate_years))

    s_x = pd.Series(index=index, data=np.arange(x_start_actuals, x_start_actuals+index.size))+shifted_origin
    if model_requires_scaling(model):
        s_x = s_x / scaling_factor

    return s_x


# Forecast Selection Functions

def get_aic_c(fit_error, n, n_params):
    """
    This function implements the corrected Akaike Information Criterion (AICc), taking as input
    a given fit error and data/model degrees of freedom. We assume that the residuals of the candidate model
    are distributed according to independent identical normal distributions with zero mean. Hence, we can use
    define the AICc as

    .. math::

        AICc = AIC + \\frac{2k(k+1)}{n-k-1} = 2k + n \\log\\left(\\frac{E}{n}\\right) + \\frac{2k(k+1)}{n-k-1},

    where :math:`k` and :math:`n` denotes the model and data degrees of freedom respectively, and :math:`E`
    denotes the residual error of the fit.

    :param fit_error: Residual error of the fit
    :type fit_error: float
    :param n: Data degrees of freedom
    :type n: int
    :param n_params: Model degrees of freedom
    :type n_params: int
    :return: Corrected Akaike Information Criterion (AICc)
    :rtype: float

    Note:

    - see AIC in `Wikipedia article on the AIC <https://en.wikipedia.org/wiki/Akaike_information_criterion>`_.

    """
    # First, deal with corner cases that can blow things up with division by zero
    if (n <= n_params + 1) or (n == 0):
        aux = n - n_params - 1
        raise ValueError(
            'ERROR: Time series too short for AIC_C: (n = ' + str(n) + ', n - n_params - 1 = ' + str(aux) + ')')
    elif fit_error == 0.0:
        if n_params == 1:
            aicc = -float("inf")
        else:
            # This can lead to suboptimal model selection when we have multiple perfect fits - we use a patch instead
            # aicc = -float("inf")
            fit_error = 10 ** -320
            aicc = n * math.log(fit_error / n) + 2 * n_params + (2 * n_params * (n_params + 1) / (n - n_params - 1))

    else:
        # Actual calculation of the AICc
        aicc = n * math.log(fit_error / n) + 2 * n_params + (2 * n_params * (n_params + 1) / (n - n_params - 1))

    # logger.info('DEBUG: getting aicc, fit_error: %s, n: %s, n_params: %s, aicc: %s', fit_error, n, n_params, aicc)
    return aicc


def get_s_aic_c_best_result_key(s_aic_c):
    # Required because aic_c can be -inf, that value is not compatible with pd.Series.argmin()
    if s_aic_c.empty or s_aic_c.isnull().all():
        return None
    if (s_aic_c.values == -np.inf).any():
        (key_best_result,) = (s_aic_c == -np.inf).nonzero()
        key_best_result = s_aic_c.index[key_best_result.min()]
    else:
        key_best_result = s_aic_c.argmin()
    return key_best_result


def detect_freq(a_date):
    if isinstance(a_date, pd.DataFrame):
        if 'date' not in a_date.columns:
            return None
        else:
            a_date = a_date.date
    s_date = pd.Series(a_date).sort_values().drop_duplicates()
    min_date_delta = s_date.diff().min()
    if pd.isnull(min_date_delta):
        return None
    elif min_date_delta == pd.Timedelta(1, unit='h'):
        return 'H'
    elif min_date_delta == pd.Timedelta(7, unit='D'):
        # Weekly seasonality - need to determine day of week
        min_date_wday = s_date.min().weekday()
        return dict_wday_name.get(min_date_wday, 'W')
    elif min_date_delta >= pd.Timedelta(28, unit='d') and \
            min_date_delta <= pd.Timedelta(31, unit='d'):
        # MS is month start, M is month end. We use MS if all dates match first of month
        if s_date.dt.day.max() == 1:
            return 'MS'
        else:
            return 'M'
    elif min_date_delta >= pd.Timedelta(89, unit='d') and \
            min_date_delta <= pd.Timedelta(92, unit='d'):
        return 'Q'
    elif min_date_delta >= pd.Timedelta(365, unit='d') and \
            min_date_delta <= pd.Timedelta(366, unit='d'):
        # YS is month start, Y is month end. We use MS if all dates match first of month
        if s_date.dt.day.max() == 1 and s_date.dt.month.max() == 1:
            return 'YS'
        else:
            return 'Y'
    elif min_date_delta >= pd.Timedelta(23, unit='h'):
            #and min_date_delta <= pd.Timedelta(1, unit='d')\
        return 'D'
    else:
        return None


def interpolate_df(df, include_mask=False):
    # In a dataframe with date gaps, replace gaps with interpolation
    if not 'date' in df.columns:    # interpolate by x column
        if df.x.diff().nunique <=1:
            return df
        else:
            df_result = (
                df.set_index('x')
                    .reindex(pd.RangeIndex(df.x.min(), df.x.max()+1, name='x'))
                    .interpolate()
                    .reset_index()
            )

    else:   # df has date column - interpolate by date
        s_date_diff = df.date.diff()
        if s_date_diff.pipe(pd.isnull).all():
            s_date_diff_first = None
        else:
            s_date_diff_first = s_date_diff.loc[s_date_diff.first_valid_index()]
        freq = detect_freq(df)
        # If space between samples is constant, no interpolation is required
        # Exception: in sparse series with date gaps, we can randomly get gaps that are constant but
        # don't match any real period, e.g. 8 days

        if s_date_diff.nunique() <=1 and not (freq == 'D' and s_date_diff_first>pd.to_timedelta(1, 'day')):
            # TODO: Add additional check for e.g. 2-sample series with 8-day gap
            return df
        df_result = (
            df.set_index('date')
                .asfreq(freq)
                .interpolate()
                .reset_index()
        )
    if 'x' in df.columns:
        df_result['x'] = df_result['x'].astype(df.x.dtype)
        if include_mask:
            df_result['is_gap_filled'] = ~df_result.x.isin(df.x)
    return df_result

# -*- coding: utf-8 -*-
#
# License:          This module is released under the terms of the LICENSE file 
#                   contained within this applications INSTALL directory

"""
Functions to plot forecast outputs
"""

# -- Coding Conventions
#    http://www.python.org/dev/peps/pep-0008/   -   Use the Python style guide
#    http://sphinx.pocoo.org/rest.html          -   Use Restructured Text for docstrings

# -- Public Imports
from tempfile import NamedTemporaryFile
import os
import matplotlib.pyplot as plt
import logging
import numpy as np

# -- Globals
logger = logging.getLogger(__name__)


# -- Functions

# ----- Utility functions
def logger_info(msg, data):
    # Convenience function for easier log typing
    logger.info(msg + '\n%s', data)


def df_string_to_unicode(df):
    # In a dataframe, convert any string columns to unicode strings
    df = df.copy()
    columns_str = df.dtypes == basestring
    if not columns_str.any():
        return df
    for col in df.columns[columns_str]:
        df[col] = df[col].astype('unicode')
    return df


def to_feather(df, file_path):
    # Save dataframe as feather file. Formats strings on unicode, for compatibility with R. Drops index.
    df.reset_index(drop=True).pipe(df_string_to_unicode).to_feather(file_path)


def pix_to_in(width_px=None, height_px=None, dpi=300):
    # Utility function to use pixel dimensions rather than ggplot's physical dims
    dpi = float(dpi)

    width_in = width_px / dpi if width_px is not None else np.NaN
    height_in = height_px / dpi if height_px is not None else np.NaN
    # print width_in, height_in
    return width_in, height_in


def has_pi (df_fcast):
    # Check if a forecast table includes prediction intervals
    return 'q5' in df_fcast.columns

# ---- Plotting functions


def _plot_forecast_create(df_fcast, width=None, height=None, title=None, dpi=70):
    """
    Creates matplotlib plot from forecast dataframe

    :param df_fcast:
        | Forecast Dataframe with the following columns:
        | - date (timestamp)
        | - model (str) : ID for the forecast model
        | - y (float) : Value of the time series in that sample
        | - is_actuals (bool) : True for actuals samples, False for forecasted samples
    :type df_fcast: pandas.DataFrame
    :param title: Plot title
    :type title: str
    :param width: plot width, in pixels
    :type width: int
    :param height: plot height, in pixels
    :type height: int
    :param dpi: plot dpi
    :type dpi: int
    :return: The plot
    :rtype: matplotlib plot instance
    """
    # Default palette from ggplot
    act_col = '#00BFC4'
    for_col = '#F8766D'
    plt.style.use('ggplot')
    figsize = (width / dpi, height / dpi)

    # Clean actuals - weights do not get plotted
    df_fcast = df_fcast.loc[df_fcast.model != 'weight']

    # create the DatetimeIndex
    df_fcast = df_fcast.set_index('date')

    if 'source' in df_fcast.columns:
        just_one = False
        sources = df_fcast.loc[df_fcast['is_actuals'], 'source'].unique()
        num_plots = len(sources)
        nrows = int(np.ceil(np.sqrt(num_plots)))
        ncols = int(np.ceil(1. * num_plots / nrows))
    else:
        # Only one set of actuals and forecast needed
        just_one = True
        sources = ['y']
        nrows = 1
        ncols = 1

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, dpi=dpi, squeeze=False)
    fig.canvas.set_window_title(title)

    x = 0
    y = 0
    for src in sources:
        ax = axes[x, y]

        # Filter the specific source is subplots
        if just_one:
            source_filt = True
        else:
            source_filt = df_fcast['source'] == src

        actuals, = ax.plot(df_fcast.loc[source_filt & df_fcast['is_actuals'], :].index,
                           df_fcast.loc[source_filt & df_fcast['is_actuals'], 'y'],
                           color=act_col, marker='o', linestyle='None', label='Actuals')
        forecast, = ax.plot(df_fcast.loc[source_filt & ~df_fcast['is_actuals'], :].index,
                            df_fcast.loc[source_filt & ~df_fcast['is_actuals'], 'y'],
                            color=for_col, marker='None', linestyle='solid', label='Forecast')

        # Fill area between 5th and 95th prediction interval
        if ('q5' in df_fcast.columns) and ('q95' in df_fcast.columns):
            where_to_fill = (source_filt &
                             (~df_fcast['is_actuals']) &
                             (~df_fcast['q5'].isnull()) &
                             (~df_fcast['q95'].isnull()))
            ax.fill_between(df_fcast.index, df_fcast['q5'], df_fcast['q95'],
                            where=where_to_fill,
                            facecolor=for_col, alpha=0.2)

        if ('q20' in df_fcast.columns) and ('q80' in df_fcast.columns):
            # Fill area between 20th and 80th prediction interval
            where_to_fill_2 = (source_filt &
                               (~df_fcast['is_actuals']) &
                               (~df_fcast['q20'].isnull()) &
                               (~df_fcast['q80'].isnull()))
            ax.fill_between(df_fcast.index, df_fcast['q20'], df_fcast['q80'],
                            where=where_to_fill_2,
                            facecolor=for_col, alpha=0.2)

        if not just_one:
            # Set the title of each subplot as per source name
            ax.set_title(src)

        ax.legend(handles=[actuals, forecast],
                  labels=['Actuals', 'Forecast'], loc='upper left')

        y += 1
        if y >= ncols:
            # New row
            y = 0
            x += 1

    # Now make the rest of the graphs invisible
    while x < nrows:
        while y < ncols:
            axes[x, y].set_visible(False)
            y += 1
        # New row
        y = 0
        x += 1

    return plt.Figure


def plot_forecast_save(df_fcast, file_path, width=None, height=None, title=None, dpi=70):
    """
    Generates matplotlib plot and saves as file

    :param df_fcast:
        | Forecast Dataframe with the following columns:
        | - date (timestamp)
        | - model (str) : ID for the forecast model
        | - y (float) : Value of the time series in that sample
        | - is_actuals (bool) : True for actuals samples, False for forecasted samples
    :type df_fcast: pandas.DataFrame
    :param file_path: File path for output
    :type file_path: basestring
    :param width: Image width, in pixels
    :type width: int
    :param height: Image height, in pixels
    :type height: int
    :param title: Plot title
    :type title: basestring
    :param dpi: Image dpi
    :type dpi: Image dpi
    :param device: 'png' or 'pdf'
    :type device: str
    """

    fig = _plot_forecast_create(df_fcast, width, height, title, dpi)

    dirname, fname = os.path.split(file_path)
    if not os.path.exists(dirname):
        logger.error('Path missing {}'.format(file_path))
        os.makedirs(dirname)
    plt.savefig(file_path, dpi=dpi)


def plot_forecast(df_fcast, width=None, height=None, title=None, dpi=70):
    """
    Generates plot and shows in an ipython notebook

    :param df_fcast:
        | Forecast Dataframe with the following columns:
        | - date (timestamp)
        | - model (str) : ID for the forecast model
        | - y (float) : Value of the time series in that sample
        | - is_actuals (bool) : True for actuals samples, False for forecasted samples
    :type df_fcast: pandas.DataFrame
    :param width: Image width, in pixels
    :type width: int
    :param height: Image height, in pixels
    :type height: int
    :param title: Plot title
    :type title: str
    :param dpi: Image dpi
    :type dpi: Image dpi
    :return: Ipython image, to display in a notebook
    :rtype: Ipython.display.Image
    """
    try:
        from IPython.display import Image
    except ImportError:
        logger.info('IPython not available, skipping...')
        return None

    file_plot = NamedTemporaryFile()
    plot_forecast_save(df_fcast, file_plot.name, width, height, title, dpi)
    return Image(filename=file_plot.name, format='png')

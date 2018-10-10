# -*- coding: utf-8 -*-
#
# License:          This module is released under the terms of the LICENSE file
#                   contained within this applications INSTALL directory

"""
Functions to plot forecast outputs
"""

# -- Coding Conventions
#    http://www.python.org/dev/peps/pep-0008/   -   Use the Python style guide
# http://sphinx.pocoo.org/rest.html          -   Use Restructured Text for
# docstrings

# -- Public Imports
from tempfile import NamedTemporaryFile
import os
import matplotlib.pyplot as plt
import logging
import numpy as np
import pandas as pd

# -- Globals
logger = logging.getLogger(__name__)

try:
    import plotly as py
    from plotly import tools
    import plotly.graph_objs as go
    plotly_exists = True
except ImportError:
    logger.info('Plotly not available, skipping...')
    plotly_exists = False


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
    # Save dataframe as feather file. Formats strings on unicode, for
    # compatibility with R. Drops index.
    df.reset_index(drop=True).pipe(df_string_to_unicode).to_feather(file_path)


def pix_to_in(width_px=None, height_px=None, dpi=300):
    # Utility function to use pixel dimensions rather than ggplot's physical
    # dims
    dpi = float(dpi)

    width_in = width_px / dpi if width_px is not None else np.NaN
    height_in = height_px / dpi if height_px is not None else np.NaN
    # print width_in, height_in
    return width_in, height_in


def has_pi(df_fcast):
    # Check if a forecast table includes prediction intervals
    return 'q5' in df_fcast.columns


# ---- Plotting functions

def _matplotlib_forecast_create(df_fcast, facet, sources, nrows, ncols,
                                width=None, height=None, title=None, dpi=70,
                                show_legend=True):
    """
    Creates matplotlib plot from forecast dataframe

    :param df_fcast:
        | Forecast Dataframe with the following columns:
        | - date (timestamp)
        | - model (str) : ID for the forecast model
        | - y (float) : Value of the time series in that sample
        | - is_actuals (bool) : True for actuals samples, False for forecasted samples # noqa
    :type df_fcast: pandas.DataFrame
    :param facet: Indicates whether a facet grid will be required
    :type facet: bool
    :param sources: Includes the various sources
    :type sources:
    :param nrows: Number of rows
    :type nrows: int
    :param ncols: Number of cols
    :type ncols: int
    :param title: Plot title
    :type title: str
    :param width: plot width, in pixels
    :type width: int
    :param height: plot height, in pixels
    :type height: int
    :param dpi: plot dpi
    :type dpi: int
    :param show_legend: Indicates whether legends will be displayed
    :type show_legend: bool

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
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize,
                             dpi=dpi, squeeze=False)
    fig.canvas.set_window_title(title)

    x = 0
    y = 0
    for src in sources:
        ax = axes[x, y]

        # Filter the specific source is subplots
        if ~facet:
            source_filt = True
        else:
            source_filt = df_fcast['source'] == str(src)

        actuals, = ax.plot(
            df_fcast.loc[source_filt & df_fcast['is_actuals'], :].index,
            df_fcast.loc[source_filt & df_fcast['is_actuals'], 'y'],
            color=act_col, marker='o', linestyle='None', label='Actuals')
        forecast, = ax.plot(
            df_fcast.loc[source_filt & ~df_fcast['is_actuals'], :].index,
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

        if facet:
            # Set the title of each subplot as per source name
            ax.set_title(src)

        if show_legend:
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


def _plotly_forecast_create(df_fcast, facet, sources, nrows, ncols,
                            width=None, height=None, title=None,
                            show_legend=False):
    """
    Creates matplotlib plot from forecast dataframe

    :param df_fcast:
        | Forecast Dataframe with the following columns:
        | - date (timestamp)
        | - model (str) : ID for the forecast model
        | - y (float) : Value of the time series in that sample
        | - is_actuals (bool) : True for actuals samples, False for forecasted samples # noqa
    :type df_fcast: pandas.DataFrame
    :param facet: Indicates whether a facet grid will be required
    :type facet: bool
    :param sources: Includes the various sources
    :type sources:
    :param nrows: Number of rows
    :type nrows: int
    :param ncols: Number of cols
    :type ncols: int
    :param title: Plot title
    :type title: str
    :param width: plot width, in pixels
    :type width: int
    :param height: plot height, in pixels
    :type height: int
    :param show_legend: Indicates whether legends will be displayed
    :type show_legend: bool

    :return: The plot
    :rtype: plotly plot instance
    """

    if facet:
        fig = tools.make_subplots(rows=nrows, cols=ncols,
                                  subplot_titles=sources)
    else:
        fig = tools.make_subplots(rows=nrows, cols=ncols)

    x = 1
    y = 1
    for src in sources:
        # Filter the specific source is subplots
        if ~facet:
            source_filt = True
            actuals_name = 'Actuals'
            forecasts_name = 'Forecast'
        else:
            source_filt = df_fcast['source'] == src
            actuals_name = str(src) + ' Actuals'
            forecasts_name = str(src) + ' Forecast'

        actuals = go.Scatter(
            x=df_fcast.loc[source_filt & df_fcast['is_actuals']].date,
            y=df_fcast.loc[source_filt & df_fcast['is_actuals']].y,
            name=actuals_name,
            line=dict(color='#00BFC4'),
            mode='markers',
            opacity=0.8)

        fig.append_trace(actuals, x, y)

        forecast = go.Scatter(
            x=df_fcast.loc[source_filt & ~df_fcast['is_actuals']].date,
            y=df_fcast.loc[source_filt & ~df_fcast['is_actuals']].y,
            name=forecasts_name,
            line=dict(color='#F8766D'),
            mode='lines',
            opacity=0.8)

        fig.append_trace(forecast, x, y)

        # Fill area between 5th and 95th prediction interval
        if ('q5' in df_fcast.columns) and ('q95' in df_fcast.columns):
            q5 = go.Scatter(
                x=df_fcast.loc[source_filt & ~df_fcast['is_actuals']].date,
                y=df_fcast.loc[source_filt & ~df_fcast['is_actuals']].q5,
                name="5% PI",
                line=dict(color='#F8766D', width=0),
                mode='lines',
                showlegend=False,
                opacity=0.8)

            fig.append_trace(q5, x, y)

            q95 = go.Scatter(
                x=df_fcast.loc[source_filt & ~df_fcast['is_actuals']].date,
                y=df_fcast.loc[source_filt & ~df_fcast['is_actuals']].q95,
                name="95% PI",
                fill='tonexty',
                fillcolor='rgba(248,118,109,0.2)',
                line=dict(color='#F8766D', width=0),
                mode='lines',
                showlegend=False,
                opacity=0.4)
            fig.append_trace(q95, x, y)

        # Fill area between 5th and 95th prediction interval
        if ('q20' in df_fcast.columns) and ('q80' in df_fcast.columns):
            q20 = go.Scatter(
                x=df_fcast.loc[source_filt & ~df_fcast['is_actuals']].date,
                y=df_fcast.loc[source_filt & ~df_fcast['is_actuals']].q20,
                name="20% PI",
                line=dict(color='#F8766D', width=0),
                showlegend=False,
                mode='lines',
                opacity=0.8)

            fig.append_trace(q20, x, y)

            q80 = go.Scatter(
                x=df_fcast.loc[source_filt & ~df_fcast['is_actuals']].date,
                y=df_fcast.loc[source_filt & ~df_fcast['is_actuals']].q80,
                name="80% PI",
                fill='tonexty',
                fillcolor='rgba(248,118,109,0.2)',
                line=dict(color='#F8766D', width=0),
                mode='lines',
                showlegend=False,
                opacity=0.6)

            fig.append_trace(q80, x, y)

        y += 1
        if y > ncols:
            # New row
            y = 1
            x += 1

    fig['layout'].update(autosize=False,
                         width=width,
                         height=height,
                         title=title,
                         showlegend=show_legend,
                         legend=dict(traceorder='normal',
                                     font=dict(family='sans-serif',
                                               size=12,
                                               color='#000'),
                                     bgcolor='#E2E2E2',
                                     bordercolor='#FFFFFF',
                                     borderwidth=0),
                         paper_bgcolor='#FFFFFF',
                         plot_bgcolor='#E2E2E2')

    if facet:
        fig['layout'].update(xaxis=dict(rangeslider=dict(visible=True),
                                        type='date'))

    return fig


def plot_forecast(df_fcast, path, output='png', width=None,
                  height=None, title=None, dpi=70, show_legend=True,
                  auto_open=False):
    """
    Generates matplotlib or plotly plot and saves it respectively as png or
    html

    :param df_fcast:
        | Forecast Dataframe with the following columns:
        | - date (timestamp)
        | - model (str) : ID for the forecast model
        | - y (float) : Value of the time series in that sample
        | - is_actuals (bool) : True for actuals samples, False for forecasted samples # noqa
    :type df_fcast: pandas.DataFrame
    :param path: File path for output
    :type path: basestring
    :param output: ...
    :type output: ...
    :param width: Image width, in pixels
    :type width: int
    :param height: Image height, in pixels
    :type height: int
    :param title: Plot title
    :type title: basestring
    :param dpi: Image dpi
    :type dpi: int
    :param show_legend: Indicates whether legends will be displayed
    :type show_legend: bool
    :param auto_open: indicates whether the output will be displayed
                      automatically
    :type auto_open: bool
    """

    assert isinstance(df_fcast, pd.DataFrame)

    if 'source' in df_fcast.columns:
        facet = False
        sources = df_fcast.loc[df_fcast['is_actuals'], 'source'].unique()
        num_plots = len(sources)
        nrows = int(np.ceil(np.sqrt(num_plots)))
        ncols = int(np.ceil(1. * num_plots / nrows))
    else:
        # Only one set of actuals and forecast needed
        facet = True
        sources = ['y']
        nrows = 1
        ncols = 1

    if output == 'png':
        fig = _matplotlib_forecast_create(df_fcast, facet, sources, nrows,
                                          ncols, width, height, title, dpi,
                                          show_legend)
        path = '{}.png'.format(path)
        dirname, fname = os.path.split(path)
        if not os.path.exists(dirname):
            logger.error('Path missing {}'.format(path))
            os.makedirs(dirname)
        plt.savefig(path, dpi=dpi)
        # TODO address the autoopen issue
    elif output == 'html':
        if plotly_exists:
            fig = _plotly_forecast_create(df_fcast, facet, sources, nrows,
                                          ncols, width, height, title,
                                          show_legend)
            path = '{}.html'.format(path)
            py.offline.plot(fig, filename=path, show_link=False,
                            auto_open=auto_open)
        else:
            logger.error('Please install plotly library to enable this '
                         'feature.')
    elif output == 'jupyter':
        if plotly_exists:
            fig = _plotly_forecast_create(df_fcast, facet, sources, nrows,
                                          ncols, width, height, title,
                                          show_legend)
            py.offline.iplot(fig, show_link=False)
    else:
        logger.error('Wrong exporting format provided. Either png or html '
                     'formats are supported at the moment.')
        return 0


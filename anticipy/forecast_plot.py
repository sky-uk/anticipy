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
import os
import logging
import numpy as np
import pandas as pd
import webbrowser

# -- Globals
logger = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt

    _matplotlib_imported = True
except ImportError:
    logger.info('Matplotlib not available, skipping importing library...')
    _matplotlib_imported = False

try:
    import plotly as py
    from plotly import tools
    from plotly import subplots
    import plotly.graph_objs as go

    _plotly_imported = True
except ImportError:
    logger.info('Plotly not available, skipping importing library...')
    _plotly_imported = False

try:
    import IPython

    _ipython_imported = True
except ImportError:
    logger.info('IPython not available, skipping importing library...')
    _ipython_imported = False


# ---- Plotting functions
def _matplotlib_forecast_create(df_fcast, subplots, sources, nrows, ncols,
                                width=None, height=None, title=None, dpi=70,
                                show_legend=True,
                                include_interval=False):
    """
    Creates matplotlib plot from forecast dataframe

    :param df_fcast:
      |  Forecast Dataframe with the following columns:
      |  - date (timestamp)
      |  - model (str) : ID for the forecast model
      |  - y (float) : Value of the time series in that sample
      |  - is_actuals (bool) : True for actuals samples, False for forecast
    :type df_fcast: pandas.DataFrame
    :param subplots: Indicates whether a facet grid will be required
    :type subplots: bool
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
    assert _matplotlib_imported, 'Error: matplotlib not installed. Please ' \
                                 'run pip install plotly, then import.'
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
        if not subplots:
            source_filt = True
        else:
            source_filt = df_fcast['source'] == src

        actuals, = ax.plot(
            df_fcast.loc[source_filt & df_fcast['is_actuals'], :].index,
            df_fcast.loc[source_filt & df_fcast['is_actuals'], 'y'],
            color=act_col, marker='o', linestyle='None', label='Actuals')
        forecast, = ax.plot(
            df_fcast.loc[source_filt & ~df_fcast['is_actuals'], :].index,
            df_fcast.loc[source_filt & ~df_fcast['is_actuals'], 'y'],
            color=for_col, marker='None', linestyle='solid', label='Forecast')

        # Fill area between 5th and 95th prediction interval
        if include_interval and \
                ('q5' in df_fcast.columns) and ('q95' in df_fcast.columns):
            where_to_fill = (source_filt &
                             (~df_fcast['is_actuals']) &
                             (~df_fcast['q5'].isnull()) &
                             (~df_fcast['q95'].isnull()))
            ax.fill_between(df_fcast.index, df_fcast['q5'], df_fcast['q95'],
                            where=where_to_fill,
                            facecolor=for_col, alpha=0.2)

        if include_interval and\
                ('q20' in df_fcast.columns) and ('q80' in df_fcast.columns):
            # Fill area between 20th and 80th prediction interval
            where_to_fill_2 = (source_filt &
                               (~df_fcast['is_actuals']) &
                               (~df_fcast['q20'].isnull()) &
                               (~df_fcast['q80'].isnull()))
            ax.fill_between(df_fcast.index, df_fcast['q20'], df_fcast['q80'],
                            where=where_to_fill_2,
                            facecolor=for_col, alpha=0.2)

        if subplots:
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


def _plotly_forecast_create(df_fcast, subplots, sources, nrows, ncols,
                            width=None, height=None, title=None,
                            show_legend=False, add_rangeslider=False,
                            include_interval=False,
                            pi_q1=5,
                            pi_q2=20
                            ):
    """
    Creates matplotlib plot from forecast dataframe

    :param df_fcast:
      |  Forecast Dataframe with the following columns:
      |  - date (timestamp)
      |  - model (str) : ID for the forecast model
      |  - y (float) : Value of the time series in that sample
      |  - is_actuals (bool) : True for actuals samples, False for forecast
    :type df_fcast: pandas.DataFrame
    :param subplots: Indicates whether a facet grid will be required
    :type subplots: bool
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
    :param add_rangeslider:
    :type add_rangeslider: bool
    :param include_interval:
    :type include_interval: bool
    :param pi_q1: Percentile for outer prediction interval (defaults to 5%-95%)
    :type pi_q1: int
    :param pi_q2: Percentile for inner prediction interval (defaults to 20%-80%)
    :type pi_q2: int

    :return: The plot
    :rtype: plotly plot instance
    """
    assert _plotly_imported, 'Error: plotly not installed. Please run pip ' \
                             'install plotly, then import the library'

    vertical_spacing = 50. / height if height is not None else 0.1

    if subplots:
        titles = map(str, sources)
        fig = py.subplots.make_subplots(
            rows=nrows,
            cols=ncols,
            subplot_titles=list(titles),
            print_grid=False,
            horizontal_spacing=0.08,
            vertical_spacing=vertical_spacing)
        margin_top = 60
    else:
        fig = py.subplots.make_subplots(
            rows=nrows, cols=ncols, print_grid=False)
        margin_top = 30

    x = 1
    y = 1

    # Due to plotly implementation details, we only show legend for 1 source
    is_first_source = True

    for src in sources:
        # Filter the specific source is subplots
        if not subplots:
            source_filt = True
            # actuals_name = 'Actuals'
            # forecasts_name = 'Forecast'
        else:
            source_filt = df_fcast['source'] == src
            # actuals_name = '{} Actuals'.format(str(src))
            # forecasts_name = '{} Forecast'.format(str(src))

        actuals_name = 'Actuals'
        forecasts_name = 'Forecast'

        actuals = go.Scatter(
            x=df_fcast.loc[source_filt & df_fcast['is_actuals']].date,
            y=df_fcast.loc[source_filt & df_fcast['is_actuals']].y,
            name=actuals_name,
            line=dict(color='rgba(0,191,196,0.2)'),
            marker=dict(color='rgba(0,191,196,0.9)', size=3),
            mode='lines+markers',
            opacity=0.8,
            legendgroup='actuals',
            showlegend=is_first_source,
        )

        fig.add_trace(actuals, x, y)

        forecast = go.Scatter(
            x=df_fcast.loc[source_filt & ~df_fcast['is_actuals']].date,
            y=df_fcast.loc[source_filt & ~df_fcast['is_actuals']].y,
            name=forecasts_name,
            line=dict(color='rgba(248,118,109,0.4)', width=1),
            marker=dict(color='rgba(248,118,109,0.9)', size=3),
            mode='lines+markers',
            legendgroup='forecast',
            showlegend=is_first_source,
        )

        fig.add_trace(forecast, x, y)
        for pi_q in [pi_q1, pi_q2]:
            # Fill prediction interval area
            str_q_low = 'q{}'.format(pi_q)
            str_q_hi = 'q{}'.format(100 - pi_q)
            if include_interval and \
                    (str_q_low in df_fcast.columns) and \
                    (str_q_hi in df_fcast.columns):
                q_low = go.Scatter(
                    x=df_fcast.loc[source_filt & ~df_fcast['is_actuals']].date,
                    y=df_fcast.loc[source_filt & ~df_fcast['is_actuals']]
                    [str_q_low],
                    name="{}% PI".format(pi_q),
                    line=dict(color='#F8766D', width=0),
                    mode='lines',
                    showlegend=False,
                    legendgroup='forecast')

                fig.add_trace(q_low, x, y)

                q_hi = go.Scatter(
                    x=df_fcast.loc[source_filt & ~df_fcast['is_actuals']].date,
                    y=df_fcast.loc[source_filt & ~df_fcast['is_actuals']]
                    [str_q_hi],
                    name="{}% PI".format(100-pi_q),
                    fill='tonexty',
                    fillcolor='rgba(248,118,109,0.2)',
                    line=dict(color='#F8766D', width=0),
                    mode='lines',
                    showlegend=False,
                    legendgroup='forecast')
                fig.add_trace(q_hi, x, y)

        y += 1
        if y > ncols:
            # New row
            y = 1
            x += 1
        is_first_source = False

    fig['layout'].update(autosize=False,
                         width=width,
                         height=height,
                         title=title,
                         showlegend=show_legend,
                         legend=dict(traceorder='normal',
                                     font=dict(family='sans-serif',
                                               size=12,
                                               color='#000'),
                                     bordercolor='#FFFFFF',
                                     borderwidth=0,
                                     orientation='h'),
                         # Using {} instead of  dict () because
                         # 'l' variable raises PEP-8 warn
                         margin={'l': 0, 'r': 0, 't': margin_top, 'b': 0},
                         paper_bgcolor='#FFFFFF',
                         plot_bgcolor='#E2E2E2',
                         )

    def set_axis_format(layout):
        # Update all axes in layout to have automargin=True
        dict_format = dict(
            automargin=True,
            tickfont=dict(size=10)
        )
        dict_update = {k: dict_format for k in layout
                       if k.startswith('xaxis') or k.startswith('yaxis')}
        layout.update(dict_update)

    set_axis_format(fig['layout'])

    if not subplots and add_rangeslider:
        fig['layout'].update(xaxis=dict(rangeslider=dict(visible=True),
                                        type='date'))

    return fig


def plot_forecast(df_fcast, output='html', path=None, width=None, height=None,
                  title=None, dpi=70, show_legend=True, auto_open=False,
                  include_interval=False,
                  pi_q1=5,
                  pi_q2=20
                  ):
    """
    Generates matplotlib or plotly plot and saves it respectively as png or
    html

    :param df_fcast:
      |  Forecast Dataframe with the following columns:
      |  - date (timestamp)
      |  - model (str) : ID for the forecast model
      |  - y (float) : Value of the time series in that sample
      |  - is_actuals (bool) : True for actuals samples, False for forecast
    :type df_fcast: pandas.DataFrame
    :param output: Indicates the output type (html=Default, png or jupyter)
    :type output: basestring
    :param path: File path for output
    :type path: basestring
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
    :param auto_open: Indicates whether the output will be displayed
                      automatically
    :type auto_open: bool
    :param pi_q1: Percentile for outer prediction interval (defaults to 5%-95%)
    :type pi_q1: int
    :param pi_q2: Percentile for inner prediction interval (defaults to 20%-80%)
    :type pi_q2: int
    :return: Success or failure code.
    :rtype: int
    """

    assert isinstance(df_fcast, pd.DataFrame)
    add_rangeslider = False   # Feature currently disabled

    if not path and (output == 'html' or output == 'png'):
        logger.error('No export path provided.')
        return 1

    if 'source' in df_fcast.columns and df_fcast.source.nunique() > 1:
        subplots = True
        sources = df_fcast.loc[df_fcast['is_actuals'], 'source'].unique()
        num_plots = len(sources)
        nrows = int(np.ceil(np.sqrt(num_plots)))
        ncols = int(np.ceil(1. * num_plots / nrows))
    else:
        # Only one set of actuals and forecast needed
        subplots = False
        sources = ['y']
        nrows = 1
        ncols = 1

    if output == 'png':
        if _matplotlib_imported:
            fig = _matplotlib_forecast_create(df_fcast, subplots, sources,
                                              nrows, ncols, width, height,
                                              title, dpi, show_legend,
                                              include_interval)

            path = '{}.png'.format(path)
            dirname, fname = os.path.split(path)
            if dirname != '' and not os.path.exists(dirname):
                logger.error('Path missing {}'.format(path))
                os.makedirs(dirname)
            plt.savefig(path, dpi=dpi)

            if auto_open:
                fileurl = 'file://{}'.format(path)
                webbrowser.open(fileurl, new=2, autoraise=True)
        else:
            logger.error('Please install matplotlib library to enable this '
                         'feature.')
    elif output == 'html':
        if _plotly_imported:
            fig = _plotly_forecast_create(df_fcast, subplots, sources, nrows,
                                          ncols, width, height, title,
                                          show_legend,
                                          add_rangeslider,
                                          include_interval, pi_q1, pi_q2)
            path = '{}.html'.format(path)
            py.offline.plot(fig, filename=path, show_link=False,
                            auto_open=auto_open, include_plotlyjs='cdn')
        else:
            logger.error('Please install plotly library to enable this '
                         'feature.')
    elif output == 'jupyter':
        if _plotly_imported and _ipython_imported:
            py.offline.init_notebook_mode(connected=True)
            fig = _plotly_forecast_create(df_fcast, subplots, sources, nrows,
                                          ncols, width, height, title,
                                          show_legend,
                                          add_rangeslider,
                                          include_interval, pi_q1, pi_q2)
            return py.offline.iplot(fig, show_link=False)
        else:
            logger.error('Please make sure that both plotly and ipython '
                         'libraries are installed to enable this feature.')
    else:
        logger.error('Wrong exporting format provided. Either png, html or '
                     'jupyter formats are supported at the moment.')
        return 1

    return 0

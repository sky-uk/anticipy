# -*- coding: utf-8 -*-
#
# License:          This module is released under the terms of the LICENSE file 
#                   contained within this applications INSTALL directory

"""
    __high_level_module_description_here__
"""

# -- Coding Conventions
#    http://www.python.org/dev/peps/pep-0008/   -   Use the Python style guide
#    http://sphinx.pocoo.org/rest.html          -   Use Restructured Text for docstrings

# -- Public Imports
import logging
import pandas as pd
import os
import forecast
import forecast_plot
import argparse

# -- Private Imports

# -- Globals
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# -- Exception classes


# -- Functions
def logger_info(msg, data):
    # Convenience function for easier log typing
    logger.info(msg + '\n%s', data)


def run_forecast_app(path_in, path_out=None, forecast_years=2.0,
                     col_name_y='y', col_name_weight='weight',
                     col_name_x='x', col_name_date='date',
                     col_name_source='source',
                     include_all_fits=False
                     ):
    assert path_in is not None and os.path.exists(path_in), 'path_in needs to be a string pointing to a valid file path'
    assert not os.path.isdir(path_in)

    file_name = os.path.basename(path_in)
    file_name_p1 = file_name.split('.')[0]

    logger_info('file_name', file_name)
    logger_info('file_name p1', file_name_p1)

    if path_out is None:
        path_out = path_in
    assert os.path.exists(path_out)

    path_folder = os.path.dirname(path_out)

    logger_info('dir name', path_folder)

    path_data = os.path.join(path_folder, file_name_p1+'_fcast.csv')
    path_metadata = os.path.join(path_folder, file_name_p1+'_metadata.csv')
    path_plot = os.path.join(path_folder, file_name_p1 + '_fcast.png')

    logger_info('path_data', path_data)
    logger_info('path_metadata', path_metadata)
    logger_info('path_plot', path_plot)

    df_y = pd.read_csv(path_in)

    if col_name_date in df_y:  # Need to parse date
        df_y[col_name_date] = df_y[col_name_date].pipe(pd.to_datetime)

    df_y = forecast.normalize_df(df_y, col_name_y, col_name_weight, col_name_x, col_name_date,
                                 col_name_source)

    dict_result = forecast.run_forecast(df_y, extrapolate_years=forecast_years, simplify_output=False,
                                        include_all_fits=include_all_fits)

    df_result = dict_result['data']
    df_metadata = dict_result['metadata']
    df_optimize_info = dict_result['optimize_info']

    df_result.to_csv(path_data, index=False)
    df_metadata.to_csv(path_metadata, index=False)

    try:
        forecast_plot.plot_forecast_save(df_result, path_plot, width=1920, height=1080)
    except AssertionError:
        logger.info("Couldn't generate plot - R not installed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_in', help='Path of input .csv file')
    parser.add_argument('--path_out', help='Path of output folder - defaults to folder of path_in', default=None)
    parser.add_argument('--forecast_years', help='Years in forecast interval', default=2.0, type=float)
    parser.add_argument('--col_name_y', help='Name of column for y', default='y')
    parser.add_argument('--col_name_date', help='Name of column for date', default='date')
    parser.add_argument('--col_name_weight', help='Name of column for weight', default='weight')
    parser.add_argument('--col_name_source', help='Name of column for y', default='source')
    parser.add_argument('--col_name_x', help='Name of column for x', default='x')
    parser.add_argument('--include_all_fits', help='If true, output includes non-optimal models', action='store_true')

    args = parser.parse_args()
    logger.info('Input: path_in= %s', args.path_in)
    logger.info('Input: path_out= %s', args.path_out)
    logger.info('Input: col_name_y= %s', args.col_name_y)
    logger.info('Input: col_name_date= %s', args.col_name_date)
    logger.info('Input: col_name_x= %s', args.col_name_x)
    logger.info('Input: col_name_weight= %s', args.col_name_weight)
    logger.info('Input: col_name_source= %s', args.col_name_source)
    logger.info('Input: include_all_fits= %s', args.include_all_fits)

    run_forecast_app(args.path_in, args.path_out, args.forecast_years,
                     args.col_name_y, args.col_name_weight, args.col_name_x, args.col_name_date,
                     args.col_name_source, args.include_all_fits)

    # run_forecast_app('/Users/pec21/Downloads/file1.csv','/Users/pec21/Downloads/',
    #                   col_name_y='occup_erl', col_name_source='bend_name')


# -- Main
if __name__ == '__main__':
    main()

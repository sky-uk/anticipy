# -*- coding: utf-8 -*-
#
# License:          This module is released under the terms of the LICENSE file
#                   contained within this applications INSTALL directory

"""
    Class and functions to test pandas dataframes and series
"""

# -- Coding Conventions
#    http://www.python.org/dev/peps/pep-0008/   -   Use the Python style guide
#    http://sphinx.pocoo.org/rest.html          -   Use Restructured Text for docstrings

# -- Public Imports
import unittest
import numpy as np
import pandas as pd
import pandas.util.testing as pdt
import logging

# -- Globals
logger = logging.getLogger(__name__)


# -- Exception classes

# -- Functions
def logger_info(msg, data):
    # Convenience function for easier log typing
    logger.info(msg + '\n%s', data)


def _is_dtype_categorical(x):
    if type(x) is pd.DataFrame:
        # Slightly faster than x.dtypes == 'category'
        return x.dtypes.apply(lambda x: x.name == 'category')
    else:
        # Used because x.dtype =='category' doesn't always work
        return x.dtype.name == 'category'


# -- Classes
class PandasTest(unittest.TestCase):

    def assert_frame_equal(self, left, right, ignore_index=False, compare_as_strings=False,
                           ignore_column_order=False, **kwargs):
        """
        Checks that 2 dataframes are equal

        :param left:
        :type left:
        :param right:
        :type right:
        :param ignore_index:
        :type ignore_index:
        :param compare_as_strings:
        :type compare_as_strings:
        :param kwargs:
        :type kwargs:

        """
        l = left
        r = right
        if ignore_index:
            l = l.reset_index(drop=True)
            r = r.reset_index(drop=True)
        if compare_as_strings:
            l = l.astype(str)
            r = r.astype(str)
        if ignore_column_order:
            r = r.pdu_reorder(l.columns)
        pdt.assert_frame_equal(l, r, **kwargs)

    def assert_frame_not_equal(self, left, right, ignore_index=False, **kwargs):
        if ignore_index:
            with self.assertRaises(AssertionError):
                pdt.assert_frame_equal(left.reset_index(drop=True), right.reset_index(drop=True), **kwargs)
        else:
            with self.assertRaises(AssertionError):
                pdt.assert_frame_equal(left, right, **kwargs)

    def assert_series_equal(self, left, right, ignore_index=False, compare_as_strings=False, ignore_name=True,
                            **kwargs):
        """
        Checks that 2 series are equal

        :param left:
        :type left:
        :param right:
        :type right:
        :param ignore_index:
        :type ignore_index:
        :param compare_as_strings:
        :type compare_as_strings:
        :param kwargs:
        :type kwargs:
        """
        l = left
        r = right
        pdt._check_isinstance(l, r, pd.Series)
        if ignore_index:
            l = l.reset_index(drop=True)
            r = r.reset_index(drop=True)
        if compare_as_strings:
            l = l.astype(str)
            r = r.astype(str)
        if ignore_name:
            l = l.rename(None)
            r = r.rename(None)

        if _is_dtype_categorical(l) or _is_dtype_categorical(r):
            self.assertTrue(_is_dtype_categorical(l))
            self.assertTrue(_is_dtype_categorical(r))
            self.assertTrue(r.equals(l))
            self.assertEqual(l.cat.ordered, r.cat.ordered)
        else:
            pdt.assert_series_equal(l, r, **kwargs)

    def assert_array_equal(self, left, right):
        np.testing.assert_array_equal(left, right)

# -- Main

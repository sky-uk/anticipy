# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.

import pandas as pd
import numpy as np
from anticipy import forecast


class TimeForecastSuite:
    params = [['W', 'D']]
    param_names = ['freq']

    def setup(self, freq):
        date_start = '2016-01-01'
        date_end = '2019-01-01'

        s_date_d = pd.date_range(date_start, date_end, freq='D')
        s_date_w = pd.date_range(date_start, date_end, freq='W')

        ts_mean = 100.
        ts_sd = 1.
        np.random.seed(0)  # Same random seed for all runs
        df_in_d = pd.DataFrame({'date': s_date_d,
                                'y': np.random.normal(ts_mean, ts_sd,
                                                      s_date_d.size)})
        df_in_w = pd.DataFrame({'date': s_date_w,
                                'y': np.random.normal(ts_mean, ts_sd,
                                                      s_date_w.size)})
        self.df_y = df_in_d if freq == 'D' else df_in_w

    def time_run_forecast(self, freq):
        df_result = forecast.run_forecast(
            self.df_y, extrapolate_years=2)

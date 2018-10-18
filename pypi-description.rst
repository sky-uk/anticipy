Anticipy
========

Anticipy is a tool to generate forecasts for time series. It takes a pandas Series or DataFrame as input, and
returns a DataFrame with the forecasted values for a given period of time.

Features:

* **Simple interface**. Start forecasting with a single function call on a pandas DataFrame.
* **Model selection**. If you provide different multiple models (e.g. linear, sigmoidal, exponential), the tool will
  compare them and choose the best fit for your data.
* **Trend and seasonality**. Support for weekly and monthly seasonality, among other types.
* **Calendar events**. Provide lists of special dates, such as holiday seasons or bank holidays, to improve model
  performance.
* **Data cleaning**. The library has tools to identify and remove outliers, and to detect and handle step changes in
  the data.

It is straightforward to generate a simple linear model with the tool - just call ``forecast.run_forecast(my_dataframe)``: ::

   import pandas as pd, numpy as np
   from anticipy import forecast
   
   df = pd.DataFrame({'y': np.arange(0., 5)}, index=pd.date_range('2018-01-01', periods=5, freq='D'))
   df_forecast = forecast.run_forecast(df, extrapolate_years=1)
   print(df_forecast.head(12))

Output: ::

    .        date source  is_actuals    model    y   q5  q20  q80  q95
    0  2018-01-01    src        True  actuals  0.0  NaN  NaN  NaN  NaN
    1  2018-01-02    src        True  actuals  1.0  NaN  NaN  NaN  NaN
    2  2018-01-03    src        True  actuals  2.0  NaN  NaN  NaN  NaN
    3  2018-01-04    src        True  actuals  3.0  NaN  NaN  NaN  NaN
    4  2018-01-05    src        True  actuals  4.0  NaN  NaN  NaN  NaN
    5  2018-01-01    src       False   linear  0.0  NaN  NaN  NaN  NaN
    6  2018-01-02    src       False   linear  1.0  NaN  NaN  NaN  NaN
    7  2018-01-03    src       False   linear  2.0  NaN  NaN  NaN  NaN
    8  2018-01-04    src       False   linear  3.0  NaN  NaN  NaN  NaN
    9  2018-01-05    src       False   linear  4.0  NaN  NaN  NaN  NaN
    10 2018-01-06    src       False   linear  5.0  5.0  5.0  5.0  5.0
    11 2018-01-07    src       False   linear  6.0  6.0  6.0  6.0  6.0

Documentation is available in `Read the Docs <https://anticipy.readthedocs.io/en/latest/>`_

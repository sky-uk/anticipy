[![Latest Release](https://img.shields.io/pypi/v/anticipy.svg)](https://img.shields.io/pypi/v/anticipy.svg)
[![Build Status](https://travis-ci.com/sky-uk/anticipy.svg?branch=master)](https://travis-ci.com/sky-uk/anticipy)
[![Documentation Status](https://readthedocs.org/projects/anticipy/badge/?version=latest)](https://anticipy.readthedocs.io/en/latest/?badge=latest)
[![Code Coverage](https://codecov.io/github/sky-uk/anticipy/branch/master/graph/badge.svg)](https://codecov.io/github/sky-uk/anticipy/)
[![pulls](https://img.shields.io/docker/pulls/skyuk/anticipy.svg)](https://hub.docker.com/r/skyuk/anticipy)



# Anticipy

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

It is straightforward to generate a simple linear model with the tool - just call forecast.run_forecast(my_dataframe):

```python
   import pandas as pd, numpy as np
   from anticipy import forecast
   
   df = pd.DataFrame({'y': np.arange(0., 5)}, index=pd.date_range('2018-01-01', periods=5, freq='D'))
   df_forecast = forecast.run_forecast(df, extrapolate_years=1)
   print(df_forecast.head(12))
```

Output:

```
   .        date   model             y  is_actuals
   0  2018-01-01       y  0.000000e+00        True
   1  2018-01-02       y  1.000000e+00        True
   2  2018-01-03       y  2.000000e+00        True
   3  2018-01-04       y  3.000000e+00        True
   4  2018-01-05       y  4.000000e+00        True
   5  2018-01-01  linear  5.551115e-17       False
   6  2018-01-02  linear  1.000000e+00       False
   7  2018-01-03  linear  2.000000e+00       False
   8  2018-01-04  linear  3.000000e+00       False
   9  2018-01-05  linear  4.000000e+00       False
   10 2018-01-06  linear  5.000000e+00       False
   11 2018-01-07  linear  6.000000e+00       False
```


Documentation is available in [Read the Docs](https://anticipy.readthedocs.io/en/latest/)

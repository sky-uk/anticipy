.. Adapt this file as required

********
Overview
********

Anticipy is a tool to generate forecasts for time series. It takes a pandas Series or DataFrame as input, and
returns a DataFrame with the forecasted values for a given period of time.

Features:

* **Simple interface**. Start forecasting with a single function call on a pandas DataFrame.
* **Model selection**. If you provide different models (e.g. linear, sigmoidal, exponential), the tool will
  compare them and choose the best fit for your data.
* **Trend and seasonality**. Support for weekly and monthly seasonality, among other types.
* **Calendar events**. Provide lists of special dates, such as holiday seasons or bank holidays, to improve model
  performance.
* **Data cleaning**. The library has tools to identify and remove outliers, and to detect and handle step changes in
  the data.

To get started, install the library with pip: ::

   pip install anticipy

It is straightforward to generate a simple linear model with the tool - just call
:py:func:`forecast.run_forecast(my_dataframe)`: ::

    import pandas as pd, numpy as np
    from anticipy import forecast, forecast_models

    df = pd.DataFrame({'y': np.full(20,10.0)+np.random.normal(0.0, 0.1, 20),
                       'date':pd.date_range('2018-01-01', periods=20, freq='D')})
    df_forecast = forecast.run_forecast(df, extrapolate_years=0.5)
    print(df_forecast.tail(3))

Output::

    .         date source  is_actuals   model         y        q5       q20        q80        q95
    219 2018-07-19    src       False  linear  9.490259  7.796581  8.339835  10.556202  11.689470
    220 2018-07-20    src       False  linear  9.487518  7.828049  8.362620  10.466285  11.640854
    221 2018-07-21    src       False  linear  9.484776  7.776001  8.343068  10.423964  11.696145

For more advanced usage, check the :ref:`rst_tutorial`.


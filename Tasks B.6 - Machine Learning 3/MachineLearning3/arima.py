import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt


def plot_acf_pacf(time_series):
    """
    Parameters:
    - time_series: The time series data (pandas Series).
    """
    # Plot ACF and PACF for 'Close' prices
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    # ACF plot
    plot_acf(time_series, lags=40, ax=ax[0])

    # PACF plot
    plot_pacf(time_series, lags=40, ax=ax[1])

    plt.tight_layout()
    plt.show()
    

def visualize_and_check_stationarity(time_series):
    """
    Visualizes the provided time series with its rolling mean and standard deviation.
    Also performs the Augmented Dickey-Fuller (ADF) test to check for stationarity.

    Parameters:
    - time_series: The time series data (pandas Series).
    """
    # Plotting the original time series with rolling mean and standard deviation
    rolling_window = 30  # 30 days rolling window
    rolling_mean = time_series.rolling(window=rolling_window).mean()
    rolling_std = time_series.rolling(window=rolling_window).std()

    plt.figure(figsize=(14, 7))
    plt.plot(time_series, color='blue', label='Original Close Prices')
    plt.plot(rolling_mean, color='red', label='Rolling Mean')
    plt.plot(rolling_std, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Close Prices with Rolling Mean & Standard Deviation')
    plt.show()

    # Augmented Dickey-Fuller test
    adft = adfuller(time_series, autolag='AIC')
    output = pd.Series(adft[0:4], index=['Test Statistics', 'p-value', 'No. of lags used', 'Number of observations used'])
    for key, values in adft[4].items():
        output['critical value (%s)' % key] = values
    print(output)
    

def decompose_time_series(time_series, model='additive', freq=30):
    """
    Decomposes a time series into trend, seasonal, and residual components.

    Parameters:
    - time_series: The time series data (pandas Series).
    - model: Type of decomposition ('additive' or 'multiplicative'). Default is 'additive'.
    - freq: Frequency for seasonal decomposition. Default is 30 (monthly).

    Returns:
    - Decomposition result object.
    """

    decomposition = seasonal_decompose(time_series, model=model, period=freq)

    # Plotting the decomposed time series components
    plt.figure(figsize=(16, 8))

    plt.subplot(411)
    plt.plot(time_series, label='Original')
    plt.legend(loc='upper left')
    plt.title('Time Series Decomposition')

    plt.subplot(412)
    plt.plot(decomposition.trend, label='Trend')
    plt.legend(loc='upper left')

    plt.subplot(413)
    plt.plot(decomposition.seasonal, label='Seasonal')
    plt.legend(loc='upper left')

    plt.subplot(414)
    plt.plot(decomposition.resid, label='Residual')
    plt.legend(loc='upper left')

    plt.tight_layout()
    plt.show()

    return decomposition
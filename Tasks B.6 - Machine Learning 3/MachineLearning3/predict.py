import warnings
warnings.filterwarnings('ignore')
import pmdarima as pm
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima.arima import ADFTest
from pmdarima.datasets import load_wineind
import plotly.graph_objects as go
import numpy as np
from initialized import *
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
from datasets import create_predict_datasets
import pandas as pd

'''LSTM TEST DATA'''
# Testing prediction on train model
def lstm_predict_test(model, train_target_scaler, x_test, y_test):    
    # Predictions on test data
    predictions = model.predict(x_test)
    predictions = train_target_scaler.inverse_transform(predictions)
    
    # Finding error using RMSE formula
    rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
    print(f'RSME LSTM: {rmse}')
    
    return predictions, rmse

'''ARIMA TEST DATA'''
def arima_predict_test(train_data, test_data, start_p=0, max_p=5, start_q=0, max_q=5, m=7, seasonal=True):
    # Extracting the Close price from train and test data
    train_close = train_data['Close']
    test_close = test_data['Close']

    x_train = list(range(len(train_close)))
    x_test = list(range(len(train_close), len(train_close) + len(test_close)))

    # Visualization of training and testing data
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_train, y=train_close, mode='lines+markers', marker=dict(size=4),  name='train', marker_color='#39304A'))
    fig.add_trace(go.Scatter(x=x_test, y=test_close, mode='lines+markers', marker=dict(size=4), name='test', marker_color='#A98D75'))
    fig.update_layout(legend_orientation="h",
                  legend=dict(x=.5, xanchor="center"),
                  plot_bgcolor='#FFFFFF',  
                  xaxis=dict(gridcolor = 'lightgrey'),
                  yaxis=dict(gridcolor = 'lightgrey'),
                  title_text = f'{ticker} ARIMA data', title_x = 0.5,
                  xaxis_title="Timestep",
                  yaxis_title="Stock price",
                  margin=dict(l=0, r=0, t=30, b=0))
    fig.show()

    # Combine train and test data for ARIMA modeling
    full_data = pd.concat([train_data, test_data])

    # ARIMA modeling
    model = pm.auto_arima(train_close, 
                          start_p=start_p, 
                          d=None, 
                          start_q=start_q, 
                          max_p=max_p, 
                          max_d=5, 
                          max_q=max_q, 
                          start_P=0, 
                          D=1, 
                          start_Q=0, 
                          max_P=5, 
                          max_D=5,
                          max_Q=5,
                          m=m, 
                          seasonal=seasonal, 
                          error_action='warn', 
                          trace=True,
                          supress_warnings=True, 
                          stepwise=True,
                          random_state=20, 
                          n_fits=50)

    model.summary()

    # Predictions
    predictions = model.predict(n_periods=len(test_close))

    rmse = np.sqrt(np.mean((predictions - test_close.values) ** 2))
    print(f'RMSE ARIMA: {rmse}')

    # Visualization of historical vs predicted values
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_test, y=test_close, mode='lines+markers', name='historical', marker_color='#39304A'))
    fig.add_trace(go.Scatter(x=x_test, y=predictions, mode='lines+markers', name='predictions', marker_color='#FFAA00'))
    fig.update_layout(legend_orientation="h",
                  legend=dict(x=.5, xanchor="center"),
                  plot_bgcolor='#FFFFFF',  
                  xaxis=dict(gridcolor = 'lightgrey'),
                  yaxis=dict(gridcolor = 'lightgrey'),
                  title_text = f'{ticker} ARIMA prediction', title_x = 0.5,
                  xaxis_title="Timestep",
                  yaxis_title="Stock price",
                  margin=dict(l=0, r=0, t=30, b=0))
    fig.show()

    return predictions, rmse

'''SARIMA TEST DATA'''
def sarimax_predict_test(train_data, test_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
    # Extracting the Close price and exogenous variables from train and test data
    train_close = train_data['Close']
    test_close = test_data['Close']
    exo_train_data = train_data['Volume']
    exo_test_data = test_data['Volume']

    # Setting frequency for the time series
    train_close.index = pd.DatetimeIndex(train_close.index).to_period('D')
    test_close.index = pd.DatetimeIndex(test_close.index).to_period('D')
    exo_train_data.index = pd.DatetimeIndex(exo_train_data.index).to_period('D')
    exo_test_data.index = pd.DatetimeIndex(exo_test_data.index).to_period('D')

    x_train = list(range(len(train_close)))
    x_test = list(range(len(train_close), len(train_close) + len(test_close)))
    
    # Visualization of training and testing data
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_train, y=train_close, mode='lines+markers', marker=dict(size=4),  name='train', marker_color='#39304A'))
    fig.add_trace(go.Scatter(x=x_test, y=test_close, mode='lines+markers', marker=dict(size=4), name='test', marker_color='#A98D75'))
    fig.update_layout(legend_orientation="h",
                  legend=dict(x=.5, xanchor="center"),
                  plot_bgcolor='#FFFFFF',  
                  xaxis=dict(gridcolor = 'lightgrey'),
                  yaxis=dict(gridcolor = 'lightgrey'),
                  title_text = f'{ticker} ARIMA data', title_x = 0.5,
                  xaxis_title="Timestep",
                  yaxis_title="Stock price",
                  margin=dict(l=0, r=0, t=30, b=0))
    fig.show()

    # SARIMAX modeling
    model = SARIMAX(train_close, exog=exo_train_data, order=order, seasonal_order=seasonal_order)
    results = model.fit(disp=-1, maxiter=200, method='nm')
    print(results.summary())

    # Predictions
    predictions = results.predict(start=len(train_close),
                                  end=len(train_close) + len(test_close) - 1, 
                                  exog=exo_test_data)

    rmse = np.sqrt(mean_squared_error(test_close, predictions))
    print(f'RMSE SARIMAX: {rmse}')

    # Visualization of historical vs predicted values
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_test, y=test_close, mode='lines+markers', name='historical', marker_color='#39304A'))
    fig.add_trace(go.Scatter(x=x_test, y=predictions, mode='lines+markers', name='predictions', marker_color='#FFAA00'))
    fig.update_layout(legend_orientation="h",
                  legend=dict(x=.5, xanchor="center"),
                  plot_bgcolor='#FFFFFF',
                  xaxis=dict(gridcolor='lightgrey'),
                  yaxis=dict(gridcolor='lightgrey'),
                  title_text=f'{ticker} SARIMAX prediction', title_x=0.5,
                  xaxis_title="Timestep",
                  yaxis_title="Stock price",
                  margin=dict(l=0, r=0, t=30, b=0))
    fig.show()

    return predictions, rmse
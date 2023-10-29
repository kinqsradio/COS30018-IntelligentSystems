import numpy as np
import plotly.graph_objects as go

def plot_predictions(tick, predict_df, lstm_predictions=None, arima_predictions=None,
                     avg_predictions=None, arima_conf_int=None, sentiment=None, n=None):
    
    # If n is provided, slice the last n rows of data and predictions
    if n:
        predict_df = predict_df[-n:]
        if lstm_predictions is not None:
            lstm_predictions = lstm_predictions[-n:]
        if arima_predictions is not None:
            arima_predictions = arima_predictions[-n:]
        if avg_predictions is not None:
            avg_predictions = avg_predictions[-n:]
        if arima_conf_int is not None:
            arima_conf_int = arima_conf_int[-n:]

    # Extract the 'Close' column values
    full_data = predict_df['Close']

    fig = go.Figure()

    # Plotting the full historical data
    fig.add_trace(go.Scatter(y=full_data,
                             mode='lines',
                             name='Historical Data',
                             line=dict(color='blue')))

    if lstm_predictions is not None:
        # Overlaying the LSTM predictions
        fig.add_trace(go.Scatter(y=np.concatenate([full_data, lstm_predictions.flatten()]),
                                 mode='lines',
                                 name='LSTM Predictions',
                                 line=dict(color='green', dash='dash')))

    if arima_predictions is not None:
        # Overlaying the ARIMA predictions
        fig.add_trace(go.Scatter(y=np.concatenate([full_data, arima_predictions]),
                                 mode='lines',
                                 name='ARIMA Predictions',
                                 line=dict(color='red', dash='dash')))
        if arima_conf_int is not None:
            # Adding confidence intervals for ARIMA predictions
            fig.add_trace(go.Scatter(y=np.concatenate([full_data, arima_conf_int[:, 0]]),
                                     mode='lines',
                                     line=dict(color='red', width=0),
                                     showlegend=False,
                                     fillcolor='rgba(255,0,0,0.2)',
                                     fill='tonexty'))
            fig.add_trace(go.Scatter(y=np.concatenate([full_data, arima_conf_int[:, 1]]),
                                     mode='lines',
                                     name='ARIMA Confidence Interval',
                                     fill='tonexty',
                                     fillcolor='rgba(255,0,0,0.2)',
                                     line=dict(color='red', width=0)))

    if avg_predictions is not None:
        # Overlaying the averaged predictions
        fig.add_trace(go.Scatter(y=np.concatenate([full_data, avg_predictions]),
                                 mode='lines',
                                 name='Averaged Predictions',
                                 line=dict(color='orange', dash='dot')))

    # Adding sentiment as annotation
    if sentiment:
        fig.add_annotation(
            dict(
                text=sentiment,
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.05,
                y=0.95,
                font=dict(size=16, color="black"),
                bgcolor="yellow",
                opacity=0.7
            )
        )

    # Formatting the plot
    fig.update_layout(title=f"{tick} Historical vs Predictions",
                      yaxis_title="Stock Price",
                      plot_bgcolor='#FFFFFF',
                      xaxis=dict(gridcolor='lightgrey'),
                      yaxis=dict(gridcolor='lightgrey'),
                      margin=dict(l=0, r=0, t=30, b=0),
                      legend_orientation="h",
                      legend=dict(x=.5, xanchor="center"))

    fig.show()

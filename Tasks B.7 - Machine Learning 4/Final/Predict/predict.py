import warnings
warnings.filterwarnings('ignore')

from datetime import datetime, timedelta
import pmdarima as pm
from dataset import create_predict_datasets
import numpy as np
from news import YahooFinanceNews

def get_sentitment(results):
    neutral_score, positive_score, negative_score = 0.0, 0.0, 0.0

    for result in results:
        sentiment = result['label']
        score = result['score']
        
        if sentiment == 'neutral':
            neutral_score += score
        elif sentiment == 'positive':
            positive_score += score
        elif sentiment == 'negative':
            negative_score += score

    # Check which sentiment has the highest cumulative score
    if neutral_score > positive_score and neutral_score > negative_score:
        overall_sentiment = 'neutral'
    elif positive_score > neutral_score and positive_score > negative_score:
        overall_sentiment = 'positive'
    else:
        overall_sentiment = 'negative'

    print(f"Overall Sentiment: {overall_sentiment}")

    # Investment decision based on sentiment (example logic)
    if overall_sentiment == 'positive':
        text = "Consider buying."
    elif overall_sentiment == 'negative':
        text = "Consider selling or holding."
    else:
        text = "Stay neutral, await further signals."
        
    return text

def predictions(lstm_model, 
                finbert_nlp, 
                tick, 
                start_predict, 
                end_predict, 
                k=5, 
                step_size=30,
                start_p=0, 
                max_p=5, 
                start_q=0, 
                max_q=5, 
                m=7, 
                seasonal=True,
                multisteps=False):

    # Processing data
    predict_df, scaled_data, scaler, x_test, y_test = create_predict_datasets(start_predict, 
                                                                              end_predict, 
                                                                              tick, 
                                                                              step_size,
                                                                              n_steps=k,
                                                                              multisteps=multisteps)
    
    # Actual Prices
    actual_prices = predict_df['Close'].values
    
    # LSTM Past Predictions
    lstm_past_predictions = lstm_model.predict(x_test)
    lstm_past_predictions = scaler.inverse_transform(lstm_past_predictions)
    
    # Arima Predictions
    arima_model = pm.auto_arima(actual_prices,
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
    
    # Arima Predictions with confidence intervals
    arima_predictions, arima_conf_int = arima_model.predict(n_periods=k, return_conf_int=True)
    arima_upper_bound = arima_conf_int[:, 1]
    arima_lower_bound = arima_conf_int[:, 0]
    
    # Get news
    news = YahooFinanceNews()
    news_summary = news.fetch_news(ticker=tick, days=k)
    text = []
    # Printing titles
    for summary in news_summary:
        text.append(summary.summary)
    
    # Finbert analysing 
    results = finbert_nlp(text)
    for result in results:
        print(result)

    sentiment = get_sentitment(results)
    print(f"Suggestion: {sentiment}")
    
    # Future forecast
    input_data = x_test[-1]
    lstm_future_predictions = lstm_model.predict(input_data.reshape(1, -1, x_test.shape[-1]))
    
    # Inversely scaling
    lstm_future_predictions = scaler.inverse_transform(lstm_future_predictions)

    # Getting the last known date from the dataset and generating future dates
    last_known_date = datetime.strptime(end_predict, '%Y-%m-%d')
    future_dates = [last_known_date + timedelta(days=i) for i in range(1, k + 1)]

    for i in range(k):
        # LSTM prediction for the day
        lstm_predicted_price = lstm_future_predictions[0, i]

        # ARIMA prediction and confidence interval for the day
        arima_predicted_price = arima_predictions[i]
        arima_upper = arima_upper_bound[i]
        arima_lower = arima_lower_bound[i]

        # Print predictions
        current_date = future_dates[i]
        print(f"Date: {current_date.strftime('%Y-%m-%d')}")
        print(f"LSTM Predicted Price: {lstm_predicted_price:.2f}")
        print(f"ARIMA Predicted Price: {arima_predicted_price:.2f} (Upper: {arima_upper:.2f}, Lower: {arima_lower:.2f})")        
    
    avg_predictions = average_predictions(lstm_future_predictions[0], arima_predictions)
    return predict_df, actual_prices, sentiment, lstm_past_predictions, avg_predictions, lstm_future_predictions[0], arima_predictions, arima_conf_int


def average_predictions(lstm_predictions, arima_predictions):
    # Ensure lstm_predictions is a 1D array
    lstm_predictions = lstm_predictions.flatten()

    # Get overlapping period
    print('Get Overlapping Period')
    min_length = min(len(lstm_predictions), len(arima_predictions))  # Check the shortest length
    arima_predictions = arima_predictions[:min_length]

    print(f'LSTM shape after slicing: {lstm_predictions.shape}')
    print(f'ARIMA shape after slicing: {arima_predictions.shape}')

    # Average predictions
    print('Average Predictions')
    avg_predictions = (lstm_predictions + arima_predictions) / 2  # We don't need np.newaxis anymore because we're not adding 3 arrays together

    print(f'avg_predictions shape: {avg_predictions.shape}')

    # Checking for NaN
    print("NaN in LSTM predictions:", np.isnan(lstm_predictions).any())
    print("NaN in ARIMA predictions:", np.isnan(arima_predictions).any())

    return avg_predictions

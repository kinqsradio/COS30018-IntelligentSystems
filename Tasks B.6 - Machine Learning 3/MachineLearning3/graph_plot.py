from initialized import *
import mplfinance as mpf
import matplotlib.pyplot as plt
import pandas as pd

'''Plt Candlestic'''
def plot_candlestick(input_df, n=1):
    
    # Copy to avoid warnings
    input_df = input_df.copy()

    # Resampling the data for n trading days
    if n > 1:
        input_df = input_df.resample(f'{n}D').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()

    # Add moving averages to the dataframe
    input_df['MA50'] = input_df['Close'].rolling(window=50).mean()
    input_df['MA100'] = input_df['Close'].rolling(window=100).mean()
    input_df['MA200'] = input_df['Close'].rolling(window=200).mean()

    # Create a custom plot for the moving averages
    ap = []
    if input_df['MA50'].dropna().shape[0] > 0:
        aligned_MA50 = input_df['MA50'].dropna().reindex(input_df.index, fill_value=None)
        ap.append(mpf.make_addplot(aligned_MA50, color='orange'))
    if input_df['MA100'].dropna().shape[0] > 0:
        aligned_MA100 = input_df['MA100'].dropna().reindex(input_df.index, fill_value=None)
        ap.append(mpf.make_addplot(aligned_MA100, color='green'))
    if input_df['MA200'].dropna().shape[0] > 0:
        aligned_MA200 = input_df['MA200'].dropna().reindex(input_df.index, fill_value=None)
        ap.append(mpf.make_addplot(aligned_MA200, color='magenta'))

    # Plot the candlestick chart
    mpf.plot(input_df, type='candle', style='charles',
             title=f"{ticker} Candlestick Chart",
             ylabel='Price',
             volume=True,
             ylabel_lower='Volume',
             addplot=ap,
             show_nontrading=True)
    
'''Plot box plot'''
def plot_boxplot(input_df, n=1, k=10):
    # Copy to avoid warnings
    input_df = input_df.copy()

    # Resampling the data for n trading days
    if n > 1:
        input_df = input_df.resample(f'{n}D').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()

    # Prepare data for boxplot
    box_data = []
    labels = []
    for idx, row in input_df.iterrows():
        box_data.append([row['Low'], row['Open'], row['Close'], row['High']])
        labels.append(idx.strftime('%Y-%m-%d'))

    # Plotting
    fig, ax = plt.subplots()
    ax.boxplot(box_data, vert=True, patch_artist=True)
    ax.set_title(f'{ticker} Boxplot Chart')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')

    # Set x-axis labels and ticks
    ax.set_xticks(range(1, len(labels) + 1, k))
    ax.set_xticklabels(labels[::k], rotation=90)

    plt.show()
    

'''Plot Predicted with Testdata'''
def plot_candlestick_predicted(input_df, predicted_prices, n=4):
    # Work with a deep copy to avoid modifying the original dataframe
    input_df = input_df.copy()

    # Resampling the data for n trading days
    if n > 1:
        input_df = input_df.resample(f'{n}D').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()

    # Add moving averages to the dataframe
    input_df['MA50'] = input_df['Close'].rolling(window=50).mean()
    input_df['MA100'] = input_df['Close'].rolling(window=100).mean()
    input_df['MA200'] = input_df['Close'].rolling(window=200).mean()

    print(f"Length of MA50: {input_df['MA50'].dropna().shape[0]}")
    print(f"Length of MA100: {input_df['MA100'].dropna().shape[0]}")
    print(f"Length of MA200: {input_df['MA200'].dropna().shape[0]}")


    # Convert the index to a DatetimeIndex
    input_df.index = pd.to_datetime(input_df.index)

    # Plot the last the last predicted candles
    df_plot = input_df[-len(predicted_prices):].copy()

    print(f"Length of df_plot: {len(df_plot)}")
    print(f"Length of predicted_prices: {len(predicted_prices)}")

    # Add Predicted Prices
    # Check if predicted_prices is 2D and reshape if necessary
    if predicted_prices.ndim == 2:
        predicted_prices = predicted_prices.reshape(-1)

    # Ensure the length of predicted_prices
    # matches the length of the sliced portion of the DataFrame
    if len(predicted_prices) > len(df_plot):
        predicted_prices = predicted_prices[-len(df_plot):]  # Take only the last predictions
    elif len(predicted_prices) < len(df_plot):
        print(f"Length mismatch: predicted_prices has length {len(predicted_prices)} but df_plot has length {len(df_plot)}")
        # Align the predictions to the end of df_plot
        start_idx = len(df_plot) - len(predicted_prices)
        df_plot = df_plot[start_idx:].copy()

    df_plot['Predicted'] = predicted_prices

    # Create a custom plot for the predicted prices
    ap = []
    if input_df['MA50'].dropna().shape[0] > 0:
        aligned_MA50 = input_df['MA50'].dropna().reindex(input_df.index, fill_value=None)
        ap.append(mpf.make_addplot(aligned_MA50, color='orange'))
    if input_df['MA100'].dropna().shape[0] > 0:
        aligned_MA100 = input_df['MA100'].dropna().reindex(input_df.index, fill_value=None)
        ap.append(mpf.make_addplot(aligned_MA100, color='green'))
    if input_df['MA200'].dropna().shape[0] > 0:
        aligned_MA200 = input_df['MA200'].dropna().reindex(input_df.index, fill_value=None)
        ap.append(mpf.make_addplot(aligned_MA200, color='magenta'))

    ap.append(mpf.make_addplot(df_plot['Predicted'], color='red', linestyle='dashed'))

    # Plot the candlestick chart
    mpf.plot(df_plot, type='candle', style='charles',
             title=f"Candlestick Chart",
             ylabel='Price',
             volume=False,
             addplot=ap,
             show_nontrading=False)
    
'''Plot Full Candlesticks for Train, Test and Predicted Prices'''
def plot_candlestick_full(train_df, test_df, predicted_prices, n=1):
    # Create deep copies to avoid modifying the original dataframes
    train_df = train_df.copy()
    test_df = test_df.copy()

    # Resampling the data for n trading days
    if n > 1:
        train_df = train_df.resample(f'{n}D').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()

        test_df = test_df.resample(f'{n}D').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()

        if train_df.empty or test_df.empty:
          raise ValueError("Resampling resulted in an empty DataFrame. Try a smaller value of n.")

        # Adjust the length of predicted_prices to match test_df
        eff_length = len(test_df)
        predicted_prices = predicted_prices[-eff_length:]

    # Compute moving averages for the training data
    train_df['MA50'] = train_df[price_value].rolling(window=50).mean()
    train_df['MA100'] = train_df[price_value].rolling(window=100).mean()
    train_df['MA200'] = train_df[price_value].rolling(window=200).mean()

    # Compute moving averages for the test data
    test_df['MA50'] = test_df[price_value].rolling(window=50).mean()
    test_df['MA100'] = test_df[price_value].rolling(window=100).mean()
    test_df['MA200'] = test_df[price_value].rolling(window=200).mean()

    # Check if predicted_prices is 2D and reshape if necessary
    if predicted_prices.ndim == 2:
        predicted_prices = predicted_prices.reshape(-1)

    # Ensure the length of predicted_prices matches the length of the test data
    if len(predicted_prices) != len(test_df):
        raise ValueError(f"Length mismatch: predicted_prices has length {len(predicted_prices)} but test_df has length {len(test_df)}")

    # Add predicted prices to the test dataframe
    test_df['Predicted'] = predicted_prices

    # Concatenate train and test dataframes to form a complete dataframe for plotting
    df_plot = pd.concat([train_df, test_df])

    # Convert the index to a DatetimeIndex
    df_plot.index = pd.to_datetime(df_plot.index)

    # Create a custom plot for the predicted prices and moving averages
    ap = []
    if df_plot['MA50'].dropna().shape[0] > 0:
        aligned_MA50 = df_plot['MA50'].dropna().reindex(df_plot.index, fill_value=None)
        ap.append(mpf.make_addplot(aligned_MA50, color='orange'))

    if df_plot['MA100'].dropna().shape[0] > 0:
        aligned_MA100 = df_plot['MA100'].dropna().reindex(df_plot.index, fill_value=None)
        ap.append(mpf.make_addplot(aligned_MA100, color='green'))

    if df_plot['MA200'].dropna().shape[0] > 0:
        aligned_MA200 = df_plot['MA200'].dropna().reindex(df_plot.index, fill_value=None)
        ap.append(mpf.make_addplot(aligned_MA200, color='magenta'))

    ap.append(mpf.make_addplot(df_plot['Predicted'], color='red', linestyle='dashed'))

    # Plot the candlestick chart
    mpf.plot(df_plot, type='candle', style='charles',
            title=f"{ticker} Candlestick Chart",
            ylabel='Price',
            volume=False,
            addplot=ap,
            show_nontrading=False)

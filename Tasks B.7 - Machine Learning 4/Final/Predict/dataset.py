import os 
import joblib
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import pandas_ta as ta
import pandas_datareader as web
import numpy as np
import yfinance as yf


DATA_DIR = 'datasets/'
PREPARED_DATA_DIR = 'prepared-data/'

# Double check directory
def ensure_directory_exists(dir_path):
  # If directory not exist => create
  if not os.path.isdir(dir_path):
      os.mkdir(dir_path)
      
# Save and Load utility functions
def save_object(obj, filename):
    with open(filename, 'wb') as f:
        joblib.dump(obj, f)

def load_object(filename):
    with open(filename, 'rb') as f:
        return joblib.load(f)

# Load Raw Data
def load_data(start_date, end_date, tick, source='yahoo'):
  ensure_directory_exists(DATA_DIR)
  CSV_FILE = os.path.join(DATA_DIR, f"RawData-from-{start_date}to-{end_date}-{tick}_stock_data.csv")


  # Check if CSV file exists
  # If exist => load
  # If not exist => download
  if os.path.exists(CSV_FILE):
      print('Loading Existing Data')
      data = pd.read_csv(CSV_FILE)
  else:
      print('Downloading Data')
      data = yf.download(tick, start_date, end_date, progress=False)
      data.to_csv(CSV_FILE)

  return data

# Data Validation
def data_validation(start_date, end_date, tick):
  PREPARED_DATA_FILE = os.path.join(PREPARED_DATA_DIR, f"PreparedData-from-{start_date}to-{end_date}-{tick}_stock_data.csv")
  CSV_FILE = os.path.join(DATA_DIR, f"RawData-from-{start_date}to-{end_date}-{tick}_stock_data.csv")
  ensure_directory_exists(PREPARED_DATA_DIR)


  if os.path.exists(PREPARED_DATA_FILE):
      print('Loading Prepared Data')
      df = pd.read_csv(PREPARED_DATA_FILE)
  else:
      print('Processing Raw Data')

      # Read Raw Data File
      df = pd.read_csv(CSV_FILE)

      df['Date'] = pd.to_datetime(df['Date'])

      df.set_index('Date', inplace=True)

      # Adding indicators
      df['RSI']=ta.rsi(df.Close, length=15)
      df['EMAF']=ta.ema(df.Close, length=20)
      df['EMAM']=ta.ema(df.Close, length=100)
      df['EMAS']=ta.ema(df.Close, length=150)

      df['Target'] = df['Adj Close']-df.Open
      df['Target'] = df['Target'].shift(-1)

      df['TargetClass'] = [1 if df.Target[i]>0 else 0 for i in range(len(df))]

      df['TargetNextClose'] = df['Adj Close'].shift(-1)

      # Handle NaN in shifted columns
      df['Target'].fillna(method='ffill', inplace=True)
      df['TargetNextClose'].fillna(method='ffill', inplace=True)

      # Handle NaN in indicators (backfill or forward fill as per your requirement)
      df.fillna(method='bfill', inplace=True)

      # Drop NaN issue in data
      df.dropna(inplace=True)

      # Drop Columns
      # df.drop(['Volume','Close', 'Date'], axis=1, inplace=True)

      # Export Prepared Data
      df.to_csv(PREPARED_DATA_FILE)

  return df

# Split Data by Date or Randomly
def split_data(df, split_ratio, split_by_date=True):
    if split_by_date:
        # Split by date
        train_size = int(len(df) * split_ratio)
        train_data = df.iloc[:train_size]
        test_data = df.iloc[train_size:]
    else:
        # Split Randomly
        train_data, test_data = train_test_split(df, test_size=1-split_ratio, random_state=42)

    print(f"Train Data Shape: {train_data.shape}")
    print(f"Test Data Shape: {test_data.shape}")

    return train_data, test_data

# Scaler
def scaler_features(input_data, scale=True):
    if scale:
        scaler = MinMaxScaler(feature_range=(0, 1))

        # Reshaping if input_data is a Series or 1D numpy array
        if len(input_data.shape) == 1:
            input_data = input_data.values.reshape(-1, 1)

        scaled_data = scaler.fit_transform(input_data)
        return scaled_data, scaler
    else:
        return input_data, None
    
# Create Predict Datasets
def create_predict_datasets(start_predict, end_predict, tick, step_size=30, n_steps=5, multisteps=False):
    
    # Download or Load Raw Data
    print(f"Fetching data from {start_predict} to {end_predict}")
    data = load_data(start_predict, end_predict, tick)

    print(f"Raw data shape: {data.shape}")
    print(data.head())

    # Data Validation
    df = data_validation(start_predict, end_predict, tick)

    print(f"Processed data shape: {df.shape}")
    print(df.head())

    # Define features and target
    feature_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'RSI', 'EMAF', 'EMAM', 'EMAS']
    target_column = 'TargetNextClose'

    # Preparing Datasets
    # Scaler for features
    scaled_data, train_feature_scaler = scaler_features(df[feature_columns])
    print("Scaled data shape:", scaled_data.shape)

    # Scaler for target
    scaled_target_train, scaler = scaler_features(df[target_column].values.reshape(-1, 1))

    x_test, y_test = [], []
    if multisteps:
        for i in range(step_size, len(scaled_data) - n_steps + 1):
            x_test.append(scaled_data[i-step_size:i])
            y_test.append(scaled_target_train[i:i+n_steps])
    else:
        for i in range(step_size, len(scaled_data)):
            x_test.append(scaled_data[i-step_size:i])
            y_test.append(scaled_target_train[i])

    x_test, y_test = np.array(x_test), np.array(y_test)
    print("x_test shape:", x_test.shape)

    # For data
    if not isinstance(data.index, pd.DatetimeIndex):
        if "Date" in data.columns:
            data['Date'] = pd.to_datetime(data['Date'])
            data.set_index('Date', inplace=True)

    # For df
    if not isinstance(df.index, pd.DatetimeIndex):
        if "Date" in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)


    return df, scaled_data, scaler, x_test, y_test
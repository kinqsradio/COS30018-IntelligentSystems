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
from datasets import Dataset


DATA_DIR = 'datasets/'
PREPARED_DATA_DIR = 'prepared-data/'

'''LSTM Datasets'''
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

# Create Train Datasets
def create_datasets(start_date, end_date, tick, step_size=30, n_steps=5, split_ratio=0.8, multisteps=False):
    
    # Download or Load Raw Data
    data = load_data(start_date, end_date, tick)

    # Data Validation
    df = data_validation(start_date, end_date, tick)

    PREPARED_TRAIN = os.path.join(PREPARED_DATA_DIR, f"{tick}_xytrain-from-{start_date}to-{end_date}-{tick}_prepared_data.npz")
    TRAIN_DATA_FILE = os.path.join(PREPARED_DATA_DIR, f"TrainData-from-{start_date}to-{end_date}-{tick}_stock_data.csv")
    TEST_DATA_FILE = os.path.join(PREPARED_DATA_DIR, f"TestData-from-{start_date}to-{end_date}-{tick}_stock_data.csv")
    SCALER_FEATURE_FILE = os.path.join(PREPARED_DATA_DIR, f"FeatureScaler-from-{start_date}to-{end_date}-{tick}.pkl")
    SCALER_TARGET_FILE = os.path.join(PREPARED_DATA_DIR, f"TargetScaler-from-{start_date}to-{end_date}-{tick}.pkl")
    TRAIN_ARRAY_FILE = os.path.join(PREPARED_DATA_DIR, f"{tick}_xytrain-from-{start_date}to-{end_date}_train_arrays.npz")
    TEST_ARRAY_FILE = os.path.join(PREPARED_DATA_DIR, f"{tick}_xytrain-from-{start_date}to-{end_date}_test_arrays.npz")

    if os.path.exists(TRAIN_DATA_FILE) and os.path.exists(TEST_DATA_FILE):
        print('Loading Existing Train and Test Data')
        train_data = pd.read_csv(TRAIN_DATA_FILE)
        test_data = pd.read_csv(TEST_DATA_FILE)

        print(f"Train Data Shape: {train_data.shape}")
        print(f"Test Data Shape: {test_data.shape}")

        # Load feature and target scalers
        train_feature_scaler = load_object(SCALER_FEATURE_FILE)
        train_target_scaler = load_object(SCALER_TARGET_FILE)

        # Load x_train, y_train, x_test, y_test
        train_arrays = np.load(TRAIN_ARRAY_FILE)
        x_train = train_arrays['x_train']
        y_train = train_arrays['y_train']

        test_arrays = np.load(TEST_ARRAY_FILE)
        x_test = test_arrays['x_test']
        y_test = test_arrays['y_test']

    else:
        print('Processing Train and Test Data')
        # Split Data
        train_data, test_data = split_data(df, split_ratio)

        # Define features and target
        feature_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'RSI', 'EMAF', 'EMAM', 'EMAS']
        target_column = 'TargetNextClose'

        # Preparing Train Datasets
        # Scaler for features
        scaled_data_train, train_feature_scaler = scaler_features(train_data[feature_columns])
        # Scaler for target
        scaled_target_train, train_target_scaler = scaler_features(train_data[target_column].values.reshape(-1, 1))

        x_train, y_train = [], []
        if multisteps:
            for i in range(step_size, len(scaled_data_train) - n_steps + 1):
                x_train.append(scaled_data_train[i-step_size:i])
                y_train.append(scaled_target_train[i:i+n_steps])
        else:
            for i in range(step_size, len(scaled_data_train)):
                x_train.append(scaled_data_train[i-step_size:i])
                y_train.append(scaled_target_train[i])

        x_train, y_train = np.array(x_train), np.array(y_train)
        print(f"x_train shape: {x_train.shape}")
        print(f"y_train shape: {y_train.shape}")


        # Preparing Test Datasets
        # Use the feature scaler to scale the test data
        scaled_data_test = train_feature_scaler.transform(test_data[feature_columns])
        # Use the target scaler to scale the test target
        scaled_target_test = train_target_scaler.transform(test_data[target_column].values.reshape(-1, 1))

        x_test, y_test = [], []
        if multisteps:
            for i in range(step_size, len(scaled_data_test) - n_steps + 1):
                x_test.append(scaled_data_test[i-step_size:i])
                y_test.append(scaled_target_test[i:i+n_steps])
        else:
            for i in range(step_size, len(scaled_data_test)):
                x_test.append(scaled_data_test[i-step_size:i])
                y_test.append(scaled_target_test[i])

        x_test, y_test = np.array(x_test), np.array(y_test)
        print(f"x_test shape: {x_test.shape}")
        print(f"y_test shape: {y_test.shape}")


        # Save train_data and test_data
        train_data.to_csv(TRAIN_DATA_FILE)
        test_data.to_csv(TEST_DATA_FILE)

        # Save feature and target scalers
        save_object(train_feature_scaler, SCALER_FEATURE_FILE)
        save_object(train_target_scaler, SCALER_TARGET_FILE)

        # Save x_train, y_train, x_test, y_test
        np.savez(TRAIN_ARRAY_FILE, x_train=x_train, y_train=y_train)
        np.savez(TEST_ARRAY_FILE, x_test=x_test, y_test=y_test)

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

    # For train_data
    if not isinstance(train_data.index, pd.DatetimeIndex):
        if "Date" in train_data.columns:
            train_data['Date'] = pd.to_datetime(train_data['Date'])
            train_data.set_index('Date', inplace=True)

    # For test_data
    if not isinstance(test_data.index, pd.DatetimeIndex):
        if "Date" in test_data.columns:
            test_data['Date'] = pd.to_datetime(test_data['Date'])
            test_data.set_index('Date', inplace=True)


    return data, df, train_data, test_data, train_feature_scaler, train_target_scaler, x_train, x_test, y_train, y_test

'''Finbert Datasets'''


# label2id = {
#     "positive": 0,
#     "negative": 1,
#     "neutral": 2
# }


def create_finbert_datasets(parquet_path):
    # Read in Parquet file dataset
    df = pd.read_parquet(parquet_path)
    
    # Drop NaN
    df = df.dropna()
    
    # Remap
    # remap = {
    #     0: 2,  # neutral
    #     1: 0,  # positive
    #     2: 1   # negative
    # }
    
    # Map the labels using remap
    # df['label'] = df['label'].map(remap)
    
    # Create a Dataset from the DataFrame
    dataset = Dataset.from_pandas(df)
    
    return dataset

def finbert_dataset_splitting(dataset):
    # Splitting the dataset into training and validation sets
    datasets = dataset.train_test_split(test_size=0.1, seed=42)
    
    train_dataset = datasets["train"]
    validation_dataset = datasets["test"]
    
    return train_dataset, validation_dataset


def finbert_datasets_mapping(dataset, tokenizer, debug=True):
    dataset = dataset.map(lambda e: tokenizer(e['text'], truncation=True, padding='max_length', max_length=128), batched=True)
    if debug: print(dataset)

    return dataset


def finbert_dataset_format(dataset, 
                   column_names=["input_ids", "token_type_ids", "attention_mask", "label"], 
                   format_type="torch",
                   debug=True):
    dataset.set_format(type=format_type, columns=column_names)
    
    if debug: print(dataset)
    
    return dataset
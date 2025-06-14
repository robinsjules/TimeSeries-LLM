import pandas as pd
import os
from pathlib import Path
from typing import Optional, Tuple, List
import logging
import numpy as np
from alpha_vantage.timeseries import TimeSeries
from sklearn.preprocessing import MinMaxScaler
import talib
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, data_dir: str = 'data'):
        """
        Initialize the DataLoader with the data directory path.
        
        Args:
            data_dir (str): Directory containing the data files
        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise ValueError(f"Data directory {data_dir} does not exist")
    
    def load_ticker_data(self, ticker: str) -> pd.DataFrame:
        """
        Load data for a specific ticker from CSV.
        
        Args:
            ticker (str): Stock ticker symbol
            
        Returns:
            pd.DataFrame: DataFrame with OHLCV data
        """
        file_path = self.data_dir / f"{ticker}.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"No data file found for {ticker}")
        
        # Load and preprocess data
        data = pd.read_csv(file_path, index_col=0, parse_dates=True)
        logger.info(f"Loaded {len(data)} rows of data for {ticker}")
        
        # Ensure all required columns exist
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"Missing required columns in {ticker} data")
        
        # Sort by date
        data = data.sort_index()
        
        # Check for missing values
        missing_values = data.isnull().sum()
        if missing_values.any():
            logger.warning(f"Found missing values in {ticker} data:\n{missing_values}")
        
        return data
    
    def get_train_val_split(self, data: pd.DataFrame, 
                          train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and validation sets.
        
        Args:
            data (pd.DataFrame): Input data
            train_ratio (float): Ratio of data to use for training
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Training and validation sets
        """
        split_idx = int(len(data) * train_ratio)
        train_data = data.iloc[:split_idx]
        val_data = data.iloc[split_idx:]
        
        logger.info(f"Split data into {len(train_data)} training and {len(val_data)} validation samples")
        return train_data, val_data
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare additional features for the model.
        
        Args:
            data (pd.DataFrame): Raw OHLCV data
            
        Returns:
            pd.DataFrame: Data with additional features
        """
        # Create a copy to avoid modifying the original
        data = data.copy()
        
        # Ensure we have a continuous business day index
        business_days = pd.date_range(start=data.index[0], end=data.index[-1], freq='B')
        data = data.reindex(business_days)
        
        # Forward fill missing values for OHLCV data
        ohlcv_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        data[ohlcv_columns] = data[ohlcv_columns].fillna(method='ffill')
        
        # Calculate returns
        data['Returns'] = data['Close'].pct_change()
        
        # Calculate moving averages
        data['MA5'] = data['Close'].rolling(window=5, min_periods=1).mean()
        data['MA20'] = data['Close'].rolling(window=20, min_periods=1).mean()
        
        # Calculate volatility
        data['Volatility'] = data['Returns'].rolling(window=20, min_periods=1).std()
        
        # Calculate RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # Calculate MACD
        macd, macd_signal, macd_hist = talib.MACD(data['Close'].values)
        data['MACD'] = macd
        data['MACD_Signal'] = macd_signal
        data['MACD_Hist'] = macd_hist
        
        # Calculate Bollinger Bands
        upper, middle, lower = talib.BBANDS(data['Close'].values, timeperiod=20)
        data['BB_Upper'] = upper
        data['BB_Middle'] = middle
        data['BB_Lower'] = lower
        
        # Calculate Stochastic Oscillator
        slowk, slowd = talib.STOCH(data['High'].values, data['Low'].values, data['Close'].values)
        data['Stoch_K'] = slowk
        data['Stoch_D'] = slowd
        
        # Calculate ADX (Average Directional Index)
        data['ADX'] = talib.ADX(data['High'].values, data['Low'].values, data['Close'].values, timeperiod=14)
        
        # Calculate ATR (Average True Range)
        data['ATR'] = talib.ATR(data['High'].values, data['Low'].values, data['Close'].values, timeperiod=14)
        
        # Calculate OBV (On Balance Volume)
        data['OBV'] = talib.OBV(data['Close'].values, data['Volume'].values)
        
        # Calculate Williams %R
        data['Williams_R'] = talib.WILLR(data['High'].values, data['Low'].values, data['Close'].values, timeperiod=14)
        
        # Calculate CCI (Commodity Channel Index)
        data['CCI'] = talib.CCI(data['High'].values, data['Low'].values, data['Close'].values, timeperiod=14)
        
        # Calculate ROC (Rate of Change)
        data['ROC'] = talib.ROC(data['Close'].values, timeperiod=10)
        
        # Calculate MOM (Momentum)
        data['MOM'] = talib.MOM(data['Close'].values, timeperiod=10)
        
        # Fill any remaining NaN values with appropriate methods
        data['Returns'] = data['Returns'].fillna(0)  # First day return is 0
        data['Volatility'] = data['Volatility'].fillna(0)  # First days volatility is 0
        data['RSI'] = data['RSI'].fillna(50)  # Neutral RSI is 50
        
        # Verify no NaN values remain
        if data.isna().sum().sum() > 0:
            logger.error("NaN values remain after feature preparation:")
            logger.error(data.isna().sum())
            raise ValueError("NaN values remain after feature preparation")
        
        logger.info(f"After feature preparation: {len(data)} rows")
        logger.info(f"Date range: {data.index[0]} to {data.index[-1]}")
        logger.info(f"Number of business days: {len(data)}")
        
        # Scale features
        scaler = MinMaxScaler(feature_range=(-1, 1))
        data_scaled = pd.DataFrame(
            scaler.fit_transform(data),
            index=data.index,
            columns=data.columns
        )
        
        return data_scaled

def load_and_prepare_data(ticker: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and prepare data for a given ticker.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Tuple of (training_data, validation_data)
    """
    try:
        # Load data
        logger.info(f"Loading and preparing data for {ticker}...")
        api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        if not api_key:
            raise ValueError("ALPHA_VANTAGE_API_KEY not found in environment variables")
            
        ts = TimeSeries(key=api_key, output_format='pandas')
        data, _ = ts.get_daily(symbol=ticker, outputsize='full')
        
        # Check if data is empty
        if data.empty:
            raise ValueError(f"No data received for ticker {ticker}")
            
        # Rename columns to match expected format
        data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # Sort by date in ascending order
        data = data.sort_index()
        
        # Filter to last 5 years of data
        end_date = data.index[-1]
        start_date = end_date - pd.DateOffset(years=5)
        data = data[start_date:end_date]
        
        logger.info(f"Loaded {len(data)} rows of data for {ticker}")
        logger.info(f"Date range: {data.index[0]} to {data.index[-1]}")
        
        # Create continuous business day index
        business_days = pd.date_range(start=data.index[0], end=data.index[-1], freq='B')
        data = data.reindex(business_days)
        
        # Forward fill missing values for OHLCV data
        data[['Open', 'High', 'Low', 'Close', 'Volume']] = data[['Open', 'High', 'Low', 'Close', 'Volume']].fillna(method='ffill')
        
        # Prepare features
        data = prepare_features(data)
        
        if data.empty:
            raise ValueError("No data remaining after feature preparation")
            
        logger.info(f"After feature preparation: {len(data)} rows")
        logger.info(f"Number of business days: {len(data)}")
        
        # Split data
        train_data, val_data = get_train_val_split(data)
        logger.info(f"Split data into {len(train_data)} training and {len(val_data)} validation samples")
        
        return train_data, val_data
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def prepare_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare features for the model.
    
    Args:
        data: DataFrame containing OHLCV data
        
    Returns:
        DataFrame with additional features
    """
    try:
        # Make a copy to avoid modifying the original
        data = data.copy()
        
        # Calculate returns
        data['Returns'] = data['Close'].pct_change()
        
        # Calculate volatility
        data['Volatility'] = data['Returns'].rolling(window=20, min_periods=1).std()
        
        # Calculate RSI
        data['RSI'] = talib.RSI(data['Close'].values, timeperiod=14)
        
        # Calculate MACD
        macd, macd_signal, macd_hist = talib.MACD(data['Close'].values)
        data['MACD'] = macd
        data['MACD_Signal'] = macd_signal
        data['MACD_Hist'] = macd_hist
        
        # Calculate Bollinger Bands
        upper, middle, lower = talib.BBANDS(data['Close'].values, timeperiod=20)
        data['BB_Upper'] = upper
        data['BB_Middle'] = middle
        data['BB_Lower'] = lower
        
        # Calculate Stochastic Oscillator
        slowk, slowd = talib.STOCH(data['High'].values, data['Low'].values, data['Close'].values)
        data['Stoch_K'] = slowk
        data['Stoch_D'] = slowd
        
        # Calculate ADX (Average Directional Index)
        data['ADX'] = talib.ADX(data['High'].values, data['Low'].values, data['Close'].values, timeperiod=14)
        
        # Calculate ATR (Average True Range)
        data['ATR'] = talib.ATR(data['High'].values, data['Low'].values, data['Close'].values, timeperiod=14)
        
        # Calculate OBV (On Balance Volume)
        data['OBV'] = talib.OBV(data['Close'].values, data['Volume'].values)
        
        # Calculate Williams %R
        data['Williams_R'] = talib.WILLR(data['High'].values, data['Low'].values, data['Close'].values, timeperiod=14)
        
        # Calculate CCI (Commodity Channel Index)
        data['CCI'] = talib.CCI(data['High'].values, data['Low'].values, data['Close'].values, timeperiod=14)
        
        # Calculate ROC (Rate of Change)
        data['ROC'] = talib.ROC(data['Close'].values, timeperiod=10)
        
        # Calculate MOM (Momentum)
        data['MOM'] = talib.MOM(data['Close'].values, timeperiod=10)
        
        # Fill any remaining NaN values with appropriate methods
        data = data.fillna(method='ffill').fillna(method='bfill')
        
        # Check if we have any data left after feature preparation
        if len(data) == 0:
            raise ValueError("No data remaining after feature preparation")
            
        # Scale features
        scaler = MinMaxScaler(feature_range=(-1, 1))
        data_scaled = pd.DataFrame(
            scaler.fit_transform(data),
            index=data.index,
            columns=data.columns
        )
        
        return data_scaled
        
    except Exception as e:
        logger.error(f"Error preparing features: {str(e)}")
        raise

def get_train_val_split(data: pd.DataFrame, val_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into training and validation sets.
    
    Args:
        data: DataFrame containing the data
        val_size: Size of validation set as a fraction of total data
        
    Returns:
        Tuple of (training_data, validation_data)
    """
    try:
        if len(data) == 0:
            raise ValueError("Cannot split empty dataset")
            
        split_idx = int(len(data) * (1 - val_size))
        train_data = data[:split_idx]
        val_data = data[split_idx:]
        
        if len(train_data) == 0 or len(val_data) == 0:
            raise ValueError("Split resulted in empty training or validation set")
            
        return train_data, val_data
        
    except Exception as e:
        logger.error(f"Error splitting data: {str(e)}")
        raise
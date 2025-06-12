import pandas as pd
import os
from pathlib import Path
from typing import Optional, Tuple
import logging
import numpy as np

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
        
        return data

def load_and_prepare_data(ticker: str, 
                         train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to load and prepare data for a ticker.
    
    Args:
        ticker (str): Stock ticker symbol
        train_ratio (float): Ratio of data to use for training
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Prepared training and validation sets
    """
    loader = DataLoader()
    data = loader.load_ticker_data(ticker)
    data = loader.prepare_features(data)
    train_data, val_data = loader.get_train_val_split(data, train_ratio)
    return train_data, val_data
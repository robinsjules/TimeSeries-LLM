import pandas as pd
from pathlib import Path
import time
from datetime import datetime
from alpha_vantage.timeseries import TimeSeries
from dotenv import load_dotenv
import os
import logging
from typing import List

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def download_ticker_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Download data for a single ticker using Alpha Vantage."""
    try:
        logger.info(f"Downloading data for {ticker}")
        
        # Get API key from environment
        api_key = os.getenv('ALPHAVANTAGE_API_KEY')
        if not api_key:
            raise ValueError("ALPHAVANTAGE_API_KEY not found in .env file")
        
        # Initialize Alpha Vantage client
        ts = TimeSeries(key=api_key, output_format='pandas')
        
        # Get daily data
        data, meta_data = ts.get_daily(symbol=ticker, outputsize='full')
        
        # Rename columns to standard format
        data = data.rename(columns={
            '1. open': 'Open',
            '2. high': 'High',
            '3. low': 'Low',
            '4. close': 'Close',
            '5. volume': 'Volume'
        })
        
        # Select relevant columns
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
        
        # Convert index to datetime
        data.index = pd.to_datetime(data.index)
        
        # Filter by date range
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        data = data.loc[(data.index >= start_dt) & (data.index <= end_dt)]
        
        # Sort by date (ascending)
        data = data.sort_index()
        
        return ticker, data
        
    except Exception as e:
        logger.error(f"Error downloading {ticker}: {str(e)}")
        return ticker, None

def download_multiple_tickers(tickers: List[str], start_date: str, end_date: str):
    """Download data for multiple tickers with rate limiting."""
    # Create data directory if it doesn't exist
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    for ticker in tickers:
        ticker, data = download_ticker_data(ticker, start_date, end_date)
        
        if data is not None:
            # Save to CSV
            output_path = data_dir / f"{ticker}.csv"
            data.to_csv(output_path)
            logger.info(f"Saved {len(data)} records for {ticker} to {output_path}")
        else:
            logger.warning(f"No data saved for {ticker}")
        
        # Alpha Vantage free tier has a limit of 5 API calls per minute
        # Wait 12 seconds between calls to be safe
        time.sleep(12)

if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]  # Add tickers
    start_date = "2021-01-01"
    end_date = "2024-01-01"
    
    download_multiple_tickers(tickers, start_date, end_date) 
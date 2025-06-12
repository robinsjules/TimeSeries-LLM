import torch
import pandas as pd
import numpy as np
import os
from darts import TimeSeries
from darts.models import TFTModel
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mape, mae, rmse
from pathlib import Path
import logging
from typing import Dict, Any, Tuple, List
import warnings
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

# Import our data loader
from data.data_loader import load_and_prepare_data

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress XGBoost warnings
warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')

# Force CPU usage
os.environ['CUDA_VISIBLE_DEVICES'] = ''
torch.set_num_threads(4)  # Limit CPU threads for stability

def validate_data(data: pd.DataFrame, name: str) -> None:
    """Validate data for NaN values and infinite values."""
    logger.info(f"\nValidating {name} data:")
    logger.info(f"Shape: {data.shape}")
    logger.info(f"NaN values: {data.isna().sum().sum()}")
    logger.info(f"Infinite values: {np.isinf(data.values).sum()}")
    logger.info(f"Value range: [{data.min().min():.2f}, {data.max().max():.2f}]")
    
    # Check for any remaining NaN values
    if data.isna().sum().sum() > 0:
        raise ValueError(f"Found NaN values in {name} data")
    
    # Check for infinite values
    if np.isinf(data.values).sum() > 0:
        raise ValueError(f"Found infinite values in {name} data")

def scale_features(data: pd.DataFrame) -> Tuple[pd.DataFrame, MinMaxScaler]:
    """Scale features using MinMaxScaler."""
    # Create a copy to avoid modifying the original data
    data_copy = data.copy()
    
    # Check for NaN values before scaling
    if data_copy.isna().sum().sum() > 0:
        logger.warning("NaN values found before scaling. Filling with forward fill then backward fill.")
        data_copy = data_copy.fillna(method='ffill').fillna(method='bfill')
    
    # Check for infinite values
    if np.isinf(data_copy.values).sum() > 0:
        logger.warning("Infinite values found before scaling. Replacing with NaN and filling.")
        data_copy = data_copy.replace([np.inf, -np.inf], np.nan)
        data_copy = data_copy.fillna(method='ffill').fillna(method='bfill')
    
    # Create scaler with feature range (-1, 1)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    
    # Scale the data
    scaled_values = scaler.fit_transform(data_copy)
    
    # Create DataFrame with scaled values
    scaled_data = pd.DataFrame(
        scaled_values,
        columns=data_copy.columns,
        index=data_copy.index
    )
    
    # Final validation
    if scaled_data.isna().sum().sum() > 0:
        raise ValueError("NaN values found after scaling")
    if np.isinf(scaled_data.values).sum() > 0:
        raise ValueError("Infinite values found after scaling")
    
    return scaled_data, scaler

def create_time_series(data: pd.DataFrame) -> TimeSeries:
    """
    Create a Darts TimeSeries object from a DataFrame.
    
    Args:
        data: DataFrame with datetime index and features
        
    Returns:
        TimeSeries object
    """
    try:
        # Ensure the index is datetime
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have a datetime index")
            
        # Create a copy of the data with the index as a column
        df = data.copy()
        df['time'] = df.index
        
        # Convert Close price to float32
        df['Close'] = df['Close'].astype('float32')
        
        # Create TimeSeries object
        series = TimeSeries.from_dataframe(
            df,
            time_col='time',
            value_cols=['Close'],
            fill_missing_dates=True,
            freq='B'  # Business day frequency
        )
        
        # Validate the series
        if len(series) == 0:
            raise ValueError("Created TimeSeries is empty")
            
        logger.info(f"Created TimeSeries with {len(series)} time steps")
        logger.info(f"Time range: {series.time_index[0]} to {series.time_index[-1]}")
        logger.info(f"Frequency: {series.freq}")
        
        return series
        
    except Exception as e:
        logger.error(f"Error creating TimeSeries: {str(e)}")
        raise

def plot_training_data(train_series: TimeSeries, val_series: TimeSeries) -> None:
    """Plot training and validation data for visual inspection."""
    plt.figure(figsize=(15, 5))
    train_series.plot(label='Training')
    val_series.plot(label='Validation')
    plt.title('Training and Validation Data')
    plt.legend()
    plt.savefig('training_data.png')
    plt.close()

def validate_predictions(predictions: TimeSeries, actual: TimeSeries) -> None:
    """Validate predictions and actual values."""
    logger.info(f"\nValidating predictions:")
    logger.info(f"Predictions length: {len(predictions)}")
    logger.info(f"Actual values length: {len(actual)}")
    
    # Convert to pandas for detailed analysis
    pred_df = predictions.pd_dataframe()
    actual_df = actual.pd_dataframe()
    
    # Check for NaN values
    pred_nan = pred_df.isna().sum().sum()
    actual_nan = actual_df.isna().sum().sum()
    
    if pred_nan > 0:
        logger.warning(f"Found {pred_nan} NaN values in predictions")
        logger.warning("First few NaN locations:")
        nan_locations = pred_df[pred_df.isna().any(axis=1)].index
        logger.warning(nan_locations[:5])
    
    if actual_nan > 0:
        logger.warning(f"Found {actual_nan} NaN values in actual values")
        logger.warning("First few NaN locations:")
        nan_locations = actual_df[actual_df.isna().any(axis=1)].index
        logger.warning(nan_locations[:5])
    
    # Log value ranges
    logger.info(f"Predictions range: [{pred_df.min().min():.2f}, {pred_df.max().max():.2f}]")
    logger.info(f"Actual values range: [{actual_df.min().min():.2f}, {actual_df.max().max():.2f}]")
    
    # Log some sample values
    logger.info("\nSample predictions:")
    logger.info(pred_df.head())
    logger.info("\nSample actual values:")
    logger.info(actual_df.head())
    
    # Check for extreme values
    pred_max = pred_df.max().max()
    pred_min = pred_df.min().min()
    if abs(pred_max) > 10 or abs(pred_min) > 10:
        logger.warning(f"Predictions contain extreme values: min={pred_min:.2f}, max={pred_max:.2f}")
    
    actual_max = actual_df.max().max()
    actual_min = actual_df.min().min()
    if abs(actual_max) > 10 or abs(actual_min) > 10:
        logger.warning(f"Actual values contain extreme values: min={actual_min:.2f}, max={actual_max:.2f}")

def initialize_model(model_params: Dict) -> TFTModel:
    """
    Initialize the TFT model with the given parameters.
    
    Args:
        model_params: Dictionary of model parameters
        
    Returns:
        Initialized TFT model
    """
    try:
        logger.info("\nInitializing TFT model with parameters:")
        for key, value in model_params.items():
            logger.info(f"{key}: {value}")
            
        # Corrected add_encoders structure
        add_encoders = {
            'datetime_attribute': {'past': ['dayofweek', 'month']},
            'position': {'past': ['relative']},
        }
        
        # Initialize model
        model = TFTModel(
            input_chunk_length=model_params['input_chunk_length'],
            output_chunk_length=model_params['output_chunk_length'],
            hidden_size=model_params['hidden_size'],
            lstm_layers=model_params['lstm_layers'],
            num_attention_heads=model_params['num_attention_heads'],
            dropout=model_params['dropout'],
            batch_size=model_params['batch_size'],
            n_epochs=model_params['n_epochs'],
            add_relative_index=model_params['add_relative_index'],
            add_encoders=add_encoders,
            random_state=42,
            pl_trainer_kwargs={
                'accelerator': 'cpu',  # Use CPU instead of MPS
                'devices': 1,
                'precision': '32-true'  # Force float32 precision
            }
        )
        
        return model
        
    except Exception as e:
        logger.error(f"Error initializing model: {str(e)}")
        raise

def cross_validate_model(data: pd.DataFrame, model_params: Dict, n_splits: int = 5) -> Dict:
    """
    Perform time series cross-validation.
    
    Args:
        data: DataFrame containing the data
        model_params: Dictionary of model parameters
        n_splits: Number of splits for cross-validation
        
    Returns:
        Dictionary containing cross-validation metrics
    """
    try:
        logger.info("\nPerforming cross-validation...")
        
        # Initialize cross-validation
        tscv = TimeSeriesSplit(n_splits=n_splits)
        metrics = []
        
        # Perform cross-validation
        for train_idx, val_idx in tscv.split(data):
            # Split data
            train_data = data.iloc[train_idx]
            val_data = data.iloc[val_idx]
            
            # Create TimeSeries objects
            train_series = create_time_series(train_data)
            val_series = create_time_series(val_data)
            
            # Initialize and train model
            model = initialize_model(model_params)
            model.fit(train_series)
            
            # Generate predictions
            predictions = model.predict(n=len(val_series))
            
            # Calculate metrics
            fold_metrics = {
                'mape': mape(val_series, predictions),
                'mae': mae(val_series, predictions),
                'rmse': rmse(val_series, predictions)
            }
            
            metrics.append(fold_metrics)
            
        # Calculate average metrics
        avg_metrics = {
            'mape': np.mean([m['mape'] for m in metrics]),
            'mae': np.mean([m['mae'] for m in metrics]),
            'rmse': np.mean([m['rmse'] for m in metrics])
        }
        
        logger.info("\nCross-validation metrics:")
        for metric, value in avg_metrics.items():
            logger.info(f"{metric.upper()}: {value:.4f}")
            
        return avg_metrics
        
    except Exception as e:
        logger.error(f"Error in cross-validation: {str(e)}")
        raise

def train_model(ticker: str) -> TFTModel:
    """
    Train the TFT model on the given ticker data.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Trained TFT model
    """
    try:
        # Load and prepare data
        train_data, val_data = load_and_prepare_data(ticker)
        
        # Define model parameters
        model_params = {
            'input_chunk_length': 30,
            'output_chunk_length': 7,
            'hidden_size': 16,
            'lstm_layers': 1,
            'num_attention_heads': 1,
            'dropout': 0.05,
            'batch_size': 8,
            'n_epochs': 5,
            'add_relative_index': True
        }
        
        # Perform cross-validation
        cv_metrics = cross_validate_model(train_data, model_params)
        
        # Create TimeSeries objects for final training
        train_series = create_time_series(train_data)
        val_series = create_time_series(val_data)
        
        # Initialize and train final model
        model = initialize_model(model_params)
        model.fit(train_series)
        
        # Generate predictions
        predictions = model.predict(n=len(val_series))
        
        # Calculate final metrics
        final_metrics = {
            'mape': mape(val_series, predictions),
            'mae': mae(val_series, predictions),
            'rmse': rmse(val_series, predictions)
        }
        
        logger.info("\nFinal model metrics:")
        for metric, value in final_metrics.items():
            logger.info(f"{metric.upper()}: {value:.4f}")
        
        return model
        
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        raise

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Train model
    model = train_model('AAPL')
    logger.info("\nModel training completed!")
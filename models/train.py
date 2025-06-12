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
from typing import Dict, Any, Tuple
import warnings
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

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
    """Convert DataFrame to Darts TimeSeries with validation."""
    # Ensure the index is datetime
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)
    
    # Validate data before creating TimeSeries
    validate_data(data, "TimeSeries input")
    
    # Log/check here
    logger.info(f"Data index frequency: {data.index.freq}")
    logger.info(f"Data index contains {len(data)} points from {data.index[0]} to {data.index[-1]}")
    
    # Convert Close price to float32
    close_data = data[['Close']].astype(np.float32)
    
    # Create TimeSeries with only the 'Close' column
    # Use fill_missing_dates=False to prevent NaN creation
    series = TimeSeries.from_dataframe(
        close_data,
        fill_missing_dates=False, 
        freq='B'  # Business day frequency
    )
    
    # Validate the created TimeSeries by converting to pandas and checking
    series_df = series.pd_dataframe()
    if series_df.isna().sum().sum() > 0:
        logger.error("NaN values found in TimeSeries after creation")
        logger.error("First few rows with NaN values:")
        nan_rows = series_df[series_df.isna().any(axis=1)]
        logger.error(nan_rows.head())
        raise ValueError("TimeSeries contains NaN values after creation")
    
    # Check for infinite values
    if np.isinf(series_df.values).sum() > 0:
        logger.error("Infinite values found in TimeSeries after creation")
        logger.error("First few rows with infinite values:")
        inf_rows = series_df[np.isinf(series_df.values).any(axis=1)]
        logger.error(inf_rows.head())
        raise ValueError("TimeSeries contains infinite values after creation")
    
    return series

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

def train_model(ticker: str = 'AAPL', 
                epochs: int = 10,
                model_params: Dict[str, Any] = None) -> TFTModel:
    """
    Train a TFT model on the specified ticker data.
    
    Args:
        ticker (str): Stock ticker symbol
        epochs (int): Number of training epochs
        model_params (dict): Optional model parameters
        
    Returns:
        TFTModel: Trained model
    """
    try:
        logger.info(f"Loading and preparing data for {ticker}...")
        train_data, val_data = load_and_prepare_data(ticker)
        
        # Log original data statistics
        logger.info("\nOriginal data statistics:")
        logger.info(f"Training data range: [{train_data['Close'].min():.2f}, {train_data['Close'].max():.2f}]")
        logger.info(f"Validation data range: [{val_data['Close'].min():.2f}, {val_data['Close'].max():.2f}]")
        
        # Validate and scale the data
        train_data, train_scaler = scale_features(train_data)
        val_data, val_scaler = scale_features(val_data)
        
        # Validate data
        validate_data(train_data, "Training")
        validate_data(val_data, "Validation")
        
        logger.info(f"Training data shape: {train_data.shape}")
        logger.info(f"Validation data shape: {val_data.shape}")
        
        # Convert to Darts TimeSeries
        train_series = create_time_series(train_data)
        val_series = create_time_series(val_data)
        
        # Plot the data for visual inspection
        plot_training_data(train_series, val_series)
        
        # Log data ranges
        logger.info(f"Training data range: {train_series.time_index[0]} to {train_series.time_index[-1]}")
        logger.info(f"Validation data range: {val_series.time_index[0]} to {val_series.time_index[-1]}")
        
        # Default model parameters with very conservative settings
        default_params = {
            'input_chunk_length': 30,
            'output_chunk_length': 7,
            'hidden_size': 8,  # Very small for stability
            'lstm_layers': 1,
            'num_attention_heads': 1,
            'dropout': 0.05,   # Minimal dropout
            'batch_size': 8,   # Very small batch size
            'n_epochs': epochs,
            'add_relative_index': True,
            'add_encoders': {
                'datetime_attribute': {
                    'past': ['dayofweek', 'month']
                },
                'position': {
                    'past': ['relative']
                },
                'custom': {},
                'cyclic': {}
            },
            'pl_trainer_kwargs': {
                "accelerator": "cpu",
                "enable_progress_bar": True,
                "enable_model_summary": True,
                "max_epochs": epochs,
                "precision": "32-true",  # Ensure float32 precision
                "gradient_clip_val": 0.01,  # Very aggressive gradient clipping
                "accumulate_grad_batches": 8,  # Increased for stability
                "deterministic": True,
                "devices": 1,
                "strategy": "auto",
                "check_val_every_n_epoch": 1,  # Check validation more frequently
                "log_every_n_steps": 1  # Log more frequently
            },
            'optimizer_kwargs': {
                'lr': 1e-4,  # Very small learning rate
                'weight_decay': 1e-6,  # Minimal weight decay
                'eps': 1e-7  # Increased epsilon
            }
        }
        
        # Update with custom parameters if provided
        if model_params:
            default_params.update(model_params)
        
        logger.info("\nInitializing TFT model with parameters:")
        for key, value in default_params.items():
            if key != 'pl_trainer_kwargs' and key != 'optimizer_kwargs':
                logger.info(f"{key}: {value}")
        
        model = TFTModel(**default_params)
        
        logger.info("\nTraining model...")
        model.fit(
            series=train_series,
            val_series=val_series,
            verbose=True
        )
        
        # Evaluate model
        logger.info("\nEvaluating model...")
        predictions = model.predict(n=len(val_series))
        
        # Validate predictions
        validate_predictions(predictions, val_series)
        
        # Calculate metrics
        try:
            mape_value = mape(val_series, predictions)
            mae_value = mae(val_series, predictions)
            rmse_value = rmse(val_series, predictions)
            
            metrics = {
                'MAPE': mape_value,
                'MAE': mae_value,
                'RMSE': rmse_value
            }
            
            logger.info("\nModel metrics:")
            for metric, value in metrics.items():
                logger.info(f"{metric}: {value:.2f}")
        except Exception as e:
            logger.error(f"\nError calculating metrics: {str(e)}")
            logger.error("Raw predictions and actual values:")
            logger.error(f"Predictions: {predictions.values()}")
            logger.error(f"Actual: {val_series.values()}")
        
        # Create models directory if it doesn't exist
        models_dir = Path('models')
        models_dir.mkdir(exist_ok=True)
        
        # Save model
        model_path = str(models_dir / f"{ticker}_tft_model.pt")
        model.save(model_path)
        logger.info(f"\nModel saved to {model_path}")
        
        return model
        
    except Exception as e:
        logger.error(f"\nError during model training: {str(e)}")
        raise

if __name__ == "__main__":
    # Train model with default parameters
    model = train_model(epochs=5)
    print("\nModel training completed!")
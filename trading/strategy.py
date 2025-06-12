import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from darts import TimeSeries
import logging

logger = logging.getLogger(__name__)

def generate_signals(predictions: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """
    Generate trading signals based on model predictions.
    
    Args:
        predictions: DataFrame containing model predictions
        threshold: Signal threshold for generating buy/sell signals
        
    Returns:
        DataFrame containing trading signals
    """
    try:
        # Calculate price changes
        price_changes = predictions['Close'].pct_change()
        
        # Generate signals
        signals = pd.DataFrame(index=predictions.index)
        signals['signal'] = 0  # 0: hold, 1: buy, -1: sell
        
        # Buy signal: predicted price increase above threshold
        signals.loc[price_changes > threshold, 'signal'] = 1
        
        # Sell signal: predicted price decrease below -threshold
        signals.loc[price_changes < -threshold, 'signal'] = -1
        
        # Add prediction and actual price
        signals['predicted_price'] = predictions['Close']
        
        logger.info(f"Generated {len(signals[signals['signal'] != 0])} trading signals")
        return signals
        
    except Exception as e:
        logger.error(f"Error generating signals: {str(e)}")
        raise

def calculate_position_size(signals: pd.DataFrame,
                          capital: float,
                          risk_per_trade: float = 0.02) -> pd.DataFrame:
    """
    Calculate position sizes for each trade.
    
    Args:
        signals: DataFrame containing trading signals
        capital: Available capital
        risk_per_trade: Maximum risk per trade as a fraction of capital
        
    Returns:
        DataFrame with added position size information
    """
    signals = signals.copy()
    
    # Calculate position sizes
    signals['position_size'] = capital * risk_per_trade * signals['strength']
    
    # Adjust for signal direction
    signals['position_size'] = signals['position_size'] * signals['signal']
    
    return signals

def run_backtest(signals: pd.DataFrame, 
                validation_data: pd.DataFrame,
                position_size: float = 0.5,
                initial_capital: float = 10000.0) -> Dict[str, Any]:
    """
    Run backtest on trading signals.
    
    Args:
        signals: DataFrame containing trading signals
        validation_data: DataFrame containing actual price data
        position_size: Size of each position as fraction of capital
        initial_capital: Initial capital for backtest
        
    Returns:
        Dictionary containing backtest results
    """
    try:
        # Initialize backtest variables
        capital = initial_capital
        position = 0
        equity_curve = []
        
        # Run backtest
        for date, row in signals.iterrows():
            if row['signal'] == 1 and position == 0:  # Buy signal
                position = (capital * position_size) / validation_data.loc[date, 'Close']
                capital -= position * validation_data.loc[date, 'Close']
            elif row['signal'] == -1 and position > 0:  # Sell signal
                capital += position * validation_data.loc[date, 'Close']
                position = 0
            
            # Calculate current equity
            current_equity = capital + (position * validation_data.loc[date, 'Close'])
            equity_curve.append(current_equity)
        
        # Convert equity curve to DataFrame
        equity_curve = pd.Series(equity_curve, index=signals.index)
        
        # Calculate metrics
        returns = equity_curve.pct_change().dropna()
        total_return = (equity_curve[-1] / initial_capital) - 1
        sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std()
        max_drawdown = (equity_curve / equity_curve.cummax() - 1).min()
        
        results = {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'equity_curve': equity_curve
        }
        
        logger.info(f"Backtest completed with {total_return:.2%} total return")
        return results
        
    except Exception as e:
        logger.error(f"Error running backtest: {str(e)}")
        raise
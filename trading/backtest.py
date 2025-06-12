import pandas as pd
import numpy as np
from typing import Dict, Any
from datetime import datetime

def run_backtest(data: pd.DataFrame,
                signals: pd.DataFrame,
                initial_capital: float = 10000.0) -> Dict[str, Any]:
    """
    Run backtest on the trading strategy.
    
    Args:
        data: DataFrame containing historical price data
        signals: DataFrame containing trading signals
        initial_capital: Initial capital for the backtest
        
    Returns:
        Dictionary containing backtest results and metrics
    """
    # Initialize portfolio
    portfolio = pd.DataFrame(index=signals.index)
    portfolio['capital'] = initial_capital
    portfolio['position'] = 0
    portfolio['holdings'] = 0
    portfolio['cash'] = initial_capital
    
    # Get the last known price from historical data
    last_price = data['Close'].iloc[-1]
    
    # Simulate trades
    for i in range(len(signals)):
        signal = signals.iloc[i]
        price = signal['predicted_price']
        
        if i == 0:
            prev_position = 0
        else:
            prev_position = portfolio['position'].iloc[i-1]
        
        # Calculate new position
        if signal['signal'] == 1:  # Buy
            new_position = 1
        elif signal['signal'] == -1:  # Sell
            new_position = -1
        else:  # Hold
            new_position = prev_position
        
        # Update portfolio
        portfolio.loc[signals.index[i], 'position'] = new_position
        portfolio.loc[signals.index[i], 'holdings'] = new_position * price
        portfolio.loc[signals.index[i], 'cash'] = portfolio['cash'].iloc[i-1] if i > 0 else initial_capital
        portfolio.loc[signals.index[i], 'capital'] = portfolio['cash'].iloc[i] + portfolio['holdings'].iloc[i]
    
    # Calculate returns
    portfolio['returns'] = portfolio['capital'].pct_change()
    portfolio['returns'] = portfolio['returns'].fillna(0)
    
    # Calculate metrics
    total_return = (portfolio['capital'].iloc[-1] - initial_capital) / initial_capital
    annual_return = (1 + total_return) ** (252 / len(portfolio)) - 1
    daily_returns = portfolio['returns']
    sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
    
    # Calculate drawdown
    portfolio['cummax'] = portfolio['capital'].cummax()
    portfolio['drawdown'] = (portfolio['capital'] - portfolio['cummax']) / portfolio['cummax']
    max_drawdown = portfolio['drawdown'].min()
    
    # Calculate win rate
    trades = portfolio[portfolio['position'] != portfolio['position'].shift(1)]
    winning_trades = trades[trades['returns'] > 0]
    win_rate = len(winning_trades) / len(trades) if len(trades) > 0 else 0
    
    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'num_trades': len(trades),
        'portfolio': portfolio
    }

def generate_trade_summary(results: Dict[str, Any]) -> pd.DataFrame:
    """
    Generate a summary of all trades from the backtest.
    
    Args:
        results: Dictionary containing backtest results
        
    Returns:
        DataFrame containing trade summary
    """
    portfolio = results['portfolio']
    trades = portfolio[portfolio['position'] != portfolio['position'].shift(1)].copy()
    
    trades['trade_type'] = trades['position'].map({1: 'BUY', -1: 'SELL', 0: 'HOLD'})
    trades['profit_loss'] = trades['returns'] * trades['capital'].shift(1)
    trades['cumulative_pl'] = trades['profit_loss'].cumsum()
    
    return trades[['trade_type', 'profit_loss', 'cumulative_pl']]
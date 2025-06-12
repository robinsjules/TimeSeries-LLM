import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List
from darts import TimeSeries
import logging
import talib

logger = logging.getLogger(__name__)

class TradingStrategy:
    """Base class for trading strategies."""
    
    def __init__(self, name: str):
        self.name = name
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals."""
        raise NotImplementedError

class MACDStrategy(TradingStrategy):
    """MACD-based trading strategy."""
    
    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        super().__init__("MACD")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            # Calculate MACD
            macd, signal, hist = talib.MACD(
                data['Close'].values,
                fastperiod=self.fast_period,
                slowperiod=self.slow_period,
                signalperiod=self.signal_period
            )
            
            # Generate signals
            signals = pd.DataFrame(index=data.index)
            signals['signal'] = 0
            
            # Buy when MACD crosses above signal line
            signals.loc[hist > 0, 'signal'] = 1
            # Sell when MACD crosses below signal line
            signals.loc[hist < 0, 'signal'] = -1
            
            return signals
            
        except Exception as e:
            logger.error(f"Error in MACD strategy: {str(e)}")
            raise

class RSIStrategy(TradingStrategy):
    """RSI-based trading strategy."""
    
    def __init__(self, period: int = 14, overbought: float = 70, oversold: float = 30):
        super().__init__("RSI")
        self.period = period
        self.overbought = overbought
        self.oversold = oversold
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            # Calculate RSI
            rsi = talib.RSI(data['Close'].values, timeperiod=self.period)
            
            # Generate signals
            signals = pd.DataFrame(index=data.index)
            signals['signal'] = 0
            
            # Buy when RSI crosses below oversold
            signals.loc[rsi < self.oversold, 'signal'] = 1
            # Sell when RSI crosses above overbought
            signals.loc[rsi > self.overbought, 'signal'] = -1
            
            return signals
            
        except Exception as e:
            logger.error(f"Error in RSI strategy: {str(e)}")
            raise

class BollingerBandsStrategy(TradingStrategy):
    """Bollinger Bands-based trading strategy."""
    
    def __init__(self, period: int = 20, std_dev: float = 2.0):
        super().__init__("Bollinger Bands")
        self.period = period
        self.std_dev = std_dev
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            # Calculate Bollinger Bands
            upper, middle, lower = talib.BBANDS(
                data['Close'].values,
                timeperiod=self.period,
                nbdevup=self.std_dev,
                nbdevdn=self.std_dev
            )
            
            # Generate signals
            signals = pd.DataFrame(index=data.index)
            signals['signal'] = 0
            
            # Buy when price crosses below lower band
            signals.loc[data['Close'] < lower, 'signal'] = 1
            # Sell when price crosses above upper band
            signals.loc[data['Close'] > upper, 'signal'] = -1
            
            return signals
            
        except Exception as e:
            logger.error(f"Error in Bollinger Bands strategy: {str(e)}")
            raise

class CombinedStrategy(TradingStrategy):
    """Combined strategy using multiple indicators."""
    
    def __init__(self, strategies: List[TradingStrategy], weights: List[float] = None):
        super().__init__("Combined")
        self.strategies = strategies
        self.weights = weights if weights else [1.0/len(strategies)] * len(strategies)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            # Generate signals from each strategy
            all_signals = []
            for strategy in self.strategies:
                signals = strategy.generate_signals(data)
                all_signals.append(signals['signal'])
            
            # Combine signals
            combined_signals = pd.DataFrame(index=data.index)
            combined_signals['signal'] = 0
            
            for i, signals in enumerate(all_signals):
                combined_signals['signal'] += signals * self.weights[i]
            
            # Normalize signals
            combined_signals['signal'] = np.sign(combined_signals['signal'])
            
            return combined_signals
            
        except Exception as e:
            logger.error(f"Error in combined strategy: {str(e)}")
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
                data: pd.DataFrame,
                position_size: float = 0.5,
                initial_capital: float = 10000.0,
                transaction_cost: float = 0.001) -> Dict[str, Any]:
    """
    Run backtest on trading signals.
    
    Args:
        signals: DataFrame containing trading signals
        data: DataFrame containing price data
        position_size: Size of each position as fraction of capital
        initial_capital: Initial capital for backtest
        transaction_cost: Cost per transaction as a fraction
        
    Returns:
        Dictionary containing backtest results
    """
    try:
        # Initialize backtest variables
        capital = initial_capital
        position = 0
        equity_curve = []
        trades = []
        
        # Run backtest
        for date, row in signals.iterrows():
            price = data.loc[date, 'Close']
            
            if row['signal'] == 1 and position == 0:  # Buy signal
                # Calculate position size
                new_position = (capital * position_size) / price
                # Calculate transaction cost
                cost = new_position * price * transaction_cost
                # Update position and capital
                position = new_position
                capital -= (new_position * price + cost)
                # Record trade
                trades.append({
                    'date': date,
                    'type': 'buy',
                    'price': price,
                    'size': new_position,
                    'cost': cost
                })
            elif row['signal'] == -1 and position > 0:  # Sell signal
                # Calculate proceeds
                proceeds = position * price
                # Calculate transaction cost
                cost = proceeds * transaction_cost
                # Update position and capital
                capital += (proceeds - cost)
                position = 0
                # Record trade
                trades.append({
                    'date': date,
                    'type': 'sell',
                    'price': price,
                    'size': position,
                    'cost': cost
                })
            
            # Calculate current equity
            current_equity = capital + (position * price)
            equity_curve.append(current_equity)
        
        # Convert equity curve to Series
        equity_curve = pd.Series(equity_curve, index=signals.index)
        
        # Calculate metrics
        returns = equity_curve.pct_change().dropna()
        total_return = (equity_curve[-1] / initial_capital) - 1
        sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std()
        max_drawdown = (equity_curve / equity_curve.cummax() - 1).min()
        
        # Calculate additional metrics
        num_trades = len(trades)
        win_rate = len([t for t in trades if t['type'] == 'sell' and t['price'] > t['price']]) / num_trades if num_trades > 0 else 0
        avg_trade_return = total_return / num_trades if num_trades > 0 else 0
        
        results = {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'equity_curve': equity_curve,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'avg_trade_return': avg_trade_return,
            'trades': trades
        }
        
        logger.info(f"Backtest completed with {total_return:.2%} total return")
        return results
        
    except Exception as e:
        logger.error(f"Error running backtest: {str(e)}")
        raise
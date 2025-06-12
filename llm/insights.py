import os
from typing import Dict, Any
from darts import TimeSeries
import openai
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import logging

# Load environment variables
load_dotenv()

# Initialize OpenAI client
openai.api_key = os.getenv('OPENAI_API_KEY')

logger = logging.getLogger(__name__)

def generate_insights(predictions: pd.DataFrame, validation_data: pd.DataFrame) -> str:
    """
    Generate trading insights based on model predictions and validation data.
    
    Args:
        predictions: DataFrame containing model predictions
        validation_data: DataFrame containing actual price data
        
    Returns:
        String containing formatted insights
    """
    try:
        insights = []
        
        # Calculate prediction accuracy
        mape = np.mean(np.abs((validation_data['Close'] - predictions['Close']) / validation_data['Close'])) * 100
        rmse = np.sqrt(np.mean((validation_data['Close'] - predictions['Close'])**2))
        
        # Add accuracy metrics
        insights.append("## Model Performance")
        insights.append(f"- Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
        insights.append(f"- Root Mean Square Error (RMSE): {rmse:.2f}")
        
        # Analyze prediction trends
        pred_trend = predictions['Close'].pct_change().mean() * 100
        actual_trend = validation_data['Close'].pct_change().mean() * 100
        
        insights.append("\n## Market Analysis")
        insights.append(f"- Predicted trend: {pred_trend:.2f}% per day")
        insights.append(f"- Actual trend: {actual_trend:.2f}% per day")
        
        # Calculate volatility
        pred_vol = predictions['Close'].pct_change().std() * 100
        actual_vol = validation_data['Close'].pct_change().std() * 100
        
        insights.append(f"- Predicted volatility: {pred_vol:.2f}%")
        insights.append(f"- Actual volatility: {actual_vol:.2f}%")
        
        # Generate trading recommendations
        insights.append("\n## Trading Recommendations")
        
        if mape < 50:  # If predictions are reasonably accurate
            if pred_trend > 0:
                insights.append("- Consider long positions as the model predicts an upward trend")
            else:
                insights.append("- Consider short positions or staying in cash as the model predicts a downward trend")
            
            if pred_vol > actual_vol:
                insights.append("- Be cautious of higher than usual volatility")
            else:
                insights.append("- Market conditions appear relatively stable")
        else:
            insights.append("- Model predictions show high error rates, consider using other indicators")
            insights.append("- Exercise caution with trading decisions")
        
        # Add risk management advice
        insights.append("\n## Risk Management")
        insights.append("- Always use stop-loss orders to manage risk")
        insights.append("- Consider position sizing based on volatility")
        insights.append("- Monitor market conditions and adjust strategies accordingly")
        
        return "\n".join(insights)
        
    except Exception as e:
        logger.error(f"Error generating insights: {str(e)}")
        return "Error generating insights. Please check the logs for details."

def generate_trade_explanation(signal: Dict[str, Any]) -> str:
    """
    Generate explanation for a specific trading signal.
    
    Args:
        signal: Dictionary containing signal information
        
    Returns:
        String containing explanation of the trading signal
    """
    prompt = f"""
    Explain the following trading signal in simple terms:
    
    Signal Type: {'BUY' if signal['signal'] == 1 else 'SELL' if signal['signal'] == -1 else 'HOLD'}
    Predicted Price Change: {signal['price_change']:.1%}
    Confidence Level: {signal['confidence']}
    Signal Strength: {signal['strength']:.1%}
    
    Please provide:
    1. A clear explanation of the signal
    2. The reasoning behind it
    3. The level of confidence in the prediction
    """
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a trading expert explaining trading signals to investors."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.7
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Error generating trade explanation: {str(e)}" 
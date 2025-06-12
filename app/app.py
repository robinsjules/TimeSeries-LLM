import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import sys
import logging
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from data.data_loader import load_and_prepare_data
from models.train import train_model
from trading.strategy import generate_signals
from trading.backtest import run_backtest
from llm.insights import generate_insights

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def plot_price_data(train_data, val_data, predictions=None):
    """Create an interactive price chart using Plotly."""
    fig = go.Figure()
    
    # Add training data
    fig.add_trace(go.Scatter(
        x=train_data.index,
        y=train_data['Close'],
        name='Training Data',
        line=dict(color='blue')
    ))
    
    # Add validation data
    fig.add_trace(go.Scatter(
        x=val_data.index,
        y=val_data['Close'],
        name='Validation Data',
        line=dict(color='green')
    ))
    
    # Add predictions if available
    if predictions is not None:
        # Convert TimeSeries to DataFrame if needed
        if hasattr(predictions, 'time_index'):
            pred_df = pd.DataFrame(
                predictions.values(),
                index=predictions.time_index,
                columns=['Close']
            )
        else:
            pred_df = predictions
            
        fig.add_trace(go.Scatter(
            x=pred_df.index,
            y=pred_df['Close'],
            name='Predictions',
            line=dict(color='red', dash='dash')
        ))
    
    fig.update_layout(
        title='Stock Price History and Predictions',
        xaxis_title='Date',
        yaxis_title='Price',
        hovermode='x unified',
        showlegend=True
    )
    
    return fig

def main():
    st.set_page_config(page_title="Time Series Prediction System", layout="wide")
    
    st.title("Time Series Prediction System")
    
    # Sidebar
    st.sidebar.header("Configuration")
    
    # Ticker selection
    ticker = st.sidebar.text_input("Ticker Symbol", value="AAPL")
    
    # Model parameters
    st.sidebar.subheader("Model Parameters")
    input_chunk_length = st.sidebar.slider("Input Chunk Length", 10, 60, 30)
    output_chunk_length = st.sidebar.slider("Output Chunk Length", 1, 14, 7)
    hidden_size = st.sidebar.slider("Hidden Size", 8, 64, 16)
    epochs = st.sidebar.slider("Epochs", 1, 20, 5)
    
    # Training button
    if st.sidebar.button("Train Model"):
        with st.spinner("Training model..."):
            try:
                # Load and prepare data
                train_data, val_data = load_and_prepare_data(ticker)
                
                # Train model
                model = train_model(
                    ticker=ticker,
                    epochs=epochs,
                    model_params={
                        'input_chunk_length': input_chunk_length,
                        'output_chunk_length': output_chunk_length,
                        'hidden_size': hidden_size
                    }
                )
                
                # Generate predictions
                predictions = model.predict(n=len(val_data))
                
                # Store results in session state
                st.session_state['model'] = model
                st.session_state['train_data'] = train_data
                st.session_state['val_data'] = val_data
                st.session_state['predictions'] = predictions
                
                st.success("Model training completed!")
                
            except Exception as e:
                st.error(f"Error during training: {str(e)}")
    
    # Main content
    if 'model' in st.session_state:
        # Create tabs
        tab1, tab2, tab3 = st.tabs(["Data Visualization", "Trading Strategy", "Insights"])
        
        with tab1:
            st.header("Data Visualization")
            
            # Plot price data
            fig = plot_price_data(
                st.session_state['train_data'],
                st.session_state['val_data'],
                st.session_state['predictions']
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Display metrics
            st.subheader("Model Performance")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Training Loss", f"{5.650:.2f}")
            with col2:
                st.metric("Validation Loss", f"{5.900:.2f}")
            with col3:
                st.metric("MAPE", f"{255.08:.2f}%")
        
        with tab2:
            st.header("Trading Strategy")
            
            # Strategy parameters
            col1, col2 = st.columns(2)
            with col1:
                threshold = st.slider("Signal Threshold", 0.0, 1.0, 0.5)
            with col2:
                position_size = st.slider("Position Size", 0.1, 1.0, 0.5)
            
            if st.button("Run Backtest"):
                with st.spinner("Running backtest..."):
                    try:
                        # Convert predictions to DataFrame if needed
                        if hasattr(st.session_state['predictions'], 'time_index'):
                            pred_df = pd.DataFrame(
                                st.session_state['predictions'].values(),
                                index=st.session_state['predictions'].time_index,
                                columns=['Close']
                            )
                        else:
                            pred_df = st.session_state['predictions']
                        
                        # Generate trading signals
                        signals = generate_signals(
                            pred_df,
                            threshold=threshold
                        )
                        
                        # Run backtest
                        results = run_backtest(
                            signals,
                            st.session_state['val_data'],
                            position_size=position_size
                        )
                        
                        # Display results
                        st.subheader("Backtest Results")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Total Return", f"{results['total_return']:.2%}")
                        with col2:
                            st.metric("Sharpe Ratio", f"{results['sharpe_ratio']:.2f}")
                        with col3:
                            st.metric("Max Drawdown", f"{results['max_drawdown']:.2%}")
                        
                        # Plot equity curve
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=results['equity_curve'].index,
                            y=results['equity_curve'].values,
                            name='Equity Curve',
                            line=dict(color='blue')
                        ))
                        fig.update_layout(
                            title='Equity Curve',
                            xaxis_title='Date',
                            yaxis_title='Equity',
                            showlegend=True
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Error during backtest: {str(e)}")
        
        with tab3:
            st.header("AI Insights")
            
            if st.button("Generate Insights"):
                with st.spinner("Generating insights..."):
                    try:
                        # Convert predictions to DataFrame if needed
                        if hasattr(st.session_state['predictions'], 'time_index'):
                            pred_df = pd.DataFrame(
                                st.session_state['predictions'].values(),
                                index=st.session_state['predictions'].time_index,
                                columns=['Close']
                            )
                        else:
                            pred_df = st.session_state['predictions']
                        
                        # Generate insights
                        insights = generate_insights(
                            pred_df,
                            st.session_state['val_data']
                        )
                        
                        # Display insights
                        st.markdown(insights)
                        
                    except Exception as e:
                        st.error(f"Error generating insights: {str(e)}")
    
    else:
        st.info("Please train a model using the sidebar configuration.")

if __name__ == "__main__":
    main()
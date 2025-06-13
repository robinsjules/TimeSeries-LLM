# Time Series Prediction System

A comprehensive time series prediction and trading system that combines machine learning with trading strategies and AI-powered insights. The system uses the Temporal Fusion Transformer (TFT) model for time series forecasting and includes a user-friendly Streamlit interface.

## Features

- **Time Series Prediction**: Uses TFT model for accurate price predictions
- **Trading Strategy**: Implements multiple trading strategies (MACD, RSI, Bollinger Bands)
- **Backtesting**: Comprehensive backtesting system with performance metrics
- **AI Insights**: GPT-3.5-turbo powered market analysis and insights
- **Interactive UI**: User-friendly Streamlit interface for easy interaction
- **Data Management**: Automated data downloading and preprocessing

## Prerequisites

- Python 3.8 or higher
- Alpha Vantage API key (for market data)
- OpenAI API key (for insights generation)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/TimeSeries-LLM.git
cd TimeSeries-LLM
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the package:
```bash
pip install -e .
```

4. Create a `.env` file in the project root with your API keys:
```
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
OPENAI_API_KEY=your_openai_key
```

## Usage

1. Download historical data:
```bash
python scripts/download_data.py
```

2. Run the Streamlit app:
```bash
streamlit run app/app.py
```

3. Access the application at `http://localhost:8501`

## Application Components

### Data Management
- Automated data downloading from Alpha Vantage
- Data preprocessing and feature engineering
- Support for multiple tickers

### Model Training
- Temporal Fusion Transformer (TFT) model
- Customizable model parameters
- Cross-validation support
- Performance metrics tracking

### Trading Strategies
- MACD Strategy
- RSI Strategy
- Bollinger Bands Strategy
- Combined Strategy with customizable weights

### Backtesting
- Comprehensive performance metrics
- Equity curve visualization
- Trade summary generation
- Risk metrics calculation

### AI Insights
- GPT-3.5-turbo powered market analysis
- Performance insights
- Trading strategy recommendations

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Darts](https://github.com/unit8co/darts) for time series forecasting
- [Alpha Vantage](https://www.alphavantage.co/) for market data
- [Streamlit](https://streamlit.io/) for the web interface
- [OpenAI](https://openai.com/) for AI insights (GPT-3.5-turbo)
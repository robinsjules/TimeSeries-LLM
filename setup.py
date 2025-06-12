from setuptools import setup, find_packages

setup(
    name="timeseries-llm",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "darts>=0.25.0",
        "torch>=2.2.0",
        "pandas>=2.0.3",
        "numpy>=1.26.0",
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "alpha_vantage>=2.3.1",
        "streamlit>=1.28.0",
        "openai>=1.6.1",
        "python-dotenv>=1.0.0",
        "matplotlib>=3.7.2",
        "plotly>=5.18.0",
        "ta>=0.10.2",
    ],
    python_requires=">=3.8",
) 
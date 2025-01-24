import pandas as pd
import numpy as np
import yfinance as yf
import sys
import os
from return_analyzer import ReturnAnalyzer



# Configuration
benchmark_symbol = "^NSEI"
risk_free_rate = 0.03

# Generate sample portfolio returns
dates = pd.date_range(start="2020-01-01", end="2024-12-31", freq='D')
portfolio_returns = pd.Series(np.random.normal(0.0005, 0.01, len(dates)), index=dates)

# Fetch benchmark data
benchmark_data = yf.download(benchmark_symbol, start="2019-01-01", end="2024-12-31")
benchmark_returns = benchmark_data['Adj Close'].pct_change().dropna()



# Initialize analyzer
analyzer = ReturnAnalyzer(
    returns=portfolio_returns,
    benchmark=benchmark_returns,
    risk_free_rate=risk_free_rate
)

# Generate full report
analyzer.generate_full_report()
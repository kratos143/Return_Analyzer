import pandas as pd
import numpy as np
import yfinance as yf
import sys
import os
from return_analyzer import ReturnAnalyzer

benchmark_symbol = "^NSEI"

# Example returns data (replace with actual data)
dates = pd.date_range(start="2019-01-01", end="2022-12-31", freq='D')
portfolio_returns = pd.Series(np.random.randn(len(dates)) / 100, index=dates)

# Fetch historical data for the benchmark (NIFTY 50)
benchmark_data = yf.download(benchmark_symbol, start="2019-01-01", end="2022-12-31")
benchmark_returns = benchmark_data['Adj Close'].pct_change().dropna()

# Align the portfolio returns with the benchmark returns
portfolio_returns, benchmark_returns = portfolio_returns.align(benchmark_returns, join='inner')

# Initialize the ReturnAnalyzer with the portfolio returns and benchmark returns
analyzer = ReturnAnalyzer(portfolio_returns, benchmark_returns)

# Plot the monthly returns heatmap
analyzer.plot_monthly_returns_heatmap()

# Plot cumulative returns
analyzer.plot_cumulative_returns()

# Plot rolling volatility
analyzer.plot_rolling_volatility()

# Plot rolling Sharpe ratio
analyzer.plot_rolling_sharpe_ratio()
analyzer.generate_full_report()
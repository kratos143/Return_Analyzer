import pandas as pd
import numpy as np

def generate_example_data():
    """
    Generate example data for the ReturnAnalyzer.
    """
    # Generate example portfolio returns data
    dates = pd.date_range(start="2022-01-01", end="2022-12-31", freq='D')
    portfolio_returns = pd.Series(np.random.randn(len(dates)) / 100, index=dates)
    
    # Generate example benchmark returns data
    benchmark_returns = pd.Series(np.random.randn(len(dates)) / 100, index=dates)
    
    return portfolio_returns, benchmark_returns

def main():
    portfolio_returns, benchmark_returns = generate_example_data()
    
    print("Example Portfolio Returns Data:")
    print(portfolio_returns.head())
    
    print("\nExample Benchmark Returns Data:")
    print(benchmark_returns.head())
    
    print("\nData Format Instructions:")
    print("1. The data should be in the form of a pandas Series with a datetime index.")
    print("2. The portfolio returns and benchmark returns should be daily returns.")
    print("3. The returns should be in decimal form (e.g., 0.01 for 1%).")
    print("4. The datetime index should cover the same date range for both portfolio and benchmark returns.")
    print("5. Missing data should be handled appropriately (e.g., using forward fill or interpolation).")

if __name__ == "__main__":
    main()
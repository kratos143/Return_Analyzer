import pandas as pd
import numpy as np
import yfinance as yf

def generate_example_data():
    """
    Generate example data for the ReturnAnalyzer.
    
    Returns:
    --------
    tuple: (portfolio_returns, benchmark_returns)
        portfolio_returns: pd.Series
            Daily returns with datetime index
        benchmark_returns: pd.Series 
            Daily benchmark returns with datetime index
    """
    # Generate example portfolio returns data
    dates = pd.date_range(start="2022-01-01", end="2022-12-31", freq='D')
    portfolio_returns = pd.Series(
        np.random.normal(0.0005, 0.01, len(dates)), 
        index=dates,
        name='Portfolio Returns'
    )
    
    # Generate example benchmark returns data (e.g., NIFTY 50)
    benchmark_data = yf.download("AAPL", start="2022-01-01", end="2022-12-31")
    benchmark_returns = benchmark_data['Adj Close'].pct_change().dropna()
    benchmark_returns.name = 'Benchmark Returns'
    
    return portfolio_returns, benchmark_returns

def main():
    """Show example data format and usage."""
    portfolio_returns, benchmark_returns = generate_example_data()
    
    print("Required Data Format:")
    print("\n1. Portfolio Returns Series:")
    print("- Type:", type(portfolio_returns))
    print("- Index Type:", type(portfolio_returns.index))
    print("- Sample Data:")
    print(portfolio_returns.head())
    
    print("\n2. Benchmark Returns Series (Optional):")
    print("- Type:", type(benchmark_returns))
    print("- Index Type:", type(benchmark_returns.index))
    print("- Sample Data:")
    print(benchmark_returns.head())
    
    print("\n3. Risk-Free Rate:")
    print("- Type: float")
    print("- Example: risk_free_rate = 0.03  # 3% annual rate")
    
    print("\nUsage Example:")
    print("""
    from return_analyzer import ReturnAnalyzer
    
    # Initialize analyzer
    analyzer = ReturnAnalyzer(
        returns=portfolio_returns,
        benchmark=benchmark_returns,
        risk_free_rate=0.03
    )
    """)

if __name__ == "__main__":
    main()
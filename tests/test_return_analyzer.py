import pandas as pd
import numpy as np
from return_analyzer import ReturnAnalyzer

def test_return_analyzer():
    # Example returns data
    dates = pd.date_range(start="2022-01-01", end="2022-12-31", freq='D')
    returns = pd.Series(np.random.randn(len(dates)) / 100, index=dates)
    
    # Create an instance of ReturnAnalyzer
    analyzer = ReturnAnalyzer(returns=returns, risk_free_rate=0.02)
    
    # Calculate metrics
    metrics = analyzer.calculate_metrics()
    
    # Check if metrics are calculated correctly
    assert not metrics.empty, "Metrics calculation failed"
    assert 'Total Return' in metrics.index, "Total Return not calculated"
    assert 'CAGR' in metrics.index, "CAGR not calculated"
    assert 'Sharpe Ratio' in metrics.index, "Sharpe Ratio not calculated"
    assert 'Max Drawdown' in metrics.index, "Max Drawdown not calculated"

if __name__ == "__main__":
    test_return_analyzer()
    print("All tests passed!")
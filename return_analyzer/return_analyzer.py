import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional

class ReturnAnalyzer:
    """
    A comprehensive tool for analyzing financial returns data.
    """
    def __init__(self, returns: pd.Series, benchmark: Optional[pd.Series] = None, risk_free_rate: float = 0.0):
        """
        Initialize the ReturnAnalyzer with returns data.
        
        Parameters:
        -----------
        returns : pd.Series
            Daily returns series with datetime index
        benchmark : pd.Series, optional
            Benchmark returns series with datetime index
        risk_free_rate : float, optional
            Risk-free rate for performance metrics, default is 0.0
        """
        if benchmark is not None:
            self.returns, self.benchmark = returns.align(benchmark, join='inner')
        else:
            self.returns = returns
            self.benchmark = None
        self.risk_free_rate = risk_free_rate
        self.daily_rf = (1 + risk_free_rate) ** (1/252) - 1
        
        # Validate input data
        self._validate_data()
        
    def _validate_data(self):
        """Validate input data format and alignment."""
        if not isinstance(self.returns.index, pd.DatetimeIndex):
            raise ValueError("Returns must have a datetime index")
        
        if self.benchmark is not None:
            if not isinstance(self.benchmark.index, pd.DatetimeIndex):
                raise ValueError("Benchmark must have a datetime index")
            
            # Align benchmark with returns
            self.returns, self.benchmark = self.returns.align(self.benchmark, join='inner')

    def calculate_metrics(self) -> pd.DataFrame:
        """Calculate key performance metrics."""
        metrics = {}
        
        # Basic statistics
        metrics['Total Return'] = self.calculate_total_return()*100
        metrics['CAGR'] = self.calculate_cagr()*100
        metrics['Annual Volatility'] = self.calculate_volatility()*100
        metrics['Sharpe Ratio'] = self.calculate_sharpe_ratio()
        metrics['Sortino Ratio'] = self.calculate_sortino_ratio()
        metrics['Max Drawdown'] = self.calculate_max_drawdown()*100
        metrics['Max Drawdown Duration'] = self.calculate_max_drawdown_duration()
        metrics['Skewness'] = stats.skew(self.returns.dropna())
        metrics['Kurtosis'] = stats.kurtosis(self.returns.dropna())
        metrics['Value at Risk (95%)'] = self.calculate_var()*100
        metrics['Conditional VaR (95%)'] = self.calculate_cvar()*100
        metrics['Win Rate'] = self.calculate_win_rate()*100
        
        if self.benchmark is not None:
            metrics['Alpha'] = self.calculate_alpha()
            metrics['Beta'] = self.calculate_beta()
            metrics['Information Ratio'] = self.calculate_information_ratio()
            metrics['Tracking Error'] = self.calculate_tracking_error()
            
        return pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])

    def calculate_total_return(self) -> float:
        """Calculate total return."""
        return ((1 + self.returns).prod() - 1)
    
    def calculate_cagr(self) -> float:
        """Calculate Compound Annual Growth Rate."""
        total_years = (self.returns.index[-1] - self.returns.index[0]).days / 365.25
        return ((1 + self.calculate_total_return()) ** (1/total_years) - 1)
    
    def calculate_volatility(self, annualize: bool = True) -> float:
        """Calculate return volatility."""
        vol = self.returns.std()
        if annualize:
            vol *= np.sqrt(252)
        return vol
    
    def calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio."""
        excess_returns = self.returns - self.daily_rf
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()
    
    def calculate_sortino_ratio(self) -> float:
        """Calculate Sortino ratio."""
        excess_returns = self.returns - self.daily_rf
        downside_returns = excess_returns[excess_returns < 0]
        downside_std = np.sqrt(252) * np.sqrt(np.mean(downside_returns**2))
        return (excess_returns.mean() * 252) / downside_std if downside_std != 0 else np.nan
    
    def calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        cum_returns = (1 + self.returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdowns = cum_returns / rolling_max - 1
        return drawdowns.min()
    
    def calculate_max_drawdown_duration(self) -> int:
        """Calculate maximum drawdown duration in days."""
        cum_returns = (1 + self.returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdown_periods = (rolling_max != cum_returns)
        
        if not drawdown_periods.any():
            return 0
            
        drawdown_groups = (drawdown_periods != drawdown_periods.shift()).cumsum()
        drawdown_duration = drawdown_groups.value_counts()
        return drawdown_duration.max() if not drawdown_duration.empty else 0
    
    def calculate_var(self, confidence: float = 0.95) -> float:
        """Calculate Value at Risk."""
        return -np.percentile(self.returns, (1 - confidence) * 100)
    
    def calculate_cvar(self, confidence: float = 0.95) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)."""
        var = self.calculate_var(confidence)
        return -self.returns[self.returns <= -var].mean()
    
    def calculate_win_rate(self) -> float:
        """Calculate win rate (percentage of positive returns)."""
        return (self.returns > 0).mean()
    
    def calculate_beta(self) -> Optional[float]:
        """Calculate beta relative to benchmark."""
        if self.benchmark is None:
            return None
        return stats.linregress(self.benchmark, self.returns)[0]
    
    def calculate_alpha(self) -> Optional[float]:
        """Calculate alpha relative to benchmark."""
        if self.benchmark is None:
            return None
        beta = self.calculate_beta()
        return (self.returns.mean() - self.daily_rf) - beta * (self.benchmark.mean() - self.daily_rf)
    
    def calculate_information_ratio(self) -> Optional[float]:
        """Calculate information ratio."""
        if self.benchmark is None:
            return None
        active_returns = self.returns - self.benchmark
        return np.sqrt(252) * active_returns.mean() / active_returns.std()
    
    def calculate_tracking_error(self) -> Optional[float]:
        """Calculate tracking error."""
        if self.benchmark is None:
            return None
        return np.sqrt(252) * (self.returns - self.benchmark).std()

    def plot_cumulative_returns(self):
        cumulative_returns = (1 + self.returns).cumprod() - 1
        plt.figure(figsize=(10, 6))
        plt.plot(cumulative_returns, label='Portfolio')
        if self.benchmark is not None:
            cumulative_benchmark = (1 + self.benchmark).cumprod() - 1
            plt.plot(cumulative_benchmark, label='Benchmark')
        plt.title('Cumulative Returns')
        plt.legend()
        plt.show()

    def plot_drawdown(self):
        """Plot drawdown over time."""
        plt.figure(figsize=(12, 6))
        cum_returns = (1 + self.returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max
        
        plt.plot(drawdown.index, drawdown)
        plt.title('Drawdown')
        plt.xlabel('Date')
        plt.ylabel('Drawdown')
        plt.grid(True)
        plt.tight_layout()
        
    def plot_monthly_returns_heatmap(self):
        # Resample returns to month-end frequency
        monthly_returns = self.returns.resample('M').agg(lambda x: (1 + x).prod() - 1)
        
        # Create a pivot table for the heatmap
        monthly_returns.index = monthly_returns.index.to_period('M')
        heatmap_data = monthly_returns.to_frame(name='returns')
        heatmap_data['year'] = heatmap_data.index.year
        heatmap_data['month'] = heatmap_data.index.month
        heatmap_pivot = heatmap_data.pivot(index='year', columns='month', values='returns')
        
        # Plot the heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(heatmap_pivot, annot=True, fmt=".2%", cmap='RdYlGn', center=0)
        plt.title('Monthly Returns Heatmap')
        plt.show()

    def plot_rolling_metrics(self, window: int = 252):
        """Plot rolling Sharpe ratio, volatility, and beta."""
        fig, axes = plt.subplots(3, 1, figsize=(12, 12))
        
        # Rolling Sharpe Ratio
        rolling_sharpe = self.returns.rolling(window=window).apply(
            lambda x: np.sqrt(252) * (x.mean() - self.daily_rf) / x.std()
        )
        axes[0].plot(rolling_sharpe.index, rolling_sharpe)
        axes[0].set_title(f'Rolling Sharpe Ratio ({window} days)')
        axes[0].grid(True)
        
        # Rolling Volatility
        rolling_vol = self.returns.rolling(window=window).std() * np.sqrt(252)
        axes[1].plot(rolling_vol.index, rolling_vol)
        axes[1].set_title(f'Rolling Volatility ({window} days)')
        axes[1].grid(True)
        
        # Rolling Beta
        if self.benchmark is not None:
            rolling_beta = self.returns.rolling(window=window).cov(self.benchmark) / \
                         self.benchmark.rolling(window=window).var()
            axes[2].plot(rolling_beta.index, rolling_beta)
            axes[2].set_title(f'Rolling Beta ({window} days)')
            axes[2].grid(True)
        
        plt.tight_layout()

    def plot_rolling_volatility(self, window: int = 21):
        rolling_volatility = self.returns.rolling(window=window).std() * np.sqrt(252)
        plt.figure(figsize=(10, 6))
        plt.plot(rolling_volatility, label='Portfolio')
        if self.benchmark is not None:
            rolling_benchmark_volatility = self.benchmark.rolling(window=window).std() * np.sqrt(252)
            plt.plot(rolling_benchmark_volatility, label='Benchmark')
        plt.title(f'{window}-Day Rolling Volatility')
        plt.legend()
        plt.show()

    def plot_rolling_sharpe_ratio(self, window: int = 21):
        rolling_sharpe = (self.returns.rolling(window=window).mean() / self.returns.rolling(window=window).std()) * np.sqrt(252)
        plt.figure(figsize=(10, 6))
        plt.plot(rolling_sharpe, label='Portfolio')
        if self.benchmark is not None:
            rolling_benchmark_sharpe = (self.benchmark.rolling(window=window).mean() / self.benchmark.rolling(window=window).std()) * np.sqrt(252)
            plt.plot(rolling_benchmark_sharpe, label='Benchmark')
        plt.title(f'{window}-Day Rolling Sharpe Ratio')
        plt.legend()
        plt.show()

    def generate_full_report(self):
        """Generate a full analysis report with metrics and plots."""
        # Calculate metrics
        metrics = self.calculate_metrics()
        print("\nPerformance Metrics:")
        print("===================")
        print(metrics)
        
        # Generate plots
        self.plot_cumulative_returns()
        self.plot_drawdown()
        self.plot_monthly_returns_heatmap()
        self.plot_rolling_metrics()
        plt.show()
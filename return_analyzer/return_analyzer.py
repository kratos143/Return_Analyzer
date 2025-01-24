import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple
from statsmodels.stats.diagnostic import acorr_ljungbox  # For autocorrelation test

class ReturnAnalyzer:
    def __init__(self, returns: pd.Series, benchmark: Optional[pd.Series | pd.DataFrame] = None, risk_free_rate: float = 0.0):
        """Initialize the ReturnAnalyzer with returns data."""
        # Validate input types
        if not isinstance(returns, pd.Series):
            raise TypeError("returns must be a pandas Series")

        # Handle benchmark DataFrame conversion
        if isinstance(benchmark, pd.DataFrame):
            benchmark = benchmark.squeeze()  # Convert DataFrame to Series
        
        if benchmark is not None and not isinstance(benchmark, pd.Series):
            raise TypeError("benchmark must be a pandas Series")

        # Convert returns to daily if needed
        if isinstance(returns.index, pd.DatetimeIndex):
            has_time = (returns.index.time != pd.Timestamp('00:00:00').time()).any()
            if has_time:
                returns = self.convert_to_daily(returns)
        
        # Convert benchmark to daily if needed
        if benchmark is not None:
            if isinstance(benchmark.index, pd.DatetimeIndex):
                has_time = (benchmark.index.time != pd.Timestamp('00:00:00').time()).any()
                if has_time:
                    benchmark = self.convert_to_daily(benchmark)
            
            # Ensure both series have datetime index
            returns.index = pd.to_datetime(returns.index)
            benchmark.index = pd.to_datetime(benchmark.index)
            
            # Align returns and benchmark
            self.returns, self.benchmark = returns.align(benchmark, join='inner')
        else:
            self.returns = returns
            self.benchmark = None
            
        self.risk_free_rate = risk_free_rate
        self.daily_rf = (1 + risk_free_rate) ** (1/252) - 1
        self._validate_data()

    def convert_to_daily(self, returns: pd.Series) -> pd.Series:
        """
        Convert intraday returns to daily returns by summing returns for each date.
        
        Parameters:
        -----------
        returns : pd.Series
            Returns series with datetime index including time
            
        Returns:
        --------
        pd.Series
            Daily returns series with date-only index
        """
        # Ensure datetime index
        returns.index = pd.to_datetime(returns.index)
        # Resample to daily frequency and sum returns
        daily_returns = returns.resample('D').sum()
        # Remove days with no trades (zero returns)
        daily_returns = daily_returns[daily_returns != 0]
        return daily_returns

    def _validate_data(self):
        """Validate the input data."""
        if len(self.returns) == 0:
            raise ValueError("Returns series is empty")
        if self.benchmark is not None and len(self.benchmark) == 0:
            raise ValueError("Benchmark series is empty")
        if not isinstance(self.returns.index, pd.DatetimeIndex):
            raise ValueError("Returns must have a datetime index")
            
        if self.returns.isnull().any():
            raise ValueError("Returns series contains NaN values")
            
        if self.benchmark is not None:
            if not isinstance(self.benchmark.index, pd.DatetimeIndex):
                raise ValueError("Benchmark must have a datetime index")
            if self.benchmark.isnull().any():
                raise ValueError("Benchmark series contains NaN values")

    def calculate_metrics(self) -> pd.DataFrame:
        """Calculate comprehensive performance metrics."""
        metrics = {}
        
        # Basic statistics
        metrics['Total Return (%)'] = self.calculate_total_return() * 100
        metrics['CAGR (%)'] = self.calculate_cagr() * 100
        metrics['Annual Volatility (%)'] = self.calculate_volatility() * 100
        metrics['Sharpe Ratio'] = self.calculate_sharpe_ratio()
        metrics['Sortino Ratio'] = self.calculate_sortino_ratio()
        metrics['Max Drawdown (%)'] = self.calculate_max_drawdown() * 100
        metrics['Max Drawdown Duration (Days)'] = self.calculate_max_drawdown_duration()
        metrics['Skewness'] = stats.skew(self.returns.dropna())
        metrics['Excess Kurtosis'] = stats.kurtosis(self.returns.dropna())
        metrics['Value at Risk (95%) (%)'] = self.calculate_var() * 100
        metrics['Conditional VaR (95%) (%)'] = self.calculate_cvar() * 100
        metrics['Win Rate (%)'] = self.calculate_win_rate() * 100
        
        # Advanced risk metrics
        metrics['Calmar Ratio'] = self.calculate_calmar_ratio()
        metrics['Omega Ratio'] = self.calculate_omega_ratio()
        metrics['Ulcer Index'] = self.calculate_ulcer_index()
        metrics['Profit Factor'] = self.calculate_profit_factor()
        metrics['Kelly Criterion (%)'] = self.calculate_kelly_criterion() * 100
        metrics['Tail Ratio'] = self.calculate_tail_ratio()
        metrics['Gain to Pain Ratio'] = self.calculate_gain_to_pain()

        # Statistical tests
        jb = stats.jarque_bera(self.returns.dropna())
        metrics['Jarque-Bera Statistic'] = jb[0]
        metrics['JB p-value'] = jb[1]
        
        try:
            lb_test = acorr_ljungbox(self.returns.dropna(), lags=5, return_df=True)
            metrics['Ljung-Box p-value (lag=5)'] = lb_test['lb_pvalue'].values[-1]
        except ImportError:
            pass

        # Benchmark-relative metrics
        if self.benchmark is not None:
            metrics['Alpha'] = self.calculate_alpha()
            metrics['Beta'] = self.calculate_beta()
            metrics['Information Ratio'] = self.calculate_information_ratio()
            metrics['Tracking Error (%)'] = self.calculate_tracking_error() * 100
            up_capture, down_capture = self.calculate_capture_ratios()
            metrics['Up Capture Ratio (%)'] = up_capture
            metrics['Down Capture Ratio (%)'] = down_capture
            metrics['Active Share (%)'] = self.calculate_active_share() * 100

        return pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])

    # Core calculations -----------------------------------------------------------
    def calculate_total_return(self) -> float:
        """Calculate cumulative return for the entire period."""
        return (1 + self.returns).prod() - 1
    
    def calculate_cagr(self) -> float:
        """Calculate Compound Annual Growth Rate."""
        total_days = (self.returns.index[-1] - self.returns.index[0]).days
        total_years = total_days / 365.25
        return (1 + self.calculate_total_return()) ** (1/total_years) - 1
    
    def calculate_volatility(self, annualize: bool = True) -> float:
        """Calculate return volatility."""
        vol = self.returns.std()
        return vol * np.sqrt(252) if annualize else vol
    
    def calculate_sharpe_ratio(self) -> float:
        """Calculate annualized Sharpe ratio."""
        excess_returns = self.returns - self.daily_rf
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()
    
    def calculate_sortino_ratio(self) -> float:
        """Calculate annualized Sortino ratio."""
        excess_returns = self.returns - self.daily_rf
        downside_returns = excess_returns[excess_returns < 0]
        downside_std = np.sqrt(np.mean(downside_returns**2)) * np.sqrt(252)
        return excess_returns.mean() * 252 / downside_std if downside_std != 0 else np.nan
    
    def calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        cumulative = (1 + self.returns).cumprod()
        peak = cumulative.expanding().max()
        return (cumulative / peak - 1).min()
    
    def calculate_max_drawdown_duration(self) -> int:
        """Calculate maximum duration of drawdown in days."""
        cumulative = (1 + self.returns).cumprod()
        peak = cumulative.expanding().max()
        drawdown = (cumulative < peak).astype(int)
        duration = drawdown * (drawdown.groupby((drawdown != drawdown.shift()).cumsum()).cumcount() + 1)
        return duration.max()
    
    def calculate_var(self, confidence: float = 0.95) -> float:
        """Calculate Value at Risk (VaR)."""
        return -np.percentile(self.returns, 100 * (1 - confidence))
    
    def calculate_cvar(self, confidence: float = 0.95) -> float:
        """Calculate Conditional Value at Risk (CVaR)."""
        var = self.calculate_var(confidence)
        return -self.returns[self.returns <= -var].mean()
    
    def calculate_win_rate(self) -> float:
        """Calculate percentage of positive returns."""
        return (self.returns > 0).mean()
    
    # Advanced metrics ------------------------------------------------------------
    def calculate_calmar_ratio(self) -> float:
        """Calculate Calmar ratio (CAGR / Max Drawdown)."""
        cagr = self.calculate_cagr()
        max_dd = abs(self.calculate_max_drawdown())
        return cagr / max_dd if max_dd != 0 else np.nan
    
    def calculate_omega_ratio(self, threshold: float = None) -> float:
        """Calculate Omega ratio relative to threshold."""
        threshold = self.daily_rf if threshold is None else threshold
        excess = self.returns - threshold
        gain = excess[excess > 0].sum()
        loss = -excess[excess < 0].sum()
        return gain / loss if loss != 0 else np.nan
    
    def calculate_ulcer_index(self) -> float:
        """Calculate Ulcer Index."""
        drawdown = self._compute_drawdown_series()
        return np.sqrt(np.mean(drawdown**2))
    
    def calculate_profit_factor(self) -> float:
        """Calculate Profit Factor (gross profit / gross loss)."""
        gross_profit = self.returns[self.returns > 0].sum()
        gross_loss = -self.returns[self.returns < 0].sum()
        return gross_profit / gross_loss if gross_loss != 0 else np.nan
    
    def calculate_kelly_criterion(self) -> float:
        """Calculate Kelly Criterion optimal position size."""
        win_rate = self.calculate_win_rate()
        avg_win = self.returns[self.returns > 0].mean()
        avg_loss = self.returns[self.returns < 0].mean()
        return (win_rate - (1 - win_rate) / (avg_win / abs(avg_loss))) if avg_loss != 0 else np.nan
    
    def calculate_tail_ratio(self, confidence: float = 0.95) -> float:
        """Calculate ratio of (upper tail / lower tail) returns."""
        upper = abs(np.percentile(self.returns, 100 * confidence))
        lower = abs(np.percentile(self.returns, 100 * (1 - confidence)))
        return upper / lower if lower != 0 else np.nan
    
    def calculate_gain_to_pain(self) -> float:
        """Calculate Gain to Pain ratio."""
        total_return = self.calculate_total_return()
        total_loss = -self.returns[self.returns < 0].sum()
        return total_return / total_loss if total_loss != 0 else np.nan
    
    def calculate_capture_ratios(self) -> Tuple[float, float]:
        """Calculate up and down capture ratios."""
        aligned_returns, aligned_bm = self.returns.align(self.benchmark, join='inner')
        up_periods = aligned_bm > 0
        down_periods = aligned_bm < 0
        
        up_capture = (aligned_returns[up_periods].mean() / aligned_bm[up_periods].mean()) * 100 if up_periods.any() else np.nan
        down_capture = (aligned_returns[down_periods].mean() / aligned_bm[down_periods].mean()) * 100 if down_periods.any() else np.nan
        
        return up_capture, down_capture
    
    def calculate_active_share(self) -> float:
        """Calculate Active Share relative to benchmark."""
        return np.abs(self.returns - self.benchmark).mean() * np.sqrt(252)
    
    def calculate_alpha(self) -> float:
        """Calculate annualized alpha."""
        beta = self.calculate_beta()
        excess_return = self.returns.mean() - self.daily_rf
        benchmark_excess = self.benchmark.mean() - self.daily_rf
        return (excess_return - beta * benchmark_excess) * 252
    
    def calculate_beta(self) -> float:
        """Calculate beta relative to benchmark."""
        covariance = np.cov(self.returns, self.benchmark)[0, 1]
        variance = self.benchmark.var()
        return covariance / variance
    
    def calculate_information_ratio(self) -> float:
        """Calculate annualized Information Ratio."""
        active_returns = self.returns - self.benchmark
        return active_returns.mean() / active_returns.std() * np.sqrt(252)
    
    def calculate_tracking_error(self) -> float:
        """Calculate annualized Tracking Error."""
        return (self.returns - self.benchmark).std() * np.sqrt(252)
    
    # Visualization methods -------------------------------------------------------
    def plot_cumulative_returns(self):
        """Plot cumulative returns vs benchmark."""
        plt.figure(figsize=(12, 6))
        cum_returns = (1 + self.returns).cumprod()
        plt.plot(cum_returns, label='Portfolio')
        
        if self.benchmark is not None:
            cum_bm = (1 + self.benchmark).cumprod()
            plt.plot(cum_bm, label='Benchmark')
            
        plt.title('Cumulative Returns')
        plt.ylabel('Growth of $1')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def plot_drawdown(self):
        """Plot drawdown over time."""
        drawdown = self._compute_drawdown_series()
        plt.figure(figsize=(12, 6))
        plt.fill_between(drawdown.index, drawdown * 100, 0, color='red', alpha=0.3)
        plt.title('Drawdown Chart')
        plt.ylabel('Drawdown (%)')
        plt.grid(True)
        plt.show()
    
    def plot_return_distribution(self, bins: int = 50):
        """Plot return distribution with KDE."""
        plt.figure(figsize=(10, 6))
        sns.histplot(self.returns, bins=bins, kde=True, stat='density')
        plt.title('Return Distribution')
        plt.xlabel('Daily Returns')
        plt.show()
    
    def plot_qq(self):
        """Plot Q-Q plot against normal distribution."""
        plt.figure(figsize=(10, 6))
        stats.probplot(self.returns.dropna(), dist="norm", plot=plt)
        plt.title('Q-Q Plot')
        plt.show()
    
    def plot_rolling_volatility(self, window: int = 21):
        """Plot rolling volatility."""
        rolling_vol = self.returns.rolling(window).std() * np.sqrt(252)
        plt.figure(figsize=(12, 6))
        plt.plot(rolling_vol, label='Portfolio')
        
        if self.benchmark is not None:
            bm_vol = self.benchmark.rolling(window).std() * np.sqrt(252)
            plt.plot(bm_vol, label='Benchmark')
            
        plt.title(f'{window}-Day Rolling Volatility')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def plot_capture_ratio(self, window: int = 126):
        """
        Plot rolling up and down capture ratios.
        
        Parameters:
        -----------
        window : int, optional
            Rolling window size in days (default: 126)
        """
        if self.benchmark is None:
            print("No benchmark provided. Capture ratio cannot be calculated.")
            return
        
        # Align returns and benchmark
        aligned_returns, aligned_bm = self.returns.align(self.benchmark, join='inner')
        
        # Calculate rolling capture ratios
        rolling_up = aligned_returns.rolling(window).mean() / aligned_bm.rolling(window).mean()
        rolling_down = aligned_returns[aligned_bm < 0].rolling(window).mean() / aligned_bm[aligned_bm < 0].rolling(window).mean()
        
        # Plot the results
        plt.figure(figsize=(12, 6))
        rolling_up.plot(label='Up Capture Ratio')
        rolling_down.plot(label='Down Capture Ratio')
        plt.title(f'{window}-Day Rolling Capture Ratios')
        plt.ylabel('Capture Ratio')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_monthly_returns_heatmap(self):
        """
        Plot a heatmap of monthly returns.
        """
        # Resample to monthly returns
        monthly_returns = self.returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        
        # Create dataframe for heatmap
        heatmap_data = pd.DataFrame({
            'Year': monthly_returns.index.year,
            'Month': monthly_returns.index.month_name(),
            'Returns': monthly_returns
        })
        
        # Pivot to matrix format
        heatmap_pivot = heatmap_data.pivot_table(
            index='Year',
            columns='Month',
            values='Returns',
            sort=False
        )
        
        # Order months chronologically
        month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                      'July', 'August', 'September', 'October', 'November', 'December']
        heatmap_pivot = heatmap_pivot.reindex(columns=month_order)

        plt.figure(figsize=(14, 8))
        sns.heatmap(heatmap_pivot * 100, 
                   annot=True, 
                   fmt=".2f", 
                   cmap='RdYlGn',
                   center=0,
                   linewidths=0.5,
                   cbar_kws={'label': 'Return (%)'})
        plt.title('Monthly Returns Heatmap (%)')
        plt.tight_layout()
        plt.show()


        

    def plot_rolling_factors(self, window: int = 262):
        """
        Plot rolling Alpha, Beta, and Sharpe Ratio.
        
        Parameters:
        -----------
        window : int
            Rolling window size in days (default: 126 ~6 months)
        """
        fig, ax = plt.subplots(3, 1, figsize=(14, 12))
        
        # Rolling Sharpe Ratio
        def rolling_sharpe_ratio(x):
            return (x.mean() - self.daily_rf) / x.std() * np.sqrt(252)
        
        rolling_sharpe = self.returns.rolling(window).apply(rolling_sharpe_ratio, raw=False)
        ax[0].plot(rolling_sharpe, label='Sharpe Ratio')
        ax[0].set_title(f'{window}-Day Rolling Sharpe Ratio')
        ax[0].grid(True)

        if self.benchmark is not None:
            # Align data
            aligned_returns, aligned_bm = self.returns.align(self.benchmark, join='inner')
            
            # Rolling Beta
            rolling_beta = aligned_returns.rolling(window).cov(aligned_bm) / aligned_bm.rolling(window).var()
            ax[1].plot(rolling_beta, label='Beta', color='orange')
            ax[1].set_title(f'{window}-Day Rolling Beta')
            ax[1].grid(True)
            
            # Rolling Alpha
            rolling_alpha = (aligned_returns.rolling(window).mean() - self.daily_rf) - \
                           rolling_beta * (aligned_bm.rolling(window).mean() - self.daily_rf)
            rolling_alpha *= 252  # Annualize
            ax[2].plot(rolling_alpha, label='Alpha', color='green')
            ax[2].set_title(f'{window}-Day Rolling Alpha (Annualized)')
            ax[2].grid(True)

        plt.tight_layout()
        plt.show()



    
    def generate_full_report(self):
        """Generate comprehensive analysis report."""
        print("Performance Metrics:")
        print(self.calculate_metrics())
        
        self.plot_cumulative_returns()
        self.plot_drawdown()
        self.plot_return_distribution()
        self.plot_qq()
        self.plot_rolling_volatility()
        self.plot_monthly_returns_heatmap()  # New heatmap
        self.plot_rolling_factors()          # New factors plot
        
        if self.benchmark is not None:
            self.plot_capture_ratio()
        
        plt.show()
    
    # Helper methods --------------------------------------------------------------
    def _compute_drawdown_series(self) -> pd.Series:
        """Compute drawdown series."""
        cumulative = (1 + self.returns).cumprod()
        peak = cumulative.expanding().max()
        return (cumulative / peak) - 1

# performance_analyzer.py
# Purpose: To load the results of a backtest, calculate a comprehensive suite of
#          performance and risk metrics, and generate detailed visualizations.
#          This is the final step in the strategy evaluation pipeline.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
import json
import config
from data_handler import DataHandler

# Make sure you have TA-Lib installed. If not, run: pip install TA-Lib
try:
    import talib
except ImportError:
    print("Warning: TA-Lib library not found. Comparison strategies will not be available.")
    talib = None

class BacktestingAgent:
    """
    A comprehensive agent for in-depth performance and risk analysis of a trading strategy.
    It takes the raw trade logs and equity curve from a backtest and generates a full suite
    of quantitative metrics and visualizations.
    """
    def __init__(self, initial_capital, data, equity_curve, trades, benchmark_name):
        self.initial_capital = initial_capital
        self.data = data
        self.equity_curve = equity_curve
        self.benchmark_name = benchmark_name
        self.risk_free_rate_annual = config.RISK_FREE_RATE
        self.risk_free_rate_daily = self.risk_free_rate_annual / 252
        
        self.dates_entry = [trades[i] for i in range(0, len(trades), 2)]
        self.trades = [trades[i] for i in range(1, len(trades), 2)]
        self.trade_exit = [trade[4] for trade in self.trades]
        
        self.trade_durations = [(self.trade_exit[i] - self.dates_entry[i]).total_seconds() / 60 for i in range(len(self.dates_entry))]
        self.trade_returns = [trade[3] for trade in self.trades]
        
        self.calculate_daily_returns()

    def calculate_daily_returns(self):
        """Calculates daily returns for both the strategy and the benchmark."""
        equity_by_trade = pd.Series(self.equity_curve[1:], index=pd.to_datetime(self.trade_exit))
        
        full_date_range = pd.date_range(start=self.data.index.min().date(), end=self.data.index.max().date(), freq='D')
        daily_equity = equity_by_trade.resample('D').last()
        daily_equity = daily_equity.reindex(full_date_range).ffill()
        daily_equity.iloc[0] = self.initial_capital
        daily_equity.ffill(inplace=True)

        self.strategy_daily_returns = daily_equity.pct_change().dropna()
        self.benchmark_daily_returns = self.data['Close'].resample('D').last().pct_change().dropna()
        
        self.strategy_daily_returns, self.benchmark_daily_returns = self.strategy_daily_returns.align(self.benchmark_daily_returns, join='inner')
        self.strategy_daily_excess_returns = self.strategy_daily_returns - self.risk_free_rate_daily
        self.benchmark_daily_excess_returns = self.benchmark_daily_returns - self.risk_free_rate_daily

    def run_all_analysis(self):
        """Orchestrates the entire analysis pipeline."""
        self.run_performance_metrics()
        self.run_risk_metrics()
        self.run_trade_statistics()
        self.run_risk_adjusted_metrics()
        self.run_additional_metrics()
        if talib:
            self.run_common_strategies()
        else:
            print("\nSkipping common strategy comparison: TA-Lib not installed.")

    # --- Performance Metrics Group ---
    def run_performance_metrics(self):
        print("\n" + "="*50)
        print("PERFORMANCE METRICS")
        print("="*50)
        self.calculate_total_return()
        self.calculate_cagr()
        self.calculate_annualized_return()
        self.calculate_and_print_periodic_returns()
        self.calculate_mean_std_returns()
        self.create_return_summary_table()
        self.plot_cumulative_return_curve()
        self.plot_quarterly_returns_comparison()

    def calculate_total_return(self):
        self.total_return_strategy = ((self.equity_curve[-1] - self.initial_capital) / self.initial_capital) * 100
        daily_benchmark_prices = self.data['Close'].resample('D').last().dropna()
        self.total_return_benchmark = ((daily_benchmark_prices.iloc[-1] - daily_benchmark_prices.iloc[0]) / daily_benchmark_prices.iloc[0]) * 100 if not daily_benchmark_prices.empty else 0

    def calculate_cagr(self):
        num_years = (self.data.index[-1] - self.data.index[0]).days / 365.25
        self.cagr_strategy = ((self.equity_curve[-1] / self.initial_capital) ** (1 / num_years)) - 1 if num_years > 0 else 0
        daily_benchmark_prices = self.data['Close'].resample('D').last().dropna()
        self.cagr_benchmark = ((daily_benchmark_prices.iloc[-1] / daily_benchmark_prices.iloc[0]) ** (1 / num_years)) - 1 if num_years > 0 and not daily_benchmark_prices.empty else 0

    def calculate_annualized_return(self):
        self.annualized_return_strategy = self.strategy_daily_returns.mean() * 252
        self.annualized_return_benchmark = self.benchmark_daily_returns.mean() * 252

    def calculate_periodic_returns(self, period='ME'):
        return self.strategy_daily_returns.resample(period).apply(lambda x: (x + 1).prod() - 1)

    def calculate_and_print_periodic_returns(self):
        self.strategy_monthly_returns = self.calculate_periodic_returns(period='ME')
        self.benchmark_monthly_returns = self.benchmark_daily_returns.resample('ME').apply(lambda x: (x + 1).prod() - 1)
        self.strategy_quarterly_returns = self.calculate_periodic_returns(period='QE')
        self.benchmark_quarterly_returns = self.benchmark_daily_returns.resample('QE').apply(lambda x: (x + 1).prod() - 1)
        self.strategy_annual_returns = self.calculate_periodic_returns(period='YE')
        self.benchmark_annual_returns = self.benchmark_daily_returns.resample('YE').apply(lambda x: (x + 1).prod() - 1)

    def calculate_mean_std_returns(self):
        self.mean_daily_return_strategy = self.strategy_daily_returns.mean() * 100
        self.std_daily_return_strategy = self.strategy_daily_returns.std() * 100
        self.mean_daily_return_benchmark = self.benchmark_daily_returns.mean() * 100
        self.std_daily_return_benchmark = self.benchmark_daily_returns.std() * 100
    
    # --- Trade Statistics Group ---
    def run_trade_statistics(self):
        print("\n" + "="*50)
        print("TRADE STATISTICS")
        print("="*50)
        self.calculate_trade_statistics()
        self.create_trade_summary_table()
        self.calculate_profitability_by_trade_type()

    def calculate_trade_statistics(self):
        self.total_trades = len(self.trades)
        self.winning_trades = [t for t in self.trades if t[3] > 0]
        self.losing_trades = [t for t in self.trades if t[3] < 0]
        self.breakeven_trades = [t for t in self.trades if t[3] == 0]
        self.win_rate = len(self.winning_trades) / self.total_trades * 100 if self.total_trades > 0 else 0
        self.loss_rate = len(self.losing_trades) / self.total_trades * 100 if self.total_trades > 0 else 0
        self.breakeven_rate = len(self.breakeven_trades) / self.total_trades * 100 if self.total_trades > 0 else 0
        self.avg_profit_pct = np.mean([t[3] for t in self.winning_trades]) * 100 if self.winning_trades else 0
        self.avg_loss_pct = np.mean([t[3] for t in self.losing_trades]) * 100 if self.losing_trades else 0
        total_profit = sum(t[3] for t in self.winning_trades)
        total_loss = abs(sum(t[3] for t in self.losing_trades))
        self.profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        self.risk_reward_ratio = abs(self.avg_profit_pct / self.avg_loss_pct) if self.avg_loss_pct != 0 else float('inf')
        self.expected_value = (self.win_rate / 100 * self.avg_profit_pct) + (self.loss_rate / 100 * self.avg_loss_pct)
        self.max_consecutive_wins = self._calculate_max_consecutive(self.trade_returns, 'win')
        self.max_consecutive_losses = self._calculate_max_consecutive(self.trade_returns, 'loss')
        self.avg_trade_duration = np.mean(self.trade_durations) if self.trade_durations else 0
        self.avg_winning_duration = np.mean([self.trade_durations[i] for i, t in enumerate(self.trades) if t[3] > 0]) if self.winning_trades else 0
        self.avg_losing_duration = np.mean([self.trade_durations[i] for i, t in enumerate(self.trades) if t[3] < 0]) if self.losing_trades else 0
        self.trade_frequency_yearly = (self.total_trades / ((self.data.index[-1] - self.data.index[0]).days / 365.25)) if (self.data.index[-1] - self.data.index[0]).days > 0 else 0
        self.trade_frequency_quarterly = self.trade_frequency_yearly / 4

    # --- Risk & Volatility Metrics Group ---
    def run_risk_metrics(self):
        print("\n" + "="*50)
        print("RISK & VOLATILITY METRICS")
        print("="*50)
        self.annualized_volatility_strategy = self.calculate_annualized_volatility(self.strategy_daily_returns)
        self.annualized_volatility_benchmark = self.calculate_annualized_volatility(self.benchmark_daily_returns)
        self.max_drawdown_strategy = self.calculate_max_drawdown(self.equity_curve)
        self.max_drawdown_benchmark = self.calculate_max_drawdown(self.data['Close'].resample('D').last().dropna().tolist())
        self.average_drawdown_strategy, _ = self.calculate_average_drawdown(self.equity_curve)
        self.average_drawdown_benchmark, _ = self.calculate_average_drawdown(self.data['Close'].resample('D').last().dropna().tolist())
        self.skewness_strategy = skew(self.strategy_daily_returns)
        self.kurtosis_strategy = kurtosis(self.strategy_daily_returns)
        self.skewness_benchmark = skew(self.benchmark_daily_returns)
        self.kurtosis_benchmark = kurtosis(self.benchmark_daily_returns)
        self.create_volatility_summary_table()
        self.plot_drawdowns_combined()

    # --- Risk-Adjusted Metrics Group ---
    def run_risk_adjusted_metrics(self):
        print("\n" + "="*50)
        print("RISK-ADJUSTED PERFORMANCE METRICS")
        print("="*50)
        self.sharpe_ratio = self.calculate_sharpe_ratio(self.strategy_daily_excess_returns)
        self.sortino_ratio = self.calculate_sortino_ratio(self.strategy_daily_excess_returns)
        self.beta = self.calculate_beta(self.strategy_daily_returns, self.benchmark_daily_returns)
        self.alpha = self.calculate_alpha(self.strategy_daily_returns, self.benchmark_daily_returns, self.risk_free_rate_daily)
        self.exposure = self.calculate_exposure(self.strategy_daily_returns)
        self.create_performance_summary_table()

    # --- Additional Metrics Group ---
    def run_additional_metrics(self):
        print("\n" + "="*50)
        print("ADDITIONAL METRICS")
        print("="*50)
        self.var_1_strategy = self.calculate_var(self.strategy_daily_returns, 0.01)
        self.var_5_strategy = self.calculate_var(self.strategy_daily_returns, 0.05)
        self.worst_year_return_strategy = self.calculate_worst_year_return(self.strategy_daily_returns)
        self.create_risk_and_correlation_summary_table()

    # --- Comparison Strategies Group ---
    def run_common_strategies(self):
        print("\n" + "="*50)
        print("COMPARISON WITH COMMON STRATEGIES")
        print("="*50)
        self.plot_strategy_comparison()
        self.create_strategy_summary_table()
    
    # --- Helper & Calculation Methods ---
    def calculate_annualized_volatility(self, daily_returns):
        return daily_returns.std() * np.sqrt(252) * 100

    def calculate_max_drawdown(self, equity_curve):
        equity_series = pd.Series(equity_curve)
        peak = equity_series.cummax()
        drawdown = (equity_series - peak) / peak
        return drawdown.min() * 100 if not drawdown.empty else 0

    def calculate_average_drawdown(self, equity_curve):
        equity_series = pd.Series(equity_curve)
        peak = equity_series.cummax()
        drawdowns = (equity_series - peak) / peak * 100
        return drawdowns.mean(), drawdowns.median()

    def calculate_sharpe_ratio(self, daily_excess_returns):
        return (daily_excess_returns.mean() / daily_excess_returns.std()) * np.sqrt(252) if daily_excess_returns.std() > 0 else 0
    
    def calculate_sortino_ratio(self, daily_excess_returns):
        downside_std = daily_excess_returns[daily_excess_returns < 0].std()
        return (daily_excess_returns.mean() / downside_std) * np.sqrt(252) if downside_std > 0 and not np.isnan(downside_std) else 0
        
    def calculate_beta(self, strategy_returns, benchmark_returns):
        covariance = strategy_returns.cov(benchmark_returns)
        variance = benchmark_returns.var()
        return covariance / variance if variance > 0 else 0

    def calculate_alpha(self, strategy_returns, benchmark_returns, risk_free_rate):
        avg_strategy_return = strategy_returns.mean()
        avg_benchmark_return = benchmark_returns.mean()
        beta = self.calculate_beta(strategy_returns, benchmark_returns)
        alpha = avg_strategy_return - (risk_free_rate + beta * (avg_benchmark_return - risk_free_rate))
        return alpha * 252

    def calculate_exposure(self, strategy_returns):
        return sum(strategy_returns != 0) / len(strategy_returns) * 100 if len(strategy_returns) > 0 else 0

    def calculate_var(self, returns, confidence_level=0.01):
        if returns.empty: return 0
        return np.percentile(returns, confidence_level * 100)

    def calculate_worst_year_return(self, returns):
        if returns.empty: return 0
        annual_returns = returns.resample('YE').apply(lambda x: (x + 1).prod() - 1)
        return annual_returns.min() * 100

    def _calculate_max_consecutive(self, returns, kind='win'):
        max_streak, current_streak = 0, 0
        for r in returns:
            is_streak = (kind == 'win' and r > 0) or (kind == 'loss' and r < 0)
            current_streak = current_streak + 1 if is_streak else 0
            max_streak = max(max_streak, current_streak)
        return max(max_streak, current_streak)

    def calculate_profitability_by_trade_type(self):
        long_trades = [t for t in self.trades if t[0] == 'long']
        short_trades = [t for t in self.trades if t[0] == 'short']
        long_returns = [t[3] for t in long_trades]
        short_returns = [t[3] for t in short_trades]
        
        profitability = {
            'Long Trades': {
                'Total Trades': len(long_trades),
                'Win Rate %': (sum(1 for r in long_returns if r > 0) / len(long_trades) * 100) if long_trades else 0,
                'Net Profit (in %R)': sum(long_returns) * 100
            },
            'Short Trades': {
                'Total Trades': len(short_trades),
                'Win Rate %': (sum(1 for r in short_returns if r > 0) / len(short_trades) * 100) if short_trades else 0,
                'Net Profit (in %R)': sum(short_returns) * 100
            }
        }
        print(pd.DataFrame(profitability).T)
        
    # --- Table Creation Methods ---
    def create_return_summary_table(self):
        summary = {
            "Metric": ["Total Return", "CAGR", "Annualized Return", "Avg Monthly Return"],
            "Strategy": [f"{self.total_return_strategy:.2f}%", f"{self.cagr_strategy * 100:.2f}%", f"{self.annualized_return_strategy * 100:.2f}%", f"{self.strategy_monthly_returns.mean()*100:.2f}%"],
            self.benchmark_name: [f"{self.total_return_benchmark:.2f}%", f"{self.cagr_benchmark * 100:.2f}%", f"{self.annualized_return_benchmark * 100:.2f}%", f"{self.benchmark_monthly_returns.mean()*100:.2f}%"]
        }
        print(pd.DataFrame(summary).set_index("Metric"))

    def create_trade_summary_table(self):
        summary = {
            "Metric": ["Total Trades", "Win Rate", "Profit Factor", "Avg Profit %", "Avg Loss %", "Risk:Reward Ratio", "Expectancy %", "Max Consecutive Wins", "Max Consecutive Losses", "Avg Duration (Min)", "Trades per Year"],
            "Value": [self.total_trades, f"{self.win_rate:.2f}%", f"{self.profit_factor:.2f}", f"{self.avg_profit_pct:.2f}%", f"{self.avg_loss_pct:.2f}%", f"{self.risk_reward_ratio:.2f}", f"{self.expected_value:.4f}%", self.max_consecutive_wins, self.max_consecutive_losses, f"{self.avg_trade_duration:.2f}", f"{self.trade_frequency_yearly:.2f}"]
        }
        print(pd.DataFrame(summary).set_index("Metric"))

    def create_volatility_summary_table(self):
        summary = {
            "Metric": ["Annualized Volatility", "Maximum Drawdown", "Average Drawdown", "Skewness", "Kurtosis"],
            "Strategy": [f"{self.annualized_volatility_strategy:.2f}%", f"{self.max_drawdown_strategy:.2f}%", f"{self.average_drawdown_strategy:.2f}%", f"{self.skewness_strategy:.2f}", f"{self.kurtosis_strategy:.2f}"],
            self.benchmark_name: [f"{self.annualized_volatility_benchmark:.2f}%", f"{self.max_drawdown_benchmark:.2f}%", f"{self.average_drawdown_benchmark:.2f}%", f"{self.skewness_benchmark:.2f}", f"{self.kurtosis_benchmark:.2f}"]
        }
        print(pd.DataFrame(summary).set_index("Metric"))

    def create_performance_summary_table(self):
        summary = {
            "Metric": ["Sharpe Ratio", "Sortino Ratio", "Alpha (Annualized)", "Beta", "Market Exposure"],
            "Strategy": [f"{self.sharpe_ratio:.2f}", f"{self.sortino_ratio:.2f}", f"{self.alpha*100:.2f}%", f"{self.beta:.2f}", f"{self.exposure:.2f}%"]
        }
        print(pd.DataFrame(summary).set_index("Metric"))
    
    def create_risk_and_correlation_summary_table(self):
        self.var_1_benchmark = self.calculate_var(self.benchmark_daily_returns, 0.01)
        self.var_5_benchmark = self.calculate_var(self.benchmark_daily_returns, 0.05)
        self.worst_year_benchmark = self.calculate_worst_year_return(self.benchmark_daily_returns)
        self.correlation_yearly_returns = self.strategy_annual_returns.corr(self.benchmark_annual_returns)
        summary = {
            "Metric": ["1% VaR (Daily)", "5% VaR (Daily)", "Worst Year", "Yearly Correlation"],
            "Strategy": [f"{self.var_1_strategy:.2f}%", f"{self.var_5_strategy:.2f}%", f"{self.worst_year_return_strategy:.2f}%", f"{self.correlation_yearly_returns:.2f}"],
            self.benchmark_name: [f"{self.var_1_benchmark:.2f}%", f"{self.var_5_benchmark:.2f}%", f"{self.worst_year_benchmark:.2f}%", "1.00"]
        }
        print(pd.DataFrame(summary).set_index("Metric"))

    # --- Plotting Methods ---
    def plot_cumulative_return_curve(self):
        equity_series = pd.Series(self.equity_curve, index=pd.to_datetime([self.data.index[0]] + self.trade_exit))
        daily_equity = equity_series.resample('D').last().ffill()
        daily_benchmark = self.data['Close'].resample('D').last().dropna()
        benchmark_performance = (daily_benchmark / daily_benchmark.iloc[0]) * self.initial_capital
        
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(daily_equity.index, daily_equity.values, label="Strategy Equity Curve", color="blue")
        ax.plot(benchmark_performance.index, benchmark_performance.values, label=f"{self.benchmark_name} Buy & Hold", color="gray", linestyle='--')
        ax.set_title("Strategy Equity Curve vs. Benchmark", fontsize=16)
        ax.set_xlabel("Date")
        ax.set_ylabel("Equity ($)")
        ax.legend()
        plt.show()

    def plot_quarterly_returns_comparison(self):
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=(14, 7))
        (self.strategy_quarterly_returns * 100).plot(kind='bar', ax=ax, color='blue', alpha=0.7, label='Strategy')
        (self.benchmark_quarterly_returns * 100).plot(kind='line', ax=ax, color='gray', marker='o', label=self.benchmark_name)
        ax.set_title("Quarterly Returns Comparison", fontsize=16)
        ax.set_ylabel("Return (%)")
        ax.set_xlabel("Quarter")
        ax.legend()
        plt.show()
    
    def plot_drawdowns_combined(self):
        equity_series = pd.Series(self.equity_curve, index=pd.to_datetime([self.data.index[0]] + self.trade_exit)).resample('D').last().ffill()
        peak = equity_series.cummax()
        strategy_drawdowns = ((equity_series - peak) / peak * 100)
        
        benchmark_equity = self.data['Close'].resample('D').last().dropna()
        benchmark_peak = benchmark_equity.cummax()
        benchmark_drawdowns = ((benchmark_equity - benchmark_peak) / benchmark_peak * 100)
        
        plt.figure(figsize=(14, 7))
        plt.plot(strategy_drawdowns.index, strategy_drawdowns, label="Strategy Drawdown", color="red")
        plt.fill_between(strategy_drawdowns.index, strategy_drawdowns, 0, color="red", alpha=0.3)
        plt.plot(benchmark_drawdowns.index, benchmark_drawdowns, label=f"{self.benchmark_name} Drawdown", color="gray", linestyle='--')
        plt.title("Drawdown Curves: Strategy vs. Benchmark")
        plt.ylabel("Drawdown (%)")
        plt.xlabel("Date")
        plt.legend()
        plt.show()

    # --- Comparison Strategy Methods (TA-Lib dependent) ---
    def mean_reversion_strategy(self):
        df = self.data.copy()
        df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
        df['Signal'] = np.where(df['RSI'] < 30, 1, np.where(df['RSI'] > 70, -1, 0))
        df['Strategy_Returns'] = df['Signal'].shift(1) * df['Close'].pct_change()
        return (1 + df['Strategy_Returns']).cumprod() - 1

    def moving_average_strategy(self):
        df = self.data.copy()
        df['SMA50'] = df['Close'].rolling(50).mean()
        df['SMA200'] = df['Close'].rolling(200).mean()
        df['Signal'] = np.where(df['SMA50'] > df['SMA200'], 1, -1)
        df['Strategy_Returns'] = df['Signal'].shift(1) * df['Close'].pct_change()
        return (1 + df['Strategy_Returns']).cumprod() - 1

    def momentum_strategy(self):
        df = self.data.copy()
        df['MACD'], df['Signal'], _ = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
        df['Signal'] = np.where(df['MACD'] > df['Signal'], 1, -1)
        df['Strategy_Returns'] = df['Signal'].shift(1) * df['Close'].pct_change()
        return (1 + df['Strategy_Returns']).cumprod() - 1

    def plot_strategy_comparison(self):
        mr_returns = self.mean_reversion_strategy()
        ma_returns = self.moving_average_strategy()
        mom_returns = self.momentum_strategy()
        strategy_returns_cum = (1 + self.strategy_daily_returns).cumprod() - 1
        
        plt.figure(figsize=(14, 8))
        plt.plot(strategy_returns_cum, label='Your FVG Strategy', color='black', linewidth=2)
        plt.plot(mr_returns, label='Mean Reversion (RSI)', linestyle='--')
        plt.plot(ma_returns, label='Moving Average Crossover', linestyle='--')
        plt.plot(mom_returns, label='Momentum (MACD)', linestyle='--')
        plt.title('Your Strategy vs. Common Benchmarks')
        plt.ylabel('Cumulative Returns')
        plt.xlabel('Date')
        plt.legend()
        plt.show()

    def create_strategy_summary_table(self):
        # This can be expanded to show a table of returns for the comparison strategies
        pass

    def run_common_strategies(self):
        self.plot_strategy_comparison()


# --- Main Execution Block ---
def main():
    """Main function to load backtest results and run the full performance analysis."""
    print("--- Loading Backtest Results for Final Analysis ---")
    
    data_agent = DataHandler({'source': 'csv', 'file_path': config.BACKTEST_DATA_FILE})
    df_backtest_raw = data_agent.get_data()
    
    try:
        with open(config.TRADE_RESULTS_FILE, 'r') as f:
            results = json.load(f)
        entries = results['entries']
        final_exits = results['final_exits']
        print(f"Loaded {len(entries)} trades from '{config.TRADE_RESULTS_FILE}'")
    except FileNotFoundError:
        print(f"Error: Trade results file not found. Please run backtester.py first.")
        return

    if df_backtest_raw is not None and entries:
        df_backtest_raw.reset_index(inplace=True)
        
        # --- ADAPTER LOGIC ---
        equity = [config.INITIAL_CAPITAL]
        for i in range(len(entries)):
            entry_price = entries[i]['entry']
            exit_price = final_exits[i]['price']
            side = entries[i]['side']
            percent_return = (exit_price / entry_price - 1) if side == 'LONG' else (entry_price / exit_price - 1)
            equity.append(equity[-1] * (1 + percent_return))
        
        formatted_trades = []
        for i in range(len(entries)):
            entry_datetime = df_backtest_raw['datetime'].iat[entries[i]['bar']]
            exit_datetime = df_backtest_raw['datetime'].iat[final_exits[i]['bar']]
            side = entries[i]['side'].lower()
            entry_price = entries[i]['entry']
            exit_price = final_exits[i]['price']
            percent_return = (exit_price / entry_price - 1) if side == 'long' else (entry_price / exit_price - 1)
            trade_tuple = (side, entry_price, exit_price, percent_return, exit_datetime)
            formatted_trades.append(entry_datetime)
            formatted_trades.append(trade_tuple)
        
        df_backtest_raw.set_index('datetime', inplace=True)
        
        analysis_agent = BacktestingAgent(
            initial_capital=config.INITIAL_CAPITAL,
            data=df_backtest_raw,
            equity_curve=equity,
            trades=formatted_trades,
            benchmark_name=config.BENCHMARK_NAME
        )
        
        analysis_agent.run_all_analysis()

if __name__ == "__main__":
    main()
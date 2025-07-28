# chart_visualizer.py
# Purpose: To load the results of a backtest and the corresponding price data,
#          and generate a high-quality candlestick chart visualizing the last
#          7000 bars with all trade entries and exits clearly marked.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from mplfinance.original_flavor import candlestick_ohlc
import config
from data_handler import DataHandler

class ChartVisualizationAgent:
    """
    An agent dedicated to creating detailed visualizations of backtesting results
    on a candlestick chart.
    """
    def __init__(self, price_data, entries, final_exits):
        """
        Initializes the agent with the necessary data.

        Args:
            price_data (pd.DataFrame): The DataFrame with OHLC data and a DatetimeIndex.
            entries (list): The list of entry trade dictionaries from the backtest.
            final_exits (list): The list of exit trade dictionaries from the backtest.
        """
        self.df = price_data
        self.entries = entries
        self.final_exits = final_exits

    def _prepare_plot_data(self, num_rows=config.CHART_VISUALIZATION_BARS):
        """
        Prepares the data for plotting by slicing the last N rows and mapping
        trade data to the correct indices for visualization.
        """
        print(f"Preparing plot data for the last {num_rows} bars...")
        
        # 1. Slice the most recent data for plotting
        self.plot_df = self.df.iloc[-num_rows:].copy()
        self.plot_df.reset_index(inplace=True) # Use integer index for plotting
        self.plot_df['x'] = np.arange(len(self.plot_df))
        
        # 2. Prepare OHLC data for mplfinance
        self.ohlc = self.plot_df[['x', 'Open', 'High', 'Low', 'Close']].values
        
        # 3. Filter and map trades that occurred within this sliced window
        plot_start_index = len(self.df) - num_rows
        
        self.plot_wins = []
        self.plot_losses = []
        self.plot_long_entries = []
        self.plot_short_entries = []

        num_trades = min(len(self.entries), len(self.final_exits))
        for i in range(num_trades):
            entry = self.entries[i]
            exit_trade = self.final_exits[i]
            
            # Check if the trade's entry bar is within our plotting window
            if entry['bar'] >= plot_start_index:
                # Adjust the bar number to the new, sliced index
                plot_bar_index = entry['bar'] - plot_start_index
                
                # Categorize entries
                if entry['side'] == 'LONG':
                    self.plot_long_entries.append((plot_bar_index, entry['entry']))
                else:
                    self.plot_short_entries.append((plot_bar_index, entry['entry']))

                # Categorize exits
                if exit_trade['type'] == 'TP':
                    self.plot_wins.append((exit_trade['bar'] - plot_start_index, exit_trade['price']))
                else: # SL
                    self.plot_losses.append((exit_trade['bar'] - plot_start_index, exit_trade['price']))

    def plot_strategy_chart(self):
        """
        Generates and displays the final candlestick chart with all strategy markers.
        """
        self._prepare_plot_data() # Prepare the data first

        if self.plot_df.empty:
            print("No data available for plotting.")
            return

        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=(16, 8))
        
        # Plot the candlesticks
        candlestick_ohlc(ax, self.ohlc, width=0.6, colorup='green', colordown='red', alpha=0.8)

        # Plot Long Entries (Blue Up Arrow)
        if self.plot_long_entries:
            x, y = zip(*self.plot_long_entries)
            ax.scatter(x, y, label='Long Entry', marker='^', color='blue', s=100, zorder=5, edgecolor='black')

        # Plot Short Entries (Orange Down Arrow)
        if self.plot_short_entries:
            x, y = zip(*self.plot_short_entries)
            ax.scatter(x, y, label='Short Entry', marker='v', color='orange', s=100, zorder=5, edgecolor='black')

        # Plot Wins (Green Circle)
        if self.plot_wins:
            x, y = zip(*self.plot_wins)
            ax.scatter(x, y, label='Take Profit', marker='o', color='lime', s=120, zorder=5, edgecolor='black')

        # Plot Losses (Red 'X')
        if self.plot_losses:
            x, y = zip(*self.plot_losses)
            ax.scatter(x, y, label='Stop Loss', marker='x', color='red', s=120, zorder=5)

        # Formatting the chart
        ax.set_title(f"FVG Strategy Backtest Results ({config.BENCHMARK_NAME} - Last 7000 Bars)", fontsize=16)
        ax.set_ylabel("Price", fontsize=12)
        ax.set_xlabel("Date/Time", fontsize=12)
        
        # Format x-axis to show dates
        tick_indices = np.linspace(0, len(self.plot_df) - 1, 8, dtype=int)
        ax.set_xticks(tick_indices)
        ax.set_xticklabels(self.plot_df['datetime'].iloc[tick_indices].dt.strftime('%Y-%m-%d %H:%M'), rotation=45, ha='right')

        # Create a clean legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper left')

        plt.tight_layout()
        plt.show()

# --- Main Execution Block ---
def main():
    """Main function to run the chart visualization."""
    print("--- Loading Backtest Results for Visualization ---")
    
    data_agent = DataHandler({'source': 'csv', 'file_path': config.BACKTEST_DATA_FILE})
    df_backtest_raw = data_agent.get_data()
    
    try:
        with open(config.TRADE_RESULTS_FILE, 'r') as f:
            results = json.load(f)
        entries = results['entries']
        final_exits = results['final_exits']
        print(f"Loaded {len(entries)} trades from '{config.TRADE_RESULTS_FILE}'")
    except FileNotFoundError:
        print(f"Error: Trade results file not found. Please run the backtester first.")
        return

    if df_backtest_raw is not None and entries:
        visualizer = ChartVisualizationAgent(
            price_data=df_backtest_raw,
            entries=entries,
            final_exits=final_exits
        )
        visualizer.plot_strategy_chart()

if __name__ == "__main__":
    main()
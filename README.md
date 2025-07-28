# Algorithmic Trading of ICT Market Structure: A Quantitative Approach to Price Action using a Machine Learning Confirmation Layer

## System Overview

This project implements an agent-based trading system that:

* Identifies ICT-inspired Fair Value Gaps (FVGs) on price charts.
* Uses a Random Forest classifier to filter for high-probability trades.
* Executes backtests with rigorous risk management (e.g., 2:1 RR, volatility filters).
* Generates in-depth performance analytics and clear visualizations.

## Agentic Workflow

-   **Data Handler Agent:** Imports and cleans OHLC data. By default, it uses `nq21-24.csv` for training and `nq25.csv` for forward testing. It can slice data based on parameters in `config.py`.
-   **Training Dataset Builder:** Runs a preliminary backtest of the core FVG strategy on historical data. It labels all potential trades as wins or losses and extracts a rich set of features (RSI, ADX, FVG size, etc.) for each trade setup.
-   **Model Trainer Agent:** Trains a Random Forest model on the labeled dataset to predict trade outcomes. It automatically optimizes the prediction probability threshold to maximize the F1-score, balancing precision and recall.
-   **Main Backtest Agent:** Runs the FVG strategy on new, unseen data, but with a crucial difference: it uses the trained ML model as a final confirmation step before entering a trade. It saves the results to `trade_results.json`.
-   **Performance Analyzer:** Takes the backtest results and computes a comprehensive suite of metrics (Sharpe Ratio, Sortino Ratio, drawdowns, etc.), comparing the strategy's performance against a buy-and-hold benchmark.
-   **Chart Visualizer:** Plots annotated candlestick charts of the backtest, clearly marking all entry and exit points for the last 7,000 bars.
-   **Main Agent:** Acts as the central orchestrator, providing a simple command-line interface (CLI) to run the entire pipeline or any individual agent.

## Key Features

* **Modular Design:** Each agent is a self-contained unit responsible for a specific task (data handling, ML training, backtesting, etc.), making the system easy to understand and modify.
* **Configuration-Driven:** All key strategy parameters (e.g., FVG displacement factor, RSI window, risk-reward ratio) are centralized in `config.py` for easy tweaking and experimentation.
* **Adaptive ML Filter:** The model learns to identify high-probability setups by analyzing a combination of technical indicators and structural market features.
* **Low Drawdown Profile:** In testing, the strategy demonstrated a significantly lower drawdown (-1.43%) compared to the Nasdaq 100 benchmark (-22.85%).

## Repository Structure

```plaintext
ICT-ML-Fusion-Trader/
│
├── Agents/
│   ├── data_handler.py             # Data ingestion/cleaning
│   ├── build_training_dataset.py   # FVG backtest + feature engineering
│   ├── model_trainer.py            # Random Forest training/evaluation
│   ├── main_run_backtest.py        # ML-filtered backtest
│   ├── performance_analyzer.py     # Metrics/analytics
│   ├── chart_visualizer.py         # Trade visualization
│   ├── main.py                     # CLI orchestrator
│   ├── config.py                   # Hyperparameters/file paths
│   └── install_libraries.py        # Dependency installer
│
├── Data/
│   ├── nq21-24.csv                 # Training data (2021–2024)
│   └── nq25.csv                    # Forward test data (2025)
│
└── README.md                       # You are here
```

## Backtest Summary (6-Month Forward Test)

| Metric          | Strategy  |  
|:----------------|:----------|
| Win Rate        | 52.00%    | 
| Profit Factor   | 2.41      |  
| Expectancy      | 0.1003%   | 
| Sharpe Ratio    | 2.18      | 
| Sortino Ratio   | 5.50      |  
| Beta            | -0.01     |  
| Max DD          | -1.43%    |    


## Quick Start

1.  **Clone the repo:**
    ```bash
    git clone [https://github.com/Taghi-Guliyev/ICT-ML-Trading-Strategy.git](https://github.com/Taghi-Guliyev/ICT-ML-Trading-Strategy.git)
    ```

2.  **Install dependencies:**
    ```bash
    python Agents/install_libraries.py
    ```

3.  **Run the pipeline:**
    ```bash
    python Agents/main.py
    ```
    *(Select option 6 to execute the full workflow from data processing to final analysis.)*

## Customization

* **Data:** Replace `nq21-24.csv` and `nq25.csv` in the `Data/` folder with your own OHLC data. Ensure it follows the same format.
* **Parameters:** Adjust strategy and model parameters in `config.py` (e.g., `FVG_DISPLACEMENT_FACTOR`, `RR_RATIO`, `RSI_WINDOW`).
* **Model:** The Random Forest model should be retrained if you switch to a different asset or timeframe to ensure its predictions are relevant.

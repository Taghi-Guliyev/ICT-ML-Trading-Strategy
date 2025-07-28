# config.py
# Purpose: Central configuration file for the entire ICT-ML trading strategy project.
# All file paths, strategy parameters, and model settings are defined here,
# allowing for easy modification and experimentation without changing the core logic of the agents.

import pandas as pd
import numpy as np

# ==============================================================================
# 1. FILE & DIRECTORY SETTINGS
# ==============================================================================
# -- Input Data --
# Data for training the model (historical in-sample period)
TRAINING_DATA_FILE = 'nq21-24.csv'
# Data for the final, unseen backtest (out-of-sample period)
BACKTEST_DATA_FILE = 'nq25.csv'

# -- Data Slicing Configuration --
# Set to True to use only the second half of the data for the TRAINING set.
SLICE_TRAINING_DATA = True
# Set to False to use the FULL dataset for the final BACKTEST set.
SLICE_BACKTEST_DATA = False

# -- Intermediate & Output Files --
# The CSV file where features for the ML model will be stored
ML_FEATURES_FILE = 'trading_features_for_ml.csv'
# Final trade results from the out-of-sample backtest
TRADE_RESULTS_FILE = 'trade_results.json'

# -- Model Artifacts --
# Path to save/load the trained Random Forest model
MODEL_PATH = 'rf_model_nq.joblib'
# Path to save/load the optimal prediction threshold
THRESHOLD_PATH = 'optimal_threshold_nq.joblib'

# ==============================================================================
# 2. FEATURE ENGINEERING PARAMETERS
# ==============================================================================
# -- Technical Indicator Settings --
RSI_WINDOW = 14
ADX_WINDOW = 14
MOMENTUM_WINDOW = 10
ATR_WINDOW = 14
STOCHASTIC_WINDOW = 14
STOCHASTIC_SMOOTH = 3
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BBANDS_WINDOW = 20
BBANDS_STD_DEV = 2
SMA_WINDOW = 50
PSAR_STEP = 0.02
PSAR_MAX_STEP = 0.2

# -- Custom Feature Settings --
BODY_SIZE_AVG_WINDOW = 20

# -- Higher-Timeframe (HTF) Settings --
HTF_TIMEFRAME = '15T'  # e.g., '15T' for 15 minutes, '1H' for 1 hour
HTF_EMA_WINDOW = 200

# ==============================================================================
# 3. STRATEGY & BACKTESTING PARAMETERS
# ==============================================================================
# -- Core FVG Strategy Logic --
FVG_DISPLACEMENT_FACTOR = 1.5  # FVG candle body must be > 1.5x the average
FVG_EXPIRY_BARS = 30           # How long an FVG remains valid
ENTRY_WINDOW_BARS = 15         # Max bars after FVG formation to look for entry
LIQ_GRAB_WINDOW = 14           # Lookback window for liquidity grabs
RR_RATIO = 2.0                 # Minimum required risk-to-reward ratio for entry
BE_LVL_RISK_FACTOR = 1.25      # Breakeven level calculation factor (e.g., 1.25 * risk)

# -- Volatility Regime Filter --
VOL_RANK_WINDOW = 100          # Lookback window for volatility percentile ranking
VOL_REGIME_BINS = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
VOL_REGIME_LABELS = ['1_Very_Low', '2_Low', '3_Normal', '4_High', '5_Very_High']
VOL_REGIMES_TO_AVOID = ['4_High', '5_Very_High']

# -- Swing Pivot Detection --
SWING_PIVOT_ORDER = 10         # Lookback/lookforward bars for detecting a swing high/low
SWING_PIVOT_PCT_DIFF = 0.001   # Minimum % difference between consecutive pivots

# ==============================================================================
# 4. MACHINE LEARNING MODEL PARAMETERS
# ==============================================================================
# The exact list of features the model will be trained on and use for prediction.
# This list is the single source of truth for the trainer and the backtester.
MODEL_FEATURES = [
    'candle_size_ratio', 'fvg_size_norm', 'fvg_age', 'dist_to_htf_ema',
    'RSI', 'ADX', 'Momentum', 'Stochastic', 'MACD_diff', 'BB_width',
    'Close_minus_SMA50', 'PSAR_diff', 'sin_minute', 'direction'
]

# -- Model Training Settings --
TRAIN_TEST_SPLIT_RATIO = 0.8   # 80% for training, 20% for testing/validation
RF_N_ESTIMATORS = 150          # Number of trees in the Random Forest
RF_RANDOM_STATE = 42           # Ensures reproducibility

# -- Threshold Optimization --
THRESHOLD_SEARCH_RANGE = np.arange(0.1, 0.5, 0.01)

# ==============================================================================
# 5. ANALYSIS & VISUALIZATION PARAMETERS
# ==============================================================================
INITIAL_CAPITAL = 10000
BENCHMARK_NAME = "Nasdaq 100"  # Name of the benchmark for comparison
RISK_FREE_RATE = 0.045         # Annualized risk-free rate for Sharpe/Sortino ratios

# -- Charting Settings --
CHART_VISUALIZATION_BARS = 7000 # Number of recent bars to plot in the chart_visualizer
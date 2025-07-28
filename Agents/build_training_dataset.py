# build_training_dataset.py
# Purpose: To load historical data (2021-2024), run the FVG strategy to find all
#          possible trade setups, engineer a comprehensive set of features for
#          each setup, and save the final labeled dataset for model training.

import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import ta
import config
from data_handler import DataHandler

class FeatureEngineeringAgent:
    """
    Handles all data transformation, from calculating technical indicators and
    custom features to identifying structural points like swing pivots.
    """
    def __init__(self, df):
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("DataFrame must have a DatetimeIndex.")
        self.df = df.copy()

    def _add_technical_indicators(self):
        print("Calculating technical indicators...")
        self.df['RSI'] = ta.momentum.RSIIndicator(close=self.df['Close'], window=config.RSI_WINDOW).rsi()
        self.df['ADX'] = ta.trend.ADXIndicator(high=self.df['High'], low=self.df['Low'], close=self.df['Close'], window=config.ADX_WINDOW).adx()
        self.df['Momentum'] = ta.momentum.ROCIndicator(close=self.df['Close'], window=config.MOMENTUM_WINDOW).roc()
        self.df['ATR'] = ta.volatility.AverageTrueRange(high=self.df['High'], low=self.df['Low'], close=self.df['Close'], window=config.ATR_WINDOW).average_true_range()
        self.df['Stochastic'] = ta.momentum.StochasticOscillator(high=self.df['High'], low=self.df['Low'], close=self.df['Close'], window=config.STOCHASTIC_WINDOW, smooth_window=config.STOCHASTIC_SMOOTH).stoch()
        macd = ta.trend.MACD(close=self.df['Close'], window_slow=config.MACD_SLOW, window_fast=config.MACD_FAST, window_sign=config.MACD_SIGNAL)
        self.df['MACD_diff'] = macd.macd_diff()
        bb = ta.volatility.BollingerBands(close=self.df['Close'], window=config.BBANDS_WINDOW, window_dev=config.BBANDS_STD_DEV)
        self.df['BB_width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / self.df['Close']
        sma50 = ta.trend.SMAIndicator(close=self.df['Close'], window=config.SMA_WINDOW).sma_indicator()
        self.df['Close_minus_SMA50'] = self.df['Close'] - sma50
        psar = ta.trend.PSARIndicator(high=self.df['High'], low=self.df['Low'], close=self.df['Close'], step=config.PSAR_STEP, max_step=config.PSAR_MAX_STEP)
        self.df['PSAR_diff'] = psar.psar() - self.df['Close']

    def _add_custom_features(self):
        print("Calculating custom features...")
        self.df['body_size'] = abs(self.df['Open'] - self.df['Close'])
        self.df['avg_body_size'] = self.df['body_size'].rolling(window=config.BODY_SIZE_AVG_WINDOW).mean().shift(1)
        self.df['candle_size_ratio'] = self.df['body_size'] / self.df['avg_body_size']
        self.df['minute'] = self.df.index.minute
        self.df['sin_minute'] = np.sin(2 * np.pi * self.df['minute'] / 60)
        
    def _add_htf_and_normalized_features(self):
        print("Calculating HTF bias and normalizing features...")
        df_15m = self.df['Close'].resample(config.HTF_TIMEFRAME).last().to_frame()
        df_15m['ema200'] = df_15m['Close'].ewm(span=config.HTF_EMA_WINDOW, adjust=False).mean()
        self.df['htf_bullish'] = (df_15m['ema200'] < df_15m['Close']).reindex(self.df.index, method='ffill')
        self.df['htf_bullish'].fillna(False, inplace=True)
        self.df['dist_to_htf_ema'] = abs(self.df['Close'] - df_15m['ema200'].reindex(self.df.index, method='ffill')) / self.df['ATR']
        self.df['Close_minus_SMA50'] = abs(self.df['Close_minus_SMA50']) / self.df['ATR']
        self.df['PSAR_diff'] = abs(self.df['PSAR_diff']) / self.df['ATR']

    def generate_features(self):
        """Orchestrates the entire feature generation process."""
        self._add_technical_indicators()
        self._add_custom_features()
        self._add_htf_and_normalized_features()
        self.df.dropna(inplace=True)
        print(f"Feature generation complete. Final shape: {self.df.shape}")
        return self.df

def run_feature_collection_backtest(df):
    """
    Runs the FVG strategy to find all possible trade setups, assigns the outcome (target),
    and returns a clean DataFrame ready for the ML model.
    """
    print("Running backtest to generate labeled training data...")
    df.reset_index(inplace=True)
    n = len(df)
    
    # --- Swing Pivots Calculation ---
    k, diff = config.SWING_PIVOT_ORDER, config.SWING_PIVOT_PCT_DIFF
    lows = df['Low'].values
    raw_lows = argrelextrema(lows, np.less, order=k)[0]
    sig_sw_l = []
    last = None
    for p in raw_lows:
        lvl = df['Low'].iat[p]
        if last is None or abs(lvl - last) / last >= diff:
            sig_sw_l.append({'pivot': p, 'level': lvl, 'expire': n - 1})
            last = lvl
    for s in sig_sw_l:
        s['expire'] = next((j for j in range(s['pivot'] + 1, n) if df['Low'].iat[j] <= s['level']), n - 1)

    highs = df['High'].values
    raw_highs = argrelextrema(highs, np.greater, order=k)[0]
    sig_sw_h = []
    last = None
    for p in raw_highs:
        lvl = df['High'].iat[p]
        if last is None or abs(lvl - last) / last >= diff:
            sig_sw_h.append({'pivot': p, 'level': lvl, 'expire': n - 1})
            last = lvl
    for s in sig_sw_h:
        s['expire'] = next((j for j in range(s['pivot'] + 1, n) if df['High'].iat[j] >= s['level']), n - 1)
        
    # --- Backtesting Loop ---
    long_merged, short_merged, entries, final_exits = [], [], [], []
    current, long_entered, short_entered = None, set(), set()
    ml_data_collector = []

    for i in range(3, n):
        # FVG Detection
        if df['High'].iat[i-1] < df['Low'].iat[i-3]:
            if df['body_size'].iat[i-2] > (df['avg_body_size'].iat[i] * config.FVG_DISPLACEMENT_FACTOR):
                gap = {'idx':i-1, 'top':df['High'].iat[i-1], 'bot':df['Low'].iat[i-3], 'sl':df['Low'].iat[i-1], 'expire':min(i-1+config.FVG_EXPIRY_BARS, n-1)}
                if long_merged and long_merged[-1]['idx']==gap['idx']-1: long_merged[-1].update(gap)
                else: long_merged.append(gap)
        
        if df['Low'].iat[i-1] > df['High'].iat[i-3]:
            if df['body_size'].iat[i-2] > (df['avg_body_size'].iat[i] * config.FVG_DISPLACEMENT_FACTOR):
                gap = {'idx': i-1, 'top': df['Low'].iat[i-1], 'bot': df['High'].iat[i-3], 'sl': df['High'].iat[i-1], 'expire':min(i-1+config.FVG_EXPIRY_BARS, n-1)}
                if short_merged and short_merged[-1]['idx'] == gap['idx'] - 1: short_merged[-1].update(gap)
                else: short_merged.append(gap)

        if current is None:
            j = i - 1
            long_can_enter, short_can_enter = False, False
            be_lvl = None 

            # Long Entry Logic
            long_latest = next((g for g in reversed(long_merged) if g['idx'] <= j <= g['expire']), None)
            if (long_latest and long_latest['idx'] not in long_entered and df['Close'].iat[j] > long_latest['bot'] and j - long_latest['idx'] <= config.ENTRY_WINDOW_BARS):
                if df['htf_bullish'].iat[j]:
                    valid_swing_low = next((s for s in reversed(sig_sw_l) if s['pivot'] <= j <= s['expire']), None)
                    if valid_swing_low and df['Low'].iloc[j-config.LIQ_GRAB_WINDOW:j+1].min() < valid_swing_low['level']:
                        entry_approx = df['Close'].iat[j]
                        rr2 = entry_approx - long_latest['sl']
                        valh = [h for h in reversed(sig_sw_h) if h['pivot'] <= j <= h['expire'] and h['level'] > entry_approx]
                        if valh and max([h['level'] - entry_approx for h in valh]) >= config.RR_RATIO * rr2:
                            distances = sorted([h['level'] - entry_approx for h in valh])
                            nearest = max(distances[:2]) if len(distances) > 1 else (distances[0] if distances else 0)
                            be_lvl = entry_approx + min(config.BE_LVL_RISK_FACTOR * rr2, nearest)
                            if df['High'].iat[j] <= be_lvl:
                                long_can_enter = True
            
            # Short Entry Logic
            short_latest = next((g for g in reversed(short_merged) if g['idx'] <= j <= g['expire']), None)
            if (short_latest and short_latest['idx'] not in short_entered and df['Close'].iat[j] < short_latest['bot'] and j - short_latest['idx'] <= config.ENTRY_WINDOW_BARS):
                if not df['htf_bullish'].iat[j]:
                    valid_swing_high = next((s for s in reversed(sig_sw_h) if s['pivot'] <= j <= s['expire']), None)
                    if valid_swing_high and df['High'].iloc[j-config.LIQ_GRAB_WINDOW:j+1].max() > valid_swing_high['level']:
                        entry_approx = df['Close'].iat[j]
                        rr2 = short_latest['sl'] - entry_approx
                        vall = [l for l in reversed(sig_sw_l) if l['pivot'] <= j <= l['expire'] and l['level'] < entry_approx]
                        if vall and max([entry_approx - l['level'] for l in vall]) >= config.RR_RATIO * rr2:
                            distances = sorted([entry_approx - l['level'] for l in vall])
                            nearest = max(distances[:2]) if len(distances) > 1 else (distances[0] if distances else 0)
                            be_lvl = entry_approx - min(config.BE_LVL_RISK_FACTOR * rr2, nearest)
                            if df['Low'].iat[j] >= be_lvl:
                                short_can_enter = True

            if long_can_enter or short_can_enter:
                side_str, latest_fvg = ('LONG', long_latest) if long_can_enter else ('SHORT', short_latest)
                entry_price = df['Open'].iat[i]
                features = {
                    'entry_bar': i, 'side': side_str,
                    'candle_size_ratio': df['candle_size_ratio'].iat[j],
                    'fvg_size_norm': abs(latest_fvg['top'] - latest_fvg['bot']) / df['ATR'].iat[j] if df['ATR'].iat[j] > 0 else 0,
                    'fvg_age': j - latest_fvg['idx'],
                    'dist_to_htf_ema': df['dist_to_htf_ema'].iat[j],
                    'RSI': df['RSI'].iat[j], 'ADX': df['ADX'].iat[j], 'Momentum': df['Momentum'].iat[j],
                    'Stochastic': df['Stochastic'].iat[j], 'MACD_diff': df['MACD_diff'].iat[j],
                    'BB_width': df['BB_width'].iat[j], 'Close_minus_SMA50': df['Close_minus_SMA50'].iat[j],
                    'PSAR_diff': df['PSAR_diff'].iat[j], 'sin_minute': df['sin_minute'].iat[j],
                    'direction': 1 if side_str == 'LONG' else 0
                }
                ml_data_collector.append(features)
                
                tp_price = entry_price + (entry_price - latest_fvg['sl']) * 3
                if side_str == 'SHORT':
                    tp_price = entry_price - (latest_fvg['sl'] - entry_price) * 3
                current = {'bar':i, 'entry':entry_price, 'sl':latest_fvg['sl'], 'tp':tp_price, 'side':side_str, 'be_lvl':be_lvl}
                entries.append(current.copy())
                (long_entered if side_str == 'LONG' else short_entered).add(latest_fvg['idx'])
        else: # Exit Logic
            e = current
            if e['side'] == 'LONG':
                if df['Low'].iat[i] <= e['sl']: final_exits.append({'bar':i, 'price':e['sl'], 'type':'SL'}); current = None
                elif df['High'].iat[i] >= e['tp']: final_exits.append({'bar':i, 'price':e['tp'], 'type':'TP'}); current = None
            elif e['side'] == 'SHORT':
                if df['High'].iat[i] >= e['sl']: final_exits.append({'bar':i, 'price':e['sl'], 'type':'SL'}); current = None
                elif df['Low'].iat[i] <= e['tp']: final_exits.append({'bar':i, 'price':e['tp'], 'type':'TP'}); current = None
    
    if not ml_data_collector:
        print("Warning: No trade setups were found during data collection.")
        return pd.DataFrame()

    features_df = pd.DataFrame(ml_data_collector)
    outcomes = {entries[i]['bar']: final_exits[i]['type'] for i in range(len(final_exits))}
    features_df['outcome'] = features_df['entry_bar'].map(outcomes)
    features_df['target'] = features_df['outcome'].apply(lambda x: 1 if x == 'TP' else 0)
    features_df.dropna(subset=['outcome'], inplace=True)
    return features_df

def main():
    """Main function to run the data collection and feature engineering pipeline."""
    print("--- Starting Feature Generation Process for Model Training ---")

    config_training = {
        'source': 'csv',
        'file_path': config.TRAINING_DATA_FILE,
        'slice_data': config.SLICE_TRAINING_DATA # <-- This line is the fix
    }
    
    training_data_agent = DataHandler(config_training)
    df_train_raw = training_data_agent.get_data()

    if df_train_raw is not None:
        feature_agent = FeatureEngineeringAgent(df_train_raw)
        df_train_features = feature_agent.generate_features()
        training_dataset = run_feature_collection_backtest(df_train_features)
        
        if not training_dataset.empty:
            training_dataset.to_csv(config.ML_FEATURES_FILE, index=False)
            print(f"\nSuccessfully generated and saved training dataset with {len(training_dataset)} samples to '{config.ML_FEATURES_FILE}'")

if __name__ == "__main__":
    main()
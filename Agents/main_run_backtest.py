# backtester.py
# Purpose: To simulate the FVG trading strategy on new, unseen out-of-sample data
# using the pre-trained Random Forest model as a final trade filter. The results
# are saved for final analysis by the performance_analyzer.py agent.

import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import ta
import joblib
import json
import config
from data_handler import DataHandler
from build_training_dataset import FeatureEngineeringAgent

class BacktestingAgent:
    """
    An agent that runs the trading strategy with a machine learning model filter
    on unseen historical data to simulate live performance.
    """
    def __init__(self, df, model, threshold):
        self.df = df
        self.model = model
        self.threshold = threshold
        self.df.reset_index(inplace=True)
        self.n = len(self.df)

    def _prepare_backtest(self):
        """Prepares necessary data structures like swing pivots using config parameters."""
        print("Preparing backtest environment (calculating swing pivots)...")
        # --- CONNECTED TO CONFIG ---
        k_l, zz_l = config.SWING_PIVOT_ORDER, config.SWING_PIVOT_PCT_DIFF
        lows = self.df['Low'].values
        raw_lows = argrelextrema(lows, np.less, order=k_l)[0]
        self.sig_sw_l = []
        last = None
        for p in raw_lows:
            lvl = self.df['Low'].iat[p]
            if last is None or abs(lvl-last)/last >= zz_l:
                self.sig_sw_l.append({'pivot':p,'level':lvl,'expire':None})
                last = lvl
        for s in self.sig_sw_l:
            exp = next((j for j in range(s['pivot']+1,self.n) if self.df['Low'].iat[j]<=s['level']), self.n-1)
            s['expire'] = exp

        k_h, zz_h = config.SWING_PIVOT_ORDER, config.SWING_PIVOT_PCT_DIFF
        highs = self.df['High'].values
        raw_highs = argrelextrema(highs, np.greater, order=k_h)[0]
        self.sig_sw_h = []
        last = None
        for p in raw_highs:
            lvl = self.df['High'].iat[p]
            if last is None or abs(lvl-last)/last >= zz_h:
                self.sig_sw_h.append({'pivot':p,'level':lvl,'expire':None})
                last = lvl
        for s in self.sig_sw_h:
            exp = next((j for j in range(s['pivot']+1,self.n) if self.df['High'].iat[j]>=s['level']), self.n-1)
            s['expire'] = exp

    def _get_features(self, j, side, latest_fvg):
        """Helper function to assemble the feature dictionary for the model."""
        direction = 1 if side == 'LONG' else 0
        return {
            'candle_size_ratio': self.df['candle_size_ratio'].iat[j],
            'fvg_size_norm': abs(latest_fvg['top'] - latest_fvg['bot']) / self.df['ATR'].iat[j] if self.df['ATR'].iat[j] > 0 else 0,
            'fvg_age': j - latest_fvg['idx'],
            'dist_to_htf_ema': self.df['dist_to_htf_ema'].iat[j],
            'RSI': self.df['RSI'].iat[j],
            'ADX': self.df['ADX'].iat[j],
            'Momentum': self.df['Momentum'].iat[j],
            'Stochastic': self.df['Stochastic'].iat[j],
            'MACD_diff': self.df['MACD_diff'].iat[j],
            'BB_width': self.df['BB_width'].iat[j],
            'Close_minus_SMA50': self.df['Close_minus_SMA50'].iat[j],
            'PSAR_diff': self.df['PSAR_diff'].iat[j],
            'sin_minute': self.df['sin_minute'].iat[j],
            'direction': direction
        }

    def run(self):
        """Executes the bar-by-bar backtest, applying the ML model as a filter."""
        self._prepare_backtest()
        print("Running main backtest loop with ML filter...")

        long_merged, short_merged, entries, final_exits = [], [], [], []
        current, long_entered, short_entered = None, set(), set()

        for i in range(3, self.n):
            # FVG Detection
            if self.df['High'].iat[i-1] < self.df['Low'].iat[i-3]:
                fvg_candle_body = self.df['body_size'].iat[i-2]
                avg_body = self.df['avg_body_size'].iat[i]
                if fvg_candle_body > (avg_body * config.FVG_DISPLACEMENT_FACTOR):
                    gap = {'idx':i-1, 'top':self.df['High'].iat[i-1], 'bot':self.df['Low'].iat[i-3], 'sl':self.df['Low'].iat[i-1], 'expire':min(i-1+config.FVG_EXPIRY_BARS, self.n-1)}
                    if long_merged and long_merged[-1]['idx']==gap['idx']-1: long_merged[-1].update(gap)
                    else: long_merged.append(gap)
            
            if self.df['Low'].iat[i-1] > self.df['High'].iat[i-3]:
                fvg_candle_body = self.df['body_size'].iat[i-2]
                avg_body = self.df['avg_body_size'].iat[i]
                if fvg_candle_body > (avg_body * config.FVG_DISPLACEMENT_FACTOR):
                    gap = {'idx': i-1, 'top': self.df['Low'].iat[i-1], 'bot': self.df['High'].iat[i-3], 'sl': self.df['High'].iat[i-1], 'expire': min(i-1+config.FVG_EXPIRY_BARS, self.n-1)}
                    if short_merged and short_merged[-1]['idx'] == gap['idx'] - 1: short_merged[-1].update(gap)
                    else: short_merged.append(gap)

            if current is None:
                j = i - 1
                long_can_enter, short_can_enter = False, False
                
                # Long Entry Logic
                long_latest = next((g for g in reversed(long_merged) if g['idx'] <= j <= g['expire']), None)
                if (long_latest and long_latest['idx'] not in long_entered and self.df['Close'].iat[j] > long_latest['bot'] and j - long_latest['idx'] <= config.ENTRY_WINDOW_BARS):
                    if self.df['htf_bullish'].iat[j]:
                        valid_swing_low = next((s for s in reversed(self.sig_sw_l) if s['pivot'] <= j <= s['expire']), None)
                        if valid_swing_low and self.df['Low'].iloc[j-config.LIQ_GRAB_WINDOW:j+1].min() < valid_swing_low['level']:
                            entry_approx = self.df['Close'].iat[j]
                            rr2 = entry_approx - long_latest['sl']
                            valh = [h for h in reversed(self.sig_sw_h) if h['pivot'] <= j <= h['expire'] and h['level'] > entry_approx]
                            if valh and max([h['level'] - entry_approx for h in valh]) >= config.RR_RATIO * rr2:
                                distances = sorted([h['level'] - entry_approx for h in valh])
                                nearest = max(distances[:2]) if len(distances) > 1 else distances[0]
                                be_lvl = entry_approx + min(config.BE_LVL_RISK_FACTOR * rr2, nearest)
                                if self.df['High'].iat[j] <= be_lvl:
                                    features = self._get_features(j, 'LONG', long_latest)
                                    feature_values = pd.DataFrame([features], columns=config.MODEL_FEATURES)
                                    win_probability = self.model.predict_proba(feature_values)[0][1]
                                    if win_probability >= self.threshold:
                                        long_can_enter = True

                # Short Entry Logic
                short_latest = next((g for g in reversed(short_merged) if g['idx'] <= j <= g['expire']), None)
                if (short_latest and short_latest['idx'] not in short_entered and self.df['Close'].iat[j] < short_latest['bot'] and j - short_latest['idx'] <= config.ENTRY_WINDOW_BARS):
                    if not self.df['htf_bullish'].iat[j]:
                        valid_swing_high = next((s for s in reversed(self.sig_sw_h) if s['pivot'] <= j <= s['expire']), None)
                        if valid_swing_high and self.df['High'].iloc[j-config.LIQ_GRAB_WINDOW:j+1].max() > valid_swing_high['level']:
                            entry_approx = self.df['Close'].iat[j]
                            rr2 = short_latest['sl'] - entry_approx
                            vall = [l for l in reversed(self.sig_sw_l) if l['pivot'] <= j <= l['expire'] and l['level'] < entry_approx]
                            if vall and max([entry_approx - l['level'] for l in vall]) >= config.RR_RATIO * rr2:
                                distances = sorted([entry_approx - l['level'] for l in vall])
                                nearest = max(distances[:2]) if len(distances) > 1 else distances[0]
                                be_lvl = entry_approx - min(config.BE_LVL_RISK_FACTOR * rr2, nearest)
                                if self.df['Low'].iat[j] >= be_lvl:
                                    features = self._get_features(j, 'SHORT', short_latest)
                                    feature_values = pd.DataFrame([features], columns=config.MODEL_FEATURES)
                                    win_probability = self.model.predict_proba(feature_values)[0][1]
                                    if win_probability >= self.threshold:
                                        short_can_enter = True
                
                # Execute Entry
                if long_can_enter:
                    entry_price = self.df['Open'].iat[i]
                    tp_price = entry_price + (entry_price - long_latest['sl']) * 3 # Simplified TP
                    current = {'bar':i, 'entry':entry_price, 'sl':long_latest['sl'], 'tp':tp_price, 'side':'LONG'}
                    entries.append(current.copy())
                    long_entered.add(long_latest['idx'])
                elif short_can_enter:
                    entry_price = self.df['Open'].iat[i]
                    tp_price = entry_price - (short_latest['sl'] - entry_price) * 3
                    current = {'bar':i, 'entry':entry_price, 'sl':short_latest['sl'], 'tp':tp_price, 'side':'SHORT'}
                    entries.append(current.copy())
                    short_entered.add(short_latest['idx'])
            else: # Exit Logic
                e = current
                if e['side'] == 'LONG':
                    if self.df['Low'].iat[i] <= e['sl']:
                        final_exits.append({'bar':i, 'price':e['sl'], 'type':'SL'}); current = None
                    elif self.df['High'].iat[i] >= e['tp']:
                        final_exits.append({'bar':i, 'price':e['tp'], 'type':'TP'}); current = None
                elif e['side'] == 'SHORT':
                    if self.df['High'].iat[i] >= e['sl']:
                        final_exits.append({'bar':i, 'price':e['sl'], 'type':'SL'}); current = None
                    elif self.df['Low'].iat[i] <= e['tp']:
                        final_exits.append({'bar':i, 'price':e['tp'], 'type':'TP'}); current = None
                        
        return entries, final_exits

# --- Main Execution Block ---
def main():
    """Main function to run the out-of-sample backtest with the ML filter."""
    print("--- Starting Final Backtest on Unseen Data ---")

    try:
        model = joblib.load(config.MODEL_PATH)
        threshold = joblib.load(config.THRESHOLD_PATH)
    except FileNotFoundError:
        print(f"Error: Model files not found. Please run 'build_training_dataset.py' and 'model_trainer.py' first.")
        return

    data_agent = DataHandler({'source': 'csv', 'file_path': config.BACKTEST_DATA_FILE})
    df_raw = data_agent.get_data()

    if df_raw is not None:
        feature_agent = FeatureEngineeringAgent(df_raw)
        df_features = feature_agent.generate_features()

        backtester = BacktestingAgent(df_features, model, threshold)
        entries, final_exits = backtester.run()
        
        results = {'entries': entries, 'final_exits': final_exits}
        try:
            with open(config.TRADE_RESULTS_FILE, 'w') as f:
                def convert(o):
                    if isinstance(o, (np.int64, np.int32)): return int(o)  
                    raise TypeError
                json.dump(results, f, indent=4, default=convert)
            print(f"\nBacktest complete. Found {len(entries)} trades.")
            print(f"Trade results saved to '{config.TRADE_RESULTS_FILE}'")
        except Exception as e:
            print(f"Error saving results to JSON: {e}")

if __name__ == "__main__":
    main()
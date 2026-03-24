import pandas as pd
import numpy as np
import pandas_ta as ta
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. Data Loader (Multi-Timeframe)
# ==========================================
class DataLoader:
    def __init__(self, filepath):
        self.filepath = filepath

    def load_data(self):
        try:
            print(f"Loading Raw Data from {self.filepath}...")
            df = pd.read_csv(self.filepath)
            
            # Smart Map
            rename_map = {
                'Open time': 'Timestamp', 'open_time': 'Timestamp',
                'Taker buy base asset volume': 'Taker_Buy_Vol'
            }
            df = df.rename(columns=rename_map)
            
            # Date Parse
            try:
                df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            except:
                df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
            df = df.set_index('Timestamp')
            
            # Numeric
            cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Taker_Buy_Vol']
            for c in cols:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors='coerce')
            
            df = df.dropna()
            print(f"Loaded {len(df)} rows.")
            return df
        except Exception as e:
            print(f"Error: {e}")
            return pd.DataFrame()

# ==========================================
# 2. Feature Factory (1H Trend + 5m Entry)
# ==========================================
class FeatureFactory:
    def __init__(self, df):
        self.df = df.copy()

    def engineer_features(self):
        print("Engineering Multi-Timeframe Matrix...")
        
        # --- MACRO (1-Hour) ---
        # Resample to 1H
        df_1h = self.df.resample('1h').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last',
            'Volume': 'sum', 'Taker_Buy_Vol': 'sum'
        }).dropna()
        
        # 1H Indicators (Trend Direction)
        df_1h['EMA_50_1H'] = ta.ema(df_1h['Close'], length=50)
        
        # 1H Order Flow (Whale Bias)
        buy_vol = df_1h['Taker_Buy_Vol']
        sell_vol = df_1h['Volume'] - buy_vol
        df_1h['Flow_Bias_1H'] = np.where(buy_vol > sell_vol, 1, -1) # 1 = Bulls, -1 = Bears
        
        # CRITICAL: Shift 1H data forward by 1 hour.
        # At 10:15 (5m candle), we only know the 1H trend from the 09:00-10:00 candle.
        df_1h_shifted = df_1h.shift(1)
        
        # --- MICRO (5-Minute) ---
        # Resample to 5m
        df_5m = self.df.resample('5min').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last',
            'Volume': 'sum', 'Taker_Buy_Vol': 'sum'
        }).dropna()
        
        # 5m Indicators (Entry Signals)
        # RSI for Pullbacks
        df_5m['RSI_5m'] = ta.rsi(df_5m['Close'], length=14)
        
        # Merge 1H Trend onto 5m Data
        # ffill() propagates the last known 1H trend to all 12 of the 5m candles in that hour
        merged_df = df_5m.join(df_1h_shifted[['EMA_50_1H', 'Flow_Bias_1H']], how='left')
        merged_df = merged_df.ffill()
        merged_df.dropna(inplace=True)
        
        return merged_df

# ==========================================
# 3. Strategy Engine V20 (Fractal Flow)
# ==========================================
class StrategyEngine:
    def __init__(self, df):
        self.df = df
        self.test_df = df.copy()
        self.results = []

    def _backtest(self, name, signals):
        price = self.test_df['Close'].values
        timestamps = self.test_df.index
        sigs = signals.values
        
        in_pos = False
        entry_price = 0.0
        trades = []
        fee = 0.0006
        
        for i in range(1, len(price)):
            if in_pos:
                # Exit Signal
                if sigs[i] == -1:
                    pnl = (price[i] - entry_price) / entry_price - (fee * 2)
                    trades.append(pnl)
                    in_pos = False
                # Hard Stop (3% safety)
                elif price[i] < entry_price * 0.97:
                    pnl = -0.03 - (fee * 2)
                    trades.append(pnl)
                    in_pos = False
            
            elif not in_pos:
                if sigs[i] == 1:
                    entry_price = price[i]
                    in_pos = True

        if not trades: return
        
        trades = np.array(trades)
        wins = trades[trades > 0]
        losses = trades[trades <= 0]
        
        net_profit = (np.prod(1 + trades) - 1) * 100
        pf = np.sum(wins) / np.abs(np.sum(losses)) if len(losses) > 0 else 0
        win_rate = len(wins) / len(trades) * 100
        
        is_golden = (pf >= 1.5)

        self.results.append({
            'Strategy': name,
            'Net Profit %': round(net_profit, 2),
            'PF': round(pf, 2),
            'Win Rate %': round(win_rate, 1),
            'Trades': len(trades),
            'Golden': is_golden
        })

    def run_hybrid_strategies(self):
        print("Running V20 Hybrid Strategies...")
        df = self.test_df
        
        # --- STRAT 1: Trend Alignment + RSI Dip ---
        # Logic: 
        # 1. 1H Trend is UP (Price > EMA 50)
        # 2. 1H Order Flow is Bullish
        # 3. 5m RSI < 30 (Buying the dip in a verified uptrend)
        
        trend_up = (df['Close'] > df['EMA_50_1H'])
        flow_bull = (df['Flow_Bias_1H'] == 1)
        dip_buy = (df['RSI_5m'] < 30)
        
        s1_entry = trend_up & flow_bull & dip_buy
        
        # Exit: 5m RSI becomes overbought (RSI > 70)
        s1_exit = (df['RSI_5m'] > 70)
        
        self._run_strat("Hybrid_Flow_Dip_RSI30", s1_entry, s1_exit)
        
        # --- STRAT 2: Aggressive Dip (RSI < 40) ---
        # Same logic, but more aggressive entry
        dip_aggressive = (df['RSI_5m'] < 40)
        s2_entry = trend_up & flow_bull & dip_aggressive
        self._run_strat("Hybrid_Flow_Dip_RSI40", s2_entry, s1_exit)

    def _run_strat(self, name, entry_cond, exit_cond):
        signals = pd.Series(0, index=self.test_df.index)
        signals[entry_cond] = 1
        signals[exit_cond] = -1
        signals = signals.replace(0, np.nan).ffill().fillna(0)
        self._backtest(name, signals)

# ==========================================
# 4. Main
# ==========================================
def main():
    print("Initializing Framework V20 (The Fractal Flow)...")
    
    loader = DataLoader('Data/btc_1m_orderflow.csv') 
    df = loader.load_data()
    
    if df.empty: return

    ff = FeatureFactory(df)
    processed_df = ff.engineer_features()

    engine = StrategyEngine(processed_df)
    engine.run_hybrid_strategies()

    results_df = pd.DataFrame(engine.results)
    if not results_df.empty:
        results_df.sort_values(by=['PF'], ascending=False, inplace=True)
        print("\n" + "="*50)
        print("HYBRID LEADERBOARD")
        print("="*50)
        print(results_df.to_string(index=False))
        
        golden = results_df[results_df['Golden']==True]
        if not golden.empty:
             print("\n=== 🏆 GOLDEN MODEL FOUND 🏆 ===")
             print(golden.iloc[0].to_string())
    else:
        print("\nNo trades generated.")

if __name__ == "__main__":
    main()

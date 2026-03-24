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
    def __init__(self, df, fee=0.0006, stop_loss_pct=0.03):
        self.df = df
        self.test_df = df.copy()
        self.results = []
        self.fee = fee
        self.stop_loss_pct = stop_loss_pct

    # Changed 'signals' to 'entries' and 'exits'
    def _backtest(self, name, entries, exits):
        price = self.test_df['Close'].values
        
        # Convert pandas boolean series to fast numpy arrays
        entries_arr = entries.values
        exits_arr = exits.values
        
        in_pos = False
        entry_price = 0.0
        trades = []
        
        for i in range(1, len(price)):
            if in_pos:
                # Exit Signal (Checking the boolean array directly)
                if exits_arr[i]:
                    pnl = (price[i] - entry_price) / entry_price - (self.fee * 2)
                    trades.append(pnl)
                    in_pos = False
                # Hard Stop
                elif price[i] < entry_price * (1 - self.stop_loss_pct):
                    pnl = -self.stop_loss_pct - (self.fee * 2)
                    trades.append(pnl)
                    in_pos = False
            
            elif not in_pos:
                # Entry Signal (Checking the boolean array directly)
                if entries_arr[i]:
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
        trend_up = (df['Close'] > df['EMA_50_1H'])
        flow_bull = (df['Flow_Bias_1H'] == 1)
        dip_buy = (df['RSI_5m'] < 30)
        
        s1_entry = trend_up & flow_bull & dip_buy
        s1_exit = (df['RSI_5m'] > 70)
        
        self._run_strat("Hybrid_Flow_Dip_RSI30", s1_entry, s1_exit)
        
        # --- STRAT 2: Aggressive Dip (RSI < 40) ---
        dip_aggressive = (df['RSI_5m'] < 40)
        s2_entry = trend_up & flow_bull & dip_aggressive
        self._run_strat("Hybrid_Flow_Dip_RSI40", s2_entry, s1_exit)

    def _run_strat(self, name, entry_cond, exit_cond):
        # We no longer build a merged signal line. 
        # We pass the boolean conditions straight to the backtester!
        self._backtest(name, entry_cond, exit_cond)

# ==========================================
# 4. Main
# ==========================================
def main():
    print("Initializing Framework V20 (The Fractal Flow)...")
    
    # Extracted configuration variables to the top
    data_filepath = 'Data/btc_1m_orderflow.csv'
    exchange_fee = 0.0006
    max_stop_loss = 0.03
    
    loader = DataLoader(data_filepath) 
    df = loader.load_data()
    
    if df.empty: return

    ff = FeatureFactory(df)
    processed_df = ff.engineer_features()

    # Pass the variables into the engine
    engine = StrategyEngine(processed_df, fee=exchange_fee, stop_loss_pct=max_stop_loss)
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

import pandas as pd
import numpy as np
import pandas_ta as ta
import matplotlib.pyplot as plt
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')
# Use a nice style for plots
plt.style.use('bmh') 

# ==========================================
# 1. Data Loader (Same as V20)
# ==========================================
class DataLoader:
    def __init__(self, filepath):
        self.filepath = filepath

    def load_data(self):
        try:
            print(f"Loading Raw Data from {self.filepath}...")
            df = pd.read_csv(self.filepath)
            
            rename_map = {
                'Open time': 'Timestamp', 'open_time': 'Timestamp',
                'Taker buy base asset volume': 'Taker_Buy_Vol'
            }
            df.rename(columns=rename_map, inplace=True)
            
            try:
                df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            except:
                df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
            df.set_index('Timestamp', inplace=True)
            
            cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Taker_Buy_Vol']
            for c in cols:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors='coerce')
            
            df.dropna(inplace=True)
            print(f"Loaded {len(df)} rows.")
            return df
        except Exception as e:
            print(f"Error: {e}")
            return pd.DataFrame()

# ==========================================
# 2. Feature Factory (V20 Hybrid Logic)
# ==========================================
class FeatureFactory:
    def __init__(self, df):
        self.df = df.copy()

    def engineer_features(self):
        print("Engineering Multi-Timeframe Matrix...")
        
        # --- MACRO (1-Hour) ---
        df_1h = self.df.resample('1h').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last',
            'Volume': 'sum', 'Taker_Buy_Vol': 'sum'
        }).dropna()
        
        df_1h['EMA_50_1H'] = ta.ema(df_1h['Close'], length=50)
        
        buy_vol = df_1h['Taker_Buy_Vol']
        sell_vol = df_1h['Volume'] - buy_vol
        df_1h['Flow_Bias_1H'] = np.where(buy_vol > sell_vol, 1, -1)
        
        df_1h_shifted = df_1h.shift(1)
        
        # --- MICRO (5-Minute) ---
        df_5m = self.df.resample('5min').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last',
            'Volume': 'sum', 'Taker_Buy_Vol': 'sum'
        }).dropna()
        
        df_5m['RSI_5m'] = ta.rsi(df_5m['Close'], length=14)
        
        merged_df = df_5m.join(df_1h_shifted[['EMA_50_1H', 'Flow_Bias_1H']], how='left')
        merged_df.ffill(inplace=True)
        merged_df.dropna(inplace=True)
        
        return merged_df

# ==========================================
# 3. Strategy Engine (With Visualization Data)
# ==========================================
class StrategyEngine:
    def __init__(self, df):
        self.df = df
        self.test_df = df.copy()
        self.results = []
        self.equity_curves = {} # Store curves for plotting
        self.trade_logs = {}    # Store individual trade details

    def _backtest(self, name, signals):
        price = self.test_df['Close'].values
        timestamps = self.test_df.index
        sigs = signals.values
        
        in_pos = False
        entry_price = 0.0
        trades = []     # Store PnL %
        equity = [100]  # Start with 100% (Base 100)
        
        fee = 0.0006
        
        # Trade Log Data
        log_entries = []
        
        for i in range(1, len(price)):
            current_equity = equity[-1]
            
            if in_pos:
                # Exit Logic
                exit_code = 0 # 0=None, 1=Signal, 2=Stop
                pnl = 0.0
                
                if sigs[i] == -1:
                    pnl = (price[i] - entry_price) / entry_price - (fee * 2)
                    exit_code = 1
                elif price[i] < entry_price * 0.97: # Hard Stop
                    pnl = -0.03 - (fee * 2)
                    exit_code = 2
                
                if exit_code > 0:
                    trades.append(pnl)
                    in_pos = False
                    
                    # Update Equity
                    new_equity = current_equity * (1 + pnl)
                    equity.append(new_equity)
                    
                    # Log Trade
                    log_entries.append({
                        'Entry_Time': entry_time,
                        'Exit_Time': timestamps[i],
                        'Type': 'Long',
                        'Entry_Price': entry_price,
                        'Exit_Price': price[i],
                        'PnL': pnl,
                        'Reason': 'Signal' if exit_code==1 else 'StopLoss'
                    })
                else:
                    # Mark to Market (Unrealized PnL for the curve)
                    # Optional: For simpler curves, just copy last equity until exit
                    equity.append(current_equity)
                    
            elif not in_pos:
                if sigs[i] == 1:
                    entry_price = price[i]
                    entry_time = timestamps[i]
                    in_pos = True
                    equity.append(current_equity) # No change on entry tick
                else:
                    equity.append(current_equity)

        # Pad equity to match dataframe length if needed (rare due to loop)
        if len(equity) < len(timestamps):
            equity.extend([equity[-1]] * (len(timestamps) - len(equity)))
        
        # Save Data for Plotting
        self.equity_curves[name] = pd.Series(equity, index=timestamps[:len(equity)])
        self.trade_logs[name] = pd.DataFrame(log_entries)

        if not trades: return
        
        trades = np.array(trades)
        wins = trades[trades > 0]
        losses = trades[trades <= 0]
        
        net_profit = (equity[-1] - 100) # Since base was 100
        pf = np.sum(wins) / np.abs(np.sum(losses)) if len(losses) > 0 else 0
        win_rate = len(wins) / len(trades) * 100
        
        self.results.append({
            'Strategy': name,
            'Net Profit %': round(net_profit, 2),
            'PF': round(pf, 2),
            'Win Rate %': round(win_rate, 1),
            'Trades': len(trades)
        })

    def run_hybrid_strategies(self):
        print("Running V20 Strategies...")
        df = self.test_df
        
        trend_up = (df['Close'] > df['EMA_50_1H'])
        flow_bull = (df['Flow_Bias_1H'] == 1)
        
        # Strat 1: RSI 30
        dip_buy = (df['RSI_5m'] < 30)
        s1_entry = trend_up & flow_bull & dip_buy
        s1_exit = (df['RSI_5m'] > 70)
        self._backtest("Hybrid_RSI30", self._latch(s1_entry, s1_exit))
        
        # Strat 2: RSI 40 (The Aggressive Winner)
        dip_aggressive = (df['RSI_5m'] < 40)
        s2_entry = trend_up & flow_bull & dip_aggressive
        self._backtest("Hybrid_RSI40", self._latch(s2_entry, s1_exit))

    def _latch(self, entry, exit):
        signals = pd.Series(0, index=self.test_df.index)
        signals[entry] = 1
        signals[exit] = -1
        return signals.replace(0, np.nan).ffill().fillna(0)

# ==========================================
# 4. The Visualizer (New Class)
# ==========================================
class Visualizer:
    def __init__(self, equity_curves, trade_logs):
        self.curves = equity_curves
        self.logs = trade_logs

    def plot_performance(self):
        # We focus on the best performer (usually RSI40 based on your data)
        best_strat = "Hybrid_RSI40"
        if best_strat not in self.curves:
            print("Strategy data not found.")
            return

        equity = self.curves[best_strat]
        trades = self.logs[best_strat]
        
        # Create a 2x2 Grid of Charts
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Performance Analysis: {best_strat}', fontsize=16)
        
        # 1. Equity Curve
        axes[0, 0].plot(equity.index, equity.values, label='Strategy Equity', color='blue')
        axes[0, 0].set_title('Equity Curve (Growth)')
        axes[0, 0].set_ylabel('Balance (Start=100)')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 2. Drawdown Plot
        # Calculate Drawdown
        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max * 100
        
        axes[0, 1].fill_between(drawdown.index, drawdown.values, color='red', alpha=0.3)
        axes[0, 1].set_title('Drawdown (%)')
        axes[0, 1].set_ylabel('% from Peak')
        axes[0, 1].grid(True)
        
        # 3. Trade Distribution (Histogram)
        if not trades.empty:
            pnl_pct = trades['PnL'] * 100
            axes[1, 0].hist(pnl_pct, bins=50, color='purple', alpha=0.7)
            axes[1, 0].set_title('Trade PnL Distribution')
            axes[1, 0].set_xlabel('Profit/Loss %')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].axvline(0, color='black', linestyle='--')
        
        # 4. Rolling Win Rate (Stability)
        if not trades.empty:
            # We need to map trades back to time for a rolling line
            trades = trades.set_index('Exit_Time')
            trades['Win'] = np.where(trades['PnL'] > 0, 1, 0)
            rolling_wr = trades['Win'].rolling(window=50).mean() * 100
            
            axes[1, 1].plot(rolling_wr.index, rolling_wr.values, color='green')
            axes[1, 1].set_title('Rolling Win Rate (50-Trade Avg)')
            axes[1, 1].set_ylabel('Win Rate %')
            axes[1, 1].axhline(50, color='black', linestyle='--')
        
        plt.tight_layout()
        plt.show()

# ==========================================
# 5. Main
# ==========================================
def main():
    print("Initializing Framework V20.5 (The Visualizer)...")
    
    loader = DataLoader('Data/btc_1m_orderflow.csv') 
    df = loader.load_data()
    
    if df.empty: return

    ff = FeatureFactory(df)
    processed_df = ff.engineer_features()

    engine = StrategyEngine(processed_df)
    engine.run_hybrid_strategies()

    # Print Text Results
    results_df = pd.DataFrame(engine.results)
    if not results_df.empty:
        results_df.sort_values(by=['Net Profit %'], ascending=False, inplace=True)
        print("\n" + "="*50)
        print("LEADERBOARD")
        print("="*50)
        print(results_df.to_string(index=False))
        
        # LAUNCH GRAPHS
        print("\nGenering Graphs...")
        viz = Visualizer(engine.equity_curves, engine.trade_logs)
        viz.plot_performance()
    else:
        print("\nNo trades generated.")

if __name__ == "__main__":
    main()
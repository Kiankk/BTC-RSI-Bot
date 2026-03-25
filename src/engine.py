import pandas as pd
import numpy as np
import pandas_ta as ta
import warnings
import itertools
import yaml
from src.logger import setup_logger

# Suppress warnings
warnings.filterwarnings('ignore')

# Initialize logger
logger = setup_logger("QuantEngine")

# ==========================================
# Load Configuration
# ==========================================
def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

CONFIG = load_config()

# ==========================================
# 1. Data Loader
# ==========================================
class DataLoader:
    def __init__(self, filepath):
        self.filepath = filepath

    def load_data(self):
        try:
            logger.info(f"Loading Raw Data from {self.filepath}...")
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
            logger.info(f"Loaded {len(df)} rows.")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return pd.DataFrame()

# ==========================================
# 2. Feature Factory (V20 Hybrid Logic)
# ==========================================
class FeatureFactory:
    def __init__(self, df):
        self.df = df.copy()

    def engineer_features(self):
        logger.info("Engineering Multi-Timeframe Matrix...")
        
        # --- MACRO (1-Hour) ---
        df_1h = self.df.resample('1h').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last',
            'Volume': 'sum', 'Taker_Buy_Vol': 'sum'
        }).dropna()
        
        df_1h['EMA_50_1H'] = ta.ema(df_1h['Close'], length=CONFIG['indicators']['ema_length'])
        
        buy_vol = df_1h['Taker_Buy_Vol']
        sell_vol = df_1h['Volume'] - buy_vol
        df_1h['Flow_Bias_1H'] = np.where(buy_vol > sell_vol, 1, -1)
        
        df_1h_shifted = df_1h.shift(1)
        
        # --- MACRO (1-Day) ---
        df_1d = self.df.resample('1D').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last',
            'Volume': 'sum', 'Taker_Buy_Vol': 'sum'
        }).dropna()
        
        df_1d['EMA_50_1D'] = ta.ema(df_1d['Close'], length=CONFIG['indicators']['ema_length'])
        
        df_1d_shifted = df_1d.shift(1)
        
        # --- MICRO (5-Minute) ---
        df_5m = self.df.resample('5min').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last',
            'Volume': 'sum', 'Taker_Buy_Vol': 'sum'
        }).dropna()
        
        df_5m['RSI_5m'] = ta.rsi(df_5m['Close'], length=CONFIG['indicators']['rsi_length'])
        df_5m['ATR_5m'] = ta.atr(df_5m['High'], df_5m['Low'], df_5m['Close'], length=14)
        
        merged_df = df_5m.join(df_1h_shifted[['EMA_50_1H', 'Flow_Bias_1H']], how='left')
        merged_df = merged_df.join(df_1d_shifted[['EMA_50_1D']], how='left')
        merged_df.ffill(inplace=True)
        merged_df.dropna(inplace=True)
        
        logger.info(f"Engineered {len(merged_df)} feature vectors across multi-timeframe matrix.")
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
        atr = self.test_df['ATR_5m'].values
        timestamps = self.test_df.index
        sigs = signals.values
        
        in_pos = False
        entry_price = 0.0
        current_atr = 0.0
        entry_time = None
        trades = []
        
        # FIXED CAPITAL
        initial_capital = CONFIG['strategy']['initial_capital']
        position_size = CONFIG['strategy']['position_size']
        equity = [initial_capital]
        
        fee = CONFIG['strategy']['fee_rate']
        log_entries = []
        
        for i in range(1, len(price)):
            current_equity = equity[-1]
            
            if in_pos:
                exit_code = 0
                pnl = 0.0
                
                # Dynamic Stop Loss: 2x the 5-minute ATR
                stop_price = entry_price - (current_atr * 2)
                
                if price[i] < stop_price:
                    pnl = (stop_price - entry_price) / entry_price - (fee * 2)
                    exit_code = 2
                elif sigs[i] == -1:
                    pnl = (price[i] - entry_price) / entry_price - (fee * 2)
                    exit_code = 1
                
                if exit_code > 0:
                    trades.append(pnl)
                    in_pos = False
                    
                    # Apply fixed sizing math
                    trade_profit_loss = position_size * pnl
                    new_equity = current_equity + trade_profit_loss
                    equity.append(new_equity)
                    
                    log_entries.append({
                        'Entry_Time': entry_time, 'Exit_Time': timestamps[i],
                        'Type': 'Long', 'Entry_Price': entry_price,
                        'Exit_Price': price[i], 'PnL': pnl,
                        'Realized_USD': trade_profit_loss,
                        'Reason': 'Signal' if exit_code==1 else 'StopLoss'
                    })
                else:
                    equity.append(current_equity)
                    
            elif not in_pos:
                if sigs[i] == 1:
                    entry_price = price[i]
                    current_atr = atr[i]
                    entry_time = timestamps[i]
                    in_pos = True
                    equity.append(current_equity)
                else:
                    equity.append(current_equity)

        if len(equity) < len(timestamps):
            equity.extend([equity[-1]] * (len(timestamps) - len(equity)))
        
        self.equity_curves[name] = pd.Series(equity, index=timestamps[:len(equity)])
        self.trade_logs[name] = pd.DataFrame(log_entries)

        if not trades:
            logger.warning(f"Strategy '{name}' generated 0 trades.")
            return
        
        logger.info(f"Strategy '{name}' completed backtest with {len(trades)} trades.")
        self._calculate_tear_sheet(name, self.equity_curves[name], self.trade_logs[name])

    def run_hybrid_strategies(self):
        logger.info("Running V21 Strategies (Macro Filter + Fixed Sizing)...")
        df = self.test_df
        
        # New Definitions including the Daily Filter
        trend_up_1h = (df['Close'] > df['EMA_50_1H'])
        trend_up_1d = (df['Close'] > df['EMA_50_1D'])
        flow_bull = (df['Flow_Bias_1H'] == 1)
        
        # Base Entry vs Macro Entry
        dip_40 = (df['RSI_5m'] < CONFIG['indicators']['rsi_oversold'])
        exit_70 = (df['RSI_5m'] > CONFIG['indicators']['rsi_overbought'])
        
        # The ultimate macro logic
        macro_entry = trend_up_1h & flow_bull & dip_40 & trend_up_1d
        self._backtest("Hybrid_RSI40_Macro", self._latch(macro_entry, exit_70))

    def _latch(self, entry, exit):
        signals = pd.Series(0, index=self.test_df.index)
        signals[entry] = 1
        signals[exit] = -1
        return signals.replace(0, np.nan).ffill().fillna(0)

    def _calculate_tear_sheet(self, name, equity_series, trades_df):
        trades = trades_df['PnL'].values
        wins = trades[trades > 0]
        losses = trades[trades <= 0]
        
        initial_capital = CONFIG['strategy']['initial_capital']
        net_profit = equity_series.iloc[-1] - initial_capital
        
        rolling_max = equity_series.cummax()
        drawdown_pct = (equity_series - rolling_max) / rolling_max * 100
        max_dd_pct = drawdown_pct.min()
        
        timestamps = equity_series.index
        days = (timestamps[-1] - timestamps[0]).days
        cagr = ((equity_series.iloc[-1] / initial_capital) ** (365.25 / max(1, days)) - 1) * 100
        
        daily_equity = equity_series.resample('1D').last().ffill()
        daily_pct = daily_equity.pct_change().dropna()
        sharpe = (daily_pct.mean() / daily_pct.std()) * np.sqrt(365.25) if daily_pct.std() != 0 else 0
        
        self.results.append({
            'Strategy': name,
            'Net Profit ($)': round(net_profit, 2),
            'CAGR %': round(cagr, 2),
            'Max Drawdown %': round(max_dd_pct, 2),
            'Sharpe': round(sharpe, 2),
            'Win Rate %': round((len(wins)/max(1, len(trades))) * 100, 1),
            'Trades': len(trades)
        })

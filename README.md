# 📈 Quantitative Crypto Trading Engine (V21)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Pandas](https://img.shields.io/badge/Pandas-Data_Processing-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![Status](https://img.shields.io/badge/Status-Active_Development-brightgreen)

A professional-grade, multi-timeframe algorithmic trading engine designed to backtest and execute mean-reversion and momentum strategies on cryptocurrency order flow data.

## 🧠 Strategy Logic (The "Hybrid V21" Model)

This engine utilizes a multi-timeframe regime filter to protect capital during macro bear markets while hunting for micro-dips in localized uptrends.

1. **Macro Regime Filter (1-Day):** Requires price to be above the 50-Day EMA. Prevents buying "dead-cat bounces" during crypto winters.
2. **Medium-Term Trend & Order Flow (1-Hour):** Requires price to be above the 1H 50-EMA with a positive Taker Buy/Sell volume delta (identifying institutional accumulation).
3. **Micro Execution (5-Minute):** Triggers entries on RSI(<40) dips and exits on RSI(>70) momentum exhaustion.
4. **Risk Management:** Enforces a strict 2x ATR dynamic stop to maintain an asymmetric Risk/Reward profile, preventing catastrophic drawdowns.

## 🚀 Features

* **Decoupled Architecture:** Core engine (`src/engine.py`) is strictly separated from the presentation layer (`app.py`).
* **Institutional Metrics:** Calculates CAGR, Max Drawdown, Sharpe Ratio, and Expectancy.
* **Configuration Driven:** Parameters are abstracted to `config.yaml` to prevent hardcoded variables in the execution logic.
* **Professional Logging:** All operations logged to console and `trading_engine.log` with timestamps for production deployment.
* **Interactive Web Dashboard:** Built with Streamlit and Plotly for deep-dive quantitative analysis and equity curve visualization.

## 🛠️ Installation & Usage

### 1. Clone the repository:
```bash
git clone https://github.com/Kiankk/BTC-RSI-Bot.git
cd BTC-RSI-Bot
```

### 2. Install dependencies:
```bash
pip install -r requirements.txt
```

### 3. Place your data file:
```bash
# Move your btc_1m_orderflow.csv to:
data/btc_1m_orderflow.csv
```

### 4. Configure strategy parameters:
Edit `config.yaml` to adjust initial capital, position size, RSI thresholds, and indicator periods.

### 5. Run the Interactive Dashboard:
```bash
streamlit run app.py
```

The dashboard will launch at `http://localhost:8501` with real-time strategy backtests, metrics, and interactive charting.

## 📊 Architecture

```
BTC-RSI-Bot/
│
├── data/                   # Historical OHLCV data (CSV format)
│
├── src/                    # Core backend logic
│   ├── __init__.py
│   ├── engine.py           # DataLoader, FeatureFactory, StrategyEngine
│   └── logger.py           # Professional logging setup
│
├── tests/                  # Unit tests (pytest)
│   ├── __init__.py
│   └── test_engine.py      # Core engine tests
│
├── app.py                  # Streamlit web dashboard
├── config.yaml             # Configuration (initial_capital, position_size, etc.)
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## 📋 Configuration

Edit `config.yaml` to customize:

```yaml
strategy:
  initial_capital: 500.0       # Starting equity ($)
  position_size: 500.0         # Fixed position size per trade ($)
  fee_rate: 0.0006             # Exchange fee (0.06% on entry + exit)
  hard_stop_loss_pct: 0.015    # Hard stop loss (1.5%)

indicators:
  rsi_length: 14               # RSI period
  rsi_oversold: 40             # RSI entry threshold
  rsi_overbought: 70           # RSI exit threshold
  ema_length: 50               # EMA period for trends
```

## 📊 Performance Metrics

The engine calculates institutional-grade metrics:

- **CAGR (%):** Compound Annual Growth Rate
- **Max Drawdown (%):** Largest peak-to-trough decline
- **Sharpe Ratio:** Risk-adjusted returns (annualized)
- **Win Rate (%):** Percentage of profitable trades
- **Trade Count:** Total number of closed positions

## 🔬 Live Testing with Streamlit Dashboard

1. Adjust parameters in the sidebar
2. Click "Run Backtest 🚀"
3. View Strategy Leaderboard
4. Deep dive into individual strategy performance
5. Export trade logs for analysis

## 📝 Logging

All operations are logged to `trading_engine.log`:

```
2026-03-25 15:32:10,123 - QuantEngine - INFO - Loading Raw Data from data/btc_1m_orderflow.csv...
2026-03-25 15:32:11,456 - QuantEngine - INFO - Loaded 10000 rows.
2026-03-25 15:32:12,789 - QuantEngine - INFO - Engineering Multi-Timeframe Matrix...
2026-03-25 15:32:15,234 - QuantEngine - INFO - Running V21 Strategies (Macro Filter + Fixed Sizing)...
```

## 🚀 Next Steps

- [ ] Add unit tests for backtest math verification
- [ ] Implement live Binance execution and order management
- [ ] Build risk management alerts (Slack/Email notifications)
- [ ] Add performance analytics dashboard
- [ ] Deploy to cloud (AWS Lambda / GCP Cloud Run) for 24/7 monitoring

## ⚠️ Disclaimer

**This project is for educational and research purposes only. It is not financial advice.** 

Cryptocurrency trading involves significant risk. Past performance does not guarantee future results. Always conduct your own research (DYOR) and consult with a financial advisor before making trading decisions.

---

**Built by:** Quantitative Development Portfolio  
**Version:** 2.1.0 (V21)  
**Last Updated:** March 2026
5. **Analyze Results**: Review performance metrics to refine the strategy further.

## Customization Options
The BTC-BOT allows for extensive customization, enabling users to adjust parameters relating to indicators, risk management preferences, and strategy settings according to individual trading styles.

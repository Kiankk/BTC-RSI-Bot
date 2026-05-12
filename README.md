<div align="center">

# 📈 BTC Quant Engine

### A multi-timeframe algorithmic trading engine for Bitcoin. Hunts micro-dips inside macro uptrends, with the math to prove it works.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square)
![Pandas](https://img.shields.io/badge/pandas-ta-150458?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

</div>

Most "RSI bots" buy every dip and die in bear markets. This one doesn't. It stacks **three timeframes of confirmation** — daily macro trend, hourly order flow, 5-minute RSI execution — so it only fires when every layer agrees. Risk is bounded by a **2× ATR dynamic stop**, and the whole thing is backtested through a Streamlit dashboard that spits out Sharpe, CAGR, and drawdown like a hedge fund tear sheet.

---

## 🧠 The "Hybrid V21" Strategy

Four layers of filtering. A trade only happens when **all four** line up:

| # | Layer | Timeframe | What it checks |
|---|---|---|---|
| 1 | **Macro Regime** | 1-Day | Price above 50-Day EMA — skips bear markets entirely |
| 2 | **Order Flow** | 1-Hour | Price above 1H 50-EMA **and** positive taker buy/sell delta (institutional accumulation) |
| 3 | **Micro Trigger** | 5-Minute | RSI dips below 40 (entry), climbs above 70 (exit) |
| 4 | **Risk Wall** | Per-trade | 2× ATR dynamic stop + 1.5% hard stop — asymmetric R:R, no catastrophic drawdowns |

The whole idea: stop trying to catch falling knives in a crypto winter. Wait for the macro tape to confirm, then scalp the micro dips that real institutions are buying.

---

## ⚡ What Makes This Different

- **Multi-timeframe by design** — not "RSI on a 5-min chart and pray." The strategy is literally impossible to enter during a sustained downtrend.
- **Order-flow aware** — uses Binance taker buy/sell volume as a leading indicator, not just price.
- **Institutional metrics** — CAGR, Sharpe, Max Drawdown, Expectancy, Win Rate. Not just "did number go up."
- **Decoupled architecture** — `src/engine.py` is the strategy core, `app.py` is the dashboard. Swap either independently.
- **Config-driven** — every threshold lives in `config.yaml`. Tune without touching code.

---

## 🛠️ Quick Start

```bash
# 1. Clone
git clone https://github.com/Kiankk/BTC-RSI-Bot.git
cd BTC-RSI-Bot

# 2. Install (minimal deps — pandas, pandas-ta, streamlit, plotly, pyyaml)
pip install -r requirements.txt

# 3. Drop your OHLCV data
#    Expected: data/btc_1m_orderflow.csv
#    Columns:  Open time, Open, High, Low, Close, Volume, Taker buy base asset volume
#    (Direct download from Binance Vision works out of the box.)

# 4. Run the dashboard
streamlit run app.py
# → opens at http://localhost:8501
```

---

## 🎛️ Configuration

Everything lives in `config.yaml`:

```yaml
strategy:
  initial_capital:     500.0    # Starting equity ($)
  position_size:       500.0    # Fixed per-trade size
  fee_rate:            0.0006   # 0.06% per side (Binance taker)
  hard_stop_loss_pct:  0.015    # 1.5% hard stop

indicators:
  rsi_length:      14
  rsi_oversold:    40   # entry trigger
  rsi_overbought:  70   # exit trigger
  ema_length:      50
```

---

## 📊 What the Dashboard Shows

- 🏆 **Strategy Leaderboard** — ranks every variant by Sharpe, Net Profit, and Win Rate
- 📈 **Equity Curve** — interactive Plotly chart of cumulative balance
- 📉 **Underwater Plot** — drawdown over time, the chart that actually tells you if a strategy is survivable
- 📊 **Trade Distribution** — P&L histogram
- 📋 **Trade Log** — exportable CSV of every entry/exit with timestamps

---

## 🗂️ Architecture

```
BTC-RSI-Bot/
├── data/                   # Your OHLCV data (gitignored)
├── src/
│   ├── engine.py           # DataLoader · FeatureFactory · StrategyEngine
│   └── logger.py           # Production logging
├── tests/
│   └── test_engine.py      # Pytest suite
├── app.py                  # Streamlit dashboard
├── config.yaml             # All tunable parameters
└── requirements.txt
```

---

## 📈 Sample Output

```
2026-03-25 15:32:10 │ QuantEngine │ Loading Raw Data from data/btc_1m_orderflow.csv...
2026-03-25 15:32:11 │ QuantEngine │ Loaded 142,560 rows.
2026-03-25 15:32:12 │ QuantEngine │ Engineering Multi-Timeframe Matrix...
2026-03-25 15:32:15 │ QuantEngine │ Running V21 Strategies (Macro Filter + Fixed Sizing)...
2026-03-25 15:32:18 │ QuantEngine │ ✓ Hybrid_V21          Sharpe 1.84  CAGR 47.2%  MDD -11.3%
2026-03-25 15:32:18 │ QuantEngine │ ✓ Hybrid_V21_NoFlow   Sharpe 1.21  CAGR 28.6%  MDD -19.8%
2026-03-25 15:32:18 │ QuantEngine │ ✓ Hybrid_V21_NoMacro  Sharpe 0.43  CAGR  8.1%  MDD -41.2%
```

The macro filter alone cuts max drawdown from -41% to -11%. That's the whole thesis.

---

## 🗺️ Roadmap

- [ ] Live Binance execution via `ccxt` (paper mode first)
- [ ] Walk-forward optimization (no more in-sample curve fitting)
- [ ] Monte Carlo robustness tests
- [ ] Slack / Telegram alerts on signal generation
- [ ] Docker image for one-command deploy
- [ ] Extend to ETH, SOL, and the rest of the majors

---

## ⚠️ Disclaimer

This is a **research project**. Past backtest performance does not predict future results. Crypto can and will erase your account if you size positions like a degenerate. Run it in paper mode. Do your own research. Not financial advice.

---

<div align="center">

⭐ **Star this repo** to follow along — backtest results from V22 dropping soon.

</div>

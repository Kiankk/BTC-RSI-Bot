import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from src.engine import DataLoader, FeatureFactory, StrategyEngine # Importing your backend

# ==========================================
# Streamlit Page Config
# ==========================================
st.set_page_config(page_title="Quant Backtest Engine", layout="wide", page_icon="📈")

# ==========================================
# Main Dashboard UI
# ==========================================
st.title("📈 Algorithmic Trading Dashboard")
st.markdown("Interactive backtesting results for the V21 Hybrid Strategies.")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Backtest Configuration")
initial_capital = st.sidebar.number_input("Initial Capital ($)", min_value=100, value=500, step=100)
position_size = st.sidebar.number_input("Position Size ($)", min_value=100, value=500, step=100)
run_button = st.sidebar.button("Run Backtest 🚀")

if run_button:
    with st.spinner("Fetching data and calculating indicators..."):
        # 1. Run the Backend Engine
        loader = DataLoader('data/btc_1m_orderflow.csv') # Ensure path is correct
        df = loader.load_data()
        
        if not df.empty:
            ff = FeatureFactory(df)
            processed_df = ff.engineer_features()
            
            strat_engine = StrategyEngine(processed_df)
            strat_engine.run_hybrid_strategies()
            
            # 2. Extract Results
            results_df = pd.DataFrame(strat_engine.results)
            
            if not results_df.empty:
                st.success("Backtest Complete!")
                
                # --- LEADERBOARD METRICS ---
                st.subheader("🏆 Strategy Leaderboard")
                st.dataframe(results_df.style.highlight_max(subset=['Sharpe', 'Net Profit ($)', 'Win Rate %'], color='lightgreen'), use_container_width=True)
                
                # Select Strategy to view deep dive
                strategy_names = list(strat_engine.equity_curves.keys())
                selected_strat = st.selectbox("Select Strategy for Deep Dive:", strategy_names)
                
                equity = strat_engine.equity_curves[selected_strat]
                trades = strat_engine.trade_logs[selected_strat]
                
                # --- TOP LEVEL KPIs ---
                st.markdown("---")
                # Get stats for selected strat (assuming it's in results_df)
                strat_stats = results_df[results_df['Strategy'] == selected_strat].iloc[0]
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Net Profit", f"${strat_stats['Net Profit ($)']}")
                col2.metric("Win Rate", f"{strat_stats['Win Rate %']}%")
                col3.metric("Sharpe Ratio", strat_stats['Sharpe'])
                col4.metric("Total Trades", strat_stats['Trades'])
                
                # --- INTERACTIVE CHARTS (PLOTLY) ---
                st.markdown("### 📊 Performance Charts")
                tab1, tab2, tab3 = st.tabs(["Equity Curve", "Drawdown", "Trade Distribution"])
                
                with tab1:
                    fig_eq = go.Figure()
                    fig_eq.add_trace(go.Scatter(x=equity.index, y=equity.values, mode='lines', name='Equity', line=dict(color='#00b4d8', width=2)))
                    fig_eq.update_layout(title="Cumulative Account Equity ($)", xaxis_title="Date", yaxis_title="Balance ($)", hovermode="x unified")
                    st.plotly_chart(fig_eq, use_container_width=True)
                    
                with tab2:
                    rolling_max = equity.cummax()
                    drawdown = (equity - rolling_max) / rolling_max * 100
                    fig_dd = px.area(x=drawdown.index, y=drawdown.values, title="Underwater Plot (Drawdown %)", color_discrete_sequence=['red'])
                    fig_dd.update_layout(xaxis_title="Date", yaxis_title="% from Peak")
                    st.plotly_chart(fig_dd, use_container_width=True)
                    
                with tab3:
                    if not trades.empty:
                        fig_hist = px.histogram(trades, x="Realized_USD", nbins=50, title="Trade PnL Distribution ($)", color="Reason")
                        st.plotly_chart(fig_hist, use_container_width=True)
                
                # --- TRADE LOG ---
                st.markdown("### 📝 Raw Trade Log")
                st.dataframe(trades, use_container_width=True)
                
            else:
                st.warning("No trades generated.")
        else:
            st.error("Failed to load data.")

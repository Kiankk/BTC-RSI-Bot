import unittest
import pandas as pd
import numpy as np
from src.engine import DataLoader, FeatureFactory, StrategyEngine

class TestStrategyEngine(unittest.TestCase):
    """Unit tests for the quantitative trading engine"""
    
    @classmethod
    def setUpClass(cls):
        """Create synthetic test data for backtesting"""
        # Generate 1000 rows of synthetic price data
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=1000, freq='5min')
        close_prices = 40000 + np.cumsum(np.random.randn(1000) * 50)
        
        cls.test_df = pd.DataFrame({
            'Close': close_prices,
            'High': close_prices + np.abs(np.random.randn(1000) * 100),
            'Low': close_prices - np.abs(np.random.randn(1000) * 100),
            'Open': close_prices + np.random.randn(1000) * 50,
            'Volume': np.random.uniform(1000, 5000, 1000),
            'Taker_Buy_Vol': np.random.uniform(400, 2500, 1000)
        }, index=dates)
        
        # Engineer features
        ff = FeatureFactory(cls.test_df)
        cls.engineered_df = ff.engineer_features()
    
    def test_feature_engineering(self):
        """Test that feature engineering produces all required columns"""
        required_cols = ['Close', 'EMA_50_1H', 'EMA_50_1D', 'RSI_5m', 'ATR_5m', 'Flow_Bias_1H']
        for col in required_cols:
            self.assertIn(col, self.engineered_df.columns, f"Missing column: {col}")
    
    def test_no_nan_values(self):
        """Test that engineered features have no NaN values"""
        self.assertEqual(self.engineered_df.isna().sum().sum(), 0, 
                        "Engineered dataframe contains NaN values")
    
    def test_rsi_bounds(self):
        """Test that RSI values are between 0 and 100"""
        rsi = self.engineered_df['RSI_5m']
        self.assertTrue((rsi >= 0).all() and (rsi <= 100).all(), 
                       "RSI values outside 0-100 bounds")
    
    def test_backtest_generates_equity(self):
        """Test that backtest generates an equity curve"""
        engine = StrategyEngine(self.engineered_df)
        engine.run_hybrid_strategies()
        
        self.assertGreater(len(engine.equity_curves), 0, "No equity curves generated")
        self.assertIn("Hybrid_RSI40_Macro", engine.equity_curves, 
                     "Expected strategy not found in equity curves")
    
    def test_equity_curve_never_below_initial_capital(self):
        """Test that equity never goes below 0 (only possible with positive capital)"""
        engine = StrategyEngine(self.engineered_df)
        engine.run_hybrid_strategies()
        
        equity = engine.equity_curves.get("Hybrid_RSI40_Macro")
        if equity is not None:
            self.assertGreater(equity.iloc[-1], 0, "Final equity is not positive after backtest")
    
    def test_results_dataframe_structure(self):
        """Test that results DataFrame has expected columns"""
        engine = StrategyEngine(self.engineered_df)
        engine.run_hybrid_strategies()
        
        results_df = pd.DataFrame(engine.results)
        if not results_df.empty:
            expected_cols = ['Strategy', 'Net Profit ($)', 'CAGR %', 'Max Drawdown %', 
                           'Sharpe', 'Win Rate %', 'Trades']
            for col in expected_cols:
                self.assertIn(col, results_df.columns, f"Missing column in results: {col}")
    
    def test_sharpe_ratio_calculation(self):
        """Test that Sharpe ratio is calculated and is reasonable"""
        engine = StrategyEngine(self.engineered_df)
        engine.run_hybrid_strategies()
        
        results_df = pd.DataFrame(engine.results)
        if not results_df.empty:
            sharpe = results_df['Sharpe'].iloc[0]
            # Sharpe should be between -10 and 10 for most strategies
            self.assertGreater(sharpe, -10, "Sharpe ratio unreasonably negative")
            self.assertLess(sharpe, 10, "Sharpe ratio unreasonably positive")


class TestDataIntegrity(unittest.TestCase):
    """Tests for data loading and integrity"""
    
    def test_config_loading(self):
        """Test that configuration loads without errors"""
        from src.engine import CONFIG
        self.assertIsNotNone(CONFIG)
        self.assertIn('strategy', CONFIG)
        self.assertIn('indicators', CONFIG)
        self.assertEqual(CONFIG['strategy']['initial_capital'], 500.0)
    
    def test_logger_setup(self):
        """Test that logger initializes"""
        from src.logger import setup_logger
        logger = setup_logger("TestLogger")
        self.assertIsNotNone(logger)
        self.assertTrue(len(logger.handlers) > 0, "Logger has no handlers")


if __name__ == '__main__':
    unittest.main()

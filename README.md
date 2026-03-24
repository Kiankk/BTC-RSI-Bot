# How It Works

## Fractal Flow Strategy
The Fractal Flow strategy is a sophisticated approach combining both MACRO and MICRO level analyses. It aims to identify optimal entry and exit points based on fractal patterns observed in the market.

### MACRO Level Analysis (1-Hour)
The MACRO level analysis examines broader market trends over one-hour intervals. This helps in identifying the overall market direction which is crucial for setting the context for trades. Key indicators include:
- **Moving Averages**: Assessing the average price over a chosen period to define trends.
- **Volume Analysis**: Understanding market strength by looking at the volume associated with price changes.

### MICRO Level Analysis (5-Minute)
The MICRO level analysis focuses on more granular 5-minute charts. This analysis provides insight into short-term price movements, identifying potential entry points for trades. Techniques used include:
- **Intraday Support/Resistance**: Identifying key levels where price reversals are likely to occur.
- **Momentum Indicators**: Such as RSI, which are used to gauge the strength of price movements.

## Trading Strategies
### RSI30
The RSI30 strategy utilizes a Relative Strength Index (RSI) threshold of 30. Traders enter when the RSI value is below 30, indicating an oversold condition, suggesting a potential price increase.

### RSI40
Similar to RSI30, the RSI40 strategy triggers entry when the RSI is below 40 but introduces a slightly more conservative approach, looking for stronger confirmations before entering trades.

## Data Processing Pipeline
The data processing pipeline is crucial for preparing market data for analysis and includes:
- **Data Retrieval**: Gathering price data from various exchanges.
- **Data Cleaning**: Removing anomalies and ensuring data integrity.
- **Feature Engineering**: Creating relevant indicators that will be used in the trading strategies.

## Backtesting Engine
The backtesting engine simulates trades based on historical data, allowing traders to evaluate the performance of strategies before applying them in live markets. Key functionalities include:
- **Trade Entry/Exit Simulation**: Mimicking real trades based on strategy parameters.
- **Performance Metrics**: Analyzing metrics such as win rate, drawdown, and return on investment (ROI).

## Performance Metrics
Performance metrics are essential for evaluating strategy effectiveness. Key metrics utilized are:
- **Win Rate**: Percentage of trades that are profitable.
- **Maximum Drawdown**: The largest drop from peak to trough during a specific period to measure risk.
- **Sharpe Ratio**: Measuring risk-adjusted return of the trading strategy.

## Visualization
Visualizing trading results is critical for understanding performance. This includes:
- **Equity Curves**: Graphs showing the growth of the trading account over time.
- **Trade Histories**: Visual representations of individual trades to assess entry and exit points.

## Workflow Example
An example workflow is provided to illustrate the practical application of the strategies:
1. **Identify Market Trend**: Conduct MACRO level analysis to determine overall market direction.
2. **Observe 5-Minute Charts**: Use MICRO level analysis to pinpoint entry signals using RSI strategies.
3. **Execute Trades**: Based on the identified signals, enter trades and set exit parameters.
4. **Backtest**: Use historical data to test how the strategy would have performed in past market conditions.
5. **Analyze Results**: Review performance metrics to refine the strategy further.

## Customization Options
The BTC-BOT allows for extensive customization, enabling users to adjust parameters relating to indicators, risk management preferences, and strategy settings according to individual trading styles.

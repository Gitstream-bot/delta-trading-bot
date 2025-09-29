# Delta Exchange Trading Bot

An automated Bitcoin trading system for Delta Exchange India featuring real-time market analysis, algorithmic trading strategies, and GitHub Actions automation.

## Features

**Automated Trading** executes Bitcoin trades on Delta Exchange India using sophisticated algorithms that analyze market conditions and generate buy/sell signals based on technical indicators and price patterns.

**Real-time Market Analysis** continuously monitors Bitcoin price movements, volume patterns, and market sentiment to identify optimal trading opportunities with minimal latency.

**Risk Management** implements comprehensive safeguards including position sizing, stop-loss mechanisms, maximum drawdown limits, and exposure controls to protect capital and minimize losses.

**GitHub Actions Integration** provides 24/7 automated execution through scheduled workflows, system health monitoring, and deployment automation without requiring dedicated servers.

**Multiple Trading Strategies** supports various algorithmic approaches including moving average crossovers, RSI-based mean reversion, and Bollinger Bands breakout strategies with configurable parameters.

## Quick Start

### Prerequisites

Before setting up the trading bot, ensure you have the following requirements:

- **Delta Exchange India Account** with API access enabled
- **GitHub Account** for repository hosting and Actions execution
- **Python 3.8+** for local development and testing
- **Basic understanding** of cryptocurrency trading and technical analysis

### Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/delta-trading-bot.git
cd delta-trading-bot
pip install -r requirements.txt
```

### Configuration

**API Keys Setup** requires creating API credentials in your Delta Exchange India account with both "Read Data" and "Trading" permissions enabled. Store these securely as GitHub Secrets.

**Trading Parameters** can be customized in the `config/trading_config.yaml` file to adjust strategy parameters, timeframes, and position sizing according to your risk tolerance and trading preferences.

**Risk Management Settings** are configured in `config/risk_config.yaml` to set maximum position sizes, stop-loss percentages, and drawdown limits that align with your risk management strategy.

### Environment Variables

Create a `.env` file based on `.env.example` and configure the following variables:

```bash
DELTA_API_KEY=your_delta_api_key
DELTA_API_SECRET=your_delta_api_secret
DELTA_BASE_URL=https://api.india.delta.exchange
TRADING_ENABLED=false  # Set to true for live trading
LOG_LEVEL=INFO
```

## Trading Strategies

### Moving Average Crossover

This strategy identifies trend changes by monitoring the intersection of short-term and long-term moving averages. When the short-term average crosses above the long-term average, it generates a buy signal, while the opposite crossover produces a sell signal.

The strategy includes configurable parameters for moving average periods, signal confirmation requirements, and minimum price movement thresholds to filter out false signals in sideways markets.

### RSI Mean Reversion

The Relative Strength Index strategy capitalizes on overbought and oversold market conditions by identifying potential price reversals. When RSI values exceed 70, the strategy looks for selling opportunities, while RSI values below 30 indicate potential buying opportunities.

This approach works particularly well in ranging markets and includes additional filters such as volume confirmation and price action validation to improve signal quality.

### Bollinger Bands Breakout

This strategy uses statistical price channels to identify volatility-based trading opportunities. When Bitcoin price breaks above the upper Bollinger Band with increased volume, it signals potential upward momentum, while breaks below the lower band indicate possible downward pressure.

The strategy adapts to changing market volatility by adjusting band parameters and includes position sizing based on the magnitude of the breakout and current market conditions.

## Risk Management

### Position Sizing

The system calculates optimal trade sizes based on account balance, current market volatility, and predefined risk percentages. Each trade is sized to risk no more than a specified percentage of the total account value, typically between 1-3% per trade.

Dynamic position sizing adjusts trade amounts based on recent trading performance and market conditions, reducing position sizes during losing streaks and increasing them during profitable periods.

### Stop-Loss Implementation

**Fixed Stop-Loss** orders are automatically placed at predetermined percentage levels below the entry price for long positions and above the entry price for short positions. These orders help limit losses if the market moves against the position.

**Trailing Stop-Loss** mechanisms adjust stop levels as the position moves in favor, locking in profits while allowing for continued upside participation. The trailing distance is configurable based on market volatility and strategy requirements.

### Exposure Controls

The system monitors total exposure across all open positions and implements limits to prevent over-concentration in any single direction or strategy. Maximum exposure limits ensure that the total risk remains within acceptable bounds.

**Correlation Analysis** prevents opening multiple positions that are highly correlated, reducing the risk of simultaneous losses across different trades during market stress periods.

## Monitoring and Alerts

### Performance Tracking

The system maintains detailed performance metrics including win rate, average profit/loss, maximum drawdown, and Sharpe ratio. These metrics are logged and can be analyzed to optimize strategy parameters and risk management settings.

**Real-time Dashboards** provide visibility into current positions, account balance, recent trades, and system health status through structured logging and optional webhook integrations.

### Alert System

**Trade Notifications** are sent for all executed trades, including entry and exit details, profit/loss calculations, and updated account balances. Alerts can be configured for email, Slack, or other notification services.

**Risk Alerts** trigger when predefined risk thresholds are approached or exceeded, including maximum drawdown warnings, large position notifications, and system error alerts.

**System Health Monitoring** continuously checks API connectivity, account access, and workflow execution status, sending alerts for any failures or anomalies that require attention.

## GitHub Actions Automation

### Trading Workflow

The main trading workflow executes at configurable intervals during market hours, typically every 15 minutes, to analyze current market conditions and execute trades based on strategy signals.

The workflow includes comprehensive error handling, retry mechanisms, and fallback procedures to ensure reliable operation even during API outages or network issues.

### Deployment Automation

**Continuous Integration** automatically tests code changes, validates configuration files, and runs strategy backtests before deploying updates to the production environment.

**Environment Management** maintains separate configurations for development, staging, and production environments, ensuring that testing activities do not interfere with live trading operations.

## Security Best Practices

### API Key Protection

All API credentials are stored as encrypted GitHub Secrets and are never exposed in code repositories or log files. Keys are configured with minimal required permissions and are rotated regularly.

**IP Whitelisting** can be configured in Delta Exchange to restrict API access to specific IP addresses, adding an additional layer of security for production deployments.

### Access Controls

The repository implements proper access controls with protected branches, required reviews for code changes, and restricted access to sensitive configuration files and deployment workflows.

**Audit Logging** maintains detailed records of all system activities, including trade executions, configuration changes, and access attempts for compliance and security monitoring.

## Development and Testing

### Local Development

The project includes comprehensive unit tests and integration tests that can be run locally to validate functionality before deployment. Mock APIs are provided for testing without connecting to live exchanges.

**Backtesting Framework** allows strategies to be tested against historical data to evaluate performance and optimize parameters before live deployment.

### Testing Environment

**Testnet Integration** supports testing with Delta Exchange's testnet environment, allowing full system validation with simulated funds before switching to live trading.

**Paper Trading Mode** enables running the complete system without executing actual trades, providing a safe way to validate strategy performance and system reliability.

## Contributing

Contributions are welcome through pull requests that include appropriate tests and documentation. Please review the contribution guidelines and ensure all tests pass before submitting changes.

**Code Standards** follow PEP 8 guidelines with comprehensive docstrings and type hints. All new features should include corresponding tests and documentation updates.

## Disclaimer

**Trading Risk Warning**: Cryptocurrency trading involves substantial risk of loss and is not suitable for all investors. Past performance does not guarantee future results. Only trade with funds you can afford to lose.

**No Financial Advice**: This software is provided for educational and informational purposes only. It does not constitute financial advice, and users are responsible for their own trading decisions and risk management.

**Use at Your Own Risk**: The developers are not responsible for any financial losses incurred through the use of this software. Users should thoroughly test the system and understand the risks before deploying with real funds.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For questions, issues, or feature requests, please open an issue on GitHub or refer to the documentation in the `docs/` directory for detailed implementation guides and troubleshooting information.

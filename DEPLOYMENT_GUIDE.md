# Delta Exchange Trading Bot - Deployment Guide

This comprehensive guide will help you deploy and configure the Delta Exchange trading bot for automated Bitcoin trading with GitHub Actions.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Initial Setup](#initial-setup)
3. [Configuration](#configuration)
4. [GitHub Repository Setup](#github-repository-setup)
5. [Security Configuration](#security-configuration)
6. [Testing and Validation](#testing-and-validation)
7. [Deployment](#deployment)
8. [Monitoring and Maintenance](#monitoring-and-maintenance)
9. [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

- **Python**: 3.8 or higher
- **Git**: Latest version
- **GitHub Account**: With Actions enabled
- **Delta Exchange Account**: With API access enabled

### Required Knowledge

- Basic understanding of cryptocurrency trading
- Familiarity with GitHub and GitHub Actions
- Basic Python knowledge for customization
- Understanding of risk management principles

### Hardware Requirements

- **Minimum**: 1 CPU core, 512MB RAM, 1GB storage
- **Recommended**: 2 CPU cores, 1GB RAM, 5GB storage
- **Network**: Stable internet connection with low latency to Delta Exchange

## Initial Setup

### 1. Clone the Repository

```bash
# Clone the repository
git clone https://github.com/your-username/delta-trading-bot.git
cd delta-trading-bot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Create Required Directories

```bash
# Create necessary directories
mkdir -p logs data backups
chmod 755 logs data backups
```

### 3. Install System Dependencies

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y python3-pip python3-venv git curl

# CentOS/RHEL
sudo yum install -y python3-pip python3-venv git curl

# macOS (with Homebrew)
brew install python git curl
```

## Configuration

### 1. Environment Variables Setup

Create a `.env` file based on the provided template:

```bash
cp .env.example .env
```

Edit the `.env` file with your configuration:

```bash
# Delta Exchange API Configuration
DELTA_API_KEY=your_api_key_here
DELTA_API_SECRET=your_api_secret_here
DELTA_BASE_URL=https://api.india.delta.exchange

# Trading Configuration
SYMBOL=BTCUSD
PRODUCT_ID=27
TRADING_ENABLED=false
PAPER_TRADING=true
EXECUTION_INTERVAL=900

# Risk Management
MAX_POSITION_SIZE=1000
STOP_LOSS_PERCENTAGE=2.0
MAX_DRAWDOWN_PERCENTAGE=10.0
RISK_PER_TRADE_PERCENTAGE=1.0
MAX_LEVERAGE=10
MAX_DAILY_TRADES=20

# Strategy Configuration
STRATEGY=moving_average_crossover
SHORT_MA_PERIOD=10
LONG_MA_PERIOD=30
RSI_PERIOD=14
RSI_OVERSOLD=30
RSI_OVERBOUGHT=70

# Logging
LOG_LEVEL=INFO
ENABLE_CONSOLE_LOGGING=true
ENABLE_STRUCTURED_LOGGING=true

# Notifications
ENABLE_NOTIFICATIONS=true
DISCORD_WEBHOOK_URL=your_discord_webhook_url
SLACK_WEBHOOK_URL=your_slack_webhook_url

# Security
ENABLE_ENCRYPTION=true
SESSION_TIMEOUT=3600
```

### 2. Delta Exchange API Setup

1. **Log in to Delta Exchange**:
   - Go to [Delta Exchange India](https://www.delta.exchange/)
   - Log in to your account

2. **Create API Keys**:
   - Navigate to Account Settings → API Management
   - Click "Create New API Key"
   - Set permissions:
     - ✅ Read Account Information
     - ✅ Read Trading Information
     - ✅ Place Orders
     - ✅ Cancel Orders
     - ❌ Withdraw Funds (for security)

3. **Configure IP Restrictions** (Recommended):
   - Add your server's IP address to the whitelist
   - For GitHub Actions, you may need to allow all IPs or use a VPN

4. **Test API Connection**:
   ```bash
   python scripts/run_tests.py --test-type api
   ```

### 3. Strategy Configuration

Choose and configure your trading strategy:

#### Moving Average Crossover
```bash
STRATEGY=moving_average_crossover
SHORT_MA_PERIOD=10
LONG_MA_PERIOD=30
```

#### RSI Mean Reversion
```bash
STRATEGY=rsi_mean_reversion
RSI_PERIOD=14
RSI_OVERSOLD=30
RSI_OVERBOUGHT=70
```

#### Bollinger Bands
```bash
STRATEGY=bollinger_bands
BB_PERIOD=20
BB_STD_DEV=2.0
```

#### Multi-Strategy
```bash
STRATEGY=multi_strategy
# Combines multiple strategies with weighted signals
```

## GitHub Repository Setup

### 1. Create GitHub Repository

1. Create a new repository on GitHub
2. Push your local code to the repository:

```bash
git remote add origin https://github.com/your-username/delta-trading-bot.git
git branch -M main
git push -u origin main
```

### 2. Configure GitHub Secrets

Go to your repository → Settings → Secrets and variables → Actions

Add the following secrets:

#### Required Secrets
```
DELTA_API_KEY=your_delta_api_key
DELTA_API_SECRET=your_delta_api_secret
```

#### Optional Secrets (for notifications)
```
DISCORD_WEBHOOK_URL=your_discord_webhook_url
SLACK_WEBHOOK_URL=your_slack_webhook_url
```

#### Emergency Controls
```
EMERGENCY_STOP_DRAWDOWN=20
EMERGENCY_STOP_DAILY_LOSS=5000
EMERGENCY_CLOSE_POSITION=false
```

### 3. Configure GitHub Variables

Add the following variables for risk management:

```
MAX_POSITION_SIZE=1000
MAX_DRAWDOWN_ALERT=10
MIN_PNL_ALERT=-1000
MAX_LEVERAGE=10
MAX_OPEN_ORDERS=5
```

### 4. Enable GitHub Actions

1. Go to your repository → Actions
2. Enable Actions if not already enabled
3. The workflows should appear automatically

## Security Configuration

### 1. API Key Security

- **Never commit API keys to the repository**
- Use GitHub Secrets for sensitive data
- Regularly rotate API keys
- Monitor API key usage

### 2. IP Whitelisting

Configure IP restrictions on Delta Exchange:

```bash
# For static IP
IP_WHITELIST=your.server.ip.address
ENABLE_IP_RESTRICTION=true

# For dynamic IP (less secure)
ENABLE_IP_RESTRICTION=false
```

### 3. Encryption

Enable data encryption for sensitive information:

```bash
ENABLE_ENCRYPTION=true
```

### 4. Session Management

Configure session timeouts:

```bash
SESSION_TIMEOUT=3600  # 1 hour
```

## Testing and Validation

### 1. Run System Validation

```bash
python scripts/run_tests.py --test-type system --verbose
```

### 2. Test API Connectivity

```bash
python scripts/run_tests.py --test-type api --verbose
```

### 3. Run Unit Tests

```bash
python scripts/run_tests.py --test-type unit --verbose
```

### 4. Run Integration Tests

```bash
python scripts/run_tests.py --test-type integration --verbose
```

### 5. Run All Tests

```bash
python scripts/run_tests.py --test-type all --verbose --output test_results.json
```

### 6. Validate Configuration

```bash
python -c "
from src.utils.config import Config
config = Config()
print('Configuration valid:', config.validate_settings())
print('Trading enabled:', config.get('TRADING_ENABLED'))
print('Paper trading:', config.get('PAPER_TRADING'))
"
```

## Deployment

### 1. Paper Trading Deployment

Start with paper trading to validate the system:

```bash
# Set paper trading mode
TRADING_ENABLED=true
PAPER_TRADING=true
```

### 2. Enable GitHub Actions

The bot will run automatically based on the configured schedule:

- **Trading Bot**: Runs every 15 minutes during market hours
- **Monitoring**: Runs every 5 minutes during market hours
- **Backtesting**: Runs weekly on Sundays

### 3. Manual Execution

You can manually trigger workflows:

1. Go to Actions → Select workflow
2. Click "Run workflow"
3. Configure parameters if needed

### 4. Live Trading Deployment

**⚠️ Only enable live trading after thorough testing!**

```bash
# Enable live trading (use with extreme caution)
TRADING_ENABLED=true
PAPER_TRADING=false
```

### 5. Gradual Rollout

1. **Week 1**: Paper trading only
2. **Week 2**: Live trading with minimal position sizes
3. **Week 3+**: Gradually increase position sizes based on performance

## Monitoring and Maintenance

### 1. Monitoring Dashboard

Monitor your bot through:

- **GitHub Actions**: View execution logs and status
- **Log Files**: Check detailed logs in the `logs/` directory
- **Notifications**: Discord/Slack alerts for important events

### 2. Performance Tracking

Key metrics to monitor:

- **Total Return**: Overall profitability
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Average Trade Duration**: Time in positions

### 3. Risk Monitoring

Monitor these risk indicators:

- **Current Exposure**: Total position size
- **Leverage**: Current leverage ratio
- **Daily PnL**: Daily profit/loss
- **Open Orders**: Number of pending orders

### 4. Regular Maintenance

#### Daily Tasks
- Check execution logs
- Monitor performance metrics
- Verify API connectivity

#### Weekly Tasks
- Review strategy performance
- Analyze backtest results
- Update risk parameters if needed

#### Monthly Tasks
- Rotate API keys
- Review and update strategies
- Analyze market conditions

### 5. Alerts and Notifications

Configure alerts for:

- **Critical**: Emergency stops, API failures
- **Warning**: High drawdown, risk limit breaches
- **Info**: Trade executions, daily summaries

## Troubleshooting

### Common Issues

#### 1. API Connection Errors

**Symptoms**: "API connection failed" errors

**Solutions**:
```bash
# Check API credentials
python scripts/run_tests.py --test-type api

# Verify network connectivity
curl -I https://api.india.delta.exchange/v2/products

# Check IP whitelist settings
```

#### 2. GitHub Actions Failures

**Symptoms**: Workflow runs fail

**Solutions**:
- Check GitHub Secrets are properly set
- Verify repository permissions
- Review workflow logs for specific errors

#### 3. Trading Not Executing

**Symptoms**: Bot runs but no trades execute

**Solutions**:
```bash
# Check trading is enabled
echo $TRADING_ENABLED

# Verify strategy signals
python -c "
from src.trading.bot import DeltaTradingBot
from src.utils.config import Config
bot = DeltaTradingBot(Config())
# Check bot status
"

# Review risk management settings
```

#### 4. High Memory Usage

**Symptoms**: Bot crashes with memory errors

**Solutions**:
- Reduce historical data lookback period
- Optimize indicator calculations
- Increase server memory

#### 5. Emergency Stop Triggered

**Symptoms**: Bot stops trading automatically

**Solutions**:
1. Review the emergency conditions that triggered the stop
2. Analyze what caused the drawdown or losses
3. Adjust risk parameters if necessary
4. Manually reset emergency stop after review

### Log Analysis

#### Check Recent Logs
```bash
tail -f logs/trading_bot.log
```

#### Search for Errors
```bash
grep -i error logs/trading_bot.log | tail -20
```

#### Check Performance Logs
```bash
cat logs/performance.log | tail -10
```

### Performance Optimization

#### 1. Reduce API Calls
- Implement caching for market data
- Batch API requests where possible
- Use WebSocket connections for real-time data

#### 2. Optimize Indicators
- Use vectorized calculations
- Cache indicator results
- Reduce calculation frequency for slow indicators

#### 3. Memory Management
- Limit historical data retention
- Clean up old log files
- Use efficient data structures

### Recovery Procedures

#### 1. System Recovery
```bash
# Stop all processes
pkill -f "python.*trading"

# Check system status
python scripts/run_tests.py --test-type system

# Restart with paper trading
PAPER_TRADING=true python src/trading/bot.py
```

#### 2. Data Recovery
```bash
# Backup current data
cp -r logs/ backups/logs_$(date +%Y%m%d_%H%M%S)/
cp -r data/ backups/data_$(date +%Y%m%d_%H%M%S)/

# Restore from backup if needed
# cp -r backups/logs_YYYYMMDD_HHMMSS/ logs/
```

## Support and Resources

### Documentation
- [Delta Exchange API Documentation](https://docs.delta.exchange/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Python Trading Libraries](https://github.com/topics/algorithmic-trading)

### Community
- [Delta Exchange Community](https://t.me/deltaexchange)
- [Algorithmic Trading Forums](https://www.reddit.com/r/algotrading/)

### Professional Support
For professional support and customization:
- Create an issue in the GitHub repository
- Contact the development team
- Consider hiring a quantitative developer

---

**⚠️ Important Disclaimers:**

1. **Risk Warning**: Cryptocurrency trading involves substantial risk of loss. Never trade with money you cannot afford to lose.

2. **No Financial Advice**: This software is for educational and research purposes. It does not constitute financial advice.

3. **Testing Required**: Always test thoroughly in paper trading mode before using real money.

4. **Monitoring Required**: Automated trading systems require constant monitoring and maintenance.

5. **Regulatory Compliance**: Ensure compliance with local regulations regarding automated trading.

**📈 Happy Trading!**

Remember: The best trading system is one that you understand, test thoroughly, and monitor continuously.

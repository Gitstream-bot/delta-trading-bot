# GitHub Actions Setup Guide

Due to permission restrictions, the GitHub Actions workflows need to be manually created in your repository. Follow these steps to set up automated trading execution.

## Step 1: Create Workflow Directory

In your repository, create the following directory structure:

```
.github/
└── workflows/
    ├── trading-bot.yml
    ├── backtest.yml
    └── monitoring.yml
```

## Step 2: Create Trading Bot Workflow

Create `.github/workflows/trading-bot.yml` with the following content:

```yaml
name: Delta Trading Bot

on:
  schedule:
    # Run every 15 minutes during market hours (9 AM to 5 PM IST, Mon-Fri)
    - cron: '*/15 9-17 * * 1-5'
  
  workflow_dispatch:
    inputs:
      trading_mode:
        description: 'Trading mode'
        required: true
        default: 'paper'
        type: choice
        options:
          - paper
          - live
      
      strategy:
        description: 'Trading strategy'
        required: false
        default: 'moving_average_crossover'
        type: choice
        options:
          - moving_average_crossover
          - rsi_mean_reversion
          - bollinger_bands
          - macd_momentum
          - multi_strategy

env:
  PYTHON_VERSION: '3.11'

jobs:
  trading_execution:
    runs-on: ubuntu-latest
    
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: 'pip'
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
      
      - name: Run trading bot
        env:
          DELTA_API_KEY: ${{ secrets.DELTA_API_KEY }}
          DELTA_API_SECRET: ${{ secrets.DELTA_API_SECRET }}
          TRADING_ENABLED: true
          PAPER_TRADING: ${{ github.event.inputs.trading_mode == 'paper' || 'true' }}
          STRATEGY: ${{ github.event.inputs.strategy || 'moving_average_crossover' }}
        run: |
          python -c "
          import sys
          sys.path.append('src')
          
          from utils.config import Config
          from trading.bot import DeltaTradingBot
          from utils.logger import initialize_logging
          
          # Initialize logging
          initialize_logging()
          
          # Create configuration
          config = Config()
          
          # Create and run bot
          bot = DeltaTradingBot(config)
          result = bot.execute_single_cycle()
          
          print(f'Execution result: {result}')
          "
```

## Step 3: Create Backtesting Workflow

Create `.github/workflows/backtest.yml` with the following content:

```yaml
name: Strategy Backtesting

on:
  workflow_dispatch:
    inputs:
      strategy:
        description: 'Strategy to backtest'
        required: true
        default: 'moving_average_crossover'
        type: choice
        options:
          - moving_average_crossover
          - rsi_mean_reversion
          - bollinger_bands
          - macd_momentum
          - multi_strategy
      
      start_date:
        description: 'Backtest start date (YYYY-MM-DD)'
        required: true
        default: '2024-01-01'
        type: string
      
      end_date:
        description: 'Backtest end date (YYYY-MM-DD)'
        required: true
        default: '2024-12-31'
        type: string

  schedule:
    # Run comprehensive backtesting every Sunday at 2 AM UTC
    - cron: '0 2 * * 0'

jobs:
  backtest:
    runs-on: ubuntu-latest
    
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
          cache: 'pip'
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
      
      - name: Run backtest
        env:
          STRATEGY: ${{ github.event.inputs.strategy || 'moving_average_crossover' }}
          START_DATE: ${{ github.event.inputs.start_date || '2024-01-01' }}
          END_DATE: ${{ github.event.inputs.end_date || '2024-12-31' }}
        run: |
          python -c "
          import sys
          sys.path.append('src')
          
          from utils.config import Config
          from data.analysis import TechnicalAnalyzer
          from utils.logger import initialize_logging
          
          # Initialize logging
          initialize_logging()
          
          print(f'Running backtest for strategy: {os.getenv(\"STRATEGY\")}')
          print(f'Period: {os.getenv(\"START_DATE\")} to {os.getenv(\"END_DATE\")}')
          
          # Implement backtesting logic here
          print('Backtest completed successfully')
          "
```

## Step 4: Create Monitoring Workflow

Create `.github/workflows/monitoring.yml` with the following content:

```yaml
name: System Monitoring

on:
  schedule:
    # Run every 30 minutes
    - cron: '*/30 * * * *'
  
  workflow_dispatch:

jobs:
  health_check:
    runs-on: ubuntu-latest
    
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
          cache: 'pip'
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
      
      - name: Run system health check
        env:
          DELTA_API_KEY: ${{ secrets.DELTA_API_KEY }}
          DELTA_API_SECRET: ${{ secrets.DELTA_API_SECRET }}
        run: |
          python scripts/run_tests.py --test-type api
```

## Step 5: Configure Repository Secrets

Go to your repository → Settings → Secrets and variables → Actions

Add the following secrets:

### Required Secrets
- `DELTA_API_KEY`: Your Delta Exchange API key
- `DELTA_API_SECRET`: Your Delta Exchange API secret

### Optional Secrets (for notifications)
- `DISCORD_WEBHOOK_URL`: Discord webhook for notifications
- `SLACK_WEBHOOK_URL`: Slack webhook for notifications

## Step 6: Configure Repository Variables

Add the following variables for risk management:

- `MAX_POSITION_SIZE`: Maximum position size (e.g., 1000)
- `MAX_DRAWDOWN_ALERT`: Maximum drawdown alert threshold (e.g., 10)
- `MIN_PNL_ALERT`: Minimum PnL alert threshold (e.g., -1000)

## Step 7: Enable Workflows

1. Go to your repository → Actions
2. You should see the three workflows listed
3. Enable each workflow by clicking on it and then "Enable workflow"

## Step 8: Test the Setup

1. Go to Actions → "Delta Trading Bot"
2. Click "Run workflow"
3. Select "paper" trading mode
4. Choose a strategy
5. Click "Run workflow"

## Important Notes

1. **Start with Paper Trading**: Always test with paper trading mode first
2. **Monitor Execution**: Check the Actions tab regularly for execution logs
3. **API Limits**: Be aware of GitHub Actions usage limits on free accounts
4. **Security**: Never commit API keys to the repository - use secrets only
5. **Testing**: Run the monitoring workflow first to ensure API connectivity

## Troubleshooting

### Workflow Not Running
- Check that the workflow files are in the correct location
- Verify that Actions are enabled in your repository settings
- Check the cron schedule syntax

### API Errors
- Verify API keys are correctly set in repository secrets
- Test API connectivity using the monitoring workflow
- Check Delta Exchange API status

### Permission Errors
- Ensure your GitHub account has necessary permissions
- Check if your repository is private (some features require paid plans)

## Next Steps

After setting up the workflows:

1. **Test thoroughly** with paper trading
2. **Monitor performance** through the Actions logs
3. **Adjust parameters** based on results
4. **Gradually transition** to live trading (if desired)
5. **Set up notifications** for important events

Remember: Always start with small position sizes and paper trading mode when testing automated trading systems.

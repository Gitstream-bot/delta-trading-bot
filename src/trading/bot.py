"""
Main Trading Bot Implementation

This module contains the core trading bot logic that orchestrates market analysis,
strategy execution, and risk management for automated Bitcoin trading on Delta Exchange.
"""

import time
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from decimal import Decimal, ROUND_DOWN
from datetime import datetime, timedelta
import json

from ..api.delta_client import DeltaExchangeClient
from ..data.market_data import MarketDataCollector
from ..data.analysis import TechnicalAnalyzer
from .strategies import StrategyManager
from .risk_manager import RiskManager
from ..utils.logger import get_logger
from ..utils.config import Config


class DeltaTradingBot:
    """
    Main trading bot class that coordinates all trading activities.
    
    This class serves as the central orchestrator for the automated trading system,
    managing market data collection, strategy execution, risk management, and
    trade execution on Delta Exchange.
    """
    
    def __init__(self, config: Config):
        """
        Initialize the trading bot with configuration.
        
        Args:
            config: Configuration object containing all bot settings
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # Initialize core components
        self.delta_client = DeltaExchangeClient(config)
        self.market_data = MarketDataCollector(config, self.delta_client)
        self.analyzer = TechnicalAnalyzer(config)
        self.strategy_manager = StrategyManager(config)
        self.risk_manager = RiskManager(config, self.delta_client)
        
        # Bot state
        self.is_running = False
        self.last_execution_time = None
        self.execution_count = 0
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': Decimal('0'),
            'max_drawdown': Decimal('0'),
            'start_balance': None,
            'current_balance': None
        }
        
        # Trading state
        self.current_position = None
        self.open_orders = []
        self.last_signal = None
        self.signal_history = []
        
        self.logger.info("Delta Trading Bot initialized successfully")
    
    async def start(self) -> None:
        """
        Start the trading bot main execution loop.
        
        This method begins the continuous trading operation, including market
        monitoring, strategy execution, and risk management.
        """
        try:
            self.logger.info("Starting Delta Trading Bot")
            
            # Perform initial health checks
            if not await self._perform_health_checks():
                raise Exception("Health checks failed - cannot start bot")
            
            # Initialize performance tracking
            await self._initialize_performance_tracking()
            
            # Set bot as running
            self.is_running = True
            
            # Start main execution loop
            await self._main_execution_loop()
            
        except Exception as e:
            self.logger.error(f"Error starting trading bot: {str(e)}")
            self.is_running = False
            raise
    
    async def stop(self) -> None:
        """
        Stop the trading bot gracefully.
        
        This method safely shuts down the bot, cancels open orders if configured,
        and saves final performance metrics.
        """
        try:
            self.logger.info("Stopping Delta Trading Bot")
            self.is_running = False
            
            # Cancel open orders if configured
            if self.config.get('CANCEL_ORDERS_ON_STOP', True):
                await self._cancel_all_orders()
            
            # Save final performance metrics
            await self._save_performance_metrics()
            
            self.logger.info("Delta Trading Bot stopped successfully")
            
        except Exception as e:
            self.logger.error(f"Error stopping trading bot: {str(e)}")
            raise
    
    async def _perform_health_checks(self) -> bool:
        """
        Perform comprehensive health checks before starting trading.
        
        Returns:
            True if all health checks pass, False otherwise
        """
        try:
            self.logger.info("Performing health checks")
            
            # Check API connectivity
            if not self.delta_client.health_check():
                self.logger.error("Delta Exchange API health check failed")
                return False
            
            # Check account access and permissions
            try:
                balance_info = self.delta_client.get_account_balance()
                if not balance_info.get('success'):
                    self.logger.error("Cannot access account balance")
                    return False
            except Exception as e:
                self.logger.error(f"Account access check failed: {str(e)}")
                return False
            
            # Check trading permissions
            if self.config.get('TRADING_ENABLED', False):
                try:
                    # Try to get current position (requires trading permissions)
                    position_info = self.delta_client.get_position()
                    if not position_info.get('success'):
                        self.logger.error("Cannot access position information")
                        return False
                except Exception as e:
                    self.logger.error(f"Trading permissions check failed: {str(e)}")
                    return False
            
            # Validate risk management settings
            if not self.risk_manager.validate_settings():
                self.logger.error("Risk management settings validation failed")
                return False
            
            # Check strategy configuration
            if not self.strategy_manager.validate_configuration():
                self.logger.error("Strategy configuration validation failed")
                return False
            
            self.logger.info("All health checks passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Health check error: {str(e)}")
            return False
    
    async def _initialize_performance_tracking(self) -> None:
        """Initialize performance tracking with current account state."""
        try:
            balance_info = self.delta_client.get_account_balance()
            if balance_info.get('success'):
                # Find USD balance for performance tracking
                balances = balance_info.get('balances', [])
                usd_balance = next((b for b in balances if b.get('asset', {}).get('symbol') == 'USD'), None)
                
                if usd_balance:
                    current_balance = Decimal(str(usd_balance.get('balance', 0)))
                    self.performance_metrics['start_balance'] = current_balance
                    self.performance_metrics['current_balance'] = current_balance
                    self.logger.info(f"Performance tracking initialized with balance: {current_balance}")
                else:
                    self.logger.warning("USD balance not found for performance tracking")
            
        except Exception as e:
            self.logger.error(f"Error initializing performance tracking: {str(e)}")
    
    async def _main_execution_loop(self) -> None:
        """
        Main execution loop that runs continuously while the bot is active.
        
        This loop handles market data collection, strategy evaluation,
        trade execution, and risk management at regular intervals.
        """
        execution_interval = self.config.get('EXECUTION_INTERVAL', 900)  # 15 minutes default
        
        while self.is_running:
            try:
                loop_start_time = time.time()
                
                # Execute one trading cycle
                await self._execute_trading_cycle()
                
                # Update execution tracking
                self.last_execution_time = datetime.now()
                self.execution_count += 1
                
                # Calculate sleep time to maintain consistent intervals
                loop_duration = time.time() - loop_start_time
                sleep_time = max(0, execution_interval - loop_duration)
                
                if sleep_time > 0:
                    self.logger.debug(f"Sleeping for {sleep_time:.2f} seconds until next execution")
                    await asyncio.sleep(sleep_time)
                else:
                    self.logger.warning(f"Execution cycle took {loop_duration:.2f}s, longer than interval {execution_interval}s")
                
            except Exception as e:
                self.logger.error(f"Error in main execution loop: {str(e)}")
                # Sleep before retrying to avoid rapid error loops
                await asyncio.sleep(60)
    
    async def _execute_trading_cycle(self) -> None:
        """
        Execute one complete trading cycle.
        
        This includes market data collection, technical analysis, strategy
        evaluation, risk checks, and trade execution if conditions are met.
        """
        try:
            self.logger.debug(f"Starting trading cycle #{self.execution_count + 1}")
            
            # Step 1: Collect current market data
            market_data = await self._collect_market_data()
            if not market_data:
                self.logger.warning("Failed to collect market data, skipping cycle")
                return
            
            # Step 2: Update current position and orders
            await self._update_trading_state()
            
            # Step 3: Perform technical analysis
            analysis_results = await self._perform_technical_analysis(market_data)
            
            # Step 4: Generate trading signals
            signals = await self._generate_trading_signals(analysis_results)
            
            # Step 5: Evaluate risk management
            risk_assessment = await self._assess_risk_conditions()
            
            # Step 6: Execute trades if conditions are met
            if signals and risk_assessment.get('can_trade', False):
                await self._execute_trades(signals, risk_assessment)
            
            # Step 7: Update performance metrics
            await self._update_performance_metrics()
            
            # Step 8: Log cycle summary
            await self._log_cycle_summary(market_data, signals, risk_assessment)
            
        except Exception as e:
            self.logger.error(f"Error in trading cycle execution: {str(e)}")
            raise
    
    async def _collect_market_data(self) -> Optional[Dict[str, Any]]:
        """
        Collect current market data including price, orderbook, and historical data.
        
        Returns:
            Dictionary containing market data, or None if collection fails
        """
        try:
            # Get current price and basic market info
            current_price = self.delta_client.get_current_price()
            orderbook = self.delta_client.get_orderbook()
            
            # Collect historical data for technical analysis
            historical_data = await self.market_data.get_historical_data(
                symbol=self.config.get('SYMBOL', 'BTCUSD'),
                timeframe='1h',
                limit=200  # Enough for most technical indicators
            )
            
            market_data = {
                'current_price': current_price,
                'orderbook': orderbook,
                'historical_data': historical_data,
                'timestamp': time.time()
            }
            
            self.logger.debug(f"Market data collected - Price: {current_price}")
            return market_data
            
        except Exception as e:
            self.logger.error(f"Error collecting market data: {str(e)}")
            return None
    
    async def _update_trading_state(self) -> None:
        """Update current position and open orders information."""
        try:
            # Get current position
            position_info = self.delta_client.get_position()
            if position_info.get('success'):
                self.current_position = position_info.get('position')
            
            # Get open orders
            self.open_orders = self.delta_client.get_open_orders()
            
            position_size = self.current_position.get('size', 0) if self.current_position else 0
            self.logger.debug(f"Trading state updated - Position: {position_size}, Open orders: {len(self.open_orders)}")
            
        except Exception as e:
            self.logger.error(f"Error updating trading state: {str(e)}")
    
    async def _perform_technical_analysis(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform technical analysis on market data.
        
        Args:
            market_data: Current market data
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            historical_data = market_data.get('historical_data', [])
            current_price = market_data.get('current_price')
            
            # Calculate technical indicators
            indicators = self.analyzer.calculate_indicators(historical_data)
            
            # Analyze price patterns
            patterns = self.analyzer.detect_patterns(historical_data)
            
            # Calculate volatility and momentum
            volatility = self.analyzer.calculate_volatility(historical_data)
            momentum = self.analyzer.calculate_momentum(historical_data)
            
            analysis_results = {
                'indicators': indicators,
                'patterns': patterns,
                'volatility': volatility,
                'momentum': momentum,
                'current_price': current_price,
                'timestamp': time.time()
            }
            
            self.logger.debug("Technical analysis completed")
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"Error in technical analysis: {str(e)}")
            return {}
    
    async def _generate_trading_signals(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate trading signals based on analysis results.
        
        Args:
            analysis_results: Results from technical analysis
            
        Returns:
            List of trading signals
        """
        try:
            signals = self.strategy_manager.generate_signals(
                analysis_results,
                self.current_position,
                self.signal_history
            )
            
            # Store signal in history
            if signals:
                self.last_signal = signals[0]  # Store the primary signal
                self.signal_history.append({
                    'signal': self.last_signal,
                    'timestamp': time.time()
                })
                
                # Keep only recent signal history
                max_history = 100
                if len(self.signal_history) > max_history:
                    self.signal_history = self.signal_history[-max_history:]
            
            if signals:
                self.logger.info(f"Generated {len(signals)} trading signals")
                for signal in signals:
                    self.logger.info(f"Signal: {signal.get('action')} - {signal.get('strategy')} - Confidence: {signal.get('confidence', 0):.2f}")
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating trading signals: {str(e)}")
            return []
    
    async def _assess_risk_conditions(self) -> Dict[str, Any]:
        """
        Assess current risk conditions and trading constraints.
        
        Returns:
            Dictionary containing risk assessment results
        """
        try:
            risk_assessment = self.risk_manager.assess_risk(
                current_position=self.current_position,
                open_orders=self.open_orders,
                performance_metrics=self.performance_metrics,
                market_conditions={}  # Could include volatility, etc.
            )
            
            can_trade = risk_assessment.get('can_trade', False)
            risk_level = risk_assessment.get('risk_level', 'unknown')
            
            self.logger.debug(f"Risk assessment - Can trade: {can_trade}, Risk level: {risk_level}")
            
            return risk_assessment
            
        except Exception as e:
            self.logger.error(f"Error in risk assessment: {str(e)}")
            return {'can_trade': False, 'reason': 'Risk assessment failed'}
    
    async def _execute_trades(self, signals: List[Dict[str, Any]], 
                            risk_assessment: Dict[str, Any]) -> None:
        """
        Execute trades based on signals and risk assessment.
        
        Args:
            signals: List of trading signals
            risk_assessment: Risk assessment results
        """
        try:
            if not self.config.get('TRADING_ENABLED', False):
                self.logger.info("Trading is disabled - signals generated but not executed")
                return
            
            for signal in signals:
                try:
                    await self._execute_single_trade(signal, risk_assessment)
                except Exception as e:
                    self.logger.error(f"Error executing trade for signal {signal}: {str(e)}")
                    continue
            
        except Exception as e:
            self.logger.error(f"Error in trade execution: {str(e)}")
    
    async def _execute_single_trade(self, signal: Dict[str, Any], 
                                  risk_assessment: Dict[str, Any]) -> None:
        """
        Execute a single trade based on a signal.
        
        Args:
            signal: Trading signal to execute
            risk_assessment: Risk assessment results
        """
        try:
            action = signal.get('action')  # 'buy', 'sell', 'close'
            strategy = signal.get('strategy')
            confidence = signal.get('confidence', 0)
            
            # Calculate position size based on risk management
            position_size = self.risk_manager.calculate_position_size(
                signal=signal,
                account_balance=self.performance_metrics.get('current_balance'),
                current_price=signal.get('price')
            )
            
            if position_size <= 0:
                self.logger.warning(f"Position size calculation returned {position_size}, skipping trade")
                return
            
            self.logger.info(f"Executing {action} trade - Strategy: {strategy}, Size: {position_size}, Confidence: {confidence:.2f}")
            
            # Execute the trade based on action type
            if action == 'buy':
                await self._execute_buy_order(position_size, signal)
            elif action == 'sell':
                await self._execute_sell_order(position_size, signal)
            elif action == 'close':
                await self._execute_close_position(signal)
            else:
                self.logger.warning(f"Unknown action type: {action}")
                return
            
            # Update performance metrics
            self.performance_metrics['total_trades'] += 1
            
        except Exception as e:
            self.logger.error(f"Error executing single trade: {str(e)}")
            raise
    
    async def _execute_buy_order(self, size: Decimal, signal: Dict[str, Any]) -> None:
        """Execute a buy order."""
        try:
            order_type = signal.get('order_type', 'market')
            
            if order_type == 'market':
                result = self.delta_client.place_market_order('buy', size)
            else:
                limit_price = signal.get('limit_price')
                if not limit_price:
                    raise ValueError("Limit price required for limit orders")
                result = self.delta_client.place_limit_order('buy', size, limit_price)
            
            if result.get('success'):
                order_id = result.get('order_id')
                self.logger.info(f"Buy order executed successfully: {order_id}")
                
                # Place stop-loss if configured
                await self._place_stop_loss_if_configured(size, 'buy', signal)
            else:
                self.logger.error(f"Buy order failed: {result}")
                
        except Exception as e:
            self.logger.error(f"Error executing buy order: {str(e)}")
            raise
    
    async def _execute_sell_order(self, size: Decimal, signal: Dict[str, Any]) -> None:
        """Execute a sell order."""
        try:
            order_type = signal.get('order_type', 'market')
            
            if order_type == 'market':
                result = self.delta_client.place_market_order('sell', size)
            else:
                limit_price = signal.get('limit_price')
                if not limit_price:
                    raise ValueError("Limit price required for limit orders")
                result = self.delta_client.place_limit_order('sell', size, limit_price)
            
            if result.get('success'):
                order_id = result.get('order_id')
                self.logger.info(f"Sell order executed successfully: {order_id}")
                
                # Place stop-loss if configured
                await self._place_stop_loss_if_configured(size, 'sell', signal)
            else:
                self.logger.error(f"Sell order failed: {result}")
                
        except Exception as e:
            self.logger.error(f"Error executing sell order: {str(e)}")
            raise
    
    async def _execute_close_position(self, signal: Dict[str, Any]) -> None:
        """Close current position."""
        try:
            if not self.current_position:
                self.logger.warning("No position to close")
                return
            
            position_size = abs(float(self.current_position.get('size', 0)))
            if position_size == 0:
                self.logger.warning("Position size is zero, nothing to close")
                return
            
            # Determine side for closing (opposite of current position)
            current_side = 'buy' if float(self.current_position.get('size', 0)) > 0 else 'sell'
            close_side = 'sell' if current_side == 'buy' else 'buy'
            
            result = self.delta_client.place_market_order(close_side, position_size, reduce_only=True)
            
            if result.get('success'):
                order_id = result.get('order_id')
                self.logger.info(f"Position closed successfully: {order_id}")
            else:
                self.logger.error(f"Position close failed: {result}")
                
        except Exception as e:
            self.logger.error(f"Error closing position: {str(e)}")
            raise
    
    async def _place_stop_loss_if_configured(self, size: Decimal, side: str, signal: Dict[str, Any]) -> None:
        """Place stop-loss order if configured in signal or settings."""
        try:
            stop_loss_pct = signal.get('stop_loss_percentage') or self.config.get('STOP_LOSS_PERCENTAGE')
            
            if not stop_loss_pct:
                return
            
            current_price = self.delta_client.get_current_price()
            stop_loss_pct_decimal = Decimal(str(stop_loss_pct)) / 100
            
            if side == 'buy':
                # For long position, stop-loss is below current price
                stop_price = current_price * (1 - stop_loss_pct_decimal)
                stop_side = 'sell'
            else:
                # For short position, stop-loss is above current price
                stop_price = current_price * (1 + stop_loss_pct_decimal)
                stop_side = 'buy'
            
            result = self.delta_client.place_stop_order(
                side=stop_side,
                size=size,
                stop_price=stop_price
            )
            
            if result.get('success'):
                self.logger.info(f"Stop-loss placed at {stop_price}")
            else:
                self.logger.error(f"Failed to place stop-loss: {result}")
                
        except Exception as e:
            self.logger.error(f"Error placing stop-loss: {str(e)}")
    
    async def _cancel_all_orders(self) -> None:
        """Cancel all open orders."""
        try:
            open_orders = self.delta_client.get_open_orders()
            
            for order in open_orders:
                try:
                    order_id = order.get('id')
                    result = self.delta_client.cancel_order(order_id)
                    if result.get('success'):
                        self.logger.info(f"Cancelled order: {order_id}")
                    else:
                        self.logger.error(f"Failed to cancel order {order_id}: {result}")
                except Exception as e:
                    self.logger.error(f"Error cancelling order {order.get('id')}: {str(e)}")
            
        except Exception as e:
            self.logger.error(f"Error cancelling all orders: {str(e)}")
    
    async def _update_performance_metrics(self) -> None:
        """Update performance tracking metrics."""
        try:
            # Get current balance
            balance_info = self.delta_client.get_account_balance()
            if balance_info.get('success'):
                balances = balance_info.get('balances', [])
                usd_balance = next((b for b in balances if b.get('asset', {}).get('symbol') == 'USD'), None)
                
                if usd_balance:
                    current_balance = Decimal(str(usd_balance.get('balance', 0)))
                    self.performance_metrics['current_balance'] = current_balance
                    
                    # Calculate PnL if we have a starting balance
                    if self.performance_metrics['start_balance']:
                        total_pnl = current_balance - self.performance_metrics['start_balance']
                        self.performance_metrics['total_pnl'] = total_pnl
                        
                        # Calculate drawdown
                        if total_pnl < self.performance_metrics['max_drawdown']:
                            self.performance_metrics['max_drawdown'] = total_pnl
            
            # Update trade statistics from recent fills
            recent_fills = self.delta_client.get_fills(limit=50)
            # Process fills to update win/loss statistics
            # This would require more sophisticated tracking of trade pairs
            
        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {str(e)}")
    
    async def _save_performance_metrics(self) -> None:
        """Save performance metrics to file."""
        try:
            metrics_file = 'logs/performance_metrics.json'
            
            # Convert Decimal values to float for JSON serialization
            serializable_metrics = {}
            for key, value in self.performance_metrics.items():
                if isinstance(value, Decimal):
                    serializable_metrics[key] = float(value)
                else:
                    serializable_metrics[key] = value
            
            serializable_metrics['last_updated'] = datetime.now().isoformat()
            
            with open(metrics_file, 'w') as f:
                json.dump(serializable_metrics, f, indent=2)
            
            self.logger.info(f"Performance metrics saved to {metrics_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving performance metrics: {str(e)}")
    
    async def _log_cycle_summary(self, market_data: Dict[str, Any], 
                               signals: List[Dict[str, Any]], 
                               risk_assessment: Dict[str, Any]) -> None:
        """Log a summary of the trading cycle."""
        try:
            current_price = market_data.get('current_price', 'N/A')
            signal_count = len(signals)
            can_trade = risk_assessment.get('can_trade', False)
            position_size = self.current_position.get('size', 0) if self.current_position else 0
            
            summary = (
                f"Cycle #{self.execution_count} Summary - "
                f"Price: {current_price}, "
                f"Signals: {signal_count}, "
                f"Can Trade: {can_trade}, "
                f"Position: {position_size}, "
                f"Open Orders: {len(self.open_orders)}"
            )
            
            self.logger.info(summary)
            
        except Exception as e:
            self.logger.error(f"Error logging cycle summary: {str(e)}")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current bot status and metrics.
        
        Returns:
            Dictionary containing current bot status
        """
        return {
            'is_running': self.is_running,
            'execution_count': self.execution_count,
            'last_execution_time': self.last_execution_time.isoformat() if self.last_execution_time else None,
            'current_position': self.current_position,
            'open_orders_count': len(self.open_orders),
            'performance_metrics': {k: float(v) if isinstance(v, Decimal) else v 
                                  for k, v in self.performance_metrics.items()},
            'last_signal': self.last_signal
        }

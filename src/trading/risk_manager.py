"""
Risk Management System

This module implements comprehensive risk management for the Delta Exchange trading bot,
including position sizing, drawdown protection, exposure limits, and emergency controls.
"""

import time
import json
from typing import Dict, List, Optional, Any, Tuple
from decimal import Decimal, ROUND_DOWN
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from ..utils.logger import get_logger
from ..utils.config import Config


class RiskLevel(Enum):
    """Risk level enumeration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskAction(Enum):
    """Risk action enumeration."""
    ALLOW = "allow"
    REDUCE = "reduce"
    BLOCK = "block"
    EMERGENCY_STOP = "emergency_stop"


@dataclass
class RiskMetrics:
    """Data class for risk metrics."""
    current_exposure: Decimal
    max_exposure: Decimal
    current_drawdown: Decimal
    max_drawdown: Decimal
    position_count: int
    max_positions: int
    leverage: Decimal
    max_leverage: Decimal
    daily_pnl: Decimal
    risk_level: RiskLevel
    recommended_action: RiskAction


class RiskManager:
    """
    Comprehensive risk management system for automated trading.
    
    This class implements multiple layers of risk control including:
    - Position sizing based on volatility and account size
    - Maximum drawdown protection
    - Exposure limits and concentration risk
    - Leverage controls
    - Emergency stop mechanisms
    """
    
    def __init__(self, config: Config, delta_client):
        """
        Initialize the risk manager.
        
        Args:
            config: Configuration object containing risk parameters
            delta_client: Delta Exchange API client
        """
        self.config = config
        self.delta_client = delta_client
        self.logger = get_logger(__name__)
        
        # Risk parameters from configuration
        self.max_position_size = Decimal(str(config.get('MAX_POSITION_SIZE', 1000)))
        self.max_drawdown_pct = Decimal(str(config.get('MAX_DRAWDOWN_PERCENTAGE', 10)))
        self.risk_per_trade_pct = Decimal(str(config.get('RISK_PER_TRADE_PERCENTAGE', 1)))
        self.max_leverage = Decimal(str(config.get('MAX_LEVERAGE', 10)))
        self.max_daily_trades = int(config.get('MAX_DAILY_TRADES', 20))
        self.max_open_positions = int(config.get('MAX_OPEN_POSITIONS', 3))
        
        # Emergency controls
        self.emergency_stop_drawdown = Decimal(str(config.get('EMERGENCY_STOP_DRAWDOWN', 20)))
        self.emergency_stop_daily_loss = Decimal(str(config.get('EMERGENCY_STOP_DAILY_LOSS', 5000)))
        
        # Risk state tracking
        self.daily_trade_count = 0
        self.daily_pnl = Decimal('0')
        self.last_reset_date = datetime.now().date()
        self.risk_violations = []
        self.emergency_stop_triggered = False
        
        # Load historical risk data
        self._load_risk_state()
        
        self.logger.info("Risk manager initialized with comprehensive controls")
    
    def validate_settings(self) -> bool:
        """
        Validate risk management settings for consistency and safety.
        
        Returns:
            True if settings are valid, False otherwise
        """
        try:
            validations = [
                (self.max_position_size > 0, "Max position size must be positive"),
                (0 < self.max_drawdown_pct <= 50, "Max drawdown must be between 0-50%"),
                (0 < self.risk_per_trade_pct <= 10, "Risk per trade must be between 0-10%"),
                (1 <= self.max_leverage <= 100, "Max leverage must be between 1-100"),
                (self.max_daily_trades > 0, "Max daily trades must be positive"),
                (self.max_open_positions > 0, "Max open positions must be positive"),
                (self.emergency_stop_drawdown > self.max_drawdown_pct, 
                 "Emergency stop drawdown must be greater than max drawdown"),
            ]
            
            for is_valid, error_msg in validations:
                if not is_valid:
                    self.logger.error(f"Risk settings validation failed: {error_msg}")
                    return False
            
            self.logger.info("Risk management settings validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating risk settings: {str(e)}")
            return False
    
    def assess_risk(self, current_position: Optional[Dict[str, Any]], 
                   open_orders: List[Dict[str, Any]], 
                   performance_metrics: Dict[str, Any],
                   market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive risk assessment.
        
        Args:
            current_position: Current position information
            open_orders: List of open orders
            performance_metrics: Performance metrics
            market_conditions: Current market conditions
            
        Returns:
            Dictionary containing risk assessment results
        """
        try:
            # Reset daily counters if needed
            self._reset_daily_counters_if_needed()
            
            # Calculate current risk metrics
            risk_metrics = self._calculate_risk_metrics(
                current_position, open_orders, performance_metrics, market_conditions
            )
            
            # Determine risk level and recommended action
            risk_level, recommended_action = self._evaluate_risk_level(risk_metrics)
            
            # Check for emergency conditions
            emergency_triggered = self._check_emergency_conditions(risk_metrics)
            
            # Update risk state
            self._update_risk_state(risk_metrics, risk_level)
            
            assessment = {
                'can_trade': recommended_action in [RiskAction.ALLOW, RiskAction.REDUCE],
                'risk_level': risk_level.value,
                'recommended_action': recommended_action.value,
                'risk_metrics': self._serialize_risk_metrics(risk_metrics),
                'emergency_stop': emergency_triggered,
                'violations': self.risk_violations[-10:],  # Last 10 violations
                'assessment_timestamp': time.time()
            }
            
            # Log risk assessment
            self.logger.info(f"Risk assessment completed - Level: {risk_level.value}, "
                           f"Action: {recommended_action.value}, Can trade: {assessment['can_trade']}")
            
            return assessment
            
        except Exception as e:
            self.logger.error(f"Error in risk assessment: {str(e)}")
            return {
                'can_trade': False,
                'risk_level': RiskLevel.CRITICAL.value,
                'recommended_action': RiskAction.BLOCK.value,
                'reason': f'Risk assessment failed: {str(e)}'
            }
    
    def calculate_position_size(self, signal: Dict[str, Any], 
                              account_balance: Optional[Decimal],
                              current_price: Decimal) -> Decimal:
        """
        Calculate optimal position size based on risk parameters.
        
        Args:
            signal: Trading signal information
            account_balance: Current account balance
            current_price: Current market price
            
        Returns:
            Calculated position size
        """
        try:
            if not account_balance or account_balance <= 0:
                self.logger.warning("Invalid account balance for position sizing")
                return Decimal('0')
            
            # Base position size using fixed percentage risk
            risk_amount = account_balance * (self.risk_per_trade_pct / 100)
            
            # Get stop loss distance from signal or use default
            stop_loss_pct = Decimal(str(signal.get('stop_loss_percentage', 2)))
            stop_loss_distance = current_price * (stop_loss_pct / 100)
            
            # Calculate position size based on risk amount and stop distance
            if stop_loss_distance > 0:
                base_position_size = risk_amount / stop_loss_distance
            else:
                # Fallback to percentage of balance
                base_position_size = account_balance * Decimal('0.1')  # 10% of balance
            
            # Apply volatility adjustment
            volatility_adjustment = self._calculate_volatility_adjustment(signal)
            adjusted_size = base_position_size * volatility_adjustment
            
            # Apply confidence adjustment
            confidence = Decimal(str(signal.get('confidence', 0.5)))
            confidence_adjustment = min(confidence * 2, Decimal('1'))  # Scale 0-1
            final_size = adjusted_size * confidence_adjustment
            
            # Apply maximum position size limit
            final_size = min(final_size, self.max_position_size)
            
            # Apply minimum position size (to avoid dust trades)
            min_position_size = current_price * Decimal('0.001')  # 0.1% of price
            if final_size < min_position_size:
                final_size = Decimal('0')
            
            # Round down to avoid precision issues
            final_size = final_size.quantize(Decimal('0.01'), rounding=ROUND_DOWN)
            
            self.logger.debug(f"Position size calculated: {final_size} "
                            f"(Risk: {risk_amount}, Confidence: {confidence})")
            
            return final_size
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {str(e)}")
            return Decimal('0')
    
    def _calculate_risk_metrics(self, current_position: Optional[Dict[str, Any]], 
                               open_orders: List[Dict[str, Any]], 
                               performance_metrics: Dict[str, Any],
                               market_conditions: Dict[str, Any]) -> RiskMetrics:
        """Calculate comprehensive risk metrics."""
        try:
            # Current exposure
            position_value = Decimal('0')
            if current_position:
                position_size = Decimal(str(current_position.get('size', 0)))
                # This would need the current price to calculate value
                # For now, we'll use the position size as a proxy
                position_value = abs(position_size)
            
            # Add pending order exposure
            pending_exposure = Decimal('0')
            for order in open_orders:
                order_size = Decimal(str(order.get('size', 0)))
                pending_exposure += abs(order_size)
            
            total_exposure = position_value + pending_exposure
            
            # Drawdown metrics
            current_drawdown = abs(Decimal(str(performance_metrics.get('max_drawdown', 0))))
            
            # Daily PnL
            daily_pnl = Decimal(str(performance_metrics.get('daily_pnl', 0)))
            
            # Position count
            position_count = 1 if current_position and float(current_position.get('size', 0)) != 0 else 0
            position_count += len(open_orders)
            
            # Leverage (placeholder - would need to be calculated from actual position data)
            leverage = Decimal('1')  # This should be calculated from actual position
            
            return RiskMetrics(
                current_exposure=total_exposure,
                max_exposure=self.max_position_size,
                current_drawdown=current_drawdown,
                max_drawdown=self.max_drawdown_pct,
                position_count=position_count,
                max_positions=self.max_open_positions,
                leverage=leverage,
                max_leverage=self.max_leverage,
                daily_pnl=daily_pnl,
                risk_level=RiskLevel.LOW,  # Will be determined later
                recommended_action=RiskAction.ALLOW  # Will be determined later
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating risk metrics: {str(e)}")
            # Return safe defaults
            return RiskMetrics(
                current_exposure=Decimal('0'),
                max_exposure=self.max_position_size,
                current_drawdown=Decimal('0'),
                max_drawdown=self.max_drawdown_pct,
                position_count=0,
                max_positions=self.max_open_positions,
                leverage=Decimal('1'),
                max_leverage=self.max_leverage,
                daily_pnl=Decimal('0'),
                risk_level=RiskLevel.CRITICAL,
                recommended_action=RiskAction.BLOCK
            )
    
    def _evaluate_risk_level(self, metrics: RiskMetrics) -> Tuple[RiskLevel, RiskAction]:
        """
        Evaluate risk level and determine recommended action.
        
        Args:
            metrics: Risk metrics
            
        Returns:
            Tuple of (risk_level, recommended_action)
        """
        try:
            violations = []
            risk_score = 0
            
            # Check exposure limits
            exposure_ratio = metrics.current_exposure / metrics.max_exposure
            if exposure_ratio > Decimal('0.9'):
                violations.append("Exposure near maximum limit")
                risk_score += 3
            elif exposure_ratio > Decimal('0.7'):
                risk_score += 2
            elif exposure_ratio > Decimal('0.5'):
                risk_score += 1
            
            # Check drawdown
            drawdown_ratio = metrics.current_drawdown / metrics.max_drawdown
            if drawdown_ratio > Decimal('0.9'):
                violations.append("Drawdown near maximum limit")
                risk_score += 4
            elif drawdown_ratio > Decimal('0.7'):
                violations.append("High drawdown detected")
                risk_score += 3
            elif drawdown_ratio > Decimal('0.5'):
                risk_score += 2
            
            # Check position count
            if metrics.position_count >= metrics.max_positions:
                violations.append("Maximum positions reached")
                risk_score += 2
            elif metrics.position_count >= metrics.max_positions * 0.8:
                risk_score += 1
            
            # Check leverage
            leverage_ratio = metrics.leverage / metrics.max_leverage
            if leverage_ratio > Decimal('0.9'):
                violations.append("Leverage near maximum")
                risk_score += 3
            elif leverage_ratio > Decimal('0.7'):
                risk_score += 2
            
            # Check daily trade count
            if self.daily_trade_count >= self.max_daily_trades:
                violations.append("Daily trade limit reached")
                risk_score += 2
            elif self.daily_trade_count >= self.max_daily_trades * 0.8:
                risk_score += 1
            
            # Check daily PnL
            if metrics.daily_pnl < -self.emergency_stop_daily_loss:
                violations.append("Daily loss limit exceeded")
                risk_score += 5
            elif metrics.daily_pnl < -self.emergency_stop_daily_loss * Decimal('0.7'):
                violations.append("High daily losses")
                risk_score += 3
            
            # Store violations
            if violations:
                self.risk_violations.extend([{
                    'timestamp': time.time(),
                    'violations': violations,
                    'risk_score': risk_score
                }])
            
            # Determine risk level and action based on score
            if risk_score >= 8:
                return RiskLevel.CRITICAL, RiskAction.EMERGENCY_STOP
            elif risk_score >= 6:
                return RiskLevel.HIGH, RiskAction.BLOCK
            elif risk_score >= 4:
                return RiskLevel.MEDIUM, RiskAction.REDUCE
            elif risk_score >= 2:
                return RiskLevel.MEDIUM, RiskAction.ALLOW
            else:
                return RiskLevel.LOW, RiskAction.ALLOW
                
        except Exception as e:
            self.logger.error(f"Error evaluating risk level: {str(e)}")
            return RiskLevel.CRITICAL, RiskAction.BLOCK
    
    def _check_emergency_conditions(self, metrics: RiskMetrics) -> bool:
        """
        Check for emergency stop conditions.
        
        Args:
            metrics: Risk metrics
            
        Returns:
            True if emergency stop should be triggered
        """
        try:
            emergency_conditions = []
            
            # Check maximum drawdown
            if metrics.current_drawdown >= self.emergency_stop_drawdown:
                emergency_conditions.append(f"Drawdown {metrics.current_drawdown}% >= {self.emergency_stop_drawdown}%")
            
            # Check daily loss limit
            if metrics.daily_pnl <= -self.emergency_stop_daily_loss:
                emergency_conditions.append(f"Daily loss {metrics.daily_pnl} >= {self.emergency_stop_daily_loss}")
            
            # Check if already triggered
            if self.emergency_stop_triggered:
                emergency_conditions.append("Emergency stop already triggered")
            
            if emergency_conditions:
                self.emergency_stop_triggered = True
                self.logger.critical(f"EMERGENCY STOP TRIGGERED: {emergency_conditions}")
                
                # Save emergency state
                self._save_emergency_state(emergency_conditions)
                
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking emergency conditions: {str(e)}")
            return True  # Err on the side of caution
    
    def _calculate_volatility_adjustment(self, signal: Dict[str, Any]) -> Decimal:
        """
        Calculate position size adjustment based on market volatility.
        
        Args:
            signal: Trading signal containing market data
            
        Returns:
            Volatility adjustment factor (0.5 to 1.5)
        """
        try:
            # Get volatility from signal or use default
            volatility = signal.get('volatility', {})
            
            if isinstance(volatility, dict):
                current_vol = volatility.get('realized_volatility', 20)  # Default 20%
            else:
                current_vol = float(volatility) if volatility else 20
            
            # Define normal volatility range (10-30% for crypto)
            normal_vol_low = 10
            normal_vol_high = 30
            
            if current_vol < normal_vol_low:
                # Low volatility - can increase position size
                adjustment = Decimal('1.2')
            elif current_vol > normal_vol_high:
                # High volatility - reduce position size
                vol_ratio = current_vol / normal_vol_high
                adjustment = Decimal('1') / Decimal(str(min(vol_ratio, 2)))  # Max 50% reduction
            else:
                # Normal volatility
                adjustment = Decimal('1')
            
            # Clamp adjustment between 0.5 and 1.5
            adjustment = max(Decimal('0.5'), min(adjustment, Decimal('1.5')))
            
            return adjustment
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility adjustment: {str(e)}")
            return Decimal('1')  # Default to no adjustment
    
    def _reset_daily_counters_if_needed(self) -> None:
        """Reset daily counters if it's a new day."""
        try:
            current_date = datetime.now().date()
            
            if current_date != self.last_reset_date:
                self.daily_trade_count = 0
                self.daily_pnl = Decimal('0')
                self.last_reset_date = current_date
                
                self.logger.info(f"Daily counters reset for {current_date}")
                
        except Exception as e:
            self.logger.error(f"Error resetting daily counters: {str(e)}")
    
    def _update_risk_state(self, metrics: RiskMetrics, risk_level: RiskLevel) -> None:
        """Update internal risk state tracking."""
        try:
            # Update metrics
            metrics.risk_level = risk_level
            
            # Save risk state periodically
            self._save_risk_state()
            
        except Exception as e:
            self.logger.error(f"Error updating risk state: {str(e)}")
    
    def _serialize_risk_metrics(self, metrics: RiskMetrics) -> Dict[str, Any]:
        """Convert risk metrics to serializable dictionary."""
        return {
            'current_exposure': float(metrics.current_exposure),
            'max_exposure': float(metrics.max_exposure),
            'exposure_ratio': float(metrics.current_exposure / metrics.max_exposure) if metrics.max_exposure > 0 else 0,
            'current_drawdown': float(metrics.current_drawdown),
            'max_drawdown': float(metrics.max_drawdown),
            'drawdown_ratio': float(metrics.current_drawdown / metrics.max_drawdown) if metrics.max_drawdown > 0 else 0,
            'position_count': metrics.position_count,
            'max_positions': metrics.max_positions,
            'leverage': float(metrics.leverage),
            'max_leverage': float(metrics.max_leverage),
            'daily_pnl': float(metrics.daily_pnl),
            'daily_trade_count': self.daily_trade_count,
            'max_daily_trades': self.max_daily_trades
        }
    
    def _load_risk_state(self) -> None:
        """Load risk state from persistent storage."""
        try:
            risk_state_file = 'logs/risk_state.json'
            
            if os.path.exists(risk_state_file):
                with open(risk_state_file, 'r') as f:
                    state = json.load(f)
                
                self.daily_trade_count = state.get('daily_trade_count', 0)
                self.daily_pnl = Decimal(str(state.get('daily_pnl', 0)))
                self.emergency_stop_triggered = state.get('emergency_stop_triggered', False)
                
                # Check if state is from today
                state_date = state.get('date')
                if state_date != datetime.now().date().isoformat():
                    self._reset_daily_counters_if_needed()
                
                self.logger.info("Risk state loaded from persistent storage")
            
        except Exception as e:
            self.logger.error(f"Error loading risk state: {str(e)}")
    
    def _save_risk_state(self) -> None:
        """Save risk state to persistent storage."""
        try:
            risk_state = {
                'date': datetime.now().date().isoformat(),
                'daily_trade_count': self.daily_trade_count,
                'daily_pnl': str(self.daily_pnl),
                'emergency_stop_triggered': self.emergency_stop_triggered,
                'last_updated': datetime.now().isoformat()
            }
            
            risk_state_file = 'logs/risk_state.json'
            with open(risk_state_file, 'w') as f:
                json.dump(risk_state, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving risk state: {str(e)}")
    
    def _save_emergency_state(self, conditions: List[str]) -> None:
        """Save emergency stop state."""
        try:
            emergency_state = {
                'timestamp': datetime.now().isoformat(),
                'conditions': conditions,
                'triggered': True
            }
            
            emergency_file = 'logs/emergency_stop.json'
            with open(emergency_file, 'w') as f:
                json.dump(emergency_state, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving emergency state: {str(e)}")
    
    def reset_emergency_stop(self) -> bool:
        """
        Reset emergency stop (manual intervention required).
        
        Returns:
            True if reset successful, False otherwise
        """
        try:
            self.emergency_stop_triggered = False
            self._save_risk_state()
            
            self.logger.warning("Emergency stop manually reset")
            return True
            
        except Exception as e:
            self.logger.error(f"Error resetting emergency stop: {str(e)}")
            return False
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive risk summary.
        
        Returns:
            Dictionary containing risk summary
        """
        try:
            return {
                'daily_trade_count': self.daily_trade_count,
                'max_daily_trades': self.max_daily_trades,
                'daily_pnl': float(self.daily_pnl),
                'emergency_stop_triggered': self.emergency_stop_triggered,
                'recent_violations': self.risk_violations[-5:],  # Last 5 violations
                'risk_parameters': {
                    'max_position_size': float(self.max_position_size),
                    'max_drawdown_pct': float(self.max_drawdown_pct),
                    'risk_per_trade_pct': float(self.risk_per_trade_pct),
                    'max_leverage': float(self.max_leverage),
                    'emergency_stop_drawdown': float(self.emergency_stop_drawdown),
                    'emergency_stop_daily_loss': float(self.emergency_stop_daily_loss)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting risk summary: {str(e)}")
            return {}

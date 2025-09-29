"""
Configuration Management System

This module provides centralized configuration management for the Delta Exchange
trading bot with support for environment variables, file-based configuration,
validation, and secure handling of sensitive data.
"""

import os
import json
import yaml
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
from dataclasses import dataclass, field
from decimal import Decimal

from .logger import get_logger


@dataclass
class TradingConfig:
    """Trading-specific configuration parameters."""
    symbol: str = 'BTCUSD'
    product_id: int = 27
    trading_enabled: bool = False
    paper_trading: bool = True
    execution_interval: int = 900  # 15 minutes
    strategy: str = 'moving_average_crossover'
    
    # Risk management
    max_position_size: Decimal = field(default_factory=lambda: Decimal('1000'))
    stop_loss_percentage: Decimal = field(default_factory=lambda: Decimal('2.0'))
    max_drawdown_percentage: Decimal = field(default_factory=lambda: Decimal('10.0'))
    risk_per_trade_percentage: Decimal = field(default_factory=lambda: Decimal('1.0'))
    
    # Strategy parameters
    short_ma_period: int = 10
    long_ma_period: int = 30
    rsi_period: int = 14
    rsi_oversold: int = 30
    rsi_overbought: int = 70
    bb_period: int = 20
    bb_std_dev: Decimal = field(default_factory=lambda: Decimal('2'))


@dataclass
class APIConfig:
    """API configuration parameters."""
    delta_api_key: str = ''
    delta_api_secret: str = ''
    delta_base_url: str = 'https://api.india.delta.exchange'
    delta_testnet_url: str = 'https://cdn-ind.testnet.deltaex.org'
    request_timeout: int = 30
    api_rate_limit: int = 100


@dataclass
class LoggingConfig:
    """Logging configuration parameters."""
    log_level: str = 'INFO'
    log_file: str = 'logs/trading_bot.log'
    log_max_size: int = 10485760  # 10MB
    log_backup_count: int = 5
    enable_console_logging: bool = True
    enable_structured_logging: bool = True
    log_dir: str = 'logs'


@dataclass
class NotificationConfig:
    """Notification configuration parameters."""
    enable_notifications: bool = True
    discord_webhook_url: str = ''
    slack_webhook_url: str = ''
    email_notifications: bool = False
    smtp_server: str = ''
    smtp_port: int = 587
    email_username: str = ''
    email_password: str = ''
    notification_email: str = ''


@dataclass
class SecurityConfig:
    """Security configuration parameters."""
    ip_whitelist: str = ''
    enable_ip_restriction: bool = False
    enable_encryption: bool = True
    session_timeout: int = 3600  # 1 hour


class Config:
    """
    Centralized configuration management system.
    
    This class handles loading configuration from multiple sources:
    1. Environment variables (highest priority)
    2. Configuration files (YAML/JSON)
    3. Default values (lowest priority)
    
    It also provides validation, type conversion, and secure handling
    of sensitive configuration data.
    """
    
    def __init__(self, config_file: Optional[str] = None, env_prefix: str = ''):
        """
        Initialize the configuration manager.
        
        Args:
            config_file: Path to configuration file (optional)
            env_prefix: Prefix for environment variables (optional)
        """
        self.logger = get_logger(__name__)
        self.env_prefix = env_prefix
        self.config_data: Dict[str, Any] = {}
        self.sensitive_keys = {
            'delta_api_key', 'delta_api_secret', 'email_password',
            'discord_webhook_url', 'slack_webhook_url'
        }
        
        # Load configuration from various sources
        self._load_default_config()
        
        if config_file:
            self._load_config_file(config_file)
        
        self._load_environment_variables()
        
        # Validate configuration
        self._validate_config()
        
        self.logger.info("Configuration loaded and validated successfully")
    
    def _load_default_config(self) -> None:
        """Load default configuration values."""
        try:
            self.config_data = {
                # Trading configuration
                'SYMBOL': 'BTCUSD',
                'PRODUCT_ID': 27,
                'TRADING_ENABLED': False,
                'PAPER_TRADING': True,
                'ENVIRONMENT': 'development',
                'EXECUTION_INTERVAL': 900,
                'STRATEGY': 'moving_average_crossover',
                
                # Risk management
                'MAX_POSITION_SIZE': 1000,
                'STOP_LOSS_PERCENTAGE': 2.0,
                'MAX_DRAWDOWN_PERCENTAGE': 10.0,
                'RISK_PER_TRADE_PERCENTAGE': 1.0,
                'MAX_LEVERAGE': 10,
                'MAX_DAILY_TRADES': 20,
                'MAX_OPEN_POSITIONS': 3,
                'EMERGENCY_STOP_DRAWDOWN': 20.0,
                'EMERGENCY_STOP_DAILY_LOSS': 5000,
                
                # Strategy parameters
                'SHORT_MA_PERIOD': 10,
                'LONG_MA_PERIOD': 30,
                'RSI_PERIOD': 14,
                'RSI_OVERSOLD': 30,
                'RSI_OVERBOUGHT': 70,
                'BB_PERIOD': 20,
                'BB_STD_DEV': 2.0,
                
                # API configuration
                'DELTA_API_KEY': '',
                'DELTA_API_SECRET': '',
                'DELTA_BASE_URL': 'https://api.india.delta.exchange',
                'DELTA_TESTNET_URL': 'https://cdn-ind.testnet.deltaex.org',
                'REQUEST_TIMEOUT': 30,
                'API_RATE_LIMIT': 100,
                
                # Logging
                'LOG_LEVEL': 'INFO',
                'LOG_FILE': 'logs/trading_bot.log',
                'LOG_MAX_SIZE': 10485760,
                'LOG_BACKUP_COUNT': 5,
                'ENABLE_CONSOLE_LOGGING': True,
                'ENABLE_STRUCTURED_LOGGING': True,
                'LOG_DIR': 'logs',
                
                # Notifications
                'ENABLE_NOTIFICATIONS': True,
                'DISCORD_WEBHOOK_URL': '',
                'SLACK_WEBHOOK_URL': '',
                'EMAIL_NOTIFICATIONS': False,
                'SMTP_SERVER': '',
                'SMTP_PORT': 587,
                'EMAIL_USERNAME': '',
                'EMAIL_PASSWORD': '',
                'NOTIFICATION_EMAIL': '',
                
                # Security
                'IP_WHITELIST': '',
                'ENABLE_IP_RESTRICTION': False,
                'ENABLE_ENCRYPTION': True,
                'SESSION_TIMEOUT': 3600,
                
                # Market hours
                'MARKET_HOURS_START': '09:00',
                'MARKET_HOURS_END': '17:00',
                'TIMEZONE': 'Asia/Kolkata',
                
                # Performance monitoring
                'ENABLE_PERFORMANCE_MONITORING': True,
                'MEMORY_LIMIT_MB': 512,
                'CPU_LIMIT_PERCENTAGE': 80,
                
                # Backtesting
                'BACKTEST_START_DATE': '2024-01-01',
                'BACKTEST_END_DATE': '2024-12-31',
                'BACKTEST_INITIAL_BALANCE': 10000,
                'BACKTEST_COMMISSION': 0.001,
                
                # Development
                'DEBUG_MODE': False,
                'MOCK_TRADING': False,
                'VERBOSE_LOGGING': False
            }
            
            self.logger.debug("Default configuration loaded")
            
        except Exception as e:
            self.logger.error(f"Error loading default configuration: {str(e)}")
            raise
    
    def _load_config_file(self, config_file: str) -> None:
        """
        Load configuration from file (YAML or JSON).
        
        Args:
            config_file: Path to configuration file
        """
        try:
            config_path = Path(config_file)
            
            if not config_path.exists():
                self.logger.warning(f"Configuration file not found: {config_file}")
                return
            
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.suffix.lower() in ['.yml', '.yaml']:
                    file_config = yaml.safe_load(f)
                elif config_path.suffix.lower() == '.json':
                    file_config = json.load(f)
                else:
                    self.logger.error(f"Unsupported configuration file format: {config_path.suffix}")
                    return
            
            if file_config:
                # Convert nested configuration to flat keys
                flat_config = self._flatten_config(file_config)
                
                # Update configuration with file values
                for key, value in flat_config.items():
                    self.config_data[key.upper()] = value
                
                self.logger.info(f"Configuration loaded from file: {config_file}")
            
        except Exception as e:
            self.logger.error(f"Error loading configuration file {config_file}: {str(e)}")
            raise
    
    def _load_environment_variables(self) -> None:
        """Load configuration from environment variables."""
        try:
            env_count = 0
            
            for key in self.config_data.keys():
                env_key = f"{self.env_prefix}{key}" if self.env_prefix else key
                env_value = os.getenv(env_key)
                
                if env_value is not None:
                    # Convert environment variable to appropriate type
                    converted_value = self._convert_env_value(env_value, key)
                    self.config_data[key] = converted_value
                    env_count += 1
            
            if env_count > 0:
                self.logger.info(f"Loaded {env_count} configuration values from environment variables")
            
        except Exception as e:
            self.logger.error(f"Error loading environment variables: {str(e)}")
            raise
    
    def _flatten_config(self, config: Dict[str, Any], prefix: str = '') -> Dict[str, Any]:
        """
        Flatten nested configuration dictionary.
        
        Args:
            config: Nested configuration dictionary
            prefix: Key prefix for flattening
            
        Returns:
            Flattened configuration dictionary
        """
        flat_config = {}
        
        for key, value in config.items():
            full_key = f"{prefix}_{key}" if prefix else key
            
            if isinstance(value, dict):
                flat_config.update(self._flatten_config(value, full_key))
            else:
                flat_config[full_key] = value
        
        return flat_config
    
    def _convert_env_value(self, value: str, key: str) -> Any:
        """
        Convert environment variable string to appropriate type.
        
        Args:
            value: Environment variable value
            key: Configuration key
            
        Returns:
            Converted value
        """
        try:
            # Boolean conversion
            if value.lower() in ['true', 'false']:
                return value.lower() == 'true'
            
            # Integer conversion for known integer keys
            integer_keys = {
                'PRODUCT_ID', 'EXECUTION_INTERVAL', 'SHORT_MA_PERIOD', 'LONG_MA_PERIOD',
                'RSI_PERIOD', 'RSI_OVERSOLD', 'RSI_OVERBOUGHT', 'BB_PERIOD',
                'LOG_MAX_SIZE', 'LOG_BACKUP_COUNT', 'REQUEST_TIMEOUT', 'API_RATE_LIMIT',
                'SMTP_PORT', 'SESSION_TIMEOUT', 'MAX_DAILY_TRADES', 'MAX_OPEN_POSITIONS',
                'MEMORY_LIMIT_MB', 'CPU_LIMIT_PERCENTAGE', 'BACKTEST_INITIAL_BALANCE'
            }
            
            if key in integer_keys:
                return int(value)
            
            # Float conversion for known float keys
            float_keys = {
                'MAX_POSITION_SIZE', 'STOP_LOSS_PERCENTAGE', 'MAX_DRAWDOWN_PERCENTAGE',
                'RISK_PER_TRADE_PERCENTAGE', 'MAX_LEVERAGE', 'BB_STD_DEV',
                'EMERGENCY_STOP_DRAWDOWN', 'EMERGENCY_STOP_DAILY_LOSS',
                'BACKTEST_COMMISSION'
            }
            
            if key in float_keys:
                return float(value)
            
            # Return as string for everything else
            return value
            
        except (ValueError, TypeError) as e:
            self.logger.warning(f"Failed to convert environment variable {key}={value}: {str(e)}")
            return value
    
    def _validate_config(self) -> None:
        """Validate configuration values."""
        try:
            validations = [
                # Trading validations
                (self.config_data.get('EXECUTION_INTERVAL', 0) > 0, 
                 "Execution interval must be positive"),
                (0 < self.config_data.get('RISK_PER_TRADE_PERCENTAGE', 0) <= 10, 
                 "Risk per trade must be between 0-10%"),
                (0 < self.config_data.get('MAX_DRAWDOWN_PERCENTAGE', 0) <= 50, 
                 "Max drawdown must be between 0-50%"),
                (self.config_data.get('MAX_LEVERAGE', 0) >= 1, 
                 "Max leverage must be >= 1"),
                
                # Strategy validations
                (self.config_data.get('SHORT_MA_PERIOD', 0) > 0, 
                 "Short MA period must be positive"),
                (self.config_data.get('LONG_MA_PERIOD', 0) > self.config_data.get('SHORT_MA_PERIOD', 0), 
                 "Long MA period must be greater than short MA period"),
                (0 < self.config_data.get('RSI_OVERSOLD', 0) < self.config_data.get('RSI_OVERBOUGHT', 100), 
                 "RSI oversold must be less than overbought"),
                
                # API validations
                (self.config_data.get('REQUEST_TIMEOUT', 0) > 0, 
                 "Request timeout must be positive"),
                (self.config_data.get('API_RATE_LIMIT', 0) > 0, 
                 "API rate limit must be positive"),
                
                # Logging validations
                (self.config_data.get('LOG_MAX_SIZE', 0) > 0, 
                 "Log max size must be positive"),
                (self.config_data.get('LOG_BACKUP_COUNT', 0) >= 0, 
                 "Log backup count must be non-negative"),
            ]
            
            for is_valid, error_msg in validations:
                if not is_valid:
                    raise ValueError(f"Configuration validation failed: {error_msg}")
            
            # Validate API credentials if trading is enabled
            if self.config_data.get('TRADING_ENABLED', False):
                if not self.config_data.get('DELTA_API_KEY') or not self.config_data.get('DELTA_API_SECRET'):
                    self.logger.warning("Trading enabled but API credentials not provided")
            
            self.logger.debug("Configuration validation passed")
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {str(e)}")
            raise
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        return self.config_data.get(key.upper(), default)
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value.
        
        Args:
            key: Configuration key
            value: Configuration value
        """
        self.config_data[key.upper()] = value
    
    def get_trading_config(self) -> TradingConfig:
        """
        Get trading-specific configuration.
        
        Returns:
            TradingConfig object
        """
        return TradingConfig(
            symbol=self.get('SYMBOL'),
            product_id=self.get('PRODUCT_ID'),
            trading_enabled=self.get('TRADING_ENABLED'),
            paper_trading=self.get('PAPER_TRADING'),
            execution_interval=self.get('EXECUTION_INTERVAL'),
            strategy=self.get('STRATEGY'),
            max_position_size=Decimal(str(self.get('MAX_POSITION_SIZE'))),
            stop_loss_percentage=Decimal(str(self.get('STOP_LOSS_PERCENTAGE'))),
            max_drawdown_percentage=Decimal(str(self.get('MAX_DRAWDOWN_PERCENTAGE'))),
            risk_per_trade_percentage=Decimal(str(self.get('RISK_PER_TRADE_PERCENTAGE'))),
            short_ma_period=self.get('SHORT_MA_PERIOD'),
            long_ma_period=self.get('LONG_MA_PERIOD'),
            rsi_period=self.get('RSI_PERIOD'),
            rsi_oversold=self.get('RSI_OVERSOLD'),
            rsi_overbought=self.get('RSI_OVERBOUGHT'),
            bb_period=self.get('BB_PERIOD'),
            bb_std_dev=Decimal(str(self.get('BB_STD_DEV')))
        )
    
    def get_api_config(self) -> APIConfig:
        """
        Get API-specific configuration.
        
        Returns:
            APIConfig object
        """
        return APIConfig(
            delta_api_key=self.get('DELTA_API_KEY'),
            delta_api_secret=self.get('DELTA_API_SECRET'),
            delta_base_url=self.get('DELTA_BASE_URL'),
            delta_testnet_url=self.get('DELTA_TESTNET_URL'),
            request_timeout=self.get('REQUEST_TIMEOUT'),
            api_rate_limit=self.get('API_RATE_LIMIT')
        )
    
    def get_logging_config(self) -> LoggingConfig:
        """
        Get logging-specific configuration.
        
        Returns:
            LoggingConfig object
        """
        return LoggingConfig(
            log_level=self.get('LOG_LEVEL'),
            log_file=self.get('LOG_FILE'),
            log_max_size=self.get('LOG_MAX_SIZE'),
            log_backup_count=self.get('LOG_BACKUP_COUNT'),
            enable_console_logging=self.get('ENABLE_CONSOLE_LOGGING'),
            enable_structured_logging=self.get('ENABLE_STRUCTURED_LOGGING'),
            log_dir=self.get('LOG_DIR')
        )
    
    def get_notification_config(self) -> NotificationConfig:
        """
        Get notification-specific configuration.
        
        Returns:
            NotificationConfig object
        """
        return NotificationConfig(
            enable_notifications=self.get('ENABLE_NOTIFICATIONS'),
            discord_webhook_url=self.get('DISCORD_WEBHOOK_URL'),
            slack_webhook_url=self.get('SLACK_WEBHOOK_URL'),
            email_notifications=self.get('EMAIL_NOTIFICATIONS'),
            smtp_server=self.get('SMTP_SERVER'),
            smtp_port=self.get('SMTP_PORT'),
            email_username=self.get('EMAIL_USERNAME'),
            email_password=self.get('EMAIL_PASSWORD'),
            notification_email=self.get('NOTIFICATION_EMAIL')
        )
    
    def get_security_config(self) -> SecurityConfig:
        """
        Get security-specific configuration.
        
        Returns:
            SecurityConfig object
        """
        return SecurityConfig(
            ip_whitelist=self.get('IP_WHITELIST'),
            enable_ip_restriction=self.get('ENABLE_IP_RESTRICTION'),
            enable_encryption=self.get('ENABLE_ENCRYPTION'),
            session_timeout=self.get('SESSION_TIMEOUT')
        )
    
    def to_dict(self, include_sensitive: bool = False) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Args:
            include_sensitive: Whether to include sensitive values
            
        Returns:
            Configuration dictionary
        """
        if include_sensitive:
            return self.config_data.copy()
        else:
            # Mask sensitive values
            masked_config = {}
            for key, value in self.config_data.items():
                if key.lower() in self.sensitive_keys and value:
                    masked_config[key] = '***MASKED***'
                else:
                    masked_config[key] = value
            return masked_config
    
    def save_to_file(self, file_path: str, include_sensitive: bool = False) -> None:
        """
        Save configuration to file.
        
        Args:
            file_path: Path to save configuration
            include_sensitive: Whether to include sensitive values
        """
        try:
            config_dict = self.to_dict(include_sensitive)
            
            file_path_obj = Path(file_path)
            file_path_obj.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path_obj, 'w', encoding='utf-8') as f:
                if file_path_obj.suffix.lower() in ['.yml', '.yaml']:
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
                else:
                    json.dump(config_dict, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Configuration saved to: {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving configuration to file: {str(e)}")
            raise
    
    def reload(self, config_file: Optional[str] = None) -> None:
        """
        Reload configuration from sources.
        
        Args:
            config_file: Optional new configuration file
        """
        try:
            self.logger.info("Reloading configuration")
            
            # Clear current configuration
            self.config_data.clear()
            
            # Reload from sources
            self._load_default_config()
            
            if config_file:
                self._load_config_file(config_file)
            
            self._load_environment_variables()
            self._validate_config()
            
            self.logger.info("Configuration reloaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error reloading configuration: {str(e)}")
            raise

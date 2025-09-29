"""
Comprehensive Logging System

This module provides a centralized logging system for the Delta Exchange trading bot
with support for multiple log levels, file rotation, structured logging, and
integration with monitoring systems.
"""

import os
import sys
import logging
import logging.handlers
import json
import traceback
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Union
from pathlib import Path
import threading
from contextlib import contextmanager


class StructuredFormatter(logging.Formatter):
    """
    Custom formatter that outputs structured JSON logs for better parsing
    and integration with log aggregation systems.
    """
    
    def __init__(self, include_extra: bool = True):
        """
        Initialize the structured formatter.
        
        Args:
            include_extra: Whether to include extra fields in log records
        """
        super().__init__()
        self.include_extra = include_extra
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as structured JSON.
        
        Args:
            record: Log record to format
            
        Returns:
            JSON formatted log string
        """
        try:
            # Base log structure
            log_entry = {
                'timestamp': datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
                'level': record.levelname,
                'logger': record.name,
                'message': record.getMessage(),
                'module': record.module,
                'function': record.funcName,
                'line': record.lineno,
                'thread': record.thread,
                'thread_name': record.threadName,
                'process': record.process
            }
            
            # Add exception information if present
            if record.exc_info:
                log_entry['exception'] = {
                    'type': record.exc_info[0].__name__ if record.exc_info[0] else None,
                    'message': str(record.exc_info[1]) if record.exc_info[1] else None,
                    'traceback': traceback.format_exception(*record.exc_info)
                }
            
            # Add extra fields if enabled
            if self.include_extra:
                extra_fields = {}
                for key, value in record.__dict__.items():
                    if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                                 'filename', 'module', 'lineno', 'funcName', 'created',
                                 'msecs', 'relativeCreated', 'thread', 'threadName',
                                 'processName', 'process', 'getMessage', 'exc_info',
                                 'exc_text', 'stack_info']:
                        try:
                            # Ensure the value is JSON serializable
                            json.dumps(value)
                            extra_fields[key] = value
                        except (TypeError, ValueError):
                            extra_fields[key] = str(value)
                
                if extra_fields:
                    log_entry['extra'] = extra_fields
            
            return json.dumps(log_entry, ensure_ascii=False)
            
        except Exception as e:
            # Fallback to simple format if JSON formatting fails
            return f"LOGGING_ERROR: {str(e)} | Original: {record.getMessage()}"


class TradingLogFilter(logging.Filter):
    """
    Custom filter for trading-specific log enhancement and filtering.
    """
    
    def __init__(self, bot_id: Optional[str] = None):
        """
        Initialize the trading log filter.
        
        Args:
            bot_id: Unique identifier for this bot instance
        """
        super().__init__()
        self.bot_id = bot_id or f"bot_{os.getpid()}"
    
    def filter(self, record: logging.LogRecord) -> bool:
        """
        Filter and enhance log records.
        
        Args:
            record: Log record to filter
            
        Returns:
            True if record should be logged, False otherwise
        """
        # Add bot identifier
        record.bot_id = self.bot_id
        
        # Add trading context if available
        if hasattr(record, 'trade_id'):
            record.context = 'trading'
        elif hasattr(record, 'analysis_id'):
            record.context = 'analysis'
        elif hasattr(record, 'api_call'):
            record.context = 'api'
        else:
            record.context = 'general'
        
        # Filter out sensitive information
        message = record.getMessage()
        
        # Mask API keys and secrets
        if 'api_key' in message.lower() or 'secret' in message.lower():
            # This is a basic implementation - in production, use more sophisticated masking
            record.msg = record.msg.replace(record.args[0] if record.args else '', '***MASKED***')
            record.args = ()
        
        return True


class LogManager:
    """
    Centralized log management system for the trading bot.
    
    This class handles log configuration, file rotation, and provides
    convenient logging methods with trading-specific enhancements.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the log manager.
        
        Args:
            config: Configuration dictionary for logging settings
        """
        self.config = config or {}
        self.loggers: Dict[str, logging.Logger] = {}
        self.handlers: Dict[str, logging.Handler] = {}
        
        # Default configuration
        self.log_level = self.config.get('LOG_LEVEL', 'INFO')
        self.log_dir = Path(self.config.get('LOG_DIR', 'logs'))
        self.log_file = self.config.get('LOG_FILE', 'trading_bot.log')
        self.max_file_size = self.config.get('LOG_MAX_SIZE', 10 * 1024 * 1024)  # 10MB
        self.backup_count = self.config.get('LOG_BACKUP_COUNT', 5)
        self.enable_console = self.config.get('ENABLE_CONSOLE_LOGGING', True)
        self.enable_structured = self.config.get('ENABLE_STRUCTURED_LOGGING', True)
        
        # Create log directory
        self.log_dir.mkdir(exist_ok=True)
        
        # Initialize logging
        self._setup_logging()
    
    def _setup_logging(self) -> None:
        """Set up the logging configuration."""
        try:
            # Configure root logger
            root_logger = logging.getLogger()
            root_logger.setLevel(getattr(logging, self.log_level.upper()))
            
            # Clear existing handlers
            root_logger.handlers.clear()
            
            # Create formatters
            if self.enable_structured:
                file_formatter = StructuredFormatter(include_extra=True)
                console_formatter = StructuredFormatter(include_extra=False)
            else:
                file_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s [%(filename)s:%(lineno)d]'
                console_format = '%(asctime)s - %(levelname)s - %(message)s'
                file_formatter = logging.Formatter(file_format)
                console_formatter = logging.Formatter(console_format)
            
            # Create file handler with rotation
            file_handler = logging.handlers.RotatingFileHandler(
                filename=self.log_dir / self.log_file,
                maxBytes=self.max_file_size,
                backupCount=self.backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(getattr(logging, self.log_level.upper()))
            file_handler.setFormatter(file_formatter)
            
            # Add trading filter
            trading_filter = TradingLogFilter()
            file_handler.addFilter(trading_filter)
            
            root_logger.addHandler(file_handler)
            self.handlers['file'] = file_handler
            
            # Create console handler if enabled
            if self.enable_console:
                console_handler = logging.StreamHandler(sys.stdout)
                console_handler.setLevel(logging.INFO)
                console_handler.setFormatter(console_formatter)
                console_handler.addFilter(trading_filter)
                
                root_logger.addHandler(console_handler)
                self.handlers['console'] = console_handler
            
            # Create separate handlers for different log types
            self._setup_specialized_handlers()
            
        except Exception as e:
            print(f"Failed to setup logging: {str(e)}")
            raise
    
    def _setup_specialized_handlers(self) -> None:
        """Set up specialized handlers for different types of logs."""
        try:
            # Trading activity log
            trading_handler = logging.handlers.RotatingFileHandler(
                filename=self.log_dir / 'trading_activity.log',
                maxBytes=self.max_file_size,
                backupCount=self.backup_count,
                encoding='utf-8'
            )
            trading_handler.setLevel(logging.INFO)
            
            if self.enable_structured:
                trading_handler.setFormatter(StructuredFormatter())
            else:
                trading_handler.setFormatter(logging.Formatter(
                    '%(asctime)s - %(levelname)s - %(message)s'
                ))
            
            self.handlers['trading'] = trading_handler
            
            # Error log
            error_handler = logging.handlers.RotatingFileHandler(
                filename=self.log_dir / 'errors.log',
                maxBytes=self.max_file_size,
                backupCount=self.backup_count,
                encoding='utf-8'
            )
            error_handler.setLevel(logging.ERROR)
            
            if self.enable_structured:
                error_handler.setFormatter(StructuredFormatter())
            else:
                error_handler.setFormatter(logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s [%(filename)s:%(lineno)d]'
                ))
            
            self.handlers['error'] = error_handler
            
            # Performance log
            performance_handler = logging.handlers.RotatingFileHandler(
                filename=self.log_dir / 'performance.log',
                maxBytes=self.max_file_size,
                backupCount=self.backup_count,
                encoding='utf-8'
            )
            performance_handler.setLevel(logging.INFO)
            
            if self.enable_structured:
                performance_handler.setFormatter(StructuredFormatter())
            else:
                performance_handler.setFormatter(logging.Formatter(
                    '%(asctime)s - %(message)s'
                ))
            
            self.handlers['performance'] = performance_handler
            
        except Exception as e:
            print(f"Failed to setup specialized handlers: {str(e)}")
    
    def get_logger(self, name: str, specialized: Optional[str] = None) -> logging.Logger:
        """
        Get or create a logger with the specified name.
        
        Args:
            name: Logger name
            specialized: Type of specialized logger ('trading', 'error', 'performance')
            
        Returns:
            Configured logger instance
        """
        logger_key = f"{name}_{specialized}" if specialized else name
        
        if logger_key not in self.loggers:
            logger = logging.getLogger(name)
            
            # Add specialized handler if requested
            if specialized and specialized in self.handlers:
                logger.addHandler(self.handlers[specialized])
            
            self.loggers[logger_key] = logger
        
        return self.loggers[logger_key]
    
    def log_trade(self, trade_data: Dict[str, Any], level: str = 'INFO') -> None:
        """
        Log trading activity with structured data.
        
        Args:
            trade_data: Dictionary containing trade information
            level: Log level
        """
        try:
            trading_logger = self.get_logger('trading', 'trading')
            log_level = getattr(logging, level.upper())
            
            # Enhance trade data with metadata
            enhanced_data = {
                'event_type': 'trade',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                **trade_data
            }
            
            trading_logger.log(log_level, "Trade executed", extra=enhanced_data)
            
        except Exception as e:
            self.get_logger('error', 'error').error(f"Failed to log trade: {str(e)}")
    
    def log_performance(self, metrics: Dict[str, Any]) -> None:
        """
        Log performance metrics.
        
        Args:
            metrics: Dictionary containing performance metrics
        """
        try:
            performance_logger = self.get_logger('performance', 'performance')
            
            enhanced_metrics = {
                'event_type': 'performance',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                **metrics
            }
            
            performance_logger.info("Performance metrics", extra=enhanced_metrics)
            
        except Exception as e:
            self.get_logger('error', 'error').error(f"Failed to log performance: {str(e)}")
    
    def log_api_call(self, endpoint: str, method: str, status_code: int, 
                    response_time: float, error: Optional[str] = None) -> None:
        """
        Log API call information.
        
        Args:
            endpoint: API endpoint called
            method: HTTP method
            status_code: Response status code
            response_time: Response time in milliseconds
            error: Error message if any
        """
        try:
            api_logger = self.get_logger('api')
            
            api_data = {
                'event_type': 'api_call',
                'endpoint': endpoint,
                'method': method,
                'status_code': status_code,
                'response_time_ms': response_time,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            if error:
                api_data['error'] = error
                api_logger.error("API call failed", extra=api_data)
            else:
                api_logger.info("API call completed", extra=api_data)
                
        except Exception as e:
            self.get_logger('error', 'error').error(f"Failed to log API call: {str(e)}")
    
    @contextmanager
    def log_execution_time(self, operation_name: str, logger_name: str = 'performance'):
        """
        Context manager to log execution time of operations.
        
        Args:
            operation_name: Name of the operation being timed
            logger_name: Logger to use for timing logs
        """
        start_time = datetime.now()
        logger = self.get_logger(logger_name)
        
        try:
            logger.debug(f"Starting operation: {operation_name}")
            yield
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Operation failed: {operation_name}", extra={
                'operation': operation_name,
                'execution_time_seconds': execution_time,
                'error': str(e)
            })
            raise
        else:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Operation completed: {operation_name}", extra={
                'operation': operation_name,
                'execution_time_seconds': execution_time
            })
    
    def rotate_logs(self) -> None:
        """Manually trigger log rotation for all handlers."""
        try:
            for handler_name, handler in self.handlers.items():
                if isinstance(handler, logging.handlers.RotatingFileHandler):
                    handler.doRollover()
                    self.get_logger('system').info(f"Log rotation completed for {handler_name}")
        except Exception as e:
            self.get_logger('error', 'error').error(f"Failed to rotate logs: {str(e)}")
    
    def get_log_stats(self) -> Dict[str, Any]:
        """
        Get statistics about log files.
        
        Returns:
            Dictionary containing log file statistics
        """
        try:
            stats = {}
            
            for log_file in self.log_dir.glob('*.log'):
                file_stats = log_file.stat()
                stats[log_file.name] = {
                    'size_bytes': file_stats.st_size,
                    'size_mb': round(file_stats.st_size / (1024 * 1024), 2),
                    'modified': datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
                    'lines': self._count_lines(log_file)
                }
            
            return stats
            
        except Exception as e:
            self.get_logger('error', 'error').error(f"Failed to get log stats: {str(e)}")
            return {}
    
    def _count_lines(self, file_path: Path) -> int:
        """Count lines in a log file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return sum(1 for _ in f)
        except Exception:
            return 0
    
    def cleanup_old_logs(self, days_to_keep: int = 30) -> None:
        """
        Clean up log files older than specified days.
        
        Args:
            days_to_keep: Number of days to keep log files
        """
        try:
            cutoff_time = datetime.now().timestamp() - (days_to_keep * 24 * 3600)
            
            for log_file in self.log_dir.glob('*.log*'):
                if log_file.stat().st_mtime < cutoff_time:
                    log_file.unlink()
                    self.get_logger('system').info(f"Deleted old log file: {log_file.name}")
                    
        except Exception as e:
            self.get_logger('error', 'error').error(f"Failed to cleanup old logs: {str(e)}")


# Global log manager instance
_log_manager: Optional[LogManager] = None
_log_manager_lock = threading.Lock()


def initialize_logging(config: Optional[Dict[str, Any]] = None) -> LogManager:
    """
    Initialize the global log manager.
    
    Args:
        config: Configuration dictionary for logging
        
    Returns:
        Initialized LogManager instance
    """
    global _log_manager
    
    with _log_manager_lock:
        if _log_manager is None:
            _log_manager = LogManager(config)
    
    return _log_manager


def get_logger(name: str, specialized: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name
        specialized: Type of specialized logger
        
    Returns:
        Logger instance
    """
    global _log_manager
    
    if _log_manager is None:
        initialize_logging()
    
    return _log_manager.get_logger(name, specialized)


def log_trade(trade_data: Dict[str, Any], level: str = 'INFO') -> None:
    """
    Convenience function to log trading activity.
    
    Args:
        trade_data: Trade information
        level: Log level
    """
    global _log_manager
    
    if _log_manager is None:
        initialize_logging()
    
    _log_manager.log_trade(trade_data, level)


def log_performance(metrics: Dict[str, Any]) -> None:
    """
    Convenience function to log performance metrics.
    
    Args:
        metrics: Performance metrics
    """
    global _log_manager
    
    if _log_manager is None:
        initialize_logging()
    
    _log_manager.log_performance(metrics)

"""
Comprehensive Test Suite for Delta Trading Bot

This module contains comprehensive tests for all components of the trading system
including unit tests, integration tests, and system tests.
"""

import unittest
import asyncio
import json
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from decimal import Decimal
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

import sys
sys.path.append('src')

from utils.config import Config
from utils.logger import initialize_logging, get_logger
from api.delta_client import DeltaExchangeClient
from api.auth import AuthenticationManager
from data.market_data import MarketDataCollector
from data.indicators import TechnicalIndicators
from data.analysis import TechnicalAnalyzer
from trading.risk_manager import RiskManager, RiskLevel, RiskAction
from trading.bot import DeltaTradingBot


class TestConfig(unittest.TestCase):
    """Test configuration management system."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_config = {
            'SYMBOL': 'BTCUSD',
            'TRADING_ENABLED': False,
            'MAX_POSITION_SIZE': 1000,
            'LOG_LEVEL': 'DEBUG'
        }
    
    def test_config_initialization(self):
        """Test configuration initialization."""
        config = Config()
        
        # Test default values
        self.assertEqual(config.get('SYMBOL'), 'BTCUSD')
        self.assertEqual(config.get('TRADING_ENABLED'), False)
        self.assertIsInstance(config.get('MAX_POSITION_SIZE'), (int, float))
    
    def test_config_environment_override(self):
        """Test environment variable override."""
        with patch.dict(os.environ, {'SYMBOL': 'ETHUSD', 'TRADING_ENABLED': 'true'}):
            config = Config()
            self.assertEqual(config.get('SYMBOL'), 'ETHUSD')
            self.assertEqual(config.get('TRADING_ENABLED'), True)
    
    def test_config_file_loading(self):
        """Test configuration file loading."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.test_config, f)
            config_file = f.name
        
        try:
            config = Config(config_file=config_file)
            self.assertEqual(config.get('SYMBOL'), 'BTCUSD')
            self.assertEqual(config.get('MAX_POSITION_SIZE'), 1000)
        finally:
            os.unlink(config_file)
    
    def test_config_validation(self):
        """Test configuration validation."""
        config = Config()
        
        # Test invalid configuration
        config.set('RISK_PER_TRADE_PERCENTAGE', 15)  # Invalid: > 10%
        
        with self.assertRaises(ValueError):
            config._validate_config()


class TestAuthentication(unittest.TestCase):
    """Test authentication and security features."""
    
    def setUp(self):
        """Set up test environment."""
        self.auth_manager = AuthenticationManager()
    
    def test_api_credentials_validation(self):
        """Test API credentials validation."""
        # Valid credentials
        self.assertTrue(self.auth_manager.validate_api_credentials(
            'valid_api_key_12345', 'valid_secret_key_abcdef123456'
        ))
        
        # Invalid credentials
        self.assertFalse(self.auth_manager.validate_api_credentials('', ''))
        self.assertFalse(self.auth_manager.validate_api_credentials('short', 'short'))
        self.assertFalse(self.auth_manager.validate_api_credentials('test_key', 'test_secret'))
    
    def test_signature_generation(self):
        """Test HMAC signature generation."""
        signature = self.auth_manager.generate_signature(
            api_secret='test_secret',
            method='GET',
            timestamp='1234567890',
            path='/api/v1/test',
            query_string='param=value',
            payload=''
        )
        
        self.assertIsInstance(signature, str)
        self.assertGreater(len(signature), 0)
    
    def test_timestamp_validation(self):
        """Test timestamp validation."""
        current_timestamp = str(int(datetime.now().timestamp()))
        old_timestamp = str(int((datetime.now() - timedelta(minutes=10)).timestamp()))
        future_timestamp = str(int((datetime.now() + timedelta(minutes=1)).timestamp()))
        
        self.assertTrue(self.auth_manager.validate_timestamp(current_timestamp))
        self.assertFalse(self.auth_manager.validate_timestamp(old_timestamp))
        self.assertFalse(self.auth_manager.validate_timestamp(future_timestamp))
    
    def test_encryption_decryption(self):
        """Test data encryption and decryption."""
        test_data = "sensitive_api_key_12345"
        
        encrypted = self.auth_manager.encrypt_sensitive_data(test_data)
        decrypted = self.auth_manager.decrypt_sensitive_data(encrypted)
        
        self.assertEqual(test_data, decrypted)
        self.assertNotEqual(test_data, encrypted)


class TestTechnicalIndicators(unittest.TestCase):
    """Test technical indicators calculations."""
    
    def setUp(self):
        """Set up test environment."""
        self.indicators = TechnicalIndicators()
        
        # Create sample price data
        np.random.seed(42)  # For reproducible tests
        self.prices = pd.Series([100 + np.random.randn() * 2 for _ in range(100)])
        self.high_prices = self.prices + np.random.rand(100) * 2
        self.low_prices = self.prices - np.random.rand(100) * 2
        self.volume = pd.Series([1000 + np.random.rand() * 500 for _ in range(100)])
    
    def test_sma_calculation(self):
        """Test Simple Moving Average calculation."""
        sma = self.indicators.sma(self.prices, 20)
        
        self.assertEqual(len(sma), len(self.prices))
        self.assertTrue(pd.isna(sma.iloc[:19]).all())  # First 19 should be NaN
        self.assertFalse(pd.isna(sma.iloc[19:]).any())  # Rest should have values
    
    def test_ema_calculation(self):
        """Test Exponential Moving Average calculation."""
        ema = self.indicators.ema(self.prices, 20)
        
        self.assertEqual(len(ema), len(self.prices))
        self.assertFalse(pd.isna(ema.iloc[1:]).any())  # Should have values after first
    
    def test_rsi_calculation(self):
        """Test RSI calculation."""
        rsi = self.indicators.rsi(self.prices, 14)
        
        self.assertEqual(len(rsi), len(self.prices))
        # RSI should be between 0 and 100
        valid_rsi = rsi.dropna()
        self.assertTrue((valid_rsi >= 0).all())
        self.assertTrue((valid_rsi <= 100).all())
    
    def test_macd_calculation(self):
        """Test MACD calculation."""
        macd_data = self.indicators.macd(self.prices)
        
        self.assertIn('macd', macd_data)
        self.assertIn('signal', macd_data)
        self.assertIn('histogram', macd_data)
        
        # All series should have same length
        self.assertEqual(len(macd_data['macd']), len(self.prices))
        self.assertEqual(len(macd_data['signal']), len(self.prices))
        self.assertEqual(len(macd_data['histogram']), len(self.prices))
    
    def test_bollinger_bands_calculation(self):
        """Test Bollinger Bands calculation."""
        bb_data = self.indicators.bollinger_bands(self.prices)
        
        self.assertIn('upper', bb_data)
        self.assertIn('middle', bb_data)
        self.assertIn('lower', bb_data)
        
        # Upper should be greater than middle, middle greater than lower
        valid_data = pd.DataFrame(bb_data).dropna()
        self.assertTrue((valid_data['upper'] >= valid_data['middle']).all())
        self.assertTrue((valid_data['middle'] >= valid_data['lower']).all())
    
    def test_atr_calculation(self):
        """Test Average True Range calculation."""
        atr = self.indicators.atr(self.high_prices, self.low_prices, self.prices)
        
        self.assertEqual(len(atr), len(self.prices))
        # ATR should be positive
        valid_atr = atr.dropna()
        self.assertTrue((valid_atr >= 0).all())
    
    def test_stochastic_calculation(self):
        """Test Stochastic Oscillator calculation."""
        stoch_data = self.indicators.stochastic(self.high_prices, self.low_prices, self.prices)
        
        self.assertIn('k_percent', stoch_data)
        self.assertIn('d_percent', stoch_data)
        
        # Stochastic should be between 0 and 100
        valid_k = stoch_data['k_percent'].dropna()
        valid_d = stoch_data['d_percent'].dropna()
        
        self.assertTrue((valid_k >= 0).all() and (valid_k <= 100).all())
        self.assertTrue((valid_d >= 0).all() and (valid_d <= 100).all())


class TestMarketDataCollector(unittest.TestCase):
    """Test market data collection and processing."""
    
    def setUp(self):
        """Set up test environment."""
        self.config = Config()
        self.mock_client = Mock()
        self.collector = MarketDataCollector(self.config, self.mock_client)
    
    def test_candle_validation(self):
        """Test candle data validation."""
        # Valid candle
        valid_candle = {
            'timestamp': 1234567890,
            'open': 100.0,
            'high': 105.0,
            'low': 95.0,
            'close': 102.0,
            'volume': 1000.0
        }
        self.assertTrue(self.collector._validate_candle(valid_candle))
        
        # Invalid candle (high < low)
        invalid_candle = {
            'timestamp': 1234567890,
            'open': 100.0,
            'high': 95.0,  # High less than low
            'low': 105.0,
            'close': 102.0,
            'volume': 1000.0
        }
        self.assertFalse(self.collector._validate_candle(invalid_candle))
        
        # Missing fields
        incomplete_candle = {
            'timestamp': 1234567890,
            'open': 100.0,
            # Missing other required fields
        }
        self.assertFalse(self.collector._validate_candle(incomplete_candle))
    
    def test_timeframe_conversion(self):
        """Test timeframe to seconds conversion."""
        self.assertEqual(self.collector._timeframe_to_seconds('1m'), 60)
        self.assertEqual(self.collector._timeframe_to_seconds('5m'), 300)
        self.assertEqual(self.collector._timeframe_to_seconds('1h'), 3600)
        self.assertEqual(self.collector._timeframe_to_seconds('1d'), 86400)
    
    def test_spread_calculation(self):
        """Test spread metrics calculation."""
        mock_orderbook = {
            'success': True,
            'bids': [{'price': '100.0', 'size': '10.0'}],
            'asks': [{'price': '101.0', 'size': '15.0'}]
        }
        
        spread_metrics = self.collector._calculate_spread_metrics(mock_orderbook)
        
        self.assertEqual(spread_metrics['best_bid'], 100.0)
        self.assertEqual(spread_metrics['best_ask'], 101.0)
        self.assertEqual(spread_metrics['spread'], 1.0)
        self.assertEqual(spread_metrics['mid_price'], 100.5)


class TestRiskManager(unittest.TestCase):
    """Test risk management system."""
    
    def setUp(self):
        """Set up test environment."""
        self.config = Config()
        self.mock_client = Mock()
        self.risk_manager = RiskManager(self.config, self.mock_client)
    
    def test_risk_settings_validation(self):
        """Test risk settings validation."""
        self.assertTrue(self.risk_manager.validate_settings())
        
        # Test invalid settings
        self.risk_manager.max_drawdown_pct = Decimal('60')  # > 50%
        self.assertFalse(self.risk_manager.validate_settings())
    
    def test_position_size_calculation(self):
        """Test position size calculation."""
        signal = {
            'confidence': 0.8,
            'stop_loss_percentage': 2.0,
            'volatility': {'realized_volatility': 20}
        }
        
        account_balance = Decimal('10000')
        current_price = Decimal('50000')
        
        position_size = self.risk_manager.calculate_position_size(
            signal, account_balance, current_price
        )
        
        self.assertIsInstance(position_size, Decimal)
        self.assertGreaterEqual(position_size, Decimal('0'))
        self.assertLessEqual(position_size, self.risk_manager.max_position_size)
    
    def test_risk_assessment(self):
        """Test comprehensive risk assessment."""
        current_position = {'size': '100'}
        open_orders = [{'size': '50'}, {'size': '25'}]
        performance_metrics = {
            'total_pnl': 500,
            'max_drawdown': -5,
            'daily_pnl': 100
        }
        market_conditions = {}
        
        assessment = self.risk_manager.assess_risk(
            current_position, open_orders, performance_metrics, market_conditions
        )
        
        self.assertIn('can_trade', assessment)
        self.assertIn('risk_level', assessment)
        self.assertIn('recommended_action', assessment)
        self.assertIsInstance(assessment['can_trade'], bool)
    
    def test_emergency_conditions(self):
        """Test emergency stop conditions."""
        # Create metrics that should trigger emergency stop
        from trading.risk_manager import RiskMetrics
        
        emergency_metrics = RiskMetrics(
            current_exposure=Decimal('1000'),
            max_exposure=Decimal('1000'),
            current_drawdown=Decimal('25'),  # > emergency threshold
            max_drawdown=Decimal('10'),
            position_count=1,
            max_positions=3,
            leverage=Decimal('1'),
            max_leverage=Decimal('10'),
            daily_pnl=Decimal('-6000'),  # > emergency daily loss
            risk_level=RiskLevel.LOW,
            recommended_action=RiskAction.ALLOW
        )
        
        emergency_triggered = self.risk_manager._check_emergency_conditions(emergency_metrics)
        self.assertTrue(emergency_triggered)


class TestTechnicalAnalyzer(unittest.TestCase):
    """Test technical analysis system."""
    
    def setUp(self):
        """Set up test environment."""
        self.config = Config()
        self.analyzer = TechnicalAnalyzer(self.config)
        
        # Create sample historical data
        self.historical_data = []
        base_price = 50000
        for i in range(100):
            price_change = np.random.randn() * 100
            price = base_price + price_change
            
            candle = {
                'timestamp': 1234567890 + i * 3600,
                'open': price - 50,
                'high': price + 100,
                'low': price - 100,
                'close': price,
                'volume': 1000 + np.random.rand() * 500,
                'typical_price': price
            }
            self.historical_data.append(candle)
    
    def test_indicators_calculation(self):
        """Test technical indicators calculation."""
        indicators = self.analyzer.calculate_indicators(self.historical_data)
        
        # Should return a dictionary with indicator values
        self.assertIsInstance(indicators, dict)
        
        # Check for key indicators
        expected_indicators = ['close', 'sma_20', 'rsi', 'macd']
        for indicator in expected_indicators:
            if indicator in indicators:
                self.assertIsInstance(indicators[indicator], (int, float))
    
    def test_pattern_detection(self):
        """Test pattern detection."""
        patterns = self.analyzer.detect_patterns(self.historical_data)
        
        self.assertIsInstance(patterns, dict)
        
        # Should detect some patterns
        if patterns:
            for pattern_name, pattern_data in patterns.items():
                self.assertIsInstance(pattern_name, str)
    
    def test_volatility_calculation(self):
        """Test volatility calculation."""
        volatility = self.analyzer.calculate_volatility(self.historical_data)
        
        self.assertIsInstance(volatility, dict)
        
        if volatility:
            for vol_metric, value in volatility.items():
                self.assertIsInstance(value, (int, float))
                self.assertGreaterEqual(value, 0)  # Volatility should be non-negative
    
    def test_momentum_calculation(self):
        """Test momentum calculation."""
        momentum = self.analyzer.calculate_momentum(self.historical_data)
        
        self.assertIsInstance(momentum, dict)
        
        if momentum:
            for momentum_metric, value in momentum.items():
                self.assertIsInstance(value, (int, float))


class TestDeltaClient(unittest.TestCase):
    """Test Delta Exchange API client."""
    
    def setUp(self):
        """Set up test environment."""
        self.config = Config()
        
        # Mock the actual Delta REST client
        with patch('src.api.delta_client.DeltaRestClient'):
            self.client = DeltaExchangeClient(self.config)
    
    def test_client_initialization(self):
        """Test client initialization."""
        self.assertIsNotNone(self.client)
        self.assertEqual(self.client.symbol, 'BTCUSD')
        self.assertEqual(self.client.product_id, 27)
    
    @patch('src.api.delta_client.DeltaRestClient')
    def test_get_current_price(self, mock_client_class):
        """Test current price retrieval."""
        mock_client = Mock()
        mock_client.get_ticker.return_value = {
            'success': True,
            'result': {'close': '50000.0'}
        }
        mock_client_class.return_value = mock_client
        
        client = DeltaExchangeClient(self.config)
        price = client.get_current_price()
        
        self.assertEqual(price, Decimal('50000.0'))
    
    @patch('src.api.delta_client.DeltaRestClient')
    def test_place_market_order(self, mock_client_class):
        """Test market order placement."""
        mock_client = Mock()
        mock_client.place_order.return_value = {
            'success': True,
            'result': {'id': 'order_123'}
        }
        mock_client_class.return_value = mock_client
        
        client = DeltaExchangeClient(self.config)
        result = client.place_market_order('buy', 100)
        
        self.assertTrue(result['success'])
        self.assertEqual(result['order_id'], 'order_123')


class TestTradingBot(unittest.TestCase):
    """Test main trading bot functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.config = Config()
        
        # Mock all external dependencies
        with patch('src.trading.bot.DeltaExchangeClient'), \
             patch('src.trading.bot.MarketDataCollector'), \
             patch('src.trading.bot.TechnicalAnalyzer'), \
             patch('src.trading.bot.StrategyManager'), \
             patch('src.trading.bot.RiskManager'):
            
            self.bot = DeltaTradingBot(self.config)
    
    def test_bot_initialization(self):
        """Test bot initialization."""
        self.assertIsNotNone(self.bot)
        self.assertFalse(self.bot.is_running)
        self.assertEqual(self.bot.execution_count, 0)
    
    def test_bot_status(self):
        """Test bot status reporting."""
        status = self.bot.get_status()
        
        self.assertIsInstance(status, dict)
        self.assertIn('is_running', status)
        self.assertIn('execution_count', status)
        self.assertIn('performance_metrics', status)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system."""
    
    def setUp(self):
        """Set up integration test environment."""
        # Initialize logging for tests
        initialize_logging({'LOG_LEVEL': 'DEBUG'})
        self.logger = get_logger(__name__)
        
        # Create test configuration
        self.config = Config()
        self.config.set('TRADING_ENABLED', False)  # Ensure no real trading
        self.config.set('PAPER_TRADING', True)
    
    def test_full_system_initialization(self):
        """Test full system initialization without external dependencies."""
        try:
            # Test configuration
            config = Config()
            self.assertIsNotNone(config)
            
            # Test logging
            logger = get_logger('test')
            logger.info("Test log message")
            
            # Test technical indicators
            indicators = TechnicalIndicators()
            self.assertIsNotNone(indicators)
            
            # Test authentication
            auth = AuthenticationManager()
            self.assertIsNotNone(auth)
            
            self.logger.info("Full system initialization test passed")
            
        except Exception as e:
            self.fail(f"System initialization failed: {str(e)}")
    
    def test_data_flow(self):
        """Test data flow through the system."""
        try:
            # Create sample market data
            sample_data = []
            for i in range(50):
                candle = {
                    'timestamp': 1234567890 + i * 3600,
                    'open': 50000 + np.random.randn() * 100,
                    'high': 50100 + np.random.randn() * 100,
                    'low': 49900 + np.random.randn() * 100,
                    'close': 50000 + np.random.randn() * 100,
                    'volume': 1000 + np.random.rand() * 500
                }
                sample_data.append(candle)
            
            # Test technical analysis
            analyzer = TechnicalAnalyzer(self.config)
            indicators = analyzer.calculate_indicators(sample_data)
            
            self.assertIsInstance(indicators, dict)
            
            # Test pattern detection
            patterns = analyzer.detect_patterns(sample_data)
            self.assertIsInstance(patterns, dict)
            
            self.logger.info("Data flow test passed")
            
        except Exception as e:
            self.fail(f"Data flow test failed: {str(e)}")


class TestPerformance(unittest.TestCase):
    """Performance tests for the trading system."""
    
    def test_indicator_calculation_performance(self):
        """Test performance of indicator calculations."""
        indicators = TechnicalIndicators()
        
        # Create large dataset
        large_dataset = pd.Series([100 + np.random.randn() for _ in range(10000)])
        
        import time
        start_time = time.time()
        
        # Calculate multiple indicators
        sma = indicators.sma(large_dataset, 20)
        ema = indicators.ema(large_dataset, 20)
        rsi = indicators.rsi(large_dataset, 14)
        
        end_time = time.time()
        calculation_time = end_time - start_time
        
        # Should complete within reasonable time (< 1 second for 10k data points)
        self.assertLess(calculation_time, 1.0)
        
        # Results should have correct length
        self.assertEqual(len(sma), len(large_dataset))
        self.assertEqual(len(ema), len(large_dataset))
        self.assertEqual(len(rsi), len(large_dataset))


def run_all_tests():
    """Run all tests and return results."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestConfig,
        TestAuthentication,
        TestTechnicalIndicators,
        TestMarketDataCollector,
        TestRiskManager,
        TestTechnicalAnalyzer,
        TestDeltaClient,
        TestTradingBot,
        TestIntegration,
        TestPerformance
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result


if __name__ == '__main__':
    # Initialize logging for tests
    initialize_logging({'LOG_LEVEL': 'INFO'})
    
    # Run all tests
    test_result = run_all_tests()
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Test Summary:")
    print(f"Tests run: {test_result.testsRun}")
    print(f"Failures: {len(test_result.failures)}")
    print(f"Errors: {len(test_result.errors)}")
    print(f"Success rate: {((test_result.testsRun - len(test_result.failures) - len(test_result.errors)) / test_result.testsRun * 100):.1f}%")
    print(f"{'='*50}")
    
    # Exit with appropriate code
    exit_code = 0 if test_result.wasSuccessful() else 1
    exit(exit_code)

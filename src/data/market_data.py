"""
Market Data Collection Module

This module handles the collection, processing, and storage of market data
from Delta Exchange for technical analysis and trading decisions.
"""

import time
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from decimal import Decimal

from ..utils.logger import get_logger
from ..utils.config import Config


class MarketDataCollector:
    """
    Collects and manages market data from Delta Exchange.
    
    This class handles real-time price data, historical data collection,
    orderbook monitoring, and data preprocessing for technical analysis.
    """
    
    def __init__(self, config: Config, delta_client):
        """
        Initialize the market data collector.
        
        Args:
            config: Configuration object
            delta_client: Delta Exchange API client
        """
        self.config = config
        self.delta_client = delta_client
        self.logger = get_logger(__name__)
        
        # Data storage
        self.price_history = []
        self.orderbook_history = []
        self.volume_history = []
        
        # Configuration
        self.symbol = config.get('SYMBOL', 'BTCUSD')
        self.product_id = config.get('PRODUCT_ID', 27)
        self.max_history_length = config.get('MAX_HISTORY_LENGTH', 1000)
        
        self.logger.info(f"Market data collector initialized for {self.symbol}")
    
    async def get_current_market_data(self) -> Dict[str, Any]:
        """
        Get comprehensive current market data.
        
        Returns:
            Dictionary containing current price, orderbook, and market statistics
        """
        try:
            self.logger.debug("Collecting current market data")
            
            # Get current price
            current_price = self.delta_client.get_current_price()
            
            # Get orderbook
            orderbook = self.delta_client.get_orderbook(depth=20)
            
            # Calculate spread and market depth
            spread_info = self._calculate_spread_metrics(orderbook)
            
            # Get recent price movement
            price_change = self._calculate_price_change()
            
            market_data = {
                'timestamp': time.time(),
                'price': float(current_price),
                'orderbook': orderbook,
                'spread': spread_info,
                'price_change': price_change,
                'symbol': self.symbol
            }
            
            # Store in history
            self._store_price_data(market_data)
            
            self.logger.debug(f"Current market data collected - Price: {current_price}")
            return market_data
            
        except Exception as e:
            self.logger.error(f"Error collecting current market data: {str(e)}")
            raise
    
    async def get_historical_data(self, symbol: str, timeframe: str = '1h', 
                                limit: int = 200) -> List[Dict[str, Any]]:
        """
        Get historical OHLCV data for technical analysis.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe (1m, 5m, 15m, 1h, 4h, 1d)
            limit: Number of candles to retrieve
            
        Returns:
            List of OHLCV candles
        """
        try:
            self.logger.debug(f"Fetching historical data for {symbol} - {timeframe} - {limit} candles")
            
            # For this implementation, we'll simulate historical data
            # In a real implementation, you would call Delta Exchange's historical data API
            historical_data = await self._fetch_historical_candles(symbol, timeframe, limit)
            
            # Process and validate the data
            processed_data = self._process_historical_data(historical_data)
            
            self.logger.info(f"Retrieved {len(processed_data)} historical candles")
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Error fetching historical data: {str(e)}")
            return []
    
    async def _fetch_historical_candles(self, symbol: str, timeframe: str, 
                                      limit: int) -> List[Dict[str, Any]]:
        """
        Fetch historical candle data from Delta Exchange.
        
        Note: This is a placeholder implementation. Delta Exchange may not have
        a direct historical candles API, so you might need to use alternative
        data sources or build candles from tick data.
        """
        try:
            # Since Delta Exchange doesn't have a direct historical candles API,
            # we'll create a simulation based on current price
            current_price = float(self.delta_client.get_current_price())
            
            # Generate simulated historical data for demonstration
            candles = []
            base_time = int(time.time())
            
            # Convert timeframe to seconds
            timeframe_seconds = self._timeframe_to_seconds(timeframe)
            
            for i in range(limit):
                # Calculate timestamp for this candle
                candle_time = base_time - (i * timeframe_seconds)
                
                # Generate realistic price movement (random walk)
                price_change = np.random.normal(0, current_price * 0.001)  # 0.1% volatility
                price = current_price + (price_change * (limit - i))
                
                # Generate OHLC data
                volatility = abs(price_change) * 2
                high = price + volatility
                low = price - volatility
                open_price = price + np.random.uniform(-volatility/2, volatility/2)
                close_price = price + np.random.uniform(-volatility/2, volatility/2)
                
                # Generate volume
                volume = np.random.uniform(100, 1000)
                
                candle = {
                    'timestamp': candle_time,
                    'open': round(open_price, 2),
                    'high': round(high, 2),
                    'low': round(low, 2),
                    'close': round(close_price, 2),
                    'volume': round(volume, 2)
                }
                
                candles.append(candle)
            
            # Reverse to get chronological order
            candles.reverse()
            
            return candles
            
        except Exception as e:
            self.logger.error(f"Error fetching historical candles: {str(e)}")
            return []
    
    def _timeframe_to_seconds(self, timeframe: str) -> int:
        """Convert timeframe string to seconds."""
        timeframe_map = {
            '1m': 60,
            '5m': 300,
            '15m': 900,
            '30m': 1800,
            '1h': 3600,
            '4h': 14400,
            '1d': 86400
        }
        return timeframe_map.get(timeframe, 3600)  # Default to 1 hour
    
    def _process_historical_data(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process and validate historical data.
        
        Args:
            raw_data: Raw historical data from API
            
        Returns:
            Processed and validated data
        """
        try:
            processed_data = []
            
            for candle in raw_data:
                # Validate candle data
                if not self._validate_candle(candle):
                    continue
                
                # Add calculated fields
                processed_candle = {
                    'timestamp': candle['timestamp'],
                    'datetime': datetime.fromtimestamp(candle['timestamp']),
                    'open': float(candle['open']),
                    'high': float(candle['high']),
                    'low': float(candle['low']),
                    'close': float(candle['close']),
                    'volume': float(candle['volume']),
                    'typical_price': (float(candle['high']) + float(candle['low']) + float(candle['close'])) / 3,
                    'price_change': 0,  # Will be calculated
                    'price_change_pct': 0  # Will be calculated
                }
                
                processed_data.append(processed_candle)
            
            # Calculate price changes
            for i in range(1, len(processed_data)):
                current = processed_data[i]
                previous = processed_data[i-1]
                
                current['price_change'] = current['close'] - previous['close']
                if previous['close'] != 0:
                    current['price_change_pct'] = (current['price_change'] / previous['close']) * 100
            
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Error processing historical data: {str(e)}")
            return []
    
    def _validate_candle(self, candle: Dict[str, Any]) -> bool:
        """
        Validate a single candle data point.
        
        Args:
            candle: Candle data to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            required_fields = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            
            # Check required fields
            for field in required_fields:
                if field not in candle:
                    self.logger.warning(f"Missing field in candle: {field}")
                    return False
            
            # Check price relationships
            high = float(candle['high'])
            low = float(candle['low'])
            open_price = float(candle['open'])
            close_price = float(candle['close'])
            
            if high < low:
                self.logger.warning("High price is less than low price")
                return False
            
            if open_price > high or open_price < low:
                self.logger.warning("Open price is outside high-low range")
                return False
            
            if close_price > high or close_price < low:
                self.logger.warning("Close price is outside high-low range")
                return False
            
            # Check for reasonable values
            if any(price <= 0 for price in [high, low, open_price, close_price]):
                self.logger.warning("Invalid price values (negative or zero)")
                return False
            
            if float(candle['volume']) < 0:
                self.logger.warning("Invalid volume (negative)")
                return False
            
            return True
            
        except (ValueError, TypeError) as e:
            self.logger.warning(f"Error validating candle: {str(e)}")
            return False
    
    def _calculate_spread_metrics(self, orderbook: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate spread and market depth metrics from orderbook.
        
        Args:
            orderbook: Orderbook data
            
        Returns:
            Dictionary containing spread metrics
        """
        try:
            if not orderbook.get('success'):
                return {}
            
            bids = orderbook.get('bids', [])
            asks = orderbook.get('asks', [])
            
            if not bids or not asks:
                return {}
            
            # Get best bid and ask
            best_bid = float(bids[0]['price']) if bids else 0
            best_ask = float(asks[0]['price']) if asks else 0
            
            if best_bid == 0 or best_ask == 0:
                return {}
            
            # Calculate spread
            spread = best_ask - best_bid
            spread_pct = (spread / best_bid) * 100
            
            # Calculate market depth
            bid_depth = sum(float(bid['size']) for bid in bids[:5])  # Top 5 levels
            ask_depth = sum(float(ask['size']) for ask in asks[:5])  # Top 5 levels
            
            # Calculate mid price
            mid_price = (best_bid + best_ask) / 2
            
            return {
                'best_bid': best_bid,
                'best_ask': best_ask,
                'spread': spread,
                'spread_pct': spread_pct,
                'mid_price': mid_price,
                'bid_depth': bid_depth,
                'ask_depth': ask_depth,
                'imbalance': (bid_depth - ask_depth) / (bid_depth + ask_depth) if (bid_depth + ask_depth) > 0 else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating spread metrics: {str(e)}")
            return {}
    
    def _calculate_price_change(self) -> Dict[str, float]:
        """
        Calculate price change metrics from recent price history.
        
        Returns:
            Dictionary containing price change information
        """
        try:
            if len(self.price_history) < 2:
                return {'change': 0, 'change_pct': 0}
            
            current_price = self.price_history[-1]['price']
            previous_price = self.price_history[-2]['price']
            
            change = current_price - previous_price
            change_pct = (change / previous_price) * 100 if previous_price != 0 else 0
            
            # Calculate additional metrics if we have more history
            metrics = {
                'change': change,
                'change_pct': change_pct
            }
            
            if len(self.price_history) >= 60:  # 1 hour of minute data
                hour_ago_price = self.price_history[-60]['price']
                hour_change = current_price - hour_ago_price
                hour_change_pct = (hour_change / hour_ago_price) * 100 if hour_ago_price != 0 else 0
                
                metrics.update({
                    'hour_change': hour_change,
                    'hour_change_pct': hour_change_pct
                })
            
            if len(self.price_history) >= 1440:  # 24 hours of minute data
                day_ago_price = self.price_history[-1440]['price']
                day_change = current_price - day_ago_price
                day_change_pct = (day_change / day_ago_price) * 100 if day_ago_price != 0 else 0
                
                metrics.update({
                    'day_change': day_change,
                    'day_change_pct': day_change_pct
                })
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating price change: {str(e)}")
            return {'change': 0, 'change_pct': 0}
    
    def _store_price_data(self, market_data: Dict[str, Any]) -> None:
        """
        Store price data in history for analysis.
        
        Args:
            market_data: Market data to store
        """
        try:
            price_point = {
                'timestamp': market_data['timestamp'],
                'price': market_data['price'],
                'spread': market_data.get('spread', {}).get('spread', 0)
            }
            
            self.price_history.append(price_point)
            
            # Maintain maximum history length
            if len(self.price_history) > self.max_history_length:
                self.price_history = self.price_history[-self.max_history_length:]
            
        except Exception as e:
            self.logger.error(f"Error storing price data: {str(e)}")
    
    def get_price_statistics(self, period_minutes: int = 60) -> Dict[str, float]:
        """
        Calculate price statistics for a given period.
        
        Args:
            period_minutes: Period in minutes to calculate statistics for
            
        Returns:
            Dictionary containing price statistics
        """
        try:
            if not self.price_history:
                return {}
            
            # Filter data for the specified period
            cutoff_time = time.time() - (period_minutes * 60)
            recent_data = [p for p in self.price_history if p['timestamp'] >= cutoff_time]
            
            if not recent_data:
                return {}
            
            prices = [p['price'] for p in recent_data]
            
            statistics = {
                'count': len(prices),
                'min': min(prices),
                'max': max(prices),
                'mean': np.mean(prices),
                'median': np.median(prices),
                'std': np.std(prices),
                'range': max(prices) - min(prices),
                'range_pct': ((max(prices) - min(prices)) / min(prices)) * 100 if min(prices) > 0 else 0
            }
            
            return statistics
            
        except Exception as e:
            self.logger.error(f"Error calculating price statistics: {str(e)}")
            return {}
    
    def get_volume_profile(self, historical_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate volume profile from historical data.
        
        Args:
            historical_data: Historical candle data
            
        Returns:
            Volume profile information
        """
        try:
            if not historical_data:
                return {}
            
            # Extract price and volume data
            prices = [candle['typical_price'] for candle in historical_data]
            volumes = [candle['volume'] for candle in historical_data]
            
            # Create price bins
            min_price = min(prices)
            max_price = max(prices)
            num_bins = 20
            
            price_range = max_price - min_price
            bin_size = price_range / num_bins
            
            # Calculate volume for each price level
            volume_profile = {}
            for i, candle in enumerate(historical_data):
                price = candle['typical_price']
                volume = candle['volume']
                
                # Determine which bin this price falls into
                bin_index = int((price - min_price) / bin_size)
                bin_index = min(bin_index, num_bins - 1)  # Ensure we don't exceed bounds
                
                bin_price = min_price + (bin_index * bin_size)
                
                if bin_price not in volume_profile:
                    volume_profile[bin_price] = 0
                
                volume_profile[bin_price] += volume
            
            # Find point of control (price level with highest volume)
            poc_price = max(volume_profile.keys(), key=lambda k: volume_profile[k])
            poc_volume = volume_profile[poc_price]
            
            # Calculate value area (70% of volume)
            total_volume = sum(volume_profile.values())
            target_volume = total_volume * 0.7
            
            # Sort by volume to find value area
            sorted_levels = sorted(volume_profile.items(), key=lambda x: x[1], reverse=True)
            
            value_area_volume = 0
            value_area_prices = []
            
            for price, volume in sorted_levels:
                value_area_volume += volume
                value_area_prices.append(price)
                
                if value_area_volume >= target_volume:
                    break
            
            value_area_high = max(value_area_prices) if value_area_prices else max_price
            value_area_low = min(value_area_prices) if value_area_prices else min_price
            
            return {
                'poc_price': poc_price,
                'poc_volume': poc_volume,
                'value_area_high': value_area_high,
                'value_area_low': value_area_low,
                'total_volume': total_volume,
                'volume_profile': volume_profile
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating volume profile: {str(e)}")
            return {}
    
    def to_dataframe(self, historical_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Convert historical data to pandas DataFrame for analysis.
        
        Args:
            historical_data: Historical candle data
            
        Returns:
            Pandas DataFrame with OHLCV data
        """
        try:
            if not historical_data:
                return pd.DataFrame()
            
            df = pd.DataFrame(historical_data)
            
            # Set timestamp as index
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
            df.set_index('datetime', inplace=True)
            
            # Ensure numeric columns
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Sort by timestamp
            df.sort_index(inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error converting to DataFrame: {str(e)}")
            return pd.DataFrame()
    
    def get_market_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive market summary.
        
        Returns:
            Dictionary containing market summary information
        """
        try:
            current_data = asyncio.run(self.get_current_market_data())
            price_stats = self.get_price_statistics()
            
            summary = {
                'symbol': self.symbol,
                'current_price': current_data.get('price'),
                'timestamp': current_data.get('timestamp'),
                'spread': current_data.get('spread', {}),
                'price_change': current_data.get('price_change', {}),
                'statistics': price_stats,
                'data_points': len(self.price_history)
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating market summary: {str(e)}")
            return {}

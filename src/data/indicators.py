"""
Technical Indicators Module

This module implements various technical indicators used for market analysis
and trading signal generation in the Delta Exchange trading bot.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from decimal import Decimal
import warnings

from ..utils.logger import get_logger

# Suppress pandas warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)


class TechnicalIndicators:
    """
    Comprehensive technical indicators calculator for cryptocurrency trading.
    
    This class provides implementations of popular technical indicators
    including trend, momentum, volatility, and volume-based indicators.
    """
    
    def __init__(self):
        """Initialize the technical indicators calculator."""
        self.logger = get_logger(__name__)
    
    def sma(self, data: Union[List[float], pd.Series], period: int) -> pd.Series:
        """
        Calculate Simple Moving Average (SMA).
        
        Args:
            data: Price data (typically close prices)
            period: Number of periods for the moving average
            
        Returns:
            Pandas Series containing SMA values
        """
        try:
            if isinstance(data, list):
                data = pd.Series(data)
            
            return data.rolling(window=period, min_periods=period).mean()
            
        except Exception as e:
            self.logger.error(f"Error calculating SMA: {str(e)}")
            return pd.Series()
    
    def ema(self, data: Union[List[float], pd.Series], period: int) -> pd.Series:
        """
        Calculate Exponential Moving Average (EMA).
        
        Args:
            data: Price data (typically close prices)
            period: Number of periods for the moving average
            
        Returns:
            Pandas Series containing EMA values
        """
        try:
            if isinstance(data, list):
                data = pd.Series(data)
            
            return data.ewm(span=period, adjust=False).mean()
            
        except Exception as e:
            self.logger.error(f"Error calculating EMA: {str(e)}")
            return pd.Series()
    
    def rsi(self, data: Union[List[float], pd.Series], period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).
        
        Args:
            data: Price data (typically close prices)
            period: Number of periods for RSI calculation (default: 14)
            
        Returns:
            Pandas Series containing RSI values (0-100)
        """
        try:
            if isinstance(data, list):
                data = pd.Series(data)
            
            # Calculate price changes
            delta = data.diff()
            
            # Separate gains and losses
            gains = delta.where(delta > 0, 0)
            losses = -delta.where(delta < 0, 0)
            
            # Calculate average gains and losses
            avg_gains = gains.rolling(window=period, min_periods=period).mean()
            avg_losses = losses.rolling(window=period, min_periods=period).mean()
            
            # Calculate RS and RSI
            rs = avg_gains / avg_losses
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
            
        except Exception as e:
            self.logger.error(f"Error calculating RSI: {str(e)}")
            return pd.Series()
    
    def macd(self, data: Union[List[float], pd.Series], 
             fast_period: int = 12, slow_period: int = 26, 
             signal_period: int = 9) -> Dict[str, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        Args:
            data: Price data (typically close prices)
            fast_period: Fast EMA period (default: 12)
            slow_period: Slow EMA period (default: 26)
            signal_period: Signal line EMA period (default: 9)
            
        Returns:
            Dictionary containing MACD line, signal line, and histogram
        """
        try:
            if isinstance(data, list):
                data = pd.Series(data)
            
            # Calculate EMAs
            ema_fast = self.ema(data, fast_period)
            ema_slow = self.ema(data, slow_period)
            
            # Calculate MACD line
            macd_line = ema_fast - ema_slow
            
            # Calculate signal line
            signal_line = self.ema(macd_line, signal_period)
            
            # Calculate histogram
            histogram = macd_line - signal_line
            
            return {
                'macd': macd_line,
                'signal': signal_line,
                'histogram': histogram
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating MACD: {str(e)}")
            return {'macd': pd.Series(), 'signal': pd.Series(), 'histogram': pd.Series()}
    
    def bollinger_bands(self, data: Union[List[float], pd.Series], 
                       period: int = 20, std_dev: float = 2) -> Dict[str, pd.Series]:
        """
        Calculate Bollinger Bands.
        
        Args:
            data: Price data (typically close prices)
            period: Number of periods for moving average (default: 20)
            std_dev: Number of standard deviations (default: 2)
            
        Returns:
            Dictionary containing upper band, middle band (SMA), and lower band
        """
        try:
            if isinstance(data, list):
                data = pd.Series(data)
            
            # Calculate middle band (SMA)
            middle_band = self.sma(data, period)
            
            # Calculate standard deviation
            std = data.rolling(window=period, min_periods=period).std()
            
            # Calculate upper and lower bands
            upper_band = middle_band + (std * std_dev)
            lower_band = middle_band - (std * std_dev)
            
            return {
                'upper': upper_band,
                'middle': middle_band,
                'lower': lower_band
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating Bollinger Bands: {str(e)}")
            return {'upper': pd.Series(), 'middle': pd.Series(), 'lower': pd.Series()}
    
    def stochastic(self, high: Union[List[float], pd.Series], 
                  low: Union[List[float], pd.Series], 
                  close: Union[List[float], pd.Series], 
                  k_period: int = 14, d_period: int = 3) -> Dict[str, pd.Series]:
        """
        Calculate Stochastic Oscillator.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            k_period: %K period (default: 14)
            d_period: %D period (default: 3)
            
        Returns:
            Dictionary containing %K and %D lines
        """
        try:
            if isinstance(high, list):
                high = pd.Series(high)
            if isinstance(low, list):
                low = pd.Series(low)
            if isinstance(close, list):
                close = pd.Series(close)
            
            # Calculate %K
            lowest_low = low.rolling(window=k_period, min_periods=k_period).min()
            highest_high = high.rolling(window=k_period, min_periods=k_period).max()
            
            k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
            
            # Calculate %D (SMA of %K)
            d_percent = k_percent.rolling(window=d_period, min_periods=d_period).mean()
            
            return {
                'k_percent': k_percent,
                'd_percent': d_percent
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating Stochastic: {str(e)}")
            return {'k_percent': pd.Series(), 'd_percent': pd.Series()}
    
    def atr(self, high: Union[List[float], pd.Series], 
            low: Union[List[float], pd.Series], 
            close: Union[List[float], pd.Series], 
            period: int = 14) -> pd.Series:
        """
        Calculate Average True Range (ATR).
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: Number of periods (default: 14)
            
        Returns:
            Pandas Series containing ATR values
        """
        try:
            if isinstance(high, list):
                high = pd.Series(high)
            if isinstance(low, list):
                low = pd.Series(low)
            if isinstance(close, list):
                close = pd.Series(close)
            
            # Calculate True Range components
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            
            # True Range is the maximum of the three
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Calculate ATR as SMA of True Range
            atr = true_range.rolling(window=period, min_periods=period).mean()
            
            return atr
            
        except Exception as e:
            self.logger.error(f"Error calculating ATR: {str(e)}")
            return pd.Series()
    
    def adx(self, high: Union[List[float], pd.Series], 
            low: Union[List[float], pd.Series], 
            close: Union[List[float], pd.Series], 
            period: int = 14) -> Dict[str, pd.Series]:
        """
        Calculate Average Directional Index (ADX) and Directional Indicators.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: Number of periods (default: 14)
            
        Returns:
            Dictionary containing ADX, +DI, and -DI
        """
        try:
            if isinstance(high, list):
                high = pd.Series(high)
            if isinstance(low, list):
                low = pd.Series(low)
            if isinstance(close, list):
                close = pd.Series(close)
            
            # Calculate True Range
            atr_values = self.atr(high, low, close, period)
            
            # Calculate Directional Movement
            up_move = high - high.shift(1)
            down_move = low.shift(1) - low
            
            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
            
            plus_dm = pd.Series(plus_dm, index=high.index)
            minus_dm = pd.Series(minus_dm, index=high.index)
            
            # Calculate smoothed DM and TR
            plus_dm_smooth = plus_dm.rolling(window=period, min_periods=period).mean()
            minus_dm_smooth = minus_dm.rolling(window=period, min_periods=period).mean()
            
            # Calculate Directional Indicators
            plus_di = 100 * (plus_dm_smooth / atr_values)
            minus_di = 100 * (minus_dm_smooth / atr_values)
            
            # Calculate ADX
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(window=period, min_periods=period).mean()
            
            return {
                'adx': adx,
                'plus_di': plus_di,
                'minus_di': minus_di
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating ADX: {str(e)}")
            return {'adx': pd.Series(), 'plus_di': pd.Series(), 'minus_di': pd.Series()}
    
    def williams_r(self, high: Union[List[float], pd.Series], 
                   low: Union[List[float], pd.Series], 
                   close: Union[List[float], pd.Series], 
                   period: int = 14) -> pd.Series:
        """
        Calculate Williams %R.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: Number of periods (default: 14)
            
        Returns:
            Pandas Series containing Williams %R values
        """
        try:
            if isinstance(high, list):
                high = pd.Series(high)
            if isinstance(low, list):
                low = pd.Series(low)
            if isinstance(close, list):
                close = pd.Series(close)
            
            highest_high = high.rolling(window=period, min_periods=period).max()
            lowest_low = low.rolling(window=period, min_periods=period).min()
            
            williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
            
            return williams_r
            
        except Exception as e:
            self.logger.error(f"Error calculating Williams %R: {str(e)}")
            return pd.Series()
    
    def cci(self, high: Union[List[float], pd.Series], 
            low: Union[List[float], pd.Series], 
            close: Union[List[float], pd.Series], 
            period: int = 20) -> pd.Series:
        """
        Calculate Commodity Channel Index (CCI).
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: Number of periods (default: 20)
            
        Returns:
            Pandas Series containing CCI values
        """
        try:
            if isinstance(high, list):
                high = pd.Series(high)
            if isinstance(low, list):
                low = pd.Series(low)
            if isinstance(close, list):
                close = pd.Series(close)
            
            # Calculate Typical Price
            typical_price = (high + low + close) / 3
            
            # Calculate SMA of Typical Price
            sma_tp = typical_price.rolling(window=period, min_periods=period).mean()
            
            # Calculate Mean Deviation
            mean_deviation = typical_price.rolling(window=period, min_periods=period).apply(
                lambda x: np.mean(np.abs(x - np.mean(x)))
            )
            
            # Calculate CCI
            cci = (typical_price - sma_tp) / (0.015 * mean_deviation)
            
            return cci
            
        except Exception as e:
            self.logger.error(f"Error calculating CCI: {str(e)}")
            return pd.Series()
    
    def obv(self, close: Union[List[float], pd.Series], 
            volume: Union[List[float], pd.Series]) -> pd.Series:
        """
        Calculate On-Balance Volume (OBV).
        
        Args:
            close: Close prices
            volume: Volume data
            
        Returns:
            Pandas Series containing OBV values
        """
        try:
            if isinstance(close, list):
                close = pd.Series(close)
            if isinstance(volume, list):
                volume = pd.Series(volume)
            
            # Calculate price direction
            price_direction = np.where(close > close.shift(1), 1, 
                                     np.where(close < close.shift(1), -1, 0))
            
            # Calculate OBV
            obv = (volume * price_direction).cumsum()
            
            return obv
            
        except Exception as e:
            self.logger.error(f"Error calculating OBV: {str(e)}")
            return pd.Series()
    
    def vwap(self, high: Union[List[float], pd.Series], 
             low: Union[List[float], pd.Series], 
             close: Union[List[float], pd.Series], 
             volume: Union[List[float], pd.Series]) -> pd.Series:
        """
        Calculate Volume Weighted Average Price (VWAP).
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            volume: Volume data
            
        Returns:
            Pandas Series containing VWAP values
        """
        try:
            if isinstance(high, list):
                high = pd.Series(high)
            if isinstance(low, list):
                low = pd.Series(low)
            if isinstance(close, list):
                close = pd.Series(close)
            if isinstance(volume, list):
                volume = pd.Series(volume)
            
            # Calculate typical price
            typical_price = (high + low + close) / 3
            
            # Calculate VWAP
            cumulative_volume = volume.cumsum()
            cumulative_price_volume = (typical_price * volume).cumsum()
            
            vwap = cumulative_price_volume / cumulative_volume
            
            return vwap
            
        except Exception as e:
            self.logger.error(f"Error calculating VWAP: {str(e)}")
            return pd.Series()
    
    def fibonacci_retracement(self, high_price: float, low_price: float) -> Dict[str, float]:
        """
        Calculate Fibonacci retracement levels.
        
        Args:
            high_price: Highest price in the range
            low_price: Lowest price in the range
            
        Returns:
            Dictionary containing Fibonacci levels
        """
        try:
            price_range = high_price - low_price
            
            levels = {
                '0%': high_price,
                '23.6%': high_price - (price_range * 0.236),
                '38.2%': high_price - (price_range * 0.382),
                '50%': high_price - (price_range * 0.5),
                '61.8%': high_price - (price_range * 0.618),
                '78.6%': high_price - (price_range * 0.786),
                '100%': low_price
            }
            
            return levels
            
        except Exception as e:
            self.logger.error(f"Error calculating Fibonacci retracement: {str(e)}")
            return {}
    
    def pivot_points(self, high: float, low: float, close: float) -> Dict[str, float]:
        """
        Calculate pivot points and support/resistance levels.
        
        Args:
            high: Previous period high
            low: Previous period low
            close: Previous period close
            
        Returns:
            Dictionary containing pivot point and support/resistance levels
        """
        try:
            # Calculate pivot point
            pivot = (high + low + close) / 3
            
            # Calculate support and resistance levels
            r1 = (2 * pivot) - low
            s1 = (2 * pivot) - high
            r2 = pivot + (high - low)
            s2 = pivot - (high - low)
            r3 = high + 2 * (pivot - low)
            s3 = low - 2 * (high - pivot)
            
            return {
                'pivot': pivot,
                'r1': r1,
                'r2': r2,
                'r3': r3,
                's1': s1,
                's2': s2,
                's3': s3
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating pivot points: {str(e)}")
            return {}
    
    def ichimoku_cloud(self, high: Union[List[float], pd.Series], 
                      low: Union[List[float], pd.Series], 
                      close: Union[List[float], pd.Series]) -> Dict[str, pd.Series]:
        """
        Calculate Ichimoku Cloud components.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            
        Returns:
            Dictionary containing Ichimoku components
        """
        try:
            if isinstance(high, list):
                high = pd.Series(high)
            if isinstance(low, list):
                low = pd.Series(low)
            if isinstance(close, list):
                close = pd.Series(close)
            
            # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
            tenkan_sen = (high.rolling(window=9).max() + low.rolling(window=9).min()) / 2
            
            # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
            kijun_sen = (high.rolling(window=26).max() + low.rolling(window=26).min()) / 2
            
            # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2, shifted 26 periods ahead
            senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
            
            # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2, shifted 26 periods ahead
            senkou_span_b = ((high.rolling(window=52).max() + low.rolling(window=52).min()) / 2).shift(26)
            
            # Chikou Span (Lagging Span): Close price shifted 26 periods back
            chikou_span = close.shift(-26)
            
            return {
                'tenkan_sen': tenkan_sen,
                'kijun_sen': kijun_sen,
                'senkou_span_a': senkou_span_a,
                'senkou_span_b': senkou_span_b,
                'chikou_span': chikou_span
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating Ichimoku Cloud: {str(e)}")
            return {}
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all available indicators for a given DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all indicators added
        """
        try:
            result_df = df.copy()
            
            # Trend indicators
            result_df['sma_20'] = self.sma(df['close'], 20)
            result_df['sma_50'] = self.sma(df['close'], 50)
            result_df['ema_12'] = self.ema(df['close'], 12)
            result_df['ema_26'] = self.ema(df['close'], 26)
            
            # Momentum indicators
            result_df['rsi'] = self.rsi(df['close'])
            
            macd_data = self.macd(df['close'])
            result_df['macd'] = macd_data['macd']
            result_df['macd_signal'] = macd_data['signal']
            result_df['macd_histogram'] = macd_data['histogram']
            
            stoch_data = self.stochastic(df['high'], df['low'], df['close'])
            result_df['stoch_k'] = stoch_data['k_percent']
            result_df['stoch_d'] = stoch_data['d_percent']
            
            result_df['williams_r'] = self.williams_r(df['high'], df['low'], df['close'])
            result_df['cci'] = self.cci(df['high'], df['low'], df['close'])
            
            # Volatility indicators
            bb_data = self.bollinger_bands(df['close'])
            result_df['bb_upper'] = bb_data['upper']
            result_df['bb_middle'] = bb_data['middle']
            result_df['bb_lower'] = bb_data['lower']
            
            result_df['atr'] = self.atr(df['high'], df['low'], df['close'])
            
            # Trend strength
            adx_data = self.adx(df['high'], df['low'], df['close'])
            result_df['adx'] = adx_data['adx']
            result_df['plus_di'] = adx_data['plus_di']
            result_df['minus_di'] = adx_data['minus_di']
            
            # Volume indicators
            if 'volume' in df.columns:
                result_df['obv'] = self.obv(df['close'], df['volume'])
                result_df['vwap'] = self.vwap(df['high'], df['low'], df['close'], df['volume'])
            
            # Ichimoku Cloud
            ichimoku_data = self.ichimoku_cloud(df['high'], df['low'], df['close'])
            for key, value in ichimoku_data.items():
                result_df[f'ichimoku_{key}'] = value
            
            self.logger.info("All technical indicators calculated successfully")
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error calculating all indicators: {str(e)}")
            return df

"""
Technical Analysis Module

This module provides comprehensive technical analysis capabilities including
pattern recognition, trend analysis, and signal generation for the trading bot.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import scipy.stats as stats
from scipy.signal import find_peaks

from .indicators import TechnicalIndicators
from ..utils.logger import get_logger


class TechnicalAnalyzer:
    """
    Comprehensive technical analysis engine for cryptocurrency trading.
    
    This class combines technical indicators with pattern recognition,
    trend analysis, and market structure analysis to provide trading insights.
    """
    
    def __init__(self, config):
        """
        Initialize the technical analyzer.
        
        Args:
            config: Configuration object containing analysis parameters
        """
        self.config = config
        self.logger = get_logger(__name__)
        self.indicators = TechnicalIndicators()
        
        # Analysis parameters
        self.trend_periods = [20, 50, 100, 200]
        self.volatility_period = 20
        self.momentum_period = 14
        
        self.logger.info("Technical analyzer initialized")
    
    def calculate_indicators(self, historical_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate all technical indicators for the given historical data.
        
        Args:
            historical_data: List of OHLCV candles
            
        Returns:
            Dictionary containing all calculated indicators
        """
        try:
            if not historical_data or len(historical_data) < 50:
                self.logger.warning("Insufficient data for indicator calculation")
                return {}
            
            # Convert to DataFrame
            df = pd.DataFrame(historical_data)
            
            # Ensure required columns exist
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in df.columns:
                    self.logger.error(f"Missing required column: {col}")
                    return {}
            
            # Calculate all indicators
            df_with_indicators = self.indicators.calculate_all_indicators(df)
            
            # Extract latest values for decision making
            latest_indicators = self._extract_latest_values(df_with_indicators)
            
            # Add custom analysis
            latest_indicators.update(self._calculate_custom_indicators(df_with_indicators))
            
            self.logger.debug("Technical indicators calculated successfully")
            return latest_indicators
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {str(e)}")
            return {}
    
    def _extract_latest_values(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Extract the latest values of all indicators.
        
        Args:
            df: DataFrame with calculated indicators
            
        Returns:
            Dictionary containing latest indicator values
        """
        try:
            latest_values = {}
            
            # Get the last row (most recent data)
            if len(df) == 0:
                return latest_values
            
            latest_row = df.iloc[-1]
            
            # Extract all numeric columns
            for column in df.columns:
                if pd.api.types.is_numeric_dtype(df[column]):
                    value = latest_row[column]
                    if not pd.isna(value):
                        latest_values[column] = float(value)
            
            return latest_values
            
        except Exception as e:
            self.logger.error(f"Error extracting latest values: {str(e)}")
            return {}
    
    def _calculate_custom_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate custom indicators and analysis metrics.
        
        Args:
            df: DataFrame with OHLCV data and basic indicators
            
        Returns:
            Dictionary containing custom indicators
        """
        try:
            custom_indicators = {}
            
            if len(df) < 20:
                return custom_indicators
            
            # Price position relative to moving averages
            latest_close = df['close'].iloc[-1]
            
            if 'sma_20' in df.columns and not pd.isna(df['sma_20'].iloc[-1]):
                custom_indicators['price_vs_sma20'] = (latest_close / df['sma_20'].iloc[-1] - 1) * 100
            
            if 'sma_50' in df.columns and not pd.isna(df['sma_50'].iloc[-1]):
                custom_indicators['price_vs_sma50'] = (latest_close / df['sma_50'].iloc[-1] - 1) * 100
            
            # Moving average slopes (trend strength)
            if 'sma_20' in df.columns and len(df) >= 25:
                sma20_slope = self._calculate_slope(df['sma_20'].tail(5))
                custom_indicators['sma20_slope'] = sma20_slope
            
            if 'sma_50' in df.columns and len(df) >= 55:
                sma50_slope = self._calculate_slope(df['sma_50'].tail(5))
                custom_indicators['sma50_slope'] = sma50_slope
            
            # Bollinger Band position
            if all(col in df.columns for col in ['bb_upper', 'bb_lower', 'bb_middle']):
                bb_upper = df['bb_upper'].iloc[-1]
                bb_lower = df['bb_lower'].iloc[-1]
                bb_middle = df['bb_middle'].iloc[-1]
                
                if not any(pd.isna([bb_upper, bb_lower, bb_middle])):
                    bb_position = (latest_close - bb_lower) / (bb_upper - bb_lower)
                    custom_indicators['bb_position'] = bb_position * 100
                    
                    # Bollinger Band squeeze detection
                    bb_width = (bb_upper - bb_lower) / bb_middle
                    custom_indicators['bb_width'] = bb_width * 100
            
            # Volume analysis
            if 'volume' in df.columns and len(df) >= 20:
                avg_volume = df['volume'].tail(20).mean()
                current_volume = df['volume'].iloc[-1]
                custom_indicators['volume_ratio'] = current_volume / avg_volume if avg_volume > 0 else 1
            
            # Price momentum
            if len(df) >= 10:
                price_momentum = self._calculate_price_momentum(df['close'], periods=[5, 10])
                custom_indicators.update(price_momentum)
            
            # Volatility metrics
            volatility_metrics = self._calculate_volatility_metrics(df)
            custom_indicators.update(volatility_metrics)
            
            return custom_indicators
            
        except Exception as e:
            self.logger.error(f"Error calculating custom indicators: {str(e)}")
            return {}
    
    def _calculate_slope(self, series: pd.Series) -> float:
        """
        Calculate the slope of a price series.
        
        Args:
            series: Price series
            
        Returns:
            Slope value
        """
        try:
            if len(series) < 2:
                return 0
            
            x = np.arange(len(series))
            y = series.values
            
            # Remove NaN values
            mask = ~np.isnan(y)
            if np.sum(mask) < 2:
                return 0
            
            slope, _, _, _, _ = stats.linregress(x[mask], y[mask])
            return slope
            
        except Exception as e:
            self.logger.error(f"Error calculating slope: {str(e)}")
            return 0
    
    def _calculate_price_momentum(self, prices: pd.Series, periods: List[int]) -> Dict[str, float]:
        """
        Calculate price momentum for different periods.
        
        Args:
            prices: Price series
            periods: List of periods to calculate momentum for
            
        Returns:
            Dictionary containing momentum values
        """
        try:
            momentum = {}
            
            for period in periods:
                if len(prices) > period:
                    current_price = prices.iloc[-1]
                    past_price = prices.iloc[-period-1]
                    
                    if past_price != 0:
                        momentum_pct = ((current_price - past_price) / past_price) * 100
                        momentum[f'momentum_{period}'] = momentum_pct
            
            return momentum
            
        except Exception as e:
            self.logger.error(f"Error calculating price momentum: {str(e)}")
            return {}
    
    def _calculate_volatility_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate various volatility metrics.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dictionary containing volatility metrics
        """
        try:
            volatility_metrics = {}
            
            if len(df) < 20:
                return volatility_metrics
            
            # Historical volatility (20-period)
            returns = df['close'].pct_change().dropna()
            if len(returns) >= 20:
                hist_vol = returns.tail(20).std() * np.sqrt(252) * 100  # Annualized
                volatility_metrics['historical_volatility'] = hist_vol
            
            # Average True Range as percentage of price
            if 'atr' in df.columns and not pd.isna(df['atr'].iloc[-1]):
                atr_pct = (df['atr'].iloc[-1] / df['close'].iloc[-1]) * 100
                volatility_metrics['atr_percentage'] = atr_pct
            
            # Intraday volatility
            if all(col in df.columns for col in ['high', 'low', 'close']):
                intraday_range = ((df['high'] - df['low']) / df['close']) * 100
                avg_intraday_vol = intraday_range.tail(20).mean()
                volatility_metrics['avg_intraday_volatility'] = avg_intraday_vol
            
            return volatility_metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility metrics: {str(e)}")
            return {}
    
    def detect_patterns(self, historical_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Detect chart patterns and price formations.
        
        Args:
            historical_data: List of OHLCV candles
            
        Returns:
            Dictionary containing detected patterns
        """
        try:
            if not historical_data or len(historical_data) < 50:
                return {}
            
            df = pd.DataFrame(historical_data)
            patterns = {}
            
            # Support and resistance levels
            support_resistance = self._find_support_resistance(df)
            patterns.update(support_resistance)
            
            # Trend patterns
            trend_patterns = self._detect_trend_patterns(df)
            patterns.update(trend_patterns)
            
            # Reversal patterns
            reversal_patterns = self._detect_reversal_patterns(df)
            patterns.update(reversal_patterns)
            
            # Continuation patterns
            continuation_patterns = self._detect_continuation_patterns(df)
            patterns.update(continuation_patterns)
            
            self.logger.debug(f"Detected {len(patterns)} patterns")
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error detecting patterns: {str(e)}")
            return {}
    
    def _find_support_resistance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Find support and resistance levels using pivot points.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dictionary containing support and resistance levels
        """
        try:
            if len(df) < 20:
                return {}
            
            # Find local peaks and troughs
            highs = df['high'].values
            lows = df['low'].values
            
            # Find peaks (resistance levels)
            peaks, _ = find_peaks(highs, distance=5, prominence=np.std(highs) * 0.5)
            
            # Find troughs (support levels)
            troughs, _ = find_peaks(-lows, distance=5, prominence=np.std(lows) * 0.5)
            
            # Get recent levels
            recent_peaks = peaks[-5:] if len(peaks) >= 5 else peaks
            recent_troughs = troughs[-5:] if len(troughs) >= 5 else troughs
            
            resistance_levels = [highs[i] for i in recent_peaks]
            support_levels = [lows[i] for i in recent_troughs]
            
            current_price = df['close'].iloc[-1]
            
            # Find nearest levels
            nearest_resistance = min(resistance_levels, key=lambda x: abs(x - current_price)) if resistance_levels else None
            nearest_support = min(support_levels, key=lambda x: abs(x - current_price)) if support_levels else None
            
            return {
                'resistance_levels': resistance_levels,
                'support_levels': support_levels,
                'nearest_resistance': nearest_resistance,
                'nearest_support': nearest_support,
                'resistance_distance': ((nearest_resistance - current_price) / current_price * 100) if nearest_resistance else None,
                'support_distance': ((current_price - nearest_support) / current_price * 100) if nearest_support else None
            }
            
        except Exception as e:
            self.logger.error(f"Error finding support/resistance: {str(e)}")
            return {}
    
    def _detect_trend_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect trend-based patterns.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dictionary containing trend patterns
        """
        try:
            patterns = {}
            
            if len(df) < 50:
                return patterns
            
            # Moving average alignment
            if all(col in df.columns for col in ['sma_20', 'sma_50']):
                sma20 = df['sma_20'].iloc[-1]
                sma50 = df['sma_50'].iloc[-1]
                
                if not pd.isna(sma20) and not pd.isna(sma50):
                    if sma20 > sma50:
                        patterns['ma_alignment'] = 'bullish'
                    else:
                        patterns['ma_alignment'] = 'bearish'
            
            # Trend strength using ADX
            if 'adx' in df.columns and not pd.isna(df['adx'].iloc[-1]):
                adx_value = df['adx'].iloc[-1]
                if adx_value > 25:
                    patterns['trend_strength'] = 'strong'
                elif adx_value > 15:
                    patterns['trend_strength'] = 'moderate'
                else:
                    patterns['trend_strength'] = 'weak'
            
            # Price channel analysis
            channel_analysis = self._analyze_price_channel(df)
            patterns.update(channel_analysis)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error detecting trend patterns: {str(e)}")
            return {}
    
    def _detect_reversal_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect potential reversal patterns.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dictionary containing reversal patterns
        """
        try:
            patterns = {}
            
            if len(df) < 20:
                return patterns
            
            # RSI divergence
            if 'rsi' in df.columns:
                rsi_divergence = self._detect_rsi_divergence(df)
                if rsi_divergence:
                    patterns['rsi_divergence'] = rsi_divergence
            
            # Double top/bottom patterns
            double_patterns = self._detect_double_patterns(df)
            patterns.update(double_patterns)
            
            # Hammer and shooting star candlestick patterns
            candlestick_patterns = self._detect_candlestick_patterns(df)
            patterns.update(candlestick_patterns)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error detecting reversal patterns: {str(e)}")
            return {}
    
    def _detect_continuation_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect continuation patterns.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dictionary containing continuation patterns
        """
        try:
            patterns = {}
            
            if len(df) < 30:
                return patterns
            
            # Flag and pennant patterns
            flag_patterns = self._detect_flag_patterns(df)
            patterns.update(flag_patterns)
            
            # Triangle patterns
            triangle_patterns = self._detect_triangle_patterns(df)
            patterns.update(triangle_patterns)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error detecting continuation patterns: {str(e)}")
            return {}
    
    def _analyze_price_channel(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze price channel and trend direction.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dictionary containing channel analysis
        """
        try:
            if len(df) < 20:
                return {}
            
            # Calculate linear regression channel
            prices = df['close'].tail(20).values
            x = np.arange(len(prices))
            
            # Fit linear regression
            slope, intercept, r_value, _, _ = stats.linregress(x, prices)
            
            # Calculate channel boundaries
            residuals = prices - (slope * x + intercept)
            std_residuals = np.std(residuals)
            
            upper_channel = slope * x + intercept + (2 * std_residuals)
            lower_channel = slope * x + intercept - (2 * std_residuals)
            
            current_price = prices[-1]
            current_upper = upper_channel[-1]
            current_lower = lower_channel[-1]
            
            # Determine position within channel
            channel_position = (current_price - current_lower) / (current_upper - current_lower)
            
            return {
                'channel_slope': slope,
                'channel_r_squared': r_value ** 2,
                'channel_position': channel_position,
                'channel_upper': current_upper,
                'channel_lower': current_lower,
                'trend_direction': 'up' if slope > 0 else 'down' if slope < 0 else 'sideways'
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing price channel: {str(e)}")
            return {}
    
    def _detect_rsi_divergence(self, df: pd.DataFrame) -> Optional[str]:
        """
        Detect RSI divergence patterns.
        
        Args:
            df: DataFrame with OHLCV and RSI data
            
        Returns:
            String indicating divergence type or None
        """
        try:
            if len(df) < 20 or 'rsi' not in df.columns:
                return None
            
            recent_data = df.tail(20)
            prices = recent_data['close'].values
            rsi_values = recent_data['rsi'].values
            
            # Find recent peaks in price and RSI
            price_peaks, _ = find_peaks(prices, distance=3)
            rsi_peaks, _ = find_peaks(rsi_values, distance=3)
            
            if len(price_peaks) >= 2 and len(rsi_peaks) >= 2:
                # Check for bearish divergence (higher highs in price, lower highs in RSI)
                if (prices[price_peaks[-1]] > prices[price_peaks[-2]] and 
                    rsi_values[rsi_peaks[-1]] < rsi_values[rsi_peaks[-2]]):
                    return 'bearish_divergence'
            
            # Find recent troughs
            price_troughs, _ = find_peaks(-prices, distance=3)
            rsi_troughs, _ = find_peaks(-rsi_values, distance=3)
            
            if len(price_troughs) >= 2 and len(rsi_troughs) >= 2:
                # Check for bullish divergence (lower lows in price, higher lows in RSI)
                if (prices[price_troughs[-1]] < prices[price_troughs[-2]] and 
                    rsi_values[rsi_troughs[-1]] > rsi_values[rsi_troughs[-2]]):
                    return 'bullish_divergence'
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error detecting RSI divergence: {str(e)}")
            return None
    
    def _detect_double_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect double top and double bottom patterns.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dictionary containing double pattern information
        """
        try:
            patterns = {}
            
            if len(df) < 30:
                return patterns
            
            highs = df['high'].tail(30).values
            lows = df['low'].tail(30).values
            
            # Find peaks and troughs
            peaks, _ = find_peaks(highs, distance=5, prominence=np.std(highs) * 0.3)
            troughs, _ = find_peaks(-lows, distance=5, prominence=np.std(lows) * 0.3)
            
            # Check for double top
            if len(peaks) >= 2:
                last_two_peaks = peaks[-2:]
                peak_heights = [highs[i] for i in last_two_peaks]
                
                # Check if peaks are similar in height (within 2%)
                if abs(peak_heights[1] - peak_heights[0]) / peak_heights[0] < 0.02:
                    patterns['double_top'] = {
                        'confirmed': True,
                        'peak_levels': peak_heights,
                        'pattern_strength': 'strong' if len(peaks) == 2 else 'moderate'
                    }
            
            # Check for double bottom
            if len(troughs) >= 2:
                last_two_troughs = troughs[-2:]
                trough_levels = [lows[i] for i in last_two_troughs]
                
                # Check if troughs are similar in level (within 2%)
                if abs(trough_levels[1] - trough_levels[0]) / trough_levels[0] < 0.02:
                    patterns['double_bottom'] = {
                        'confirmed': True,
                        'trough_levels': trough_levels,
                        'pattern_strength': 'strong' if len(troughs) == 2 else 'moderate'
                    }
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error detecting double patterns: {str(e)}")
            return {}
    
    def _detect_candlestick_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect basic candlestick patterns.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dictionary containing candlestick patterns
        """
        try:
            patterns = {}
            
            if len(df) < 3:
                return patterns
            
            # Get last few candles
            recent_candles = df.tail(3)
            
            for i, (_, candle) in enumerate(recent_candles.iterrows()):
                open_price = candle['open']
                high_price = candle['high']
                low_price = candle['low']
                close_price = candle['close']
                
                body_size = abs(close_price - open_price)
                upper_shadow = high_price - max(open_price, close_price)
                lower_shadow = min(open_price, close_price) - low_price
                total_range = high_price - low_price
                
                if total_range == 0:
                    continue
                
                # Hammer pattern (bullish reversal)
                if (lower_shadow > 2 * body_size and 
                    upper_shadow < body_size * 0.1 and 
                    body_size / total_range < 0.3):
                    patterns[f'hammer_candle_{i}'] = 'bullish_reversal'
                
                # Shooting star pattern (bearish reversal)
                if (upper_shadow > 2 * body_size and 
                    lower_shadow < body_size * 0.1 and 
                    body_size / total_range < 0.3):
                    patterns[f'shooting_star_candle_{i}'] = 'bearish_reversal'
                
                # Doji pattern (indecision)
                if body_size / total_range < 0.1:
                    patterns[f'doji_candle_{i}'] = 'indecision'
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error detecting candlestick patterns: {str(e)}")
            return {}
    
    def _detect_flag_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect flag and pennant continuation patterns.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dictionary containing flag patterns
        """
        try:
            # This is a simplified implementation
            # In practice, flag detection would be more sophisticated
            patterns = {}
            
            if len(df) < 20:
                return patterns
            
            # Look for strong move followed by consolidation
            recent_data = df.tail(20)
            prices = recent_data['close'].values
            
            # Check for strong initial move (first 5 periods)
            initial_move = (prices[4] - prices[0]) / prices[0]
            
            # Check for consolidation (next 10 periods)
            consolidation_data = prices[5:15]
            consolidation_range = (max(consolidation_data) - min(consolidation_data)) / np.mean(consolidation_data)
            
            if abs(initial_move) > 0.05 and consolidation_range < 0.03:  # 5% move, 3% consolidation
                if initial_move > 0:
                    patterns['bull_flag'] = 'continuation_bullish'
                else:
                    patterns['bear_flag'] = 'continuation_bearish'
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error detecting flag patterns: {str(e)}")
            return {}
    
    def _detect_triangle_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect triangle continuation patterns.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dictionary containing triangle patterns
        """
        try:
            patterns = {}
            
            if len(df) < 20:
                return patterns
            
            recent_data = df.tail(20)
            highs = recent_data['high'].values
            lows = recent_data['low'].values
            
            # Find trend lines for highs and lows
            x = np.arange(len(highs))
            
            # Fit trend lines
            high_slope, _, high_r, _, _ = stats.linregress(x, highs)
            low_slope, _, low_r, _, _ = stats.linregress(x, lows)
            
            # Check for triangle patterns based on slope convergence
            if abs(high_r) > 0.7 and abs(low_r) > 0.7:  # Good trend line fit
                if high_slope < 0 and low_slope > 0:  # Converging lines
                    patterns['symmetrical_triangle'] = 'continuation'
                elif high_slope < 0 and abs(low_slope) < 0.001:  # Descending triangle
                    patterns['descending_triangle'] = 'bearish_continuation'
                elif abs(high_slope) < 0.001 and low_slope > 0:  # Ascending triangle
                    patterns['ascending_triangle'] = 'bullish_continuation'
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error detecting triangle patterns: {str(e)}")
            return {}
    
    def calculate_volatility(self, historical_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate various volatility measures.
        
        Args:
            historical_data: List of OHLCV candles
            
        Returns:
            Dictionary containing volatility metrics
        """
        try:
            if not historical_data or len(historical_data) < 20:
                return {}
            
            df = pd.DataFrame(historical_data)
            
            # Calculate returns
            returns = df['close'].pct_change().dropna()
            
            volatility_metrics = {
                'realized_volatility': returns.std() * np.sqrt(252) * 100,  # Annualized
                'volatility_20d': returns.tail(20).std() * np.sqrt(252) * 100,
                'volatility_5d': returns.tail(5).std() * np.sqrt(252) * 100,
            }
            
            # Parkinson volatility (using high-low range)
            if all(col in df.columns for col in ['high', 'low']):
                parkinson_vol = np.sqrt(
                    (1 / (4 * np.log(2))) * 
                    np.mean((np.log(df['high'] / df['low'])) ** 2)
                ) * np.sqrt(252) * 100
                
                volatility_metrics['parkinson_volatility'] = parkinson_vol
            
            return volatility_metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility: {str(e)}")
            return {}
    
    def calculate_momentum(self, historical_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate momentum indicators and metrics.
        
        Args:
            historical_data: List of OHLCV candles
            
        Returns:
            Dictionary containing momentum metrics
        """
        try:
            if not historical_data or len(historical_data) < 20:
                return {}
            
            df = pd.DataFrame(historical_data)
            momentum_metrics = {}
            
            # Price momentum for different periods
            periods = [1, 3, 5, 10, 20]
            for period in periods:
                if len(df) > period:
                    current_price = df['close'].iloc[-1]
                    past_price = df['close'].iloc[-period-1]
                    momentum_pct = ((current_price - past_price) / past_price) * 100
                    momentum_metrics[f'momentum_{period}d'] = momentum_pct
            
            # Rate of change
            if len(df) >= 10:
                roc = ((df['close'].iloc[-1] - df['close'].iloc[-10]) / df['close'].iloc[-10]) * 100
                momentum_metrics['rate_of_change_10d'] = roc
            
            # Momentum oscillator
            if len(df) >= 20:
                momentum_osc = df['close'].iloc[-1] - df['close'].iloc[-10]
                momentum_metrics['momentum_oscillator'] = momentum_osc
            
            return momentum_metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating momentum: {str(e)}")
            return {}

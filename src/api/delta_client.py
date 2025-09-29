"""
Delta Exchange API Client Wrapper

This module provides a comprehensive wrapper around the Delta Exchange REST API
for automated trading operations with enhanced error handling and logging.
"""

import os
import time
import logging
from typing import Dict, List, Optional, Any, Union
from decimal import Decimal

from delta_rest_client import DeltaRestClient
from delta_rest_client.constants import OrderType, TimeInForce

from ..utils.logger import get_logger
from ..utils.config import Config


class DeltaExchangeClient:
    """
    Enhanced Delta Exchange API client with comprehensive trading functionality.
    
    This class wraps the official delta-rest-client library with additional
    features including error handling, retry logic, and trading utilities.
    """
    
    def __init__(self, config: Config):
        """
        Initialize the Delta Exchange client.
        
        Args:
            config: Configuration object containing API credentials and settings
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # Initialize the Delta REST client
        self.client = DeltaRestClient(
            base_url=config.get('DELTA_BASE_URL'),
            api_key=config.get('DELTA_API_KEY'),
            api_secret=config.get('DELTA_API_SECRET')
        )
        
        self.product_id = config.get('PRODUCT_ID', 27)  # Default to BTCUSD
        self.symbol = config.get('SYMBOL', 'BTCUSD')
        
        self.logger.info(f"Delta Exchange client initialized for {self.symbol}")
    
    def get_account_balance(self) -> Dict[str, Any]:
        """
        Retrieve current account balance information.
        
        Returns:
            Dictionary containing balance information for all assets
            
        Raises:
            Exception: If API call fails or returns error
        """
        try:
            self.logger.debug("Fetching account balance")
            response = self.client.get_balances()
            
            if response.get('success'):
                balances = response.get('result', [])
                self.logger.info(f"Retrieved balance for {len(balances)} assets")
                return {
                    'success': True,
                    'balances': balances,
                    'timestamp': time.time()
                }
            else:
                error_msg = response.get('error', 'Unknown error')
                self.logger.error(f"Failed to get account balance: {error_msg}")
                raise Exception(f"Balance retrieval failed: {error_msg}")
                
        except Exception as e:
            self.logger.error(f"Error getting account balance: {str(e)}")
            raise
    
    def get_current_price(self) -> Decimal:
        """
        Get the current market price for the configured symbol.
        
        Returns:
            Current market price as Decimal
            
        Raises:
            Exception: If price retrieval fails
        """
        try:
            self.logger.debug(f"Fetching current price for {self.symbol}")
            response = self.client.get_ticker(self.symbol)
            
            if response.get('success'):
                ticker_data = response.get('result')
                current_price = Decimal(str(ticker_data.get('close', 0)))
                self.logger.debug(f"Current price for {self.symbol}: {current_price}")
                return current_price
            else:
                error_msg = response.get('error', 'Unknown error')
                self.logger.error(f"Failed to get current price: {error_msg}")
                raise Exception(f"Price retrieval failed: {error_msg}")
                
        except Exception as e:
            self.logger.error(f"Error getting current price: {str(e)}")
            raise
    
    def get_orderbook(self, depth: int = 20) -> Dict[str, Any]:
        """
        Retrieve the current orderbook for the configured product.
        
        Args:
            depth: Number of price levels to retrieve (default: 20)
            
        Returns:
            Dictionary containing bid and ask levels with prices and sizes
        """
        try:
            self.logger.debug(f"Fetching orderbook for product {self.product_id}")
            response = self.client.get_l2_orderbook(self.product_id)
            
            if response.get('success'):
                orderbook = response.get('result')
                self.logger.debug(f"Retrieved orderbook with {len(orderbook.get('buy', []))} bids and {len(orderbook.get('sell', []))} asks")
                return {
                    'success': True,
                    'bids': orderbook.get('buy', [])[:depth],
                    'asks': orderbook.get('sell', [])[:depth],
                    'timestamp': time.time()
                }
            else:
                error_msg = response.get('error', 'Unknown error')
                self.logger.error(f"Failed to get orderbook: {error_msg}")
                raise Exception(f"Orderbook retrieval failed: {error_msg}")
                
        except Exception as e:
            self.logger.error(f"Error getting orderbook: {str(e)}")
            raise
    
    def place_market_order(self, side: str, size: Union[int, float, Decimal], 
                          reduce_only: bool = False) -> Dict[str, Any]:
        """
        Place a market order.
        
        Args:
            side: Order side ('buy' or 'sell')
            size: Order size in base currency
            reduce_only: Whether this is a reduce-only order
            
        Returns:
            Dictionary containing order response information
        """
        try:
            size_str = str(size)
            self.logger.info(f"Placing market {side} order: {size_str} {self.symbol}")
            
            response = self.client.place_order(
                product_id=self.product_id,
                size=size_str,
                side=side,
                order_type=OrderType.MARKET,
                reduce_only=str(reduce_only).lower()
            )
            
            if response.get('success'):
                order_data = response.get('result')
                order_id = order_data.get('id')
                self.logger.info(f"Market order placed successfully: ID {order_id}")
                return {
                    'success': True,
                    'order_id': order_id,
                    'order_data': order_data,
                    'timestamp': time.time()
                }
            else:
                error_msg = response.get('error', 'Unknown error')
                self.logger.error(f"Failed to place market order: {error_msg}")
                raise Exception(f"Market order failed: {error_msg}")
                
        except Exception as e:
            self.logger.error(f"Error placing market order: {str(e)}")
            raise
    
    def place_limit_order(self, side: str, size: Union[int, float, Decimal], 
                         price: Union[int, float, Decimal], 
                         time_in_force: str = 'gtc',
                         reduce_only: bool = False) -> Dict[str, Any]:
        """
        Place a limit order.
        
        Args:
            side: Order side ('buy' or 'sell')
            size: Order size in base currency
            price: Limit price
            time_in_force: Time in force ('gtc', 'ioc', 'fok')
            reduce_only: Whether this is a reduce-only order
            
        Returns:
            Dictionary containing order response information
        """
        try:
            size_str = str(size)
            price_str = str(price)
            
            # Map time_in_force to TimeInForce enum
            tif_mapping = {
                'gtc': TimeInForce.GTC,
                'ioc': TimeInForce.IOC,
                'fok': TimeInForce.FOK
            }
            tif = tif_mapping.get(time_in_force.lower(), TimeInForce.GTC)
            
            self.logger.info(f"Placing limit {side} order: {size_str} {self.symbol} @ {price_str}")
            
            response = self.client.place_order(
                product_id=self.product_id,
                size=size_str,
                side=side,
                limit_price=price_str,
                order_type=OrderType.LIMIT,
                time_in_force=tif,
                reduce_only=str(reduce_only).lower()
            )
            
            if response.get('success'):
                order_data = response.get('result')
                order_id = order_data.get('id')
                self.logger.info(f"Limit order placed successfully: ID {order_id}")
                return {
                    'success': True,
                    'order_id': order_id,
                    'order_data': order_data,
                    'timestamp': time.time()
                }
            else:
                error_msg = response.get('error', 'Unknown error')
                self.logger.error(f"Failed to place limit order: {error_msg}")
                raise Exception(f"Limit order failed: {error_msg}")
                
        except Exception as e:
            self.logger.error(f"Error placing limit order: {str(e)}")
            raise
    
    def place_stop_order(self, side: str, size: Union[int, float, Decimal],
                        stop_price: Union[int, float, Decimal],
                        limit_price: Optional[Union[int, float, Decimal]] = None,
                        trail_amount: Optional[Union[int, float, Decimal]] = None) -> Dict[str, Any]:
        """
        Place a stop order (stop-loss or trailing stop).
        
        Args:
            side: Order side ('buy' or 'sell')
            size: Order size in base currency
            stop_price: Stop trigger price
            limit_price: Limit price for stop-limit orders (optional)
            trail_amount: Trailing amount for trailing stops (optional)
            
        Returns:
            Dictionary containing order response information
        """
        try:
            size_str = str(size)
            stop_price_str = str(stop_price)
            
            order_params = {
                'product_id': self.product_id,
                'size': size_str,
                'side': side,
                'stop_price': stop_price_str
            }
            
            if limit_price is not None:
                order_params['limit_price'] = str(limit_price)
                order_params['order_type'] = OrderType.LIMIT
                order_type_desc = "stop-limit"
            else:
                order_params['order_type'] = OrderType.MARKET
                order_type_desc = "stop-market"
            
            if trail_amount is not None:
                order_params['trail_amount'] = str(trail_amount)
                order_params['isTrailingStopLoss'] = True
                order_type_desc = "trailing-stop"
            
            self.logger.info(f"Placing {order_type_desc} {side} order: {size_str} {self.symbol}")
            
            response = self.client.place_stop_order(**order_params)
            
            if response.get('success'):
                order_data = response.get('result')
                order_id = order_data.get('id')
                self.logger.info(f"Stop order placed successfully: ID {order_id}")
                return {
                    'success': True,
                    'order_id': order_id,
                    'order_data': order_data,
                    'timestamp': time.time()
                }
            else:
                error_msg = response.get('error', 'Unknown error')
                self.logger.error(f"Failed to place stop order: {error_msg}")
                raise Exception(f"Stop order failed: {error_msg}")
                
        except Exception as e:
            self.logger.error(f"Error placing stop order: {str(e)}")
            raise
    
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        Cancel an open order.
        
        Args:
            order_id: ID of the order to cancel
            
        Returns:
            Dictionary containing cancellation response
        """
        try:
            self.logger.info(f"Cancelling order: {order_id}")
            
            response = self.client.cancel_order(self.product_id, order_id)
            
            if response.get('success'):
                self.logger.info(f"Order {order_id} cancelled successfully")
                return {
                    'success': True,
                    'order_id': order_id,
                    'timestamp': time.time()
                }
            else:
                error_msg = response.get('error', 'Unknown error')
                self.logger.error(f"Failed to cancel order {order_id}: {error_msg}")
                raise Exception(f"Order cancellation failed: {error_msg}")
                
        except Exception as e:
            self.logger.error(f"Error cancelling order {order_id}: {str(e)}")
            raise
    
    def get_open_orders(self) -> List[Dict[str, Any]]:
        """
        Retrieve all open orders.
        
        Returns:
            List of open orders
        """
        try:
            self.logger.debug("Fetching open orders")
            response = self.client.get_live_orders()
            
            if response.get('success'):
                orders = response.get('result', [])
                self.logger.info(f"Retrieved {len(orders)} open orders")
                return orders
            else:
                error_msg = response.get('error', 'Unknown error')
                self.logger.error(f"Failed to get open orders: {error_msg}")
                raise Exception(f"Open orders retrieval failed: {error_msg}")
                
        except Exception as e:
            self.logger.error(f"Error getting open orders: {str(e)}")
            raise
    
    def get_position(self) -> Dict[str, Any]:
        """
        Get current position for the configured product.
        
        Returns:
            Dictionary containing position information
        """
        try:
            self.logger.debug(f"Fetching position for product {self.product_id}")
            response = self.client.get_position(self.product_id)
            
            if response.get('success'):
                position = response.get('result')
                size = position.get('size', 0)
                self.logger.debug(f"Current position size: {size}")
                return {
                    'success': True,
                    'position': position,
                    'size': float(size),
                    'timestamp': time.time()
                }
            else:
                error_msg = response.get('error', 'Unknown error')
                self.logger.error(f"Failed to get position: {error_msg}")
                raise Exception(f"Position retrieval failed: {error_msg}")
                
        except Exception as e:
            self.logger.error(f"Error getting position: {str(e)}")
            raise
    
    def get_order_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieve order history.
        
        Args:
            limit: Maximum number of orders to retrieve
            
        Returns:
            List of historical orders
        """
        try:
            self.logger.debug(f"Fetching order history (limit: {limit})")
            query = {"product_id": self.product_id}
            response = self.client.order_history(query, page_size=limit)
            
            if response.get('success'):
                orders = response.get('result', [])
                self.logger.info(f"Retrieved {len(orders)} historical orders")
                return orders
            else:
                error_msg = response.get('error', 'Unknown error')
                self.logger.error(f"Failed to get order history: {error_msg}")
                raise Exception(f"Order history retrieval failed: {error_msg}")
                
        except Exception as e:
            self.logger.error(f"Error getting order history: {str(e)}")
            raise
    
    def get_fills(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieve fill history (executed trades).
        
        Args:
            limit: Maximum number of fills to retrieve
            
        Returns:
            List of trade fills
        """
        try:
            self.logger.debug(f"Fetching fills (limit: {limit})")
            query = {"product_id": self.product_id}
            response = self.client.fills(query, page_size=limit)
            
            if response.get('success'):
                fills = response.get('result', [])
                self.logger.info(f"Retrieved {len(fills)} fills")
                return fills
            else:
                error_msg = response.get('error', 'Unknown error')
                self.logger.error(f"Failed to get fills: {error_msg}")
                raise Exception(f"Fills retrieval failed: {error_msg}")
                
        except Exception as e:
            self.logger.error(f"Error getting fills: {str(e)}")
            raise
    
    def set_leverage(self, leverage: Union[int, float]) -> Dict[str, Any]:
        """
        Set leverage for the configured product.
        
        Args:
            leverage: Leverage value to set
            
        Returns:
            Dictionary containing leverage setting response
        """
        try:
            leverage_str = str(leverage)
            self.logger.info(f"Setting leverage to {leverage_str}x for product {self.product_id}")
            
            response = self.client.set_leverage(self.product_id, leverage_str)
            
            if response.get('success'):
                self.logger.info(f"Leverage set to {leverage_str}x successfully")
                return {
                    'success': True,
                    'leverage': leverage,
                    'timestamp': time.time()
                }
            else:
                error_msg = response.get('error', 'Unknown error')
                self.logger.error(f"Failed to set leverage: {error_msg}")
                raise Exception(f"Leverage setting failed: {error_msg}")
                
        except Exception as e:
            self.logger.error(f"Error setting leverage: {str(e)}")
            raise
    
    def health_check(self) -> bool:
        """
        Perform a health check to verify API connectivity.
        
        Returns:
            True if API is accessible, False otherwise
        """
        try:
            self.logger.debug("Performing API health check")
            # Try to get current price as a simple connectivity test
            self.get_current_price()
            self.logger.info("API health check passed")
            return True
        except Exception as e:
            self.logger.error(f"API health check failed: {str(e)}")
            return False

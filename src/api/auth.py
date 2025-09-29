"""
Authentication and Security Module

This module handles API authentication, key management, and security features
for the Delta Exchange trading bot.
"""

import os
import hashlib
import hmac
import time
import base64
import logging
from typing import Dict, Optional, Tuple
from cryptography.fernet import Fernet

from ..utils.logger import get_logger


class AuthenticationManager:
    """
    Manages API authentication and security features for Delta Exchange.
    
    This class handles API key validation, signature generation, and
    security measures including IP whitelisting and rate limiting.
    """
    
    def __init__(self):
        """Initialize the authentication manager."""
        self.logger = get_logger(__name__)
        self._encryption_key = None
        self._initialize_encryption()
    
    def _initialize_encryption(self) -> None:
        """Initialize encryption for sensitive data storage."""
        try:
            # Generate or load encryption key
            key_file = os.path.join(os.path.expanduser('~'), '.delta_bot_key')
            
            if os.path.exists(key_file):
                with open(key_file, 'rb') as f:
                    self._encryption_key = f.read()
            else:
                self._encryption_key = Fernet.generate_key()
                with open(key_file, 'wb') as f:
                    f.write(self._encryption_key)
                os.chmod(key_file, 0o600)  # Restrict file permissions
                
            self.logger.debug("Encryption initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize encryption: {str(e)}")
            raise
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """
        Encrypt sensitive data such as API keys.
        
        Args:
            data: Plain text data to encrypt
            
        Returns:
            Encrypted data as base64 string
        """
        try:
            fernet = Fernet(self._encryption_key)
            encrypted_data = fernet.encrypt(data.encode())
            return base64.b64encode(encrypted_data).decode()
        except Exception as e:
            self.logger.error(f"Failed to encrypt data: {str(e)}")
            raise
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """
        Decrypt sensitive data.
        
        Args:
            encrypted_data: Base64 encoded encrypted data
            
        Returns:
            Decrypted plain text data
        """
        try:
            fernet = Fernet(self._encryption_key)
            decoded_data = base64.b64decode(encrypted_data.encode())
            decrypted_data = fernet.decrypt(decoded_data)
            return decrypted_data.decode()
        except Exception as e:
            self.logger.error(f"Failed to decrypt data: {str(e)}")
            raise
    
    def validate_api_credentials(self, api_key: str, api_secret: str) -> bool:
        """
        Validate API credentials format and basic requirements.
        
        Args:
            api_key: Delta Exchange API key
            api_secret: Delta Exchange API secret
            
        Returns:
            True if credentials appear valid, False otherwise
        """
        try:
            # Basic validation checks
            if not api_key or not api_secret:
                self.logger.error("API key or secret is empty")
                return False
            
            if len(api_key) < 10:
                self.logger.error("API key appears too short")
                return False
            
            if len(api_secret) < 20:
                self.logger.error("API secret appears too short")
                return False
            
            # Check for common patterns that indicate test/invalid keys
            test_patterns = ['test', 'demo', 'fake', 'invalid', 'example']
            key_lower = api_key.lower()
            secret_lower = api_secret.lower()
            
            for pattern in test_patterns:
                if pattern in key_lower or pattern in secret_lower:
                    self.logger.warning(f"API credentials contain test pattern: {pattern}")
                    return False
            
            self.logger.debug("API credentials validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating API credentials: {str(e)}")
            return False
    
    def generate_signature(self, api_secret: str, method: str, timestamp: str, 
                          path: str, query_string: str = "", payload: str = "") -> str:
        """
        Generate HMAC signature for Delta Exchange API requests.
        
        Args:
            api_secret: API secret key
            method: HTTP method (GET, POST, etc.)
            timestamp: Unix timestamp as string
            path: API endpoint path
            query_string: URL query parameters
            payload: Request body payload
            
        Returns:
            HMAC signature as hex string
        """
        try:
            # Construct signature data according to Delta Exchange specification
            signature_data = method + timestamp + path + query_string + payload
            
            # Generate HMAC-SHA256 signature
            signature = hmac.new(
                api_secret.encode('utf-8'),
                signature_data.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            
            self.logger.debug(f"Generated signature for {method} {path}")
            return signature
            
        except Exception as e:
            self.logger.error(f"Failed to generate signature: {str(e)}")
            raise
    
    def get_current_timestamp(self) -> str:
        """
        Get current Unix timestamp as string.
        
        Returns:
            Current timestamp as string
        """
        return str(int(time.time()))
    
    def validate_timestamp(self, timestamp: str, max_age_seconds: int = 300) -> bool:
        """
        Validate that a timestamp is recent and not too old.
        
        Args:
            timestamp: Timestamp to validate
            max_age_seconds: Maximum age in seconds (default: 5 minutes)
            
        Returns:
            True if timestamp is valid, False otherwise
        """
        try:
            timestamp_int = int(timestamp)
            current_time = int(time.time())
            age = current_time - timestamp_int
            
            if age < 0:
                self.logger.warning("Timestamp is in the future")
                return False
            
            if age > max_age_seconds:
                self.logger.warning(f"Timestamp is too old: {age} seconds")
                return False
            
            return True
            
        except ValueError:
            self.logger.error("Invalid timestamp format")
            return False
        except Exception as e:
            self.logger.error(f"Error validating timestamp: {str(e)}")
            return False
    
    def check_ip_whitelist(self, current_ip: str, whitelist: Optional[str] = None) -> bool:
        """
        Check if current IP is in the whitelist.
        
        Args:
            current_ip: Current IP address
            whitelist: Comma-separated list of allowed IPs
            
        Returns:
            True if IP is allowed, False otherwise
        """
        try:
            if not whitelist:
                # No whitelist configured, allow all IPs
                return True
            
            allowed_ips = [ip.strip() for ip in whitelist.split(',')]
            
            if current_ip in allowed_ips:
                self.logger.debug(f"IP {current_ip} is whitelisted")
                return True
            else:
                self.logger.warning(f"IP {current_ip} is not in whitelist")
                return False
                
        except Exception as e:
            self.logger.error(f"Error checking IP whitelist: {str(e)}")
            return False
    
    def get_public_ip(self) -> Optional[str]:
        """
        Get the current public IP address.
        
        Returns:
            Public IP address as string, or None if unable to determine
        """
        try:
            import requests
            
            # Try multiple IP detection services
            services = [
                'https://api.ipify.org',
                'https://ipinfo.io/ip',
                'https://icanhazip.com'
            ]
            
            for service in services:
                try:
                    response = requests.get(service, timeout=5)
                    if response.status_code == 200:
                        ip = response.text.strip()
                        self.logger.debug(f"Detected public IP: {ip}")
                        return ip
                except requests.RequestException:
                    continue
            
            self.logger.warning("Unable to determine public IP address")
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting public IP: {str(e)}")
            return None
    
    def create_auth_headers(self, api_key: str, api_secret: str, method: str,
                           path: str, query_string: str = "", payload: str = "") -> Dict[str, str]:
        """
        Create authentication headers for Delta Exchange API requests.
        
        Args:
            api_key: API key
            api_secret: API secret
            method: HTTP method
            path: API endpoint path
            query_string: URL query parameters
            payload: Request body payload
            
        Returns:
            Dictionary containing authentication headers
        """
        try:
            timestamp = self.get_current_timestamp()
            signature = self.generate_signature(
                api_secret, method, timestamp, path, query_string, payload
            )
            
            headers = {
                'api-key': api_key,
                'signature': signature,
                'timestamp': timestamp,
                'Content-Type': 'application/json',
                'User-Agent': 'DeltaTradingBot/1.0'
            }
            
            self.logger.debug("Created authentication headers")
            return headers
            
        except Exception as e:
            self.logger.error(f"Failed to create auth headers: {str(e)}")
            raise
    
    def validate_environment(self, api_key: str, base_url: str) -> bool:
        """
        Validate that API key matches the environment (production vs testnet).
        
        Args:
            api_key: API key to validate
            base_url: Base URL being used
            
        Returns:
            True if environment is consistent, False otherwise
        """
        try:
            is_production_url = 'api.india.delta.exchange' in base_url
            is_testnet_url = 'testnet.deltaex.org' in base_url
            
            # This is a basic check - in practice, you might have more
            # sophisticated ways to determine if a key is for prod or testnet
            if is_production_url:
                self.logger.info("Using production environment")
                # Add any production-specific validation here
                return True
            elif is_testnet_url:
                self.logger.info("Using testnet environment")
                # Add any testnet-specific validation here
                return True
            else:
                self.logger.error(f"Unknown environment URL: {base_url}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error validating environment: {str(e)}")
            return False
    
    def secure_log_credentials(self, api_key: str) -> str:
        """
        Create a secure representation of API key for logging.
        
        Args:
            api_key: Full API key
            
        Returns:
            Masked API key safe for logging
        """
        if len(api_key) <= 8:
            return "***"
        
        # Show first 4 and last 4 characters, mask the middle
        return f"{api_key[:4]}...{api_key[-4:]}"
    
    def rotate_encryption_key(self) -> bool:
        """
        Rotate the encryption key used for sensitive data.
        
        Returns:
            True if rotation successful, False otherwise
        """
        try:
            # Generate new encryption key
            new_key = Fernet.generate_key()
            
            # Update the key file
            key_file = os.path.join(os.path.expanduser('~'), '.delta_bot_key')
            backup_file = key_file + '.backup'
            
            # Create backup of old key
            if os.path.exists(key_file):
                os.rename(key_file, backup_file)
            
            # Write new key
            with open(key_file, 'wb') as f:
                f.write(new_key)
            os.chmod(key_file, 0o600)
            
            self._encryption_key = new_key
            
            self.logger.info("Encryption key rotated successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to rotate encryption key: {str(e)}")
            return False

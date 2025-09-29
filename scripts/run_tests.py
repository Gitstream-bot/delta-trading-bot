#!/usr/bin/env python3
"""
Test Runner Script

This script provides comprehensive testing capabilities for the Delta Trading Bot
including unit tests, integration tests, performance tests, and system validation.
"""

import os
import sys
import argparse
import subprocess
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from utils.logger import initialize_logging, get_logger
from utils.config import Config


class TestRunner:
    """
    Comprehensive test runner for the trading bot system.
    
    This class provides various testing capabilities including:
    - Unit tests for individual components
    - Integration tests for system interactions
    - Performance tests for optimization
    - System validation tests
    - API connectivity tests
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the test runner.
        
        Args:
            config: Configuration object (optional)
        """
        self.config = config or Config()
        self.logger = get_logger(__name__)
        
        # Test directories
        self.project_root = Path(__file__).parent.parent
        self.tests_dir = self.project_root / 'tests'
        self.src_dir = self.project_root / 'src'
        
        # Test results
        self.test_results = {}
        
        self.logger.info("Test runner initialized")
    
    def run_unit_tests(self, verbose: bool = False) -> Dict[str, Any]:
        """
        Run unit tests for all components.
        
        Args:
            verbose: Enable verbose output
            
        Returns:
            Dictionary containing test results
        """
        self.logger.info("Running unit tests...")
        
        try:
            # Run pytest if available, otherwise use unittest
            test_command = self._get_test_command('unit', verbose)
            
            start_time = time.time()
            result = subprocess.run(
                test_command,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout
            )
            end_time = time.time()
            
            test_results = {
                'test_type': 'unit',
                'success': result.returncode == 0,
                'duration': end_time - start_time,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'return_code': result.returncode
            }
            
            if test_results['success']:
                self.logger.info(f"Unit tests passed in {test_results['duration']:.2f} seconds")
            else:
                self.logger.error(f"Unit tests failed with return code {result.returncode}")
                if verbose:
                    self.logger.error(f"STDERR: {result.stderr}")
            
            return test_results
            
        except subprocess.TimeoutExpired:
            self.logger.error("Unit tests timed out after 5 minutes")
            return {
                'test_type': 'unit',
                'success': False,
                'error': 'Test execution timed out',
                'duration': 300
            }
        except Exception as e:
            self.logger.error(f"Error running unit tests: {str(e)}")
            return {
                'test_type': 'unit',
                'success': False,
                'error': str(e),
                'duration': 0
            }
    
    def run_integration_tests(self, verbose: bool = False) -> Dict[str, Any]:
        """
        Run integration tests.
        
        Args:
            verbose: Enable verbose output
            
        Returns:
            Dictionary containing test results
        """
        self.logger.info("Running integration tests...")
        
        try:
            # Integration tests require more setup
            test_command = self._get_test_command('integration', verbose)
            
            start_time = time.time()
            result = subprocess.run(
                test_command,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes timeout
            )
            end_time = time.time()
            
            test_results = {
                'test_type': 'integration',
                'success': result.returncode == 0,
                'duration': end_time - start_time,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'return_code': result.returncode
            }
            
            if test_results['success']:
                self.logger.info(f"Integration tests passed in {test_results['duration']:.2f} seconds")
            else:
                self.logger.error(f"Integration tests failed with return code {result.returncode}")
                if verbose:
                    self.logger.error(f"STDERR: {result.stderr}")
            
            return test_results
            
        except subprocess.TimeoutExpired:
            self.logger.error("Integration tests timed out after 10 minutes")
            return {
                'test_type': 'integration',
                'success': False,
                'error': 'Test execution timed out',
                'duration': 600
            }
        except Exception as e:
            self.logger.error(f"Error running integration tests: {str(e)}")
            return {
                'test_type': 'integration',
                'success': False,
                'error': str(e),
                'duration': 0
            }
    
    def run_performance_tests(self, verbose: bool = False) -> Dict[str, Any]:
        """
        Run performance tests.
        
        Args:
            verbose: Enable verbose output
            
        Returns:
            Dictionary containing test results
        """
        self.logger.info("Running performance tests...")
        
        try:
            # Performance tests focus on timing and resource usage
            test_command = self._get_test_command('performance', verbose)
            
            start_time = time.time()
            result = subprocess.run(
                test_command,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=900  # 15 minutes timeout
            )
            end_time = time.time()
            
            test_results = {
                'test_type': 'performance',
                'success': result.returncode == 0,
                'duration': end_time - start_time,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'return_code': result.returncode
            }
            
            if test_results['success']:
                self.logger.info(f"Performance tests passed in {test_results['duration']:.2f} seconds")
            else:
                self.logger.error(f"Performance tests failed with return code {result.returncode}")
                if verbose:
                    self.logger.error(f"STDERR: {result.stderr}")
            
            return test_results
            
        except subprocess.TimeoutExpired:
            self.logger.error("Performance tests timed out after 15 minutes")
            return {
                'test_type': 'performance',
                'success': False,
                'error': 'Test execution timed out',
                'duration': 900
            }
        except Exception as e:
            self.logger.error(f"Error running performance tests: {str(e)}")
            return {
                'test_type': 'performance',
                'success': False,
                'error': str(e),
                'duration': 0
            }
    
    def validate_system_configuration(self) -> Dict[str, Any]:
        """
        Validate system configuration and dependencies.
        
        Returns:
            Dictionary containing validation results
        """
        self.logger.info("Validating system configuration...")
        
        validation_results = {
            'test_type': 'system_validation',
            'success': True,
            'checks': {},
            'errors': []
        }
        
        try:
            # Check Python version
            python_version = sys.version_info
            validation_results['checks']['python_version'] = {
                'required': '3.8+',
                'actual': f"{python_version.major}.{python_version.minor}.{python_version.micro}",
                'passed': python_version >= (3, 8)
            }
            
            if not validation_results['checks']['python_version']['passed']:
                validation_results['errors'].append("Python 3.8+ required")
                validation_results['success'] = False
            
            # Check required directories
            required_dirs = ['src', 'tests', 'logs', '.github/workflows']
            for dir_name in required_dirs:
                dir_path = self.project_root / dir_name
                validation_results['checks'][f'directory_{dir_name}'] = {
                    'required': True,
                    'exists': dir_path.exists(),
                    'passed': dir_path.exists()
                }
                
                if not dir_path.exists():
                    validation_results['errors'].append(f"Required directory missing: {dir_name}")
                    validation_results['success'] = False
            
            # Check required files
            required_files = [
                'requirements.txt',
                'README.md',
                '.env.example',
                'src/utils/config.py',
                'src/api/delta_client.py',
                'src/trading/bot.py'
            ]
            
            for file_name in required_files:
                file_path = self.project_root / file_name
                validation_results['checks'][f'file_{file_name.replace("/", "_")}'] = {
                    'required': True,
                    'exists': file_path.exists(),
                    'passed': file_path.exists()
                }
                
                if not file_path.exists():
                    validation_results['errors'].append(f"Required file missing: {file_name}")
                    validation_results['success'] = False
            
            # Check Python dependencies
            try:
                import pandas
                import numpy
                import requests
                
                validation_results['checks']['dependencies'] = {
                    'pandas': True,
                    'numpy': True,
                    'requests': True,
                    'passed': True
                }
            except ImportError as e:
                validation_results['checks']['dependencies'] = {
                    'error': str(e),
                    'passed': False
                }
                validation_results['errors'].append(f"Missing dependencies: {str(e)}")
                validation_results['success'] = False
            
            # Validate configuration
            try:
                config = Config()
                config_valid = config.validate_settings() if hasattr(config, 'validate_settings') else True
                
                validation_results['checks']['configuration'] = {
                    'valid': config_valid,
                    'passed': config_valid
                }
                
                if not config_valid:
                    validation_results['errors'].append("Configuration validation failed")
                    validation_results['success'] = False
                    
            except Exception as e:
                validation_results['checks']['configuration'] = {
                    'error': str(e),
                    'passed': False
                }
                validation_results['errors'].append(f"Configuration error: {str(e)}")
                validation_results['success'] = False
            
            # Check log directory permissions
            logs_dir = self.project_root / 'logs'
            try:
                test_file = logs_dir / 'test_write.tmp'
                test_file.write_text('test')
                test_file.unlink()
                
                validation_results['checks']['log_permissions'] = {
                    'writable': True,
                    'passed': True
                }
            except Exception as e:
                validation_results['checks']['log_permissions'] = {
                    'writable': False,
                    'error': str(e),
                    'passed': False
                }
                validation_results['errors'].append(f"Log directory not writable: {str(e)}")
                validation_results['success'] = False
            
            if validation_results['success']:
                self.logger.info("System configuration validation passed")
            else:
                self.logger.error(f"System validation failed: {validation_results['errors']}")
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Error during system validation: {str(e)}")
            return {
                'test_type': 'system_validation',
                'success': False,
                'error': str(e),
                'checks': {}
            }
    
    def test_api_connectivity(self) -> Dict[str, Any]:
        """
        Test API connectivity and authentication.
        
        Returns:
            Dictionary containing connectivity test results
        """
        self.logger.info("Testing API connectivity...")
        
        connectivity_results = {
            'test_type': 'api_connectivity',
            'success': True,
            'tests': {},
            'errors': []
        }
        
        try:
            # Test basic HTTP connectivity
            import requests
            
            # Test Delta Exchange API endpoint
            delta_url = self.config.get('DELTA_BASE_URL', 'https://api.india.delta.exchange')
            
            try:
                response = requests.get(f"{delta_url}/v2/products", timeout=10)
                connectivity_results['tests']['delta_api_reachable'] = {
                    'url': delta_url,
                    'status_code': response.status_code,
                    'passed': response.status_code == 200
                }
                
                if response.status_code != 200:
                    connectivity_results['errors'].append(f"Delta API returned status {response.status_code}")
                    connectivity_results['success'] = False
                    
            except requests.RequestException as e:
                connectivity_results['tests']['delta_api_reachable'] = {
                    'url': delta_url,
                    'error': str(e),
                    'passed': False
                }
                connectivity_results['errors'].append(f"Cannot reach Delta API: {str(e)}")
                connectivity_results['success'] = False
            
            # Test authentication if credentials are provided
            api_key = self.config.get('DELTA_API_KEY')
            api_secret = self.config.get('DELTA_API_SECRET')
            
            if api_key and api_secret:
                try:
                    # Import and test authentication
                    from api.auth import AuthenticationManager
                    
                    auth_manager = AuthenticationManager()
                    credentials_valid = auth_manager.validate_api_credentials(api_key, api_secret)
                    
                    connectivity_results['tests']['api_credentials'] = {
                        'valid': credentials_valid,
                        'passed': credentials_valid
                    }
                    
                    if not credentials_valid:
                        connectivity_results['errors'].append("API credentials validation failed")
                        connectivity_results['success'] = False
                        
                except Exception as e:
                    connectivity_results['tests']['api_credentials'] = {
                        'error': str(e),
                        'passed': False
                    }
                    connectivity_results['errors'].append(f"Authentication test failed: {str(e)}")
                    connectivity_results['success'] = False
            else:
                connectivity_results['tests']['api_credentials'] = {
                    'skipped': 'No credentials provided',
                    'passed': True  # Not a failure if no credentials
                }
            
            # Test network latency
            try:
                start_time = time.time()
                response = requests.get(f"{delta_url}/v2/products", timeout=5)
                latency = (time.time() - start_time) * 1000  # Convert to milliseconds
                
                connectivity_results['tests']['api_latency'] = {
                    'latency_ms': round(latency, 2),
                    'acceptable': latency < 1000,  # Less than 1 second
                    'passed': latency < 5000  # Less than 5 seconds
                }
                
                if latency >= 5000:
                    connectivity_results['errors'].append(f"High API latency: {latency:.2f}ms")
                    connectivity_results['success'] = False
                    
            except Exception as e:
                connectivity_results['tests']['api_latency'] = {
                    'error': str(e),
                    'passed': False
                }
                connectivity_results['errors'].append(f"Latency test failed: {str(e)}")
                connectivity_results['success'] = False
            
            if connectivity_results['success']:
                self.logger.info("API connectivity tests passed")
            else:
                self.logger.error(f"API connectivity tests failed: {connectivity_results['errors']}")
            
            return connectivity_results
            
        except Exception as e:
            self.logger.error(f"Error during API connectivity test: {str(e)}")
            return {
                'test_type': 'api_connectivity',
                'success': False,
                'error': str(e),
                'tests': {}
            }
    
    def _get_test_command(self, test_type: str, verbose: bool) -> List[str]:
        """
        Get the appropriate test command for the test type.
        
        Args:
            test_type: Type of test to run
            verbose: Enable verbose output
            
        Returns:
            List of command arguments
        """
        base_command = [sys.executable, '-m']
        
        # Try to use pytest if available
        try:
            import pytest
            base_command.extend(['pytest'])
            
            if verbose:
                base_command.append('-v')
            
            # Add test-specific arguments
            if test_type == 'unit':
                base_command.extend(['tests/test_trading_system.py::TestConfig',
                                   'tests/test_trading_system.py::TestAuthentication',
                                   'tests/test_trading_system.py::TestTechnicalIndicators',
                                   'tests/test_trading_system.py::TestRiskManager'])
            elif test_type == 'integration':
                base_command.extend(['tests/test_trading_system.py::TestIntegration'])
            elif test_type == 'performance':
                base_command.extend(['tests/test_trading_system.py::TestPerformance'])
            else:
                base_command.append('tests/')
                
        except ImportError:
            # Fall back to unittest
            base_command.extend(['unittest'])
            
            if verbose:
                base_command.append('-v')
            
            if test_type == 'unit':
                base_command.append('tests.test_trading_system')
            else:
                base_command.append('discover')
                base_command.extend(['-s', 'tests'])
        
        return base_command
    
    def run_all_tests(self, verbose: bool = False) -> Dict[str, Any]:
        """
        Run all tests and return comprehensive results.
        
        Args:
            verbose: Enable verbose output
            
        Returns:
            Dictionary containing all test results
        """
        self.logger.info("Running comprehensive test suite...")
        
        all_results = {
            'start_time': time.time(),
            'tests': {},
            'summary': {}
        }
        
        # Run system validation first
        all_results['tests']['system_validation'] = self.validate_system_configuration()
        
        # Run API connectivity test
        all_results['tests']['api_connectivity'] = self.test_api_connectivity()
        
        # Run unit tests
        all_results['tests']['unit'] = self.run_unit_tests(verbose)
        
        # Run integration tests
        all_results['tests']['integration'] = self.run_integration_tests(verbose)
        
        # Run performance tests
        all_results['tests']['performance'] = self.run_performance_tests(verbose)
        
        # Calculate summary
        all_results['end_time'] = time.time()
        all_results['total_duration'] = all_results['end_time'] - all_results['start_time']
        
        passed_tests = sum(1 for result in all_results['tests'].values() if result.get('success', False))
        total_tests = len(all_results['tests'])
        
        all_results['summary'] = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'success_rate': (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            'overall_success': passed_tests == total_tests
        }
        
        # Log summary
        if all_results['summary']['overall_success']:
            self.logger.info(f"All tests passed! ({passed_tests}/{total_tests}) "
                           f"Duration: {all_results['total_duration']:.2f}s")
        else:
            self.logger.error(f"Some tests failed. ({passed_tests}/{total_tests}) "
                            f"Success rate: {all_results['summary']['success_rate']:.1f}%")
        
        return all_results
    
    def save_test_results(self, results: Dict[str, Any], output_file: str) -> None:
        """
        Save test results to file.
        
        Args:
            results: Test results dictionary
            output_file: Output file path
        """
        try:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            self.logger.info(f"Test results saved to: {output_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving test results: {str(e)}")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Delta Trading Bot Test Runner')
    
    parser.add_argument('--test-type', choices=['unit', 'integration', 'performance', 'system', 'api', 'all'],
                       default='all', help='Type of tests to run')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    parser.add_argument('--output', '-o', help='Output file for test results (JSON)')
    parser.add_argument('--config', '-c', help='Configuration file path')
    
    args = parser.parse_args()
    
    # Initialize logging
    initialize_logging({'LOG_LEVEL': 'INFO' if not args.verbose else 'DEBUG'})
    
    # Initialize configuration
    config = Config(config_file=args.config) if args.config else Config()
    
    # Create test runner
    runner = TestRunner(config)
    
    # Run tests based on type
    if args.test_type == 'unit':
        results = runner.run_unit_tests(args.verbose)
    elif args.test_type == 'integration':
        results = runner.run_integration_tests(args.verbose)
    elif args.test_type == 'performance':
        results = runner.run_performance_tests(args.verbose)
    elif args.test_type == 'system':
        results = runner.validate_system_configuration()
    elif args.test_type == 'api':
        results = runner.test_api_connectivity()
    else:  # all
        results = runner.run_all_tests(args.verbose)
    
    # Save results if output file specified
    if args.output:
        runner.save_test_results(results, args.output)
    
    # Print summary
    if isinstance(results, dict) and 'summary' in results:
        summary = results['summary']
        print(f"\n{'='*50}")
        print(f"Test Summary:")
        print(f"Total tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed_tests']}")
        print(f"Failed: {summary['failed_tests']}")
        print(f"Success rate: {summary['success_rate']:.1f}%")
        print(f"{'='*50}")
        
        # Exit with appropriate code
        exit_code = 0 if summary['overall_success'] else 1
    else:
        # Single test result
        success = results.get('success', False) if isinstance(results, dict) else False
        exit_code = 0 if success else 1
    
    sys.exit(exit_code)


if __name__ == '__main__':
    main()

"""Test runner script for the crossvals module.

This script provides a convenient way to run all tests with different configurations.
"""

import pytest
import sys
import os

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def run_all_tests():
    """Run all tests with verbose output."""
    return pytest.main([
        "tests/crossvals/",
        "-v",
        "--tb=short",
        "--strict-markers"
    ])


def run_fast_tests():
    """Run only fast tests (excluding slow performance tests)."""
    return pytest.main([
        "tests/crossvals/",
        "-v",
        "-m", "not slow",
        "--tb=short",
        "--strict-markers"
    ])


def run_with_coverage():
    """Run tests with coverage report."""
    return pytest.main([
        "tests/crossvals/",
        "--cov=tsforecast.crossvals",
        "--cov-report=html",
        "--cov-report=term",
        "-v",
        "--tb=short",
        "--strict-markers"
    ])


def run_specific_module(module_name):
    """Run tests for a specific module.
    
    Args:
        module_name (str): Name of the test module (e.g., 'test_base_classes')
    """
    return pytest.main([
        f"tests/crossvals/{module_name}.py",
        "-v",
        "--tb=short",
        "--strict-markers"
    ])


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run crossvals tests")
    parser.add_argument(
        "--mode",
        choices=["all", "fast", "coverage"],
        default="fast",
        help="Test mode to run"
    )
    parser.add_argument(
        "--module",
        help="Specific test module to run"
    )
    
    args = parser.parse_args()
    
    if args.module:
        exit_code = run_specific_module(args.module)
    elif args.mode == "all":
        exit_code = run_all_tests()
    elif args.mode == "fast":
        exit_code = run_fast_tests()
    elif args.mode == "coverage":
        exit_code = run_with_coverage()
    
    sys.exit(exit_code)
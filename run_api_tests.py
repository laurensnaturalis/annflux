#!/usr/bin/env python3
# Copyright 2025 Intel Corporation
# Copyright 2025 Naturalis Biodiversity Center
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Test runner script for AnnFlux REST API tests
Provides convenient commands for running different test suites
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description, env=None):
    """Run a command and handle the result"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, env=env)
        print(f"\n✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description} failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"\n❌ Command not found: {cmd[0]}")
        return False


def main():
    parser = argparse.ArgumentParser(description="AnnFlux REST API Test Runner")
    parser.add_argument(
        "command",
        choices=[
            "all", "unit", "integration", "api", "security", 
            "performance", "coverage", "quick", "ci"
        ],
        help="Test command to run"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Don't clean up test artifacts"
    )
    parser.add_argument(
        "--parallel", "-p",
        action="store_true",
        help="Run tests in parallel (requires pytest-xdist)"
    )
    
    args = parser.parse_args()
    
    # Base pytest command
    pytest_cmd = ["python", "-m", "pytest"]
    
    # Ensure PYTHONPATH includes src directory
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).parent / "src") + ":" + env.get("PYTHONPATH", "")
    
    if args.verbose:
        pytest_cmd.append("-vv")
    
    if args.parallel:
        pytest_cmd.extend(["-n", "auto"])
    
    # Change to project root
    project_root = Path(__file__).parent
    original_cwd = Path.cwd()
    
    try:
        os.chdir(project_root)
        
        success = True
        
        if args.command == "all":
            # Run all tests
            cmd = pytest_cmd + [
                "src/annflux/tests/test_rest_api.py",
                "src/annflux/tests/test_api_integration.py",
                "src/annflux/tests/test_api_utils.py",
                "-m", "not slow"
            ]
            success = run_command(cmd, "All API Tests", env)
        
        elif args.command == "unit":
            # Run unit tests only
            cmd = pytest_cmd + [
                "src/annflux/tests/test_rest_api.py",
                "-m", "unit and not slow"
            ]
            success = run_command(cmd, "Unit Tests", env)
        
        elif args.command == "integration":
            # Run integration tests
            cmd = pytest_cmd + [
                "src/annflux/tests/test_api_integration.py",
                "-m", "integration and not slow"
            ]
            success = run_command(cmd, "Integration Tests", env)
        
        elif args.command == "api":
            # Run all API-related tests
            cmd = pytest_cmd + [
                "src/annflux/tests/test_rest_api.py",
                "src/annflux/tests/test_api_integration.py",
                "-m", "api and not slow"
            ]
            success = run_command(cmd, "API Tests", env)
        
        elif args.command == "security":
            # Run security tests
            cmd = pytest_cmd + [
                "src/annflux/tests/test_api_integration.py::TestAPISecurity",
                "-m", "security"
            ]
            success = run_command(cmd, "Security Tests", env)
        
        elif args.command == "performance":
            # Run performance tests
            cmd = pytest_cmd + [
                "src/annflux/tests/test_api_integration.py::TestAPIPerformance",
                "-m", "performance",
                "-s"  # Don't capture output for performance tests
            ]
            success = run_command(cmd, "Performance Tests", env)
        
        elif args.command == "coverage":
            # Run tests with coverage
            cmd = pytest_cmd + [
                "src/annflux/tests/test_rest_api.py",
                "src/annflux/tests/test_api_integration.py",
                "--cov=annflux.ui.basic",
                "--cov-report=term-missing",
                "--cov-report=html:htmlcov",
                "--cov-report=xml"
            ]
            success = run_command(cmd, "Tests with Coverage", env)
            
            if success:
                print("\n📊 Coverage report generated:")
                print("  - Terminal: displayed above")
                print("  - HTML: htmlcov/index.html")
                print("  - XML: coverage.xml")
        
        elif args.command == "quick":
            # Run quick subset of tests (no coverage)
            quick_cmd = ["python", "-m", "pytest"]
            if args.verbose:
                quick_cmd.append("-vv")
            
            cmd = quick_cmd + [
                "src/annflux/tests/test_rest_api.py::TestRestAPI::test_root_endpoint",
                "src/annflux/tests/test_rest_api.py::TestRestAPI::test_data_endpoint",
                "src/annflux/tests/test_rest_api.py::TestRestAPI::test_label_endpoint",
                "--cov=no",
                "-v"
            ]
            success = run_command(cmd, "Quick Tests", env)
        
        elif args.command == "ci":
            # CI-friendly test suite
            cmd = pytest_cmd + [
                "src/annflux/tests/test_rest_api.py",
                "src/annflux/tests/test_api_integration.py",
                "--cov=annflux.ui.basic",
                "--cov-report=xml",
                "--cov-report=term",
                "--junit-xml=test-results.xml",
                "-m", "not slow and not performance"
            ]
            success = run_command(cmd, "CI Test Suite", env)
            
            if success:
                print("\n📋 CI artifacts generated:")
                print("  - Coverage: coverage.xml")
                print("  - Test results: test-results.xml")
        
        # Cleanup if requested
        if not args.no_cleanup and success:
            cleanup_cmd = ["find", ".", "-name", "*.pyc", "-delete"]
            subprocess.run(cleanup_cmd, capture_output=True)
            
            cleanup_cmd = ["find", ".", "-name", "__pycache__", "-type", "d", "-exec", "rm", "-rf", "{}", "+"]
            subprocess.run(cleanup_cmd, capture_output=True)
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
        
    finally:
        # Restore original working directory
        os.chdir(original_cwd)


if __name__ == "__main__":
    main()

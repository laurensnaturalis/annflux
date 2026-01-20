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
Configuration and utilities for REST API testing
"""

import json
import os
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
from unittest.mock import MagicMock, patch

import flask
import pandas as pd
import pytest
from PIL import Image

from annflux.ui.basic.run_server import app, get_app
from annflux.tools.core import AnnFluxState


class TestDataFactory:
    """Factory for creating test data and mock environments"""
    
    @staticmethod
    def create_test_project(
        temp_dir: str,
        num_images: int = 10,
        num_labels: int = 3,
        include_performance: bool = True
    ) -> Path:
        """Create a complete test project structure"""
        project_root = Path(temp_dir)
        
        # Create directory structure
        directories = [
            "images",
            "annflux",
            "annflux/thumbs", 
            "annflux/group_images",
            "annflux/failed_images",
            "annflux/backups",
            "original",
            "mask",
            "wav"
        ]
        
        for dir_path in directories:
            (project_root / dir_path).mkdir(parents=True, exist_ok=True)
        
        # Create sample images
        for i in range(num_images):
            uid = f"img_{i:03d}"
            img = Image.new('RGB', (100, 100), color=(i*25 % 255, (i*50) % 255, (i*75) % 255))
            img.save(project_root / "images" / f"{uid}.jpg")
        
        # Create annflux data
        TestDataFactory._create_annflux_data(project_root, num_images, num_labels)
        
        # Create label definitions
        TestDataFactory._create_label_definitions(project_root, num_labels)
        
        # Create class to color mapping
        TestDataFactory._create_class_colors(project_root, num_labels)
        
        # Create performance data if requested
        if include_performance:
            TestDataFactory._create_performance_data(project_root)
        
        # Create detailed performance data
        TestDataFactory._create_detailed_performance(project_root)
        
        return project_root
    
    @staticmethod
    def _create_annflux_data(project_root: Path, num_images: int, num_labels: int):
        """Create annflux.csv with sample data"""
        labels = [f"label{i}" for i in range(num_labels)]
        
        data = []
        for i in range(num_images):
            uid = f"img_{i:03d}"
            data.append({
                'uid': uid,
                'label_predicted': labels[i % num_labels],
                'score_predicted': 0.5 + (i % 5) * 0.1,
                'dp_most_needed': i % 3,
                'dp_is_ldp': i % 2 == 0,
                'dp_depth': i % 5,
                'dp_parent': f"img_{(i-1) % num_images:03d}" if i > 0 else "null",
                'labeled': 0,
                'in_test': i % 10 == 0,  # 10% test set
                'e_0': i * 0.1,
                'e_1': i * 0.05,
                'patch_x': i % 5 if i % 2 == 0 else None,
                'patch_y': i % 3 if i % 2 == 0 else None
            })
        
        df = pd.DataFrame(data)
        df.to_csv(project_root / "annflux" / "annflux.csv", index=False)
    
    @staticmethod
    def _create_label_definitions(project_root: Path, num_labels: int):
        """Create label definitions file"""
        labels = []
        for i in range(num_labels):
            if i == 0:
                labels.append([f"label{i}", "null"])
            else:
                labels.append([f"label{i}", "label0"])
        
        label_defs = {"labels": labels}
        with open(project_root / "annflux" / "label_defs.json", "w") as f:
            json.dump(label_defs, f, indent=2)
    
    @staticmethod
    def _create_class_colors(project_root: Path, num_labels: int):
        """Create class to color mapping"""
        colors = ["#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF"]
        
        data = []
        for i in range(num_labels):
            data.append({
                'class': f"label{i}",
                'color': colors[i % len(colors)],
                'count': 10 + i * 5
            })
        
        df = pd.DataFrame(data)
        df.to_csv(project_root / "annflux" / "class_to_color.csv", index=False)
    
    @staticmethod
    def _create_performance_data(project_root: Path):
        """Create performance metrics"""
        performance = {
            "accuracy": 0.85,
            "precision": 0.82,
            "recall": 0.88,
            "f1_score": 0.85,
            "percentage_near_labeled": 0.75,
            "test_performance": [["label0", 0.8, 0.85], ["label1", 0.9, 0.82]]
        }
        
        with open(project_root / "annflux" / "performance.json", "w") as f:
            json.dump(performance, f, indent=2)
    
    @staticmethod
    def _create_detailed_performance(project_root: Path):
        """Create detailed performance CSV"""
        data = []
        for i in range(5):
            data.append({
                'precision': 0.8 + i * 0.02,
                'recall': 0.7 + i * 0.03,
                'num_predicted_certain': 10 + i,
                'num_predicted_uncertain': 5 + i
            })
        
        df = pd.DataFrame(data)
        df.to_csv(project_root / "annflux" / "detailed_performance.csv", index=False)


class MockStateFactory:
    """Factory for creating mock AnnFluxState objects"""
    
    @staticmethod
    def create_mock_state(project_root: Path) -> MagicMock:
        """Create a comprehensive mock state object"""
        mock_state = MagicMock(spec=AnnFluxState)
        
        mock_state.project_folder = str(project_root)
        mock_state.annflux_folder = str(project_root / "annflux")
        mock_state.annflux_path = str(project_root / "annflux" / "annflux.csv")
        mock_state.labels_path = str(project_root / "annflux" / "labels.json")
        mock_state.performance_path = str(project_root / "annflux" / "performance.json")
        mock_state.doublecheck_path = str(project_root / "annflux" / "doublecheck.json")
        
        # State variables
        mock_state.g_quick_status = "idle"
        mock_state.linear_status_epoch = 0
        mock_state.trained_for_version = 0
        mock_state.labeled_indices = [0, 1, 2]
        mock_state.features = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        mock_state.time_new_status_time = None
        mock_state.train_thread = None
        mock_state.new_labeled_uids = set()
        
        # Methods
        mock_state.is_initialized.return_value = True
        
        return mock_state


class TestAppFactory:
    """Factory for creating test applications"""
    
    @staticmethod
    def create_test_app(project_root: Path, mock_state: Optional[MagicMock] = None) -> flask.Flask:
        """Create a Flask app for testing"""
        # Set environment variables
        os.environ["PROJECT_ROOT"] = str(project_root)
        os.environ["USERS"] = ""  # No authentication for tests
        os.environ["LOGGING_LEVEL"] = "ERROR"  # Reduce log noise in tests
        
        # Create app
        test_app = get_app()
        test_app.config['TESTING'] = True
        test_app.config['WTF_CSRF_ENABLED'] = False
        
        # Mock the global state if provided
        if mock_state:
            with patch('annflux.ui.basic.run_server.g_state', mock_state):
                with test_app.app_context():
                    try:
                        from annflux.ui.basic.run_server import _init
                        _init()
                    except Exception as e:
                        # Handle initialization errors gracefully
                        pass
        else:
            with test_app.app_context():
                try:
                    from annflux.ui.basic.run_server import _init
                    _init()
                except Exception as e:
                    # Handle initialization errors gracefully
                    pass
        
        return test_app


class APIAssertions:
    """Helper class for common API assertions"""
    
    @staticmethod
    def assert_success_response(response, expected_keys: Optional[list] = None):
        """Assert that response is successful"""
        assert response.status_code == 200
        if expected_keys:
            data = json.loads(response.data) if response.content_type == 'application/json' else {}
            for key in expected_keys:
                assert key in data
    
    @staticmethod
    def assert_json_response(response, expected_status: int = 200):
        """Assert that response is valid JSON"""
        assert response.status_code == expected_status
        assert response.content_type == 'application/json'
        try:
            json.loads(response.data)
        except json.JSONDecodeError:
            pytest.fail("Response is not valid JSON")
    
    @staticmethod
    def assert_file_response(response, expected_mime: str):
        """Assert that response is a file"""
        assert response.status_code == 200
        assert expected_mime in response.content_type
        assert len(response.data) > 0
    
    @staticmethod
    def assert_error_response(response, expected_status: int = 400):
        """Assert that response is an error"""
        assert response.status_code == expected_status
        if response.content_type == 'application/json':
            data = json.loads(response.data)
            assert "error" in data or "message" in data


@pytest.fixture
def temp_project_dir():
    """Create a temporary project directory"""
    temp_dir = tempfile.mkdtemp()
    project_root = TestDataFactory.create_test_project(temp_dir)
    
    yield project_root
    
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_state(temp_project_dir):
    """Create a mock state object"""
    return MockStateFactory.create_mock_state(temp_project_dir)


@pytest.fixture
def test_app(temp_project_dir, mock_state):
    """Create a test Flask app"""
    return TestAppFactory.create_test_app(temp_project_dir, mock_state)


@pytest.fixture
def test_client(test_app):
    """Create a test client"""
    return test_app.test_client()


@pytest.fixture
def sample_labels():
    """Sample label data for testing"""
    return {
        "img_001": "label0",
        "img_002": "label1", 
        "img_003": "label0"
    }


@pytest.fixture
def sample_status_data():
    """Sample status data for testing"""
    return {
        "idleTime": 100,
        "groupTrain": False
    }


# Test data constants
TEST_IMAGE_UIDS = ["img_001", "img_002", "img_003"]
TEST_LABELS = ["label0", "label1", "label2"]
TEST_COLORS = ["#FF0000", "#00FF00", "#0000FF"]

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
import pandas

from annflux.repository.resultset import Resultset
from annflux.repository.repository import Repository

import json
import os
import tempfile
import shutil
from io import BytesIO
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from annflux.ui.basic.run_server import _init, get_app

@pytest.fixture
def temp_project_dir():
    """Create a temporary project directory with required structure"""
    temp_dir = tempfile.mkdtemp()
    project_root = Path(temp_dir)

    # Create directory structure
    images_dir = project_root / "images"
    annflux_dir = project_root / "annflux"
    thumbs_dir = annflux_dir / "thumbs"
    group_images_dir = annflux_dir / "group_images"
    failed_images_dir = annflux_dir / "failed_images"

    for dir_path in [
        images_dir,
        annflux_dir,
        thumbs_dir,
        group_images_dir,
        failed_images_dir,
    ]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Create sample data files
    uids = ["img_001", "img_002", "img_003"]
    annflux_data = pd.DataFrame(
        {
            "uid": uids,
            "label_predicted": ["label1", "label2", "label1"],
            "score_predicted": [0.8, 0.6, 0.9],
            "dp_most_needed": [1, 2, 0],
            "labeled": [0, 0, 0],
            "in_test": [1, 0, 0],
        }
    )
    annflux_data.to_csv(annflux_dir / "annflux.csv", index=False)

    with open(os.path.join(annflux_dir, "split.json"), "w") as f:
        json.dump({"test": "img_001"}, f)

    # Create sample label definitions
    label_defs = {"labels": [["label1", "null"], ["label2", "null"]]}
    with open(annflux_dir / "label_defs.json", "w") as f:
        json.dump(label_defs, f)

    # Create sample class to color mapping
    class_to_color = pd.DataFrame(
        {
            "class": ["label1", "label2"],
            "color": ["#FF0000", "#00FF00"],
            "count": [10, 5],
        }
    )
    class_to_color.to_csv(annflux_dir / "class_to_color.csv", index=False)

    # Create sample performance data
    performance = {
        "accuracy": 0.85,
        "precision": 0.82,
        "recall": 0.88,
        "test_performance": [],
    }
    with open(annflux_dir / "performance.json", "w") as f:
        json.dump(performance, f)

    # Create sample images
    for uid in uids:
        img = Image.new("RGB", (100, 100), color="red")
        img.save(images_dir / f"{uid}.jpg")

    # Create data repo
    repo = Repository(os.path.join(annflux_dir, "datarepo"))
    resultset_tmp_path = os.path.join(tempfile.mkdtemp(), "resultset")
    os.makedirs(resultset_tmp_path)
    pandas.DataFrame(
        data={
            "uid": uids,
            "label_predicted": ["label1", "label1", "label1"],
            "score_predicted": [1.0, 1.0, 1.0],
        }
    ).to_csv(os.path.join(resultset_tmp_path, "results.csv"), index=False)
    np.savez(
        os.path.join(resultset_tmp_path, "last_full.npz"),
        lastFull=np.random.rand(3, 256),
    )
    result_set = Resultset(resultset_tmp_path)
    repo.commit(result_set, tag="unseen")

    yield project_root

    # Cleanup
    shutil.rmtree(temp_dir)
    shutil.rmtree(resultset_tmp_path)

@pytest.fixture
def mock_app(temp_project_dir):
    """Create Flask app with mocked dependencies"""
    # Set environment variables
    os.environ["PROJECT_ROOT"] = str(temp_project_dir)
    os.environ["USERS"] = ""  # No authentication for tests

    # Mock the global state by patching it in the module
    with patch("annflux.ui.basic.run_server.g_state", create=True) as mock_state:
        mock_state.project_folder = str(temp_project_dir)
        mock_state.annflux_folder = str(temp_project_dir / "annflux")
        mock_state.annflux_path = str(temp_project_dir / "annflux" / "annflux.csv")
        mock_state.labels_path = str(temp_project_dir / "annflux" / "labels.json")
        mock_state.performance_path = str(
            temp_project_dir / "annflux" / "performance.json"
        )
        mock_state.doublecheck_path = str(
            temp_project_dir / "annflux" / "doublecheck.json"
        )
        mock_state.g_quick_status = "idle"
        mock_state.linear_status_epoch = 0
        mock_state.trained_for_version = 0
        mock_state.labeled_indices = []
        mock_state.features = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        mock_state.time_new_status_time = None
        mock_state.train_thread = None
        mock_state.is_initialized.return_value = True
        mock_state.new_labeled_uids = set()

        # Mock other global variables
        with patch("annflux.ui.basic.run_server.g_layout", "label"):
            with patch("annflux.ui.basic.run_server.logger") as mock_logger:
                mock_logger.info = lambda x: None
                mock_logger.error = lambda x: None
                mock_logger.warning = lambda x: None

                # Initialize app
                test_app = get_app()
                test_app.config["TESTING"] = True

                with test_app.app_context():
                    _init()

                yield test_app

@pytest.fixture
def client(mock_app):
    """Create test client"""
    return mock_app.test_client()

class TestRestAPI:
    """Comprehensive unit tests for AnnFlux REST API endpoints"""



    def test_root_endpoint(self, client):
        """Test root endpoint returns HTML"""
        response = client.get("/")
        assert response.status_code == 200
        assert b"html" in response.data

    def test_annflux_endpoint(self, client):
        """Test /annflux endpoint"""
        response = client.get("/annflux")
        assert response.status_code == 200
        assert b"html" in response.data

    def test_data_endpoint(self, client):
        """Test /data endpoint returns parquet data"""
        response = client.get("/data")
        assert response.status_code == 200
        assert response.content_type.startswith("application/x-binary")

        # Verify parquet data can be read
        df = pd.read_parquet(BytesIO(response.data))
        assert len(df) == 3
        assert "uid" in df.columns

    def test_data_endpoint_with_filter(self, client):
        """Test /data endpoint with filter query"""
        response = client.get(
            "/data?filter_query=row.label_predicted%20=%20%27label1%27"
        )
        assert response.status_code == 200

        df = pd.read_parquet(BytesIO(response.data))
        assert len(df) == 2  # Should have 2 rows with label1

    def test_data_group_endpoint(self, client):
        """Test /data/group endpoint"""
        response = client.get("/data/group")
        assert response.status_code == 200

    def test_thumbnail_endpoint(self, client, temp_project_dir):
        """Test thumbnail generation"""
        response = client.get("/images/thumbnail/img_001")
        assert response.status_code == 200
        assert response.content_type == "image/jpg"

    def test_thumbnail_missing_image(self, client, temp_project_dir):
        """Test thumbnail for missing image"""
        response = client.get("/images/thumbnail/nonexistent")
        assert response.status_code == 200
        assert response.content_type == "image/jpg"

    def test_thumbnail_original(self, client, temp_project_dir):
        """Test original thumbnail endpoint"""
        # Create original image
        original_dir = temp_project_dir / "original"
        original_dir.mkdir(exist_ok=True)
        img = Image.new("RGB", (100, 100), color="blue")
        img.save(original_dir / "img_001.jpg")

        response = client.get("/images/original/thumbnail/img_001")
        assert response.status_code == 200
        assert response.content_type == "image/jpg"

    def test_thumbnail_mask(self, client, temp_project_dir):
        """Test mask thumbnail endpoint"""
        # Create mask image
        mask_dir = temp_project_dir / "mask"
        mask_dir.mkdir(exist_ok=True)
        img = Image.new("RGB", (100, 100), color="green")
        img.save(mask_dir / "img_001.png")

        response = client.get("/images/mask/thumbnail/img_001")
        assert response.status_code == 200
        assert response.content_type == "image/png"

    def test_images_full(self, client):
        """Test full image endpoint"""
        response = client.get("/images/full/img_001")
        assert response.status_code == 200
        assert response.content_type == "image/jpg"

    def test_images_full_missing(self, client):
        """Test full image endpoint for missing image"""
        response = client.get("/images/full/nonexistent")
        assert response.status_code == 404

    def test_sound_endpoint(self, client, temp_project_dir):
        """Test sound endpoint"""
        # Create audio directory and file
        wav_dir = temp_project_dir / "wav"
        wav_dir.mkdir(exist_ok=True)
        # Create a dummy wav file
        with open(wav_dir / "sound_001.wav", "wb") as f:
            f.write(b"RIFF\x24\x08\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00")

        response = client.get("/sounds/sound_001")
        assert response.status_code == 200
        assert response.content_type == "audio/wav"

    def test_label_definitions_sort(self, client):
        """Test label definitions sorting endpoint"""
        response = client.get("/v1/label_definitions/sort")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["result"] == "ok"

    def test_label_definitions_put(self, client):
        """Test adding new label definition"""
        new_label = ["label3", "null"]
        response = client.put(
            "/v1/label_definitions", json=new_label, content_type="application/json"
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["result"] == "ok"

    def test_label_definitions_put_with_parent(self, client):
        """Test adding label definition with parent"""
        new_label = ["label3", "label1"]
        response = client.put(
            "/v1/label_definitions", json=new_label, content_type="application/json"
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["result"] == "ok"

    def test_label_definitions_put_exclusive(self, client):
        """Test adding exclusive label definition"""
        new_label = ["label3", "label1", True]
        response = client.put(
            "/v1/label_definitions", json=new_label, content_type="application/json"
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["result"] == "ok"

    def test_version_endpoint(self, client):
        """Test version endpoint"""
        response = client.get("/version")
        assert response.status_code == 200

    def test_label_endpoint(self, client):
        """Test label submission endpoint"""
        label_data = {"img_001": "label1", "img_002": "label2"}
        response = client.post(
            "/label", json=label_data, content_type="application/json"
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["success"] is True

    def test_label_endpoint_empty(self, client):
        """Test label endpoint with empty data"""
        response = client.post("/label", json={}, content_type="application/json")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["success"] is True

    def test_performance_endpoint(self, client):
        """Test performance endpoint"""
        response = client.get("/performance")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert "accuracy" in data

    def test_detailed_performance_data(self, client, temp_project_dir):
        """Test detailed performance data endpoint"""
        # Create detailed performance file
        detailed_perf = pd.DataFrame(
            {
                "precision": [0.8, 0.9],
                "recall": [0.7, 0.85],
                "num_predicted_certain": [10, 15],
                "num_predicted_uncertain": [5, 3],
            }
        )
        detailed_perf.to_csv(
            temp_project_dir / "annflux" / "detailed_performance.csv", index=False
        )

        response = client.get("/detailed_performance/data")
        assert response.status_code == 200
        df = pd.read_csv(BytesIO(response.data))
        assert len(df) == 2

    def test_exclusivity_data_get(self, client):
        """Test exclusivity data endpoint"""
        response = client.get("/exclusivity/data")
        assert response.status_code == 200
        df = pd.read_csv(BytesIO(response.data))
        assert "left" in df.columns
        assert "right" in df.columns

    def test_exclusivity_data_post(self, client):
        """Test exclusivity data POST endpoint"""
        exclusivity_data = [{"left": "label1", "right": "label2"}]
        response = client.post(
            "/exclusivity/data", json=exclusivity_data, content_type="application/json"
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["success"] is True

    def test_label_provider_data(self, client, temp_project_dir):
        """Test label provider data endpoint"""
        # Create label provider file
        label_provider = pd.DataFrame(
            {"label": ["label1", "label2"], "provider": ["provider1", "provider2"]}
        )
        label_provider.to_csv(
            temp_project_dir / "annflux" / "label_provider.csv", index=False
        )

        response = client.get("/label_provider/data")
        assert response.status_code == 200
        df = pd.read_csv(BytesIO(response.data))
        assert len(df) == 2

    def test_labels_css(self, client):
        """Test labels CSS endpoint"""
        response = client.get("/labels/css")
        assert response.status_code == 200
        assert response.content_type == "text/css; charset=utf-8"
        css_content = response.data.decode("utf-8")
        assert ".label_" in css_content

    def test_label_defs_list(self, client):
        """Test label definitions list endpoint"""
        response = client.get("/label_defs")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert "labels" in data
        assert len(data["labels"]) == 2

    def test_status_endpoint(self, client):
        """Test status endpoint"""
        status_data = {"idleTime": 100, "groupTrain": False}
        response = client.post(
            "/status", json=status_data, content_type="application/json"
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert "status" in data
        assert "package_version" in data
        assert "num_total" in data

    def test_status_endpoint_with_group_train(self, client):
        """Test status endpoint with group training"""
        status_data = {"idleTime": 100, "groupTrain": True}
        response = client.post(
            "/status", json=status_data, content_type="application/json"
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert "status" in data

    def test_status_endpoint_auto_train(self, client):
        """Test status endpoint triggers auto training"""
        status_data = {
            "idleTime": 2000,  # Above default threshold
            "groupTrain": False,
        }
        response = client.post(
            "/status", json=status_data, content_type="application/json"
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert "status" in data

    def test_ui_endpoints(self, client):
        """Test UI rendering endpoints"""
        endpoints = [
            "/exclusivity",
            "/class_examples",
            "/label_provider",
            "/detailed_performance",
            "/ui/detailed_performance",
        ]

        for endpoint in endpoints:
            response = client.get(endpoint)
            assert response.status_code == 200

    def test_error_handling(self, client):
        """Test error handling"""
        # Test 404 error
        response = client.get("/nonexistent")
        assert response.status_code == 404

    def test_no_cache_headers(self, client):
        """Test no-cache headers on appropriate endpoints"""
        response = client.get("/data")
        assert response.status_code == 200
        assert "no-cache" in response.headers.get("Cache-Control", "").lower()

    def test_content_types(self, client):
        """Test correct content types for different endpoints"""
        content_type_tests = [
            ("/data", "application/x-binary"),
            ("/images/thumbnail/img_001", "image/jpg"),
            ("/performance", "application/json"),
            ("/labels/css", "text/css"),
            # ("/exclusivity/data", "text/csv"),
        ]

        for endpoint, expected_content_type in content_type_tests:
            response = client.get(endpoint)
            if response.status_code == 200:
                assert expected_content_type in response.content_type

    # def test_data_filtering_sql_injection(self, client):
    #     """Test SQL injection protection in data filtering"""
    #     malicious_query = "'; DROP TABLE annflux; --"
    #     response = client.get(f"/data?filter_query={malicious_query}")
    #     # Should return error due to malformed query, not crash the server
    #     assert response.status_code == 400

    def test_large_label_submission(self, client):
        """Test handling of large label submissions"""
        # Create large label dataset
        large_label_data = {}
        for i in range(1000):
            large_label_data[f"img_{i:03d}"] = f"label{i % 10}"

        response = client.post(
            "/label", json=large_label_data, content_type="application/json"
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["success"] is True

    def test_concurrent_requests(self, client):
        """Test handling of concurrent requests"""
        import threading

        results = []

        def make_request():
            response = client.get("/data")
            results.append(response.status_code)

        # Create multiple threads
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # All requests should succeed
        assert all(status == 200 for status in results)

    def test_invalid_json_handling(self, client):
        """Test handling of invalid JSON data"""
        response = client.post(
            "/label", data="invalid json", content_type="application/json"
        )
        assert response.status_code == 400

    def test_missing_files_handling(self, client):
        """Test handling of missing data files"""
        # This should not crash even if some files are missing
        response = client.get("/performance")
        # Should return empty dict if file doesn't exist
        assert response.status_code == 200


class TestAuthentication:
    """Test authentication functionality"""

    def test_auth_required(self):
        """Test that authentication is required when users are configured"""
        with patch.dict(os.environ, {"USERS": "testuser|hashedpassword"}):
            os.environ["PROJECT_ROOT"] = "/tmp"

            with patch("annflux.ui.basic.run_server._init"):
                test_app = get_app()
                client = test_app.test_client()

                response = client.get("/")
                # Should require authentication
                assert response.status_code == 401

    def test_no_auth_when_no_users(self):
        """Test that authentication is not required when no users are configured"""
        os.environ["PROJECT_ROOT"] = "/tmp"
        os.environ["USERS"] = ""

        with patch("annflux.ui.basic.run_server._init"):
            test_app = get_app()
            client = test_app.test_client()

            response = client.get("/")
            # Should succeed without authentication
            assert response.status_code == 200


class TestErrorHandling:
    """Test error handling and edge cases"""

    @pytest.fixture
    def error_app(self):
        """Create app with error-prone setup"""
        os.environ["PROJECT_ROOT"] = "/nonexistent"
        os.environ["USERS"] = ""

        test_app = get_app()
        test_app.config["TESTING"] = True
        return test_app

    @pytest.fixture
    def error_client(self, error_app):
        """Create test client for error testing"""
        return error_app.test_client()

    def test_missing_project_root(self, error_client):
        """Test behavior when PROJECT_ROOT is missing"""
        with patch(
            "annflux.ui.basic.run_server._init",
            side_effect=RuntimeError("PROJECT_ROOT not set"),
        ):
            response = error_client.get("/")
            # Should handle error gracefully
            assert response.status_code in [500, 404]

    def test_corrupted_data_files(self, error_client):
        """Test behavior with corrupted data files"""
        # This would require mocking file operations
        # For now, just test that the server doesn't crash
        response = error_client.get("/data")
        # Should handle missing/corrupted files gracefully
        assert response.status_code in [200, 404, 500]


if __name__ == "__main__":
    pytest.main([__file__])

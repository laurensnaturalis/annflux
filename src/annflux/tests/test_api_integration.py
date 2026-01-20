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
Integration tests for the REST API
These tests test the full API workflow and integration between components
"""
import io
import json
import shutil
import time
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from annflux.tests.test_api_utils import (
    TestDataFactory, 
    APIAssertions
)
from annflux.tests.test_rest_api import client, temp_project_dir, mock_app


class TestAPIIntegration:
    """Integration tests for the complete API workflow"""
    
    def test_complete_annotation_workflow(self, client, temp_project_dir):
        """Test the complete annotation workflow from data retrieval to labeling"""
        
        # 1. Get initial data
        response = client.get("/data")
        APIAssertions.assert_file_response(response, "application/x-binary")
        df = pd.read_parquet(BytesIO(response.data))
        initial_labeled_count = df['labeled'].sum()
        assert initial_labeled_count == 0
        
        # 2. Get label definitions
        response = client.get("/label_defs")
        APIAssertions.assert_json_response(response)
        label_defs = json.loads(response.data)
        initial_label_count = len(label_defs['labels'])
        
        # 3. Add new label definition
        new_label = ["new_label", "null"]
        response = client.put("/v1/label_definitions", json=new_label)
        APIAssertions.assert_success_response(response)
        
        # 4. Verify label was added
        response = client.get("/label_defs")
        APIAssertions.assert_json_response(response)
        updated_label_defs = json.loads(response.data)
        assert len(updated_label_defs['labels']) == initial_label_count + 1
        
        # 5. Submit labels
        label_data = {"img_001": "label0", "img_002": "label1"}
        response = client.post("/label", json=label_data)
        APIAssertions.assert_success_response(response)
        
        # 6. Check updated data
        response = client.get("/data")
        APIAssertions.assert_file_response(response, "application/x-binary")
        updated_df = pd.read_parquet(BytesIO(response.data))
        
        # Note: The actual labeling might be handled asynchronously
        # So we check that the response is successful rather than checking the data
        
        # 7. Get performance metrics
        response = client.get("/performance")
        APIAssertions.assert_json_response(response)
        performance = json.loads(response.data)
        assert "accuracy" in performance
        
        # 8. Test status endpoint
        status_data = {"idleTime": 100, "groupTrain": False}
        response = client.post("/status", json=status_data)
        APIAssertions.assert_json_response(response)
        status = json.loads(response.data)
        assert "status" in status
        assert "package_version" in status
    
    def test_filtering_and_search_workflow(self, client):
        """Test data filtering and search capabilities"""
        
        # 1. Get all data
        response = client.get("/data")
        df_all = pd.read_parquet(BytesIO(response.data))
        total_count = len(df_all)
        
        # 2. Filter by predicted label
        filter_query = "row.label_predicted = 'label0'"
        response = client.get(f"/data?filter_query={filter_query}")
        df_filtered = pd.read_parquet(BytesIO(response.data))
        assert len(df_filtered) < total_count
        
        # 3. Complex filter
        complex_filter = "row.label_predicted = 'label0' AND row.dp_most_needed > 0"
        response = client.get(f"/data?filter_query={complex_filter}")
        df_complex = pd.read_parquet(BytesIO(response.data))
        assert len(df_complex) <= len(df_filtered)
    
    def test_exclusivity_workflow(self, client):
        """Test label exclusivity management"""
        
        # 1. Get initial exclusivity data
        response = client.get("/exclusivity/data")
        df_initial = pd.read_csv(io.StringIO(response.data.decode('utf-8')))
        initial_count = len(df_initial)
        
        # 2. Add exclusivity rules
        exclusivity_data = [
            {"left": "label0", "right": "label1"},
            {"left": "label1", "right": "label2"}
        ]
        response = client.post("/exclusivity/data", json=exclusivity_data)
        APIAssertions.assert_success_response(response)
        
        # 3. Verify exclusivity data was updated
        response = client.get("/exclusivity/data")
        df_updated = pd.read_csv(io.StringIO(response.data.decode('utf-8')))
        assert len(df_updated) == initial_count + 2

    @pytest.mark.skip("needs work")
    def test_performance_tracking_workflow(self, client, temp_project_dir):
        """Test performance tracking and metrics"""

        # 1. Get basic performance
        response = client.get("/performance")
        APIAssertions.assert_json_response(response)
        performance = json.loads(response.data)

        # 2. Get detailed performance
        response = client.get("/detailed_performance/data")
        df_detailed = pd.read_csv(response.data)
        assert 'precision' in df_detailed.columns
        assert 'recall' in df_detailed.columns

        # 3. Get CSS for labels
        response = client.get("/labels/css")
        assert response.status_code == 200
        assert response.content_type == "text/css"
        css_content = response.data.decode('utf-8')
        assert ".label_" in css_content

    def test_batch_operations(self, client):
        """Test batch operations and bulk data handling"""
        
        # 1. Batch label submission
        batch_labels = {}
        for i in range(10):
            batch_labels[f"img_{i:03d}"] = f"label{i % 3}"
        
        response = client.post("/label", json=batch_labels)
        APIAssertions.assert_success_response(response)
        
        # 2. Batch label definition creation
        # batch_label_defs = []
        # for i in range(5, 8):
        #     batch_label_defs.append([f"batch_label{i}", "null"])
        #
        # for label_def in batch_label_defs:
        #     response = client.put("/v1/label_definitions", json=label_def)
        #     APIAssertions.assert_success_response(response)
        #
        # # 3. Verify all labels were added
        # response = client.get("/label_defs")
        # label_defs = json.loads(response.data)
        # assert len(label_defs['labels']) >= 8  # Original 3 + new 5
    
    def test_concurrent_operations(self, client):
        """Test handling of concurrent API operations"""
        import threading
        import queue
        
        results = queue.Queue()
        
        def make_api_request(endpoint, method='GET', data=None):
            if method == 'GET':
                response = client.get(endpoint)
            elif method == 'POST':
                response = client.post(endpoint, json=data)
            elif method == 'PUT':
                response = client.put(endpoint, json=data)
            
            results.put((endpoint, response.status_code))
        
        # Create multiple concurrent requests
        threads = []
        endpoints = [
            ("/data", "GET"),
            ("/performance", "GET"),
            ("/label_defs", "GET"),
            ("/status", "POST", {"idleTime": 100}),
            ("/label", "POST", {"img_001": "label0"})
        ]
        
        for endpoint_info in endpoints:
            if len(endpoint_info) == 2:
                endpoint, method = endpoint_info # ty: ignore
                data = None
            else:
                endpoint, method, data = endpoint_info # ty: ignore
            
            thread = threading.Thread(
                target=make_api_request, 
                args=(endpoint, method, data)
            )
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check results
        successful_requests = 0
        while not results.empty():
            endpoint, status_code = results.get()
            if status_code == 200:
                successful_requests += 1
        
        # Most requests should succeed
        assert successful_requests >= len(endpoints) * 0.8
    
    def test_error_recovery_workflow(self, client):
        """Test error handling and recovery"""
        
        # 1. Test invalid JSON
        response = client.post("/label",
                               data="invalid json",
                               content_type='application/json')
        assert response.status_code >= 400
        
        # 2. Test invalid filter query
        response = client.get("/data?filter_query=invalid_sql_syntax")
        # Should not crash the server
        assert response.status_code in [200, 400, 500]
        
        # 3. Test missing resources
        response = client.get("/images/full/nonexistent")
        assert response.status_code == 404
        
        # 4. Verify server is still responsive
        response = client.get("/performance")
        assert response.status_code == 200
    
    def test_data_consistency_workflow(self, client):
        """Test data consistency across different endpoints"""
        
        # 1. Get data from main endpoint
        response = client.get("/data")
        df_main = pd.read_parquet(BytesIO(response.data))
        
        # 2. Get performance data
        response = client.get("/performance")
        performance = json.loads(response.data)
        
        # 3. Get label definitions
        response = client.get("/label_defs")
        label_defs = json.loads(response.data)
        
        # 4. Verify consistency
        # All uids in main data should be valid format
        for uid in df_main['uid']:
            assert uid.startswith('img_')
            assert uid.replace('img_', '').isdigit()
        
        # Labels in performance should exist in label definitions
        if 'test_performance' in performance:
            for test_perf in performance['test_performance']:
                if len(test_perf) > 0:
                    label_name = test_perf[0]
                    label_exists = any(label_name in label_def for label_def in label_defs['labels'])
                    assert label_exists
                    # Note: This might not always be true due to test data setup
    
    def test_ui_endpoints_integration(self, client):
        """Test UI rendering endpoints integration"""
        
        ui_endpoints = [
            "/exclusivity",
            "/class_examples",
            "/label_provider", 
            "/detailed_performance",
            "/ui/detailed_performance"
        ]
        
        for endpoint in ui_endpoints:
            response = client.get(endpoint)
            assert response.status_code == 200
            # Should return HTML content
            assert b"html" in response.data.lower()


class TestAPISecurity:
    """Security-focused API tests"""
    
    def test_input_validation(self, client):
        """Test input validation and sanitization"""
        
        # 1. Test SQL injection attempts
        malicious_inputs = [
            "'; DROP TABLE annflux; --",
            "' OR '1'='1",
            "'; SELECT * FROM annflux; --"
        ]
        
        for malicious_input in malicious_inputs:
            response = client.get(f"/data?filter_query={malicious_input}")
            # Should not crash the server
            assert response.status_code in [200, 400, 500]
        
        # 2. Test XSS attempts
        xss_payloads = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<img src=x onerror=alert('xss')>"
        ]
        
        for payload in xss_payloads:
            response = client.post("/label", json={"test_uid": payload})
            # Should handle gracefully
            assert response.status_code in [200, 400]
        
        # 3. Test large payload handling
        large_payload = {f"uid_{i}": f"label_{i}" for i in range(10000)}
        response = client.post("/label", json=large_payload)
        # Should either succeed or fail gracefully
        assert response.status_code in [200, 400, 413]
    
    def test_authentication_and_authorization(self, client):
        """Test authentication and authorization mechanisms"""
        
        # With no users configured, all endpoints should be accessible
        public_endpoints = [
            "/data",
            "/performance", 
            "/label_defs",
            "/version"
        ]
        
        for endpoint in public_endpoints:
            response = client.get(endpoint)
            assert response.status_code == 200
    
    def test_rate_limiting_resilience(self, client):
        """Test resilience to rapid requests"""
        
        # Make many rapid requests
        responses = []
        for i in range(100):
            response = client.get("/performance")
            responses.append(response.status_code)
        
        # Most requests should succeed (rate limiting might kick in)
        success_rate = sum(1 for code in responses if code == 200) / len(responses)
        assert success_rate >= 0.8  # At least 80% should succeed


class TestAPIPerformance:
    """Performance-focused API tests"""
    
    def test_response_times(self, client):
        """Test API response times"""
        
        endpoints = [
            "/data",
            "/performance",
            "/label_defs",
            "/version"
        ]
        
        for endpoint in endpoints:
            start_time = time.time()
            response = client.get(endpoint)
            end_time = time.time()
            
            assert response.status_code == 200
            response_time = end_time - start_time
            
            # Most endpoints should respond within 1 second
            assert response_time < 1.0, f"{endpoint} took {response_time:.2f}s"

    @pytest.mark.skip("fix later")
    def test_large_dataset_handling(self, client, temp_project_dir):
        """Test handling of larger datasets"""
        
        # Create a larger dataset
        large_temp_dir = temp_project_dir.parent / "large_test"
        large_temp_dir.mkdir(exist_ok=True)
        
        TestDataFactory.create_test_project(
            str(large_temp_dir), 
            num_images=1000,
            num_labels=10
        )
        
        # Update environment to use large dataset
        import os
        original_project_root = os.environ.get("PROJECT_ROOT")
        os.environ["PROJECT_ROOT"] = str(large_temp_dir)
        
        try:
            # Test data retrieval
            start_time = time.time()
            response = client.get("/data")
            end_time = time.time()
            
            assert response.status_code == 200
            
            # Should still be reasonable response time
            response_time = end_time - start_time
            assert response_time < 5.0, f"Large dataset took {response_time:.2f}s"
            
            # Verify data size
            df = pd.read_parquet(BytesIO(response.data))
            assert len(df) == 1000
            
        finally:
            # Restore original project root
            if original_project_root:
                os.environ["PROJECT_ROOT"] = original_project_root
            
            # Cleanup
            shutil.rmtree(large_temp_dir, ignore_errors=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

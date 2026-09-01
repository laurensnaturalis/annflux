# AnnFlux REST API Test Suite

This directory contains comprehensive unit tests for the AnnFlux REST API, providing coverage for all endpoints, error handling, security, and performance scenarios.

## Test Structure

### Test Files

- **`test_rest_api.py`** - Comprehensive unit tests for all REST API endpoints
- **`test_api_integration.py`** - Integration tests and workflow tests
- **`test_api_utils.py`** - Test utilities, factories, and helper classes
- **`test_server.py`** - Existing server tests (legacy)

### Test Categories

- **Unit Tests**: Individual endpoint testing with mocked dependencies
- **Integration Tests**: Full workflow testing and component interaction
- **Security Tests**: Input validation, authentication, and vulnerability testing
- **Performance Tests**: Response time and load testing

## Running Tests

### Quick Start

```bash
# Run all API tests
python run_api_tests.py all

# Run quick subset for development
python run_api_tests.py quick

# Run with coverage
python run_api_tests.py coverage
```

### Test Commands

| Command | Description |
|---------|-------------|
| `all` | Run all API tests (excluding slow tests) |
| `unit` | Run unit tests only |
| `integration` | Run integration tests only |
| `api` | Run all API-related tests |
| `security` | Run security-focused tests |
| `performance` | Run performance tests |
| `coverage` | Run tests with coverage reporting |
| `quick` | Run quick subset for development |
| `ci` | CI-friendly test suite with XML output |

### Options

- `--verbose, -v`: Verbose output
- `--no-cleanup`: Don't clean up test artifacts
- `--parallel, -p`: Run tests in parallel (requires pytest-xdist)

### Examples

```bash
# Run with verbose output
python run_api_tests.py unit --verbose

# Run tests in parallel
python run_api_tests.py all --parallel

# Run security tests with verbose output
python run_api_tests.py security --verbose

# CI pipeline
python run_api_tests.py ci
```

## Test Coverage

### API Endpoints Covered

#### Data Endpoints
- `GET /data` - Main data retrieval with filtering
- `GET /data/group` - Group data retrieval
- `GET /images/thumbnail/<uid>` - Thumbnail generation
- `GET /images/original/thumbnail/<uid>` - Original thumbnails
- `GET /images/mask/thumbnail/<uid>` - Mask thumbnails
- `GET /images/full/<uid>` - Full image access
- `GET /sounds/<uid>` - Audio file access

#### Label Management
- `PUT /v1/label_definitions` - Add label definitions
- `GET /v1/label_definitions/sort` - Sort label definitions
- `GET /label_defs` - List label definitions
- `POST /label` - Submit labels

#### Performance & Analytics
- `GET /performance` - Basic performance metrics
- `GET /detailed_performance/data` - Detailed performance data
- `GET /labels/css` - CSS for label styling

#### Configuration
- `GET /exclusivity/data` - Get exclusivity rules
- `POST /exclusivity/data` - Set exclusivity rules
- `GET /label_provider/data` - Label provider data
- `POST /status` - System status and training trigger

#### UI Endpoints
- `GET /` - Main UI
- `GET /annflux` - Alternative UI endpoint
- `GET /exclusivity` - Exclusivity management UI
- `GET /class_examples` - Class examples UI
- `GET /label_provider` - Label provider UI
- `GET /detailed_performance` - Detailed performance UI

### Test Scenarios

#### Happy Path Tests
- ✅ All endpoints respond correctly
- ✅ Data filtering works as expected
- ✅ Label submission and management
- ✅ Image and file serving
- ✅ Performance metrics retrieval

#### Error Handling Tests
- ✅ Invalid JSON handling
- ✅ Missing resources
- ✅ Malformed requests
- ✅ SQL injection protection
- ✅ XSS prevention

#### Edge Cases
- ✅ Large dataset handling
- ✅ Concurrent requests
- ✅ Empty data scenarios
- ✅ File system errors
- ✅ Network timeouts

#### Security Tests
- ✅ Input validation and sanitization
- ✅ Authentication mechanisms
- ✅ Rate limiting resilience
- ✅ Data exposure prevention

#### Performance Tests
- ✅ Response time validation
- ✅ Large dataset performance
- ✅ Memory usage validation
- ✅ Concurrent load testing

## Test Architecture

### Test Factories

#### TestDataFactory
Creates realistic test data and project structures:
```python
# Create test project with 100 images and 10 labels
project_root = TestDataFactory.create_test_project(
    temp_dir, num_images=100, num_labels=10
)
```

#### MockStateFactory
Creates mock AnnFluxState objects for testing:
```python
mock_state = MockStateFactory.create_mock_state(project_root)
```

#### TestAppFactory
Creates Flask test applications with proper mocking:
```python
test_app = TestAppFactory.create_test_app(project_root, mock_state)
```

### Test Utilities

#### APIAssertions
Helper methods for common API assertions:
```python
APIAssertions.assert_success_response(response, ["status", "package_version"])
APIAssertions.assert_json_response(response, 200)
APIAssertions.assert_file_response(response, "image/jpg")
```

### Fixtures

#### Standard Fixtures
- `temp_project_dir` - Temporary project with sample data
- `mock_state` - Mocked AnnFluxState object
- `test_app` - Flask test application
- `test_client` - Flask test client
- `sample_labels` - Sample label data
- `sample_status_data` - Sample status data

## Configuration

### pytest.ini
Configures pytest with:
- Coverage settings (80% minimum)
- Test markers (unit, integration, security, performance)
- Warning filters
- Source code paths

### Environment Variables for Testing
- `PROJECT_ROOT` - Set to temporary directory during tests
- `USERS` - Empty string for no authentication in tests
- `LOGGING_LEVEL` - Set to ERROR to reduce noise

## Writing New Tests

### Adding Unit Tests

```python
def test_new_endpoint(self, client):
    """Test new API endpoint"""
    response = client.get("/new/endpoint")
    APIAssertions.assert_success_response(response)
    
    data = json.loads(response.data)
    assert "expected_field" in data
```

### Adding Integration Tests

```python
def test_new_workflow(self, test_client, temp_project_dir):
    """Test complete workflow for new feature"""
    # 1. Setup initial state
    # 2. Make API calls
    # 3. Verify results
    # 4. Check side effects
```

### Test Markers

Use appropriate markers for your tests:

```python
@pytest.mark.unit
def test_specific_functionality(self):
    """Unit test for specific functionality"""
    pass

@pytest.mark.integration
def test_workflow_integration(self):
    """Integration test for complete workflow"""
    pass

@pytest.mark.security
def test_security_aspect(self):
    """Security-focused test"""
    pass

@pytest.mark.performance
def test_performance_aspect(self):
    """Performance-focused test"""
    pass

@pytest.mark.slow
def test_slow_operation(self):
    """Test that takes significant time"""
    pass
```

## Continuous Integration

### GitHub Actions Integration

```yaml
- name: Run API Tests
  run: |
    python run_api_tests.py ci
    
- name: Upload Coverage
  uses: codecov/codecov-action@v1
  with:
    file: ./coverage.xml
```

### Requirements for CI

- Python 3.11
- pytest with coverage
- All test dependencies in requirements.txt

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you're running from project root
2. **Missing Dependencies**: Install with `pip install -e .`
3. **Permission Errors**: Check file permissions for test directories
4. **Port Conflicts**: Tests use mock clients, no real ports needed

### Debugging Tests

```bash
# Run with verbose output
python run_api_tests.py unit --verbose

# Run specific test
python -m pytest src/annflux/tests/test_rest_api.py::TestRestAPI::test_data_endpoint -v

# Run with debugging
python -m pytest src/annflux/tests/test_rest_api.py -v -s --pdb
```

### Test Data Issues

Tests create temporary directories that are automatically cleaned up. If tests fail due to missing test data:

1. Check TestDataFactory.create_test_project() is working
2. Verify file permissions in temp directory
3. Ensure all required files are created in the factory

## Contributing

### Adding New Tests

1. Choose appropriate test file (unit vs integration)
2. Use existing fixtures and factories
3. Follow naming conventions
4. Add appropriate markers
5. Update documentation

### Test Standards

- All tests must be independent
- Use descriptive test names
- Test both success and failure scenarios
- Include assertions for all important conditions
- Clean up after tests (use fixtures)

## Coverage Goals

- **Minimum Coverage**: 80% of API code
- **Target Coverage**: 90% of API code
- **Critical Paths**: 100% coverage for authentication and data validation

Coverage reports are generated in:
- Terminal output (during test run)
- HTML report: `htmlcov/index.html`
- XML report: `coverage.xml` (for CI)

## Performance Benchmarks

### Response Time Targets
- Data endpoints: < 1 second
- Image endpoints: < 2 seconds
- Status endpoints: < 500ms
- UI endpoints: < 1 second

### Load Testing
- Concurrent requests: 10+ simultaneous
- Large datasets: 1000+ records
- Memory usage: < 100MB for test datasets

## Security Testing

### Vulnerability Categories Tested
- SQL Injection
- Cross-Site Scripting (XSS)
- Path Traversal
- Authentication Bypass
- Data Exposure
- Rate Limiting Bypass

### Input Validation Tests
- Malicious JSON payloads
- Oversized requests
- Special characters in parameters
- Invalid data types
- Missing required fields

This comprehensive test suite ensures the AnnFlux REST API is robust, secure, and performant across all use cases.

# Root conftest.py - for project-wide pytest configuration
import pytest

def pytest_addoption(parser):
    parser.addoption("--run-api-tests", action="store_true", default=False,
                     help="Run tests that make actual API calls")

@pytest.fixture
def config(request):
    return request.config

# Define a marker to skip API tests
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "api: mark test as requiring API access"
    )
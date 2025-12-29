"""
Pytest configuration for XREPORT E2E tests.
Provides fixtures for Playwright page objects and API client.
"""
import pytest
from playwright.sync_api import Page

# Base URLs - adjust these if your app runs on different ports
UI_BASE_URL = "http://localhost:7861"
API_BASE_URL = "http://localhost:8000"


@pytest.fixture(scope="session")
def base_url() -> str:
    """Returns the base URL of the UI."""
    return UI_BASE_URL


@pytest.fixture(scope="session")
def api_base_url() -> str:
    """Returns the base URL of the API."""
    return API_BASE_URL


@pytest.fixture
def api_context(playwright):
    """
    Creates an API request context for making direct HTTP calls.
    Useful for testing backend endpoints independently of the UI.
    """
    context = playwright.request.new_context(base_url=API_BASE_URL)
    yield context
    context.dispose()

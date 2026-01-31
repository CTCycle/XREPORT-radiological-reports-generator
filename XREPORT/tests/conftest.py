"""
Pytest configuration for XREPORT E2E tests.
Provides fixtures for Playwright page objects and API client.
"""

import os

import pytest
from playwright.sync_api import Page

# Base URLs - configured via environment variables when running tests
UI_HOST = os.environ.get("UI_HOST", "127.0.0.1")
UI_PORT = os.environ.get("UI_PORT", "7861")
FASTAPI_HOST = os.environ.get("FASTAPI_HOST", "127.0.0.1")
FASTAPI_PORT = os.environ.get("FASTAPI_PORT", "8000")

UI_BASE_URL = f"http://{UI_HOST}:{UI_PORT}"
API_BASE_URL = f"http://{FASTAPI_HOST}:{FASTAPI_PORT}"


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

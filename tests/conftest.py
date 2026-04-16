"""
Pytest configuration for XREPORT E2E tests.
Provides fixtures for Playwright page objects and API client.
"""

import os

import pytest


def _normalize_host(value: str) -> str:
    host = value.strip()
    if host == "0.0.0.0":
        return "127.0.0.1"
    return host


def _resolve_base_url(
    explicit_url_env: str,
    host_env: str,
    port_env: str,
    fallback_host: str,
    fallback_port: str,
) -> str:
    explicit = os.environ.get(explicit_url_env, "").strip()
    if explicit:
        return explicit.rstrip("/")

    host = _normalize_host(os.environ.get(host_env, fallback_host))
    port = os.environ.get(port_env, fallback_port).strip() or fallback_port
    return f"http://{host}:{port}"


UI_BASE_URL = _resolve_base_url(
    explicit_url_env="APP_TEST_FRONTEND_URL",
    host_env="UI_HOST",
    port_env="UI_PORT",
    fallback_host="127.0.0.1",
    fallback_port="7861",
)
API_BASE_URL = _resolve_base_url(
    explicit_url_env="APP_TEST_BACKEND_URL",
    host_env="FASTAPI_HOST",
    port_env="FASTAPI_PORT",
    fallback_host="127.0.0.1",
    fallback_port="8000",
)


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


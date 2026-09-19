"""
Pytest configuration for XREPORT E2E tests.
Provides fixtures for Playwright page objects and API client.
"""

import os
from pathlib import Path
import sys
import threading
from queue import Queue

import pytest

###############################################################################
def _configure_test_cache_environment() -> None:
    """Keep direct pytest invocations on the same cache layout as launchers."""
    cache_root = Path(__file__).resolve().parents[2] / "runtimes" / "cache"
    paths = {
        "XREPORT_CACHE_ROOT": cache_root,
        "XDG_CACHE_HOME": cache_root,
        "UV_CACHE_DIR": cache_root / "uv",
        "PIP_CACHE_DIR": cache_root / "pip",
        "NPM_CONFIG_CACHE": cache_root / "npm",
        "npm_config_cache": cache_root / "npm",
        "PLAYWRIGHT_BROWSERS_PATH": cache_root / "playwright-browsers",
        "PYTEST_CACHE_DIR": cache_root / "pytest",
        "PYTEST_BASETEMP": cache_root / "pytest-tmp",
        "RUFF_CACHE_DIR": cache_root / "ruff",
        "MYPY_CACHE_DIR": cache_root / "mypy",
        "PYTHONPYCACHEPREFIX": cache_root / "python",
        "COVERAGE_FILE": cache_root / "coverage" / ".coverage",
        "HF_HOME": cache_root / "huggingface",
        "HF_HUB_CACHE": cache_root / "huggingface" / "hub",
        "HF_MODULES_CACHE": cache_root / "huggingface" / "modules",
        "HF_DATASETS_CACHE": cache_root / "huggingface" / "datasets",
        "TORCH_HOME": cache_root / "torch",
        "KERAS_HOME": cache_root / "keras",
        "MPLCONFIGDIR": cache_root / "matplotlib",
    }
    for path in {value for value in paths.values() if isinstance(value, Path)}:
        path.mkdir(parents=True, exist_ok=True)
    for name, value in paths.items():
        os.environ[name] = str(value)
    sys.pycache_prefix = str(cache_root / "python")
    os.environ.pop("HF_CACHE_DIR", None)
    os.environ.pop("TRANSFORMERS_CACHE", None)


_configure_test_cache_environment()

###############################################################################
def _normalize_host(value: str) -> str:
    host = value.strip()
    if host == "0.0.0.0":
        return "127.0.0.1"
    return host

###############################################################################
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
    fallback_port="8003",
)
API_BASE_URL = _resolve_base_url(
    explicit_url_env="APP_TEST_BACKEND_URL",
    host_env="FASTAPI_HOST",
    port_env="FASTAPI_PORT",
    fallback_host="127.0.0.1",
    fallback_port="5003",
)

###############################################################################
@pytest.fixture(scope="session")
def base_url() -> str:
    """Returns the base URL of the UI."""
    return UI_BASE_URL

###############################################################################
@pytest.fixture(scope="session")
def api_base_url() -> str:
    """Returns the base URL of the API."""
    return API_BASE_URL

###############################################################################
@pytest.fixture
def api_context(playwright):
    """
    Creates an API request context for making direct HTTP calls.
    Useful for testing backend endpoints independently of the UI.
    """
    context = playwright.request.new_context(base_url=API_BASE_URL)
    yield context
    context.dispose()

###############################################################################
def run_async_in_thread(awaitable):
    """Run a coroutine on a dedicated thread to avoid cross-plugin event loop clashes."""
    result_queue: Queue[tuple[bool, object]] = Queue(maxsize=1)

    def _runner() -> None:
        import asyncio

        try:
            result_queue.put((True, asyncio.run(awaitable)))
        except BaseException as exc:  # pragma: no cover - test helper path
            result_queue.put((False, exc))

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join()

    ok, value = result_queue.get()
    if ok:
        return value
    raise value

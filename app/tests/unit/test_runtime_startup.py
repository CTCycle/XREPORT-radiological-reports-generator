from __future__ import annotations

import os
import pytest
import server.configurations.database_config as database_config
import server.configurations.environment as environment
import json
import subprocess
import sys
import server.app as app_module
import server.services.startup_validation as startup_validation
from fastapi.testclient import TestClient
from server.configurations.runtime_environment import get_runtime_environment_settings, CorsSettings, RuntimeEnvironmentSettings
from pathlib import Path
from server.common import model_cache
from server.common.runtime_layout import RuntimeLayout
from fastapi import Request
from tests.conftest import run_async_in_thread
from server.configurations.settings import DatabaseSettings, FeatureSettings, GlobalSettings, InferenceSettings, JobsSettings, ServerSettings










###############################################################################
@pytest.fixture(autouse=True)
def reset_environment_state() -> None:
    environment._environment_state.cache_clear()
    database_config.clear_database_configuration_error()
    yield
    environment._environment_state.cache_clear()
    database_config.clear_database_configuration_error()

###############################################################################
def _configure_environment_paths(tmp_path, monkeypatch: pytest.MonkeyPatch):
    settings_dir = tmp_path / "settings"
    settings_dir.mkdir()
    env_path = settings_dir / ".env"
    example_path = settings_dir / ".env.example"
    monkeypatch.setattr(environment, "ENV_FILE_PATH", env_path)
    monkeypatch.setattr(environment, "ENV_EXAMPLE_FILE_PATH", example_path)
    monkeypatch.setattr(database_config, "DATABASE_CONFIG_FILE_PATH", settings_dir.parent / "database-config.json")
    monkeypatch.setattr(database_config, "DATABASE_FILE_PATH", settings_dir.parent / "database.db")
    return env_path, example_path

###############################################################################
def test_missing_env_is_seeded_from_template_and_loaded(tmp_path, monkeypatch) -> None:
    env_path, example_path = _configure_environment_paths(tmp_path, monkeypatch)
    contents = b"XREPORT_TEST_ENV=from-template\n"
    example_path.write_bytes(contents)
    monkeypatch.delenv("XREPORT_TEST_ENV", raising=False)

    loaded_path = environment.load_environment(force=True)

    assert loaded_path == env_path
    assert env_path.read_bytes() == contents
    assert os.environ["XREPORT_TEST_ENV"] == "from-template"

###############################################################################
def test_existing_env_is_never_overwritten_by_template(tmp_path, monkeypatch) -> None:
    env_path, example_path = _configure_environment_paths(tmp_path, monkeypatch)
    env_path.write_bytes(b"XREPORT_TEST_ENV=existing\n")
    example_path.write_bytes(b"XREPORT_TEST_ENV=template\n")
    monkeypatch.delenv("XREPORT_TEST_ENV", raising=False)

    environment.load_environment(force=True)

    assert env_path.read_bytes() == b"XREPORT_TEST_ENV=existing\n"
    assert os.environ["XREPORT_TEST_ENV"] == "existing"

###############################################################################
def test_explicit_resource_root_override_survives_environment_loading(
    tmp_path, monkeypatch
) -> None:
    env_path, _ = _configure_environment_paths(tmp_path, monkeypatch)
    env_path.write_text(
        "XREPORT_RESOURCES_DIR=from-file\nXREPORT_TEST_ENV=from-file\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("XREPORT_RESOURCES_DIR", "from-process")
    monkeypatch.setenv("XREPORT_TEST_ENV", "from-process")

    environment.load_environment(force=True)

    assert os.environ["XREPORT_RESOURCES_DIR"] == "from-process"
    assert os.environ["XREPORT_TEST_ENV"] == "from-file"

###############################################################################
def test_stray_database_keys_are_ignored_and_purged(tmp_path, monkeypatch) -> None:
    env_path, _ = _configure_environment_paths(tmp_path, monkeypatch)
    env_path.write_text(
        "EMBEDDED_DATABASE=false\nDATABASE_INSERT_BATCH_SIZE=25\nXREPORT_TEST_ENV=kept\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DATABASE_HOST", "leftover-host")
    monkeypatch.setenv("EMBEDDED_DATABASE", "false")
    monkeypatch.delenv("XREPORT_TEST_ENV", raising=False)

    environment.load_environment(force=True)

    configured = database_config.read_database_configuration()
    assert configured.backend == "sqlite"
    assert configured.sqlite_path == env_path.parent.parent / "database.db"
    assert configured.insert_batch_size == 1000
    assert "DATABASE_HOST" not in os.environ
    assert "EMBEDDED_DATABASE" not in os.environ
    assert os.environ["XREPORT_TEST_ENV"] == "kept"
    assert "DATABASE_INSERT_BATCH_SIZE" in env_path.read_text(encoding="utf-8")


###############################################################################
def test_supported_runtime_settings_are_loaded_from_dotenv(tmp_path, monkeypatch) -> None:
    env_path, _ = _configure_environment_paths(tmp_path, monkeypatch)
    env_path.write_text(
        "\n".join(
            (
                "INFERENCE_HF_LOCAL_ONLY=false",
                "INFERENCE_DEVICE=cpu",
                "CORS_ALLOWED_ORIGINS=https://ui.example,https://desktop.example",
                "CORS_ALLOW_CREDENTIALS=true",
                "CORS_ALLOW_METHODS=GET,OPTIONS",
                "CORS_ALLOW_HEADERS=Authorization,X-XREPORT-Desktop-Token",
                "",
            )
        ),
        encoding="utf-8",
    )

    environment.load_environment(force=True)
    settings = get_runtime_environment_settings()

    assert settings.inference_hf_local_only is False
    assert settings.inference_device == "cpu"
    assert settings.cors.allowed_origins == (
        "https://ui.example",
        "https://desktop.example",
    )
    assert settings.cors.allow_credentials is True
    assert settings.cors.allow_methods == ("GET", "OPTIONS")
    assert settings.cors.allow_headers == (
        "Authorization",
        "X-XREPORT-Desktop-Token",
    )

    from server.app import create_app

    response = TestClient(create_app()).options(
        "/api/health",
        headers={
            "Origin": "https://ui.example",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "X-XREPORT-Desktop-Token",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "https://ui.example"







###############################################################################
def test_concurrent_service_initialization_keeps_ml_imports_lazy(
    tmp_path: Path,
) -> None:
    root = Path(__file__).parents[3]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "app")
    environment["KERAS_BACKEND"] = "torch"
    environment["MPLBACKEND"] = "Agg"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["XREPORT_RESOURCES_DIR"] = str(tmp_path / "data")
    environment.pop("XREPORT_DESKTOP", None)
    code = """
import importlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import sys

from server.configurations import get_database_settings
from server.repositories.database.initializer import initialize_database

# Service factories read the singleton row, so model normal startup first.
initialize_database(get_database_settings())

module_names = (
    "server.services.preparation",
    "server.services.training",
    "server.services.validation_runs",
    "server.services.evaluation",
    "server.repositories.serialization.model",
)

def import_module(name):
    return importlib.import_module(name)

with ThreadPoolExecutor(max_workers=len(module_names)) as pool:
    list(pool.map(import_module, module_names))

from server.repositories.serialization.model import ModelSerializer
from server.services.preparation import get_preparation_service
from server.services.training import TrainingService, get_training_runtime, get_training_service
from server.services.validation_runs import get_validation_service

with ThreadPoolExecutor(max_workers=3) as pool:
    services = list(pool.map(lambda factory: factory(), (
        get_preparation_service,
        get_training_service,
        get_validation_service,
    )))

with TemporaryDirectory() as directory:
    checkpoint = Path(directory) / "checkpoint"
    configuration = checkpoint / "configuration"
    configuration.mkdir(parents=True)
    (configuration / "configuration.json").write_text("{}", encoding="utf-8")
    (configuration / "metadata.json").write_text("{}", encoding="utf-8")
    (configuration / "session_history.json").write_text(
        '{"epochs": 2, "history": {"loss": [1.0], "val_loss": [1.1]}}',
        encoding="utf-8",
    )

    record = SimpleNamespace(name="checkpoint", path=checkpoint, artifact_complete=True)
    class CheckpointRepositoryStub:
        def list_checkpoints(self):
            return [record]

    listing = TrainingService(
        job_manager=object(),
        training_runtime=get_training_runtime(),
        checkpoint_repository=CheckpointRepositoryStub(),
    ).get_checkpoints()

heavy_modules = (
    "torch",
    "torch.utils",
    "keras",
    "torchvision",
    "transformers",
    "server.models.training.model",
    "server.models.training.dataloader",
    "server.models.inference.generator",
)
print(json.dumps({
    "factory_types": [type(service).__name__ for service in services],
    "checkpoint_epochs": listing.checkpoints[0].epochs,
    "heavy_modules": [name for name in heavy_modules if name in sys.modules],
    "serializer_loaded": "server.repositories.serialization.model" in sys.modules,
    "serializer_type": ModelSerializer.__name__,
}))
"""

    payloads = []
    repetitions = 10
    for attempt in range(repetitions):
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=root,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert completed.returncode == 0, (
            f"Concurrent service initialization failed on repetition "
            f"{attempt + 1}/{repetitions}.\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
        payloads.append(json.loads(completed.stdout.strip().splitlines()[-1]))

    for payload in payloads:
        assert payload["factory_types"] == [
            "PreparationService",
            "TrainingService",
            "ValidationService",
        ]
        assert payload["checkpoint_epochs"] == 2
        assert payload["heavy_modules"] == []
        assert payload["serializer_loaded"] is True
        assert payload["serializer_type"] == "ModelSerializer"





###############################################################################
def test_model_cache_bootstrap_overwrites_hostile_global_cache_values(
    monkeypatch,
) -> None:
    monkeypatch.setenv("HF_CACHE_DIR", r"C:\Users\Public\global-hf")
    monkeypatch.setenv("HF_HOME", r"C:\Users\Public\global-hf-home")
    monkeypatch.setenv("HF_HUB_CACHE", r"C:\Users\Public\global-hub")
    monkeypatch.setenv("TRANSFORMERS_CACHE", r"C:\Users\Public\external-transformers")

    model_cache.configure_model_cache()

    assert os.environ["XREPORT_CACHE_ROOT"] == str(model_cache.RUNTIME_CACHE_DIR)
    assert os.environ["XDG_CACHE_HOME"] == str(model_cache.RUNTIME_CACHE_DIR)
    assert os.environ["UV_CACHE_DIR"] == str(model_cache.UV_CACHE_DIR)
    assert os.environ["PIP_CACHE_DIR"] == str(model_cache.PIP_CACHE_DIR)
    assert os.environ["NPM_CONFIG_CACHE"] == str(model_cache.NPM_CACHE_DIR)
    assert os.environ["npm_config_cache"] == str(model_cache.NPM_CACHE_DIR)
    assert os.environ["PLAYWRIGHT_BROWSERS_PATH"] == str(
        model_cache.PLAYWRIGHT_BROWSERS_CACHE_DIR
    )
    assert os.environ["PYTEST_CACHE_DIR"] == str(model_cache.PYTEST_CACHE_DIR)
    assert os.environ["PYTEST_BASETEMP"] == str(model_cache.PYTEST_BASETEMP_DIR)
    assert os.environ["RUFF_CACHE_DIR"] == str(model_cache.RUFF_CACHE_DIR)
    assert os.environ["MYPY_CACHE_DIR"] == str(model_cache.MYPY_CACHE_DIR)
    assert os.environ["COVERAGE_FILE"] == str(
        model_cache.COVERAGE_CACHE_DIR / ".coverage"
    )
    assert os.environ["HF_HOME"] == str(model_cache.HUGGINGFACE_CACHE_DIR)
    assert os.environ["HF_HUB_CACHE"] == str(model_cache.HF_HUB_CACHE_DIR)
    assert os.environ["HF_MODULES_CACHE"] == str(model_cache.HF_MODULES_CACHE_DIR)
    assert os.environ["HF_DATASETS_CACHE"] == str(model_cache.HF_DATASETS_CACHE_DIR)
    assert "TRANSFORMERS_CACHE" not in os.environ
    assert os.environ["TORCH_HOME"] == str(model_cache.TORCH_CACHE_DIR)
    assert os.environ["KERAS_HOME"] == str(model_cache.KERAS_CACHE_DIR)
    assert os.environ["MPLCONFIGDIR"] == str(model_cache.MATPLOTLIB_CACHE_DIR)
    assert os.environ["PYTHONPYCACHEPREFIX"] == str(model_cache.PYTHON_CACHE_DIR)
    assert os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "1"
    assert "HF_CACHE_DIR" not in os.environ
    for path in (
        model_cache.HUGGINGFACE_CACHE_DIR,
        model_cache.HF_HUB_CACHE_DIR,
        model_cache.HF_MODULES_CACHE_DIR,
        model_cache.HF_DATASETS_CACHE_DIR,
        model_cache.TORCH_CACHE_DIR,
        model_cache.KERAS_CACHE_DIR,
        model_cache.MATPLOTLIB_CACHE_DIR,
    ):
        assert path.is_relative_to(model_cache.RUNTIME_CACHE_DIR)
        assert path.is_dir()
    for path in (
        model_cache.HF_INSTALLED_DIR,
        model_cache.HF_STAGING_DIR,
        model_cache.HF_ROLLBACK_DIR,
        model_cache.HF_METADATA_DIR,
    ):
        assert path.is_dir()
        assert not path.is_relative_to(model_cache.RUNTIME_CACHE_DIR)











def _clear_runtime_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "INFERENCE_HF_LOCAL_ONLY",
        "INFERENCE_DEVICE",
        "CORS_ALLOWED_ORIGINS",
        "CORS_ALLOW_CREDENTIALS",
        "CORS_ALLOW_METHODS",
        "CORS_ALLOW_HEADERS",
    ):
        monkeypatch.delenv(name, raising=False)


def test_runtime_environment_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_runtime_environment(monkeypatch)

    assert get_runtime_environment_settings() == RuntimeEnvironmentSettings(
        inference_hf_local_only=True,
        inference_device="auto",
        cors=CorsSettings(
            allowed_origins=(),
            allow_credentials=False,
            allow_methods=("GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"),
            allow_headers=(
                "Authorization",
                "Content-Type",
                "X-XREPORT-Desktop-Token",
            ),
        ),
    )


@pytest.mark.parametrize("value", ["1", "true", "YES", " on "])
def test_environment_boolean_accepts_true_spellings(
    value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_runtime_environment(monkeypatch)
    monkeypatch.setenv("INFERENCE_HF_LOCAL_ONLY", value)

    assert get_runtime_environment_settings().inference_hf_local_only is True


@pytest.mark.parametrize("value", ["0", "false", "NO", " off "])
def test_environment_boolean_accepts_false_spellings(
    value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_runtime_environment(monkeypatch)
    monkeypatch.setenv("INFERENCE_HF_LOCAL_ONLY", value)

    assert get_runtime_environment_settings().inference_hf_local_only is False


def test_runtime_environment_normalizes_device_and_cors_lists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_runtime_environment(monkeypatch)
    monkeypatch.setenv("INFERENCE_DEVICE", " CPU ")
    monkeypatch.setenv(
        "CORS_ALLOWED_ORIGINS", " https://ui.example, ,https://desktop.example "
    )
    monkeypatch.setenv("CORS_ALLOW_CREDENTIALS", "true")
    monkeypatch.setenv("CORS_ALLOW_METHODS", "GET, OPTIONS")
    monkeypatch.setenv("CORS_ALLOW_HEADERS", "Authorization, X-Test-Header")

    settings = get_runtime_environment_settings()

    assert settings.inference_device == "cpu"
    assert settings.cors == CorsSettings(
        allowed_origins=("https://ui.example", "https://desktop.example"),
        allow_credentials=True,
        allow_methods=("GET", "OPTIONS"),
        allow_headers=("Authorization", "X-Test-Header"),
    )


@pytest.mark.parametrize("value", ["maybe", "2", "enabled"])
def test_invalid_runtime_boolean_is_rejected(
    value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_runtime_environment(monkeypatch)
    monkeypatch.setenv("INFERENCE_HF_LOCAL_ONLY", value)

    with pytest.raises(ValueError, match="INFERENCE_HF_LOCAL_ONLY"):
        get_runtime_environment_settings()


@pytest.mark.parametrize("value", ["gpu", "mps"])
def test_invalid_device_is_rejected(
    value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_runtime_environment(monkeypatch)
    monkeypatch.setenv("INFERENCE_DEVICE", value)

    with pytest.raises(ValueError, match="INFERENCE_DEVICE"):
        get_runtime_environment_settings()


def test_blank_device_uses_auto_default(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_runtime_environment(monkeypatch)
    monkeypatch.setenv("INFERENCE_DEVICE", "   ")

    assert get_runtime_environment_settings().inference_device == "auto"


def test_wildcard_cors_origin_cannot_use_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_runtime_environment(monkeypatch)
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "https://ui.example, *")
    monkeypatch.setenv("CORS_ALLOW_CREDENTIALS", "true")

    with pytest.raises(ValueError, match="cannot include '\\*'"):
        get_runtime_environment_settings()


def test_inference_policy_is_composed_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from server.configurations.settings import (
        DEFAULT_APPLICATION_SETTINGS,
        DatabaseSettings,
        application_settings_to_server_settings,
    )

    _clear_runtime_environment(monkeypatch)
    monkeypatch.setenv("INFERENCE_HF_LOCAL_ONLY", "false")
    monkeypatch.setenv("INFERENCE_DEVICE", "cuda")
    database = DatabaseSettings(
        backend="sqlite",
        engine=None,
        host=None,
        port=None,
        database_name=None,
        username=None,
        password=None,
        ssl=False,
        ssl_ca=None,
        connect_timeout=3,
        insert_batch_size=1000,
    )

    settings = application_settings_to_server_settings(
        DEFAULT_APPLICATION_SETTINGS, database
    )

    assert settings.inference.hf_local_only is False
    assert settings.inference.device == "cuda"
    assert settings.inference.model_timeout == 600


def test_cors_is_disabled_when_origins_are_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_runtime_environment(monkeypatch)
    from server.app import create_app

    response = TestClient(create_app()).options(
        "/api/health",
        headers={
            "Origin": "https://ui.example",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert response.status_code == 405
    assert "access-control-allow-origin" not in response.headers


def test_cors_preflight_allows_configured_origin_and_rejects_other_origin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from server.app import create_app

    _clear_runtime_environment(monkeypatch)
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "https://ui.example")
    monkeypatch.setenv("CORS_ALLOW_CREDENTIALS", "true")
    monkeypatch.setenv("CORS_ALLOW_METHODS", "GET, OPTIONS")
    monkeypatch.setenv("CORS_ALLOW_HEADERS", "X-XREPORT-Desktop-Token")
    client = TestClient(create_app())
    preflight_headers = {
        "Access-Control-Request-Method": "GET",
        "Access-Control-Request-Headers": "X-XREPORT-Desktop-Token",
    }

    allowed = client.options(
        "/api/health",
        headers={"Origin": "https://ui.example", **preflight_headers},
    )
    denied = client.options(
        "/api/health",
        headers={"Origin": "https://untrusted.example", **preflight_headers},
    )

    assert allowed.status_code == 200
    assert allowed.headers["access-control-allow-origin"] == "https://ui.example"
    assert allowed.headers["access-control-allow-credentials"] == "true"
    assert "get" in allowed.headers["access-control-allow-methods"].lower()
    assert "x-xreport-desktop-token" in allowed.headers[
        "access-control-allow-headers"
    ].lower()
    assert denied.status_code == 400
    assert "access-control-allow-origin" not in denied.headers





###############################################################################
def test_source_layout_defaults_to_repository_data(monkeypatch) -> None:
    repository_root = Path(__file__).resolve().parents[3]
    monkeypatch.setenv("XREPORT_DESKTOP", "false")
    monkeypatch.setenv("XREPORT_RESOURCES_DIR", "")

    layout = RuntimeLayout.from_environment()

    assert layout.mode == "source"
    assert layout.resources_root == repository_root / "data"

###############################################################################
def test_source_layout_accepts_an_explicit_resource_root(monkeypatch, tmp_path) -> None:
    selected_root = tmp_path / "selected-resources"
    monkeypatch.setenv("XREPORT_DESKTOP", "false")
    monkeypatch.setenv("XREPORT_RESOURCES_DIR", str(selected_root))

    layout = RuntimeLayout.from_environment()

    assert layout.resources_root == selected_root.resolve()







@pytest.fixture
def client_build(tmp_path, monkeypatch):
    index = tmp_path / "index.html"
    index.write_text("<html><body>source-spa</body></html>", encoding="utf-8")
    (tmp_path / "main.js").write_text("console.log('bundle')", encoding="utf-8")
    monkeypatch.setattr(app_module, "PACKAGED_MODE", False)
    monkeypatch.setattr(app_module, "CLIENT_DIST_DIR", tmp_path)
    return tmp_path


def test_source_static_mode_serves_root_assets_and_angular_routes(
    client_build, monkeypatch
):
    monkeypatch.setenv("XREPORT_SERVE_CLIENT", "1")
    monkeypatch.delenv("XREPORT_LAUNCH_TOKEN", raising=False)
    application = app_module.create_app()
    client = TestClient(application)

    root = client.get("/")
    asset = client.get("/main.js")
    route = client.get("/reports/42")

    assert root.status_code == 200
    assert "source-spa" in root.text
    assert asset.status_code == 200
    assert asset.text == "console.log('bundle')"
    assert route.status_code == 200
    assert "source-spa" in route.text


def test_source_static_mode_keeps_api_paths_out_of_spa_fallback(
    client_build, monkeypatch
):
    monkeypatch.setenv("XREPORT_SERVE_CLIENT", "true")
    monkeypatch.delenv("XREPORT_LAUNCH_TOKEN", raising=False)
    application = app_module.create_app()
    application.state.database_ready = True
    client = TestClient(application)

    health = client.get("/api/health")
    unknown = client.get("/api/not-a-route")
    unknown_api_root = client.get("/api")

    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert unknown.status_code == 404
    assert unknown.headers["content-type"].startswith("application/json")
    assert "source-spa" not in unknown.text
    assert unknown_api_root.status_code == 404
    assert "source-spa" not in unknown_api_root.text


def test_database_recovery_health_and_spa_remain_available(client_build, monkeypatch):
    monkeypatch.setenv("XREPORT_SERVE_CLIENT", "1")
    monkeypatch.delenv("XREPORT_LAUNCH_TOKEN", raising=False)
    application = app_module.create_app()
    client = TestClient(application)

    root = client.get("/")
    health = client.get("/api/health")

    assert root.status_code == 200
    assert "source-spa" in root.text
    assert health.status_code == 200
    assert health.json()["status"] == "database_recovery"


def test_source_static_mode_rejects_traversal(client_build, monkeypatch):
    monkeypatch.setenv("XREPORT_SERVE_CLIENT", "1")
    outside = client_build.parent / "secret.txt"
    outside.write_text("must not be served", encoding="utf-8")

    response = app_module.serve_source_frontend("..\\secret.txt")

    assert response.status_code == 404
    assert "must not be served" not in response.body.decode()


def test_default_source_root_still_redirects_to_docs(client_build, monkeypatch):
    monkeypatch.delenv("XREPORT_SERVE_CLIENT", raising=False)
    monkeypatch.delenv("XREPORT_LAUNCH_TOKEN", raising=False)
    application = app_module.create_app()
    client = TestClient(application)

    response = client.get("/", follow_redirects=False)

    assert response.status_code == 307
    assert response.headers["location"].endswith("/docs")


def test_source_shutdown_requires_launcher_token(client_build, monkeypatch):
    monkeypatch.delenv("XREPORT_SERVE_CLIENT", raising=False)
    monkeypatch.setenv("XREPORT_LAUNCH_TOKEN", "launcher-secret-token")
    application = app_module.create_app()
    client = TestClient(application)

    unauthorized = client.post("/__xreport/shutdown")
    authorized = client.post(
        "/__xreport/shutdown",
        headers={"X-XREPORT-Launch-Token": "launcher-secret-token"},
    )

    assert unauthorized.status_code == 403
    assert authorized.status_code == 200
    assert application.state.desktop_shutdown_event.is_set()


def test_source_static_mode_requires_a_built_index(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "PACKAGED_MODE", False)
    monkeypatch.setattr(app_module, "CLIENT_DIST_DIR", tmp_path)
    monkeypatch.setenv("XREPORT_SERVE_CLIENT", "1")

    with pytest.raises(FileNotFoundError, match="Built Angular frontend"):
        app_module.create_app()







###############################################################################
def test_server_app_import_does_not_load_heavy_runtime_modules() -> None:
    root = Path(__file__).parents[3]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "app")
    environment["KERAS_BACKEND"] = "torch"
    environment["MPLBACKEND"] = "Agg"
    environment.pop("XREPORT_DESKTOP", None)
    code = """
import json
import sys
import server.app

heavy_modules = (
    "torch",
    "transformers",
    "keras",
    "torchvision",
    "server.models.inference.providers.huggingface",
    "server.models.training.processing",
    "server.repositories.serialization.model",
    "server.services.dataset_processing",
    "server.services.inference_runtime",
)
print(json.dumps({
    "route_count": len(server.app.app.routes),
    "loaded_heavy_modules": [name for name in heavy_modules if name in sys.modules],
}))
"""

    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout.strip().splitlines()[-1])

    assert payload["route_count"] == 27
    assert payload["loaded_heavy_modules"] == []

















###############################################################################
def _database_settings() -> DatabaseSettings:
    return DatabaseSettings(
        backend="sqlite",
        engine=None,
        host=None,
        port=None,
        database_name=None,
        username=None,
        password=None,
        ssl=False,
        ssl_ca=None,
        connect_timeout=3,
        insert_batch_size=1000,
        sqlite_path=Path("database.db").resolve(),
    )

###############################################################################
def _server_settings(database: DatabaseSettings) -> ServerSettings:
    return ServerSettings(
        database=database,
        global_settings=GlobalSettings(seed=42),
        features=FeatureSettings(allow_local_filesystem_access=True),
        jobs=JobsSettings(polling_interval=1.0),
        inference=InferenceSettings(
            hf_local_only=True,
            device="cpu",
            model_timeout=60,
        ),
    )

###############################################################################
def test_startup_validations_prepares_database_and_composes_settings_once(
    monkeypatch,
) -> None:
    database = _database_settings()
    settings = _server_settings(database)
    prepared: list[DatabaseSettings] = []
    composed: list[bool] = []

    monkeypatch.setattr(
        startup_validation,
        "prepare_database_for_startup",
        lambda value: prepared.append(value),
    )
    monkeypatch.setattr(
        startup_validation,
        "get_server_settings",
        lambda: composed.append(True) or settings,
    )
    monkeypatch.setattr(startup_validation, "_ensure_directory", lambda path: None)

    result = startup_validation.run_startup_validations(database)

    assert result is settings
    assert prepared == [database]
    assert composed == [True]

###############################################################################
def test_app_lifespan_uses_returned_settings_for_state_and_health(monkeypatch) -> None:
    database = _database_settings()
    settings = _server_settings(database)
    validation_inputs: list[DatabaseSettings] = []

    monkeypatch.setattr(app_module, "get_database_settings", lambda: database)
    monkeypatch.setattr(
        app_module,
        "run_startup_validations",
        lambda value: validation_inputs.append(value) or settings,
    )
    application = app_module.create_app()

    async def exercise_lifespan() -> None:
        async with app_module.app_lifespan(application):
            assert application.state.server_settings is settings
            request = Request(
                {
                    "type": "http",
                    "method": "GET",
                    "path": "/api/health",
                    "raw_path": b"/api/health",
                    "app": application,
                    "query_string": b"",
                    "headers": [],
                    "client": ("test", 1),
                    "server": ("127.0.0.1", 5003),
                    "scheme": "http",
                }
            )
            response = app_module.health_check(request)
            assert response.runtime_mode == "sqlite"
            assert response.runtime_port == 5003

    run_async_in_thread(exercise_lifespan())
    assert validation_inputs == [database]

from __future__ import annotations

from fastapi import Request

import server.app as app_module
import server.services.startup_validation as startup_validation
from tests.conftest import run_async_in_thread
from server.configurations.settings import (
    DatabaseSettings,
    FeatureSettings,
    GlobalSettings,
    InferenceSettings,
    JobsSettings,
    ServerSettings,
)

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
    )


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

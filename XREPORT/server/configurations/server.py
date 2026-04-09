from __future__ import annotations

from typing import Any

from XREPORT.server.configurations.settings import (
    get_app_settings,
    get_server_settings,
    reload_settings_for_tests,
)
from XREPORT.server.domain.settings import (
    DatabaseSettings,
    FeatureSettings,
    GlobalSettings,
    JobsSettings,
    JsonDatabaseSettings,
    JsonFeatureSettings,
    JsonGlobalSettings,
    JsonJobsSettings,
    ServerSettings,
)


# -----------------------------------------------------------------------------
def build_global_settings(data: dict[str, Any]) -> GlobalSettings:
    payload = JsonGlobalSettings.model_validate(data)
    return GlobalSettings(seed=payload.seed)


# -----------------------------------------------------------------------------
def build_jobs_settings(data: dict[str, Any]) -> JobsSettings:
    payload = JsonJobsSettings.model_validate(data)
    return JobsSettings(polling_interval=payload.polling_interval)


# -----------------------------------------------------------------------------
def build_feature_settings(payload: dict[str, Any] | Any) -> FeatureSettings:
    source = JsonFeatureSettings.model_validate(payload)
    return FeatureSettings(allow_local_filesystem_access=source.allow_local_filesystem_access)


# -----------------------------------------------------------------------------
def build_database_settings(payload: dict[str, Any] | Any) -> DatabaseSettings:
    source = JsonDatabaseSettings.model_validate(payload)
    if source.embedded_database:
        return DatabaseSettings(
            embedded_database=True,
            engine=None,
            host=None,
            port=None,
            database_name=None,
            username=None,
            password=None,
            ssl=False,
            ssl_ca=None,
            connect_timeout=source.connect_timeout,
            insert_batch_size=source.insert_batch_size,
        )
    return DatabaseSettings(
        embedded_database=False,
        engine=source.engine.strip().lower(),
        host=source.host,
        port=source.port,
        database_name=source.database_name,
        username=source.username,
        password=source.password,
        ssl=source.ssl,
        ssl_ca=source.ssl_ca,
        connect_timeout=source.connect_timeout,
        insert_batch_size=source.insert_batch_size,
    )


# -----------------------------------------------------------------------------
def build_server_settings(data: dict[str, Any] | Any) -> ServerSettings:
    payload = data if isinstance(data, dict) else {}
    database_payload = payload.get("database", {})
    global_payload = payload.get("global", {})
    feature_payload = payload.get("features", {})
    jobs_payload = payload.get("jobs", {})

    return ServerSettings(
        database=build_database_settings(database_payload),
        global_settings=build_global_settings(global_payload),
        features=build_feature_settings(feature_payload),
        jobs=build_jobs_settings(jobs_payload),
    )


server_settings = get_server_settings()

__all__ = [
    "DatabaseSettings",
    "FeatureSettings",
    "GlobalSettings",
    "JobsSettings",
    "ServerSettings",
    "get_app_settings",
    "get_server_settings",
    "reload_settings_for_tests",
    "build_database_settings",
    "build_feature_settings",
    "build_global_settings",
    "build_jobs_settings",
    "build_server_settings",
    "server_settings",
]

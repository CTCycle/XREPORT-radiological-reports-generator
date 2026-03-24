from __future__ import annotations

import os
from typing import Any

from XREPORT.server.configurations.base import ensure_mapping, load_configuration_data
from XREPORT.server.domain.settings import (
    DatabaseSettings,
    GlobalSettings,
    JobsSettings,
    ServerSettings,
)

from XREPORT.server.common.constants import CONFIGURATION_FILE

from XREPORT.server.common.utils.types import (
    coerce_bool,
    coerce_float,
    coerce_int,
    coerce_str_or_none,
)


# [BUILDER FUNCTIONS]
###############################################################################
def build_global_settings(data: dict[str, Any]) -> GlobalSettings:
    payload = ensure_mapping(data)
    return GlobalSettings(
        seed=coerce_int(payload.get("seed"), 42),
    )


# -----------------------------------------------------------------------------
def build_jobs_settings(data: dict[str, Any]) -> JobsSettings:
    payload = ensure_mapping(data)
    return JobsSettings(
        polling_interval=coerce_float(payload.get("polling_interval"), 1.0),
    )


# -----------------------------------------------------------------------------
def build_database_settings(payload: dict[str, Any] | Any) -> DatabaseSettings:
    database_payload = ensure_mapping(payload)
    payload_embedded = coerce_bool(database_payload.get("embedded_database"), True)
    embedded_value = os.getenv("DB_EMBEDDED") if "DB_EMBEDDED" in os.environ else None
    embedded = coerce_bool(embedded_value, payload_embedded)

    insert_batch_size = coerce_int(
        os.getenv("DB_INSERT_BATCH_SIZE"),
        1000,
        minimum=1,
    )

    if embedded:
        # External fields are ignored entirely when embedded DB is active
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
            connect_timeout=10,
            insert_batch_size=insert_batch_size,
        )

    # External DB mode
    engine_value = coerce_str_or_none(os.getenv("DB_ENGINE")) or "postgres"
    normalized_engine = engine_value.lower() if engine_value else None
    ssl = coerce_bool(os.getenv("DB_SSL"), False)

    return DatabaseSettings(
        embedded_database=False,
        engine=normalized_engine,
        host=coerce_str_or_none(os.getenv("DB_HOST")),
        port=coerce_int(
            os.getenv("DB_PORT"),
            5432,
            minimum=1,
            maximum=65535,
        ),
        database_name=coerce_str_or_none(os.getenv("DB_NAME")),
        username=coerce_str_or_none(os.getenv("DB_USER")),
        password=coerce_str_or_none(os.getenv("DB_PASSWORD")),
        ssl=ssl,
        ssl_ca=coerce_str_or_none(os.getenv("DB_SSL_CA")),
        connect_timeout=coerce_int(
            os.getenv("DB_CONNECT_TIMEOUT"),
            10,
            minimum=1,
        ),
        insert_batch_size=insert_batch_size,
    )


# -----------------------------------------------------------------------------
def build_server_settings(data: dict[str, Any] | Any) -> ServerSettings:
    payload = ensure_mapping(data)
    database_payload = ensure_mapping(payload.get("database"))
    global_payload = ensure_mapping(payload.get("global"))
    jobs_payload = ensure_mapping(payload.get("jobs"))

    return ServerSettings(
        database=build_database_settings(database_payload),
        global_settings=build_global_settings(global_payload),
        jobs=build_jobs_settings(jobs_payload),
    )


# [SERVER CONFIGURATION LOADER]
###############################################################################
def get_server_settings(config_path: str | None = None) -> ServerSettings:
    path = config_path or CONFIGURATION_FILE
    payload = load_configuration_data(path)

    return build_server_settings(payload)


server_settings = get_server_settings()

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from XREPORT.server.configurations.base import ensure_mapping, load_configuration_data

from XREPORT.server.utils.constants import CONFIGURATION_FILE

from XREPORT.server.utils.types import (
    coerce_bool,
    coerce_float,
    coerce_int,
    coerce_str,
    coerce_str_or_none,
)


# [SERVER SETTINGS]
###############################################################################
@dataclass(frozen=True)
class DatabaseSettings:
    embedded_database: bool
    engine: str | None
    host: str | None
    port: int | None
    database_name: str | None
    username: str | None
    password: str | None
    ssl: bool
    ssl_ca: str | None
    connect_timeout: int
    insert_batch_size: int
    browse_batch_size: int


# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class GlobalSettings:
    seed: int


# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class JobsSettings:
    polling_interval: float


# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class TrainingSettings:
    use_jit: bool
    jit_backend: str
    use_mixed_precision: bool
    dataloader_workers: int
    prefetch_factor: int
    pin_memory: bool
    persistent_workers: bool


# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class ServerSettings:
    database: DatabaseSettings
    global_settings: GlobalSettings
    jobs: JobsSettings
    training: TrainingSettings


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
    embedded = bool(payload.get("embedded_database", True))
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
            insert_batch_size=coerce_int(
                payload.get("insert_batch_size"), 1000, minimum=1
            ),
            browse_batch_size=coerce_int(
                payload.get("browse_batch_size"), 200, minimum=1, maximum=1000
            ),
        )

    # External DB mode
    engine_value = coerce_str_or_none(payload.get("engine")) or "postgres"
    normalized_engine = engine_value.lower() if engine_value else None
    return DatabaseSettings(
        embedded_database=False,
        engine=normalized_engine,
        host=coerce_str_or_none(payload.get("host")),
        port=coerce_int(payload.get("port"), 5432, minimum=1, maximum=65535),
        database_name=coerce_str_or_none(payload.get("database_name")),
        username=coerce_str_or_none(payload.get("username")),
        password=coerce_str_or_none(payload.get("password")),
        ssl=bool(payload.get("ssl", False)),
        ssl_ca=coerce_str_or_none(payload.get("ssl_ca")),
        connect_timeout=coerce_int(payload.get("connect_timeout"), 10, minimum=1),
        insert_batch_size=coerce_int(payload.get("insert_batch_size"), 1000, minimum=1),
        browse_batch_size=coerce_int(
            payload.get("browse_batch_size"), 200, minimum=1, maximum=1000
        ),
    )


# -----------------------------------------------------------------------------
def build_training_settings(data: dict[str, Any]) -> TrainingSettings:
    payload = ensure_mapping(data)
    jit_backend = coerce_str(payload.get("jit_backend"), "inductor")
    valid_backends = ["inductor", "eager", "aot_eager", "nvprims_nvfuser"]
    if jit_backend not in valid_backends:
        jit_backend = "inductor"
    return TrainingSettings(
        use_jit=bool(payload.get("use_jit", True)),
        jit_backend=jit_backend,
        use_mixed_precision=bool(payload.get("use_mixed_precision", False)),
        dataloader_workers=coerce_int(payload.get("dataloader_workers"), 0, minimum=0),
        prefetch_factor=coerce_int(payload.get("prefetch_factor"), 1, minimum=1),
        pin_memory=coerce_bool(payload.get("pin_memory"), False),
        persistent_workers=coerce_bool(payload.get("persistent_workers"), False),
    )


# -----------------------------------------------------------------------------
def build_server_settings(data: dict[str, Any] | Any) -> ServerSettings:
    payload = ensure_mapping(data)
    database_payload = ensure_mapping(payload.get("database"))
    global_payload = ensure_mapping(payload.get("global"))
    jobs_payload = ensure_mapping(payload.get("jobs"))
    training_payload = ensure_mapping(payload.get("training"))

    return ServerSettings(
        database=build_database_settings(database_payload),
        global_settings=build_global_settings(global_payload),
        jobs=build_jobs_settings(jobs_payload),
        training=build_training_settings(training_payload),
    )


# [SERVER CONFIGURATION LOADER]
###############################################################################
def get_server_settings(config_path: str | None = None) -> ServerSettings:
    path = config_path or CONFIGURATION_FILE
    payload = load_configuration_data(path)

    return build_server_settings(payload)


server_settings = get_server_settings()

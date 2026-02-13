from __future__ import annotations

from dataclasses import dataclass


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

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from xreport_shared.domain.settings import ApplicationSettingsResponse


###############################################################################
class DatabaseConfigurationSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    backend: Literal["sqlite", "postgresql"]
    sqlite_path: str | None = None
    engine: str | None = None
    host: str | None = None
    port: int | None = None
    database_name: str | None = None
    username: str | None = None
    password_configured: bool
    ssl: bool
    ssl_ca: str | None = None
    connect_timeout: int
    insert_batch_size: int


###############################################################################
class DatabaseConfigurationStatusResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    state: Literal["ready", "recovery", "unavailable"]
    application: str
    schema_revision: str | None = None
    schema_head: str
    active: DatabaseConfigurationSummary | None = None
    suggested: DatabaseConfigurationSummary
    connected: bool
    schema_ready: bool
    jobs_running: bool
    error: str | None = None


###############################################################################
class DatabaseConnectionTestResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    success: bool
    backend: Literal["sqlite", "postgresql"]
    connected: bool
    schema_ready: bool
    detail: str


###############################################################################
class DatabaseSwitchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    active: DatabaseConfigurationSummary
    settings: ApplicationSettingsResponse


__all__ = [
    "DatabaseConfigurationStatusResponse",
    "DatabaseConfigurationSummary",
    "DatabaseConnectionTestResponse",
    "DatabaseSwitchResponse",
]

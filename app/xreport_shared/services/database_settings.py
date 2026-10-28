from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import re
from typing import Any

from sqlalchemy import column as sqlalchemy_column
from sqlalchemy import inspect, select
from sqlalchemy import table as sqlalchemy_table

from xreport_shared.common.path import DATABASE_FILE_PATH
from xreport_shared.configurations.database_config import (
    DatabaseConfiguration,
    DatabaseConfigurationError,
    read_database_configuration_status,
    save_database_configuration,
)
from xreport_shared.configurations.settings import (
    application_settings_to_server_settings,
)
from xreport_shared.domain.database_settings import (
    DatabaseConfigurationStatusResponse,
    DatabaseConfigurationSummary,
    DatabaseConnectionTestResponse,
    DatabaseSwitchResponse,
)
from xreport_shared.repositories.application_settings import ApplicationSettingsRepository
from xreport_shared.repositories.database.backend import (
    activate_database,
    finalize_database_activation,
    rollback_database_activation,
)
from xreport_shared.repositories.database.engine import Database
from xreport_shared.repositories.database.initializer import (
    database_ownership_error,
    schema_head,
)
from xreport_shared.repositories.database.schema import application_schema
from xreport_shared.services.jobs import get_job_manager
from xreport_shared.services.settings import SettingsService


###############################################################################
class DatabaseSwitchBusyError(RuntimeError):
    """Raised when database switching would interrupt an active background job."""


###############################################################################
@dataclass(frozen=True)
class DatabaseProbe:
    connected: bool
    schema_ready: bool
    # False when the database belongs to another schema and must not be used.
    compatible: bool
    revision: str | None
    detail: str


###############################################################################
@dataclass(frozen=True)
class DatabaseSwitchResult:
    response: DatabaseSwitchResponse
    server_settings: Any


###############################################################################
class DatabaseSettingsService:
    """Independent file-backed database configuration and runtime activation."""

    # -------------------------------------------------------------------------
    def get_status(
        self,
        *,
        database_ready: bool,
        startup_error: str | None,
    ) -> DatabaseConfigurationStatusResponse:
        stored = read_database_configuration_status()
        configuration = stored.configuration
        probe: DatabaseProbe | None = None
        connection_error = stored.error or startup_error
        if configuration is not None:
            probe = self._probe(configuration)
            if not probe.connected or not probe.schema_ready:
                connection_error = probe.detail
        connected = probe is not None and probe.connected
        schema_ready = probe is not None and probe.schema_ready

        if not database_ready:
            state = "recovery"
        elif not connected or not schema_ready:
            state = "unavailable"
        else:
            state = "ready"

        return DatabaseConfigurationStatusResponse(
            state=state,
            application=application_schema().title,
            schema_revision=probe.revision if probe is not None else None,
            schema_head=schema_head(),
            active=self._summary(configuration) if configuration else None,
            suggested=self._summary(
                DatabaseConfiguration(
                    backend="sqlite",
                    sqlite_path=DATABASE_FILE_PATH,
                )
            ),
            connected=connected,
            schema_ready=schema_ready,
            jobs_running=get_job_manager().is_job_running(),
            error=connection_error,
        )

    # -------------------------------------------------------------------------
    def test_connection(
        self, candidate: DatabaseConfiguration
    ) -> DatabaseConnectionTestResponse:
        resolved = self._preserve_saved_password(candidate)
        probe = self._probe(resolved)
        detail = probe.detail
        if probe.connected and probe.compatible and not probe.schema_ready:
            detail = "Connection succeeded. The application schema will be prepared when you switch."
        return DatabaseConnectionTestResponse(
            success=probe.connected and probe.compatible,
            backend=resolved.backend,
            connected=probe.connected,
            schema_ready=probe.schema_ready,
            detail=detail,
        )

    # -------------------------------------------------------------------------
    def switch_database(
        self, candidate: DatabaseConfiguration
    ) -> DatabaseSwitchResult:
        if get_job_manager().is_job_running():
            raise DatabaseSwitchBusyError(
                "Wait for active background jobs to finish before switching databases."
            )

        resolved = self._preserve_saved_password(candidate)
        target_settings = resolved.to_database_settings()
        candidate_database: Database | None = None
        activation = None
        try:
            from xreport_shared.repositories.database.initializer import (
                prepare_database_for_startup,
            )

            prepare_database_for_startup(target_settings)
            candidate_database = Database(target_settings)
            values = ApplicationSettingsRepository(candidate_database).get_settings()
            server_settings = application_settings_to_server_settings(
                values, target_settings
            )
            from xreport_shared.services.startup_validation import run_startup_validations

            server_settings = run_startup_validations(server_settings)
            probe = self._probe(resolved)
            if not probe.connected or not probe.schema_ready:
                raise DatabaseConfigurationError(
                    "The target database did not pass its connection and schema checks: "
                    + probe.detail
                )

            response = DatabaseSwitchResponse(
                active=self._summary(resolved),
                settings=SettingsService._response(values),
            )
            activation = activate_database(candidate_database)
            save_database_configuration(resolved)
            from xreport_shared.configurations.startup import get_database_settings

            get_database_settings.cache_clear()
            finalize_database_activation(activation)
            return DatabaseSwitchResult(
                response=response,
                server_settings=server_settings,
            )
        except Exception:
            if activation is not None:
                rollback_database_activation(activation)
            elif candidate_database is not None:
                candidate_database.engine.dispose()
            raise

    # -------------------------------------------------------------------------
    @staticmethod
    def _preserve_saved_password(
        candidate: DatabaseConfiguration,
    ) -> DatabaseConfiguration:
        if candidate.backend != "postgresql" or candidate.password:
            return candidate
        stored = read_database_configuration_status().configuration
        if stored is None or stored.backend != "postgresql" or not stored.password:
            return candidate
        same_identity = (
            stored.host == candidate.host
            and stored.port == candidate.port
            and stored.database_name == candidate.database_name
            and stored.username == candidate.username
        )
        return (
            candidate.model_copy(update={"password": stored.password})
            if same_identity
            else candidate
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def _probe(configuration: DatabaseConfiguration) -> DatabaseProbe:
        schema = application_schema()
        database: Database | None = None
        try:
            database = Database(configuration.to_database_settings())
            with database.engine.connect() as connection:
                table_names = set(inspect(connection).get_table_names())
                ownership_error = database_ownership_error(table_names, schema)
                if ownership_error:
                    return DatabaseProbe(True, False, False, None, ownership_error)
                if schema.version_table not in table_names:
                    if table_names:
                        return DatabaseProbe(
                            True,
                            False,
                            False,
                            None,
                            f"The database contains tables but no {schema.title} schema.",
                        )
                    return DatabaseProbe(
                        True,
                        False,
                        True,
                        None,
                        "Connection succeeded, but the application schema is not initialized.",
                    )
                revisions = (
                    connection.execute(
                        select(sqlalchemy_column("version_num")).select_from(
                            sqlalchemy_table(schema.version_table)
                        )
                    )
                    .scalars()
                    .all()
                )
                revision = revisions[0] if len(revisions) == 1 else None
                if revision != schema_head(schema):
                    return DatabaseProbe(
                        True,
                        False,
                        True,
                        revision,
                        "Connection succeeded, but the schema revision is not current.",
                    )
                return DatabaseProbe(
                    True, True, True, revision, "Connection and application schema are ready."
                )
        except Exception as exc:  # noqa: BLE001
            return DatabaseProbe(False, False, False, None, _sanitize_database_error(exc))
        finally:
            if database is not None:
                database.engine.dispose()

    # -------------------------------------------------------------------------
    @staticmethod
    def _summary(
        configuration: DatabaseConfiguration,
    ) -> DatabaseConfigurationSummary:
        values: dict[str, Any] = configuration.model_dump(
            mode="json", exclude={"password"}
        )
        values["password_configured"] = bool(configuration.password)
        return DatabaseConfigurationSummary.model_validate(values)


###############################################################################
def _sanitize_database_error(error: Exception) -> str:
    message = str(error).splitlines()[0][:280]
    message = re.sub(r"(?i)(password\s*[=:]\s*)[^\s,;]+", r"\1<redacted>", message)
    message = re.sub(r"(?i)(://[^:/\s]+:)[^@\s]+@", r"\1<redacted>@", message)
    return message or "The database connection could not be established."


###############################################################################
@lru_cache(maxsize=1)
def get_database_settings_service() -> DatabaseSettingsService:
    return DatabaseSettingsService()


__all__ = [
    "DatabaseSettingsService",
    "DatabaseSwitchBusyError",
    "DatabaseSwitchResult",
    "get_database_settings_service",
]

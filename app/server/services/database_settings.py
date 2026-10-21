from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import re
from typing import Any

from sqlalchemy import inspect, text

from server.common.path import DATABASE_FILE_PATH, ENV_FILE_PATH
from server.configurations.database_config import (
    DatabaseConfiguration,
    DatabaseConfigurationError,
    read_database_configuration_status,
    save_database_configuration,
)
from server.configurations.settings import (
    application_settings_to_server_settings,
)
from server.domain.database_settings import (
    DatabaseConfigurationStatusResponse,
    DatabaseConfigurationSummary,
    DatabaseConnectionTestResponse,
    DatabaseSwitchResponse,
)
from server.repositories.application_settings import ApplicationSettingsRepository
from server.repositories.database.backend import (
    activate_database,
    finalize_database_activation,
    rollback_database_activation,
)
from server.repositories.database.engine import Database
from server.repositories.database.initializer import HEAD_REVISION
from server.services.jobs import get_job_manager
from server.services.settings import SettingsService


class DatabaseSwitchBusyError(RuntimeError):
    """Raised when database switching would interrupt an active background job."""


@dataclass(frozen=True)
class DatabaseSwitchResult:
    response: DatabaseSwitchResponse
    server_settings: Any


class DatabaseSettingsService:
    """Independent file-backed database configuration and runtime activation."""

    def get_status(
        self,
        *,
        database_ready: bool,
        startup_error: str | None,
    ) -> DatabaseConfigurationStatusResponse:
        stored = read_database_configuration_status()
        configuration = stored.configuration
        connected = False
        schema_ready = False
        connection_error = stored.error or startup_error
        if configuration is not None:
            connected, schema_ready, detail = self._probe(configuration)
            if not connected or not schema_ready:
                connection_error = detail

        if not database_ready:
            state = "recovery"
        elif not connected or not schema_ready:
            state = "unavailable"
        else:
            state = "ready"

        return DatabaseConfigurationStatusResponse(
            state=state,
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

    def test_connection(
        self, candidate: DatabaseConfiguration
    ) -> DatabaseConnectionTestResponse:
        resolved = self._preserve_saved_password(candidate)
        connected, schema_ready, detail = self._probe(resolved)
        if connected and not schema_ready:
            detail = "Connection succeeded. The application schema will be prepared when you switch."
        elif connected:
            detail = "Connection and application schema are ready."
        return DatabaseConnectionTestResponse(
            success=connected,
            backend=resolved.backend,
            connected=connected,
            schema_ready=schema_ready,
            detail=detail,
        )

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
            from server.repositories.database.initializer import (
                prepare_database_for_startup,
            )

            prepare_database_for_startup(target_settings)
            candidate_database = Database(target_settings)
            values = ApplicationSettingsRepository(candidate_database).get_settings()
            server_settings = application_settings_to_server_settings(
                values, target_settings
            )
            from server.services.startup_validation import run_startup_validations

            server_settings = run_startup_validations(server_settings)
            connected, schema_ready, detail = self._probe(resolved)
            if not connected or not schema_ready:
                raise DatabaseConfigurationError(
                    "The target database did not pass its connection and schema checks: "
                    + detail
                )

            response = DatabaseSwitchResponse(
                active=self._summary(resolved),
                settings=SettingsService._response(values),
            )
            activation = activate_database(candidate_database)
            save_database_configuration(resolved, environment_file=ENV_FILE_PATH)
            from server.configurations.startup import get_database_settings

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

    @staticmethod
    def _probe(
        configuration: DatabaseConfiguration,
    ) -> tuple[bool, bool, str]:
        database: Database | None = None
        try:
            database = Database(configuration.to_database_settings())
            with database.engine.connect() as connection:
                connection.execute(text("SELECT 1"))
                inspector = inspect(connection)
                if not inspector.has_table("alembic_version"):
                    return True, False, "Connection succeeded, but the application schema is not initialized."
                if not inspector.has_table("application_settings"):
                    return True, False, "Connection succeeded, but the application settings table is missing."
                revisions = connection.execute(
                    text("SELECT version_num FROM alembic_version")
                ).scalars().all()
                if revisions != [HEAD_REVISION]:
                    return (
                        True,
                        False,
                        "Connection succeeded, but the schema revision is not current.",
                    )
                return True, True, "Connection and application schema are ready."
        except Exception as exc:  # noqa: BLE001
            message = _sanitize_database_error(exc)
            return False, False, message
        finally:
            if database is not None:
                database.engine.dispose()

    @staticmethod
    def _summary(
        configuration: DatabaseConfiguration,
    ) -> DatabaseConfigurationSummary:
        values: dict[str, Any] = configuration.model_dump(
            mode="json", exclude={"password"}
        )
        values["password_configured"] = bool(configuration.password)
        return DatabaseConfigurationSummary.model_validate(values)


def _sanitize_database_error(error: Exception) -> str:
    message = str(error).splitlines()[0][:280]
    message = re.sub(r"(?i)(password\s*[=:]\s*)[^\s,;]+", r"\1<redacted>", message)
    message = re.sub(r"(?i)(://[^:/\s]+:)[^@\s]+@", r"\1<redacted>@", message)
    return message or "The database connection could not be established."


@lru_cache(maxsize=1)
def get_database_settings_service() -> DatabaseSettingsService:
    return DatabaseSettingsService()


__all__ = [
    "DatabaseSettingsService",
    "DatabaseSwitchBusyError",
    "DatabaseSwitchResult",
    "get_database_settings_service",
]

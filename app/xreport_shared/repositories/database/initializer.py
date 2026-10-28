from __future__ import annotations

# Alembic discovers autogenerate plugins during import; set its logger before
# importing Alembic so normal application startup does not emit plugin noise.
# ruff: noqa: E402

from collections.abc import Iterator
from contextlib import contextmanager
import hashlib
import logging
import re
import time

logging.getLogger("alembic.runtime.plugins").setLevel(logging.WARNING)

from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config as AlembicConfig
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from alembic.util.exc import CommandError
import sqlalchemy
from sqlalchemy import CheckConstraint, MetaData, UniqueConstraint, inspect, text
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.sql.elements import TextClause

from xreport_shared.common.utils.logger import logger
from xreport_shared.configurations import DatabaseSettings, get_database_settings
from xreport_shared.repositories.database.engine import Database
from xreport_shared.repositories.database.schema import (
    SHARED_VERSION_TABLE,
    VERSION_TABLES,
    ApplicationSchema,
    application_schema,
    profile_title,
)
from xreport_shared.repositories.database.utils import normalize_postgres_engine

###############################################################################
class DatabaseSchemaError(RuntimeError):
    """Raised when an existing database cannot be safely adopted."""

###############################################################################
class DatabaseMigrationError(RuntimeError):
    """Raised when migration discovery, locking, or execution fails."""

###############################################################################
def _postgres_database_exists_sql() -> str:
    return "SELECT 1 FROM pg_database WHERE datname=:name"

###############################################################################
def _create_database_sql(database_name: str) -> str:
    safe_database = database_name.replace('"', '""')
    return (
        f"CREATE DATABASE \"{safe_database}\" WITH ENCODING 'UTF8' TEMPLATE template0"
    )

###############################################################################
def build_postgres_connect_args(settings: DatabaseSettings) -> dict[str, str | int]:
    connect_args: dict[str, str | int] = {
        "connect_timeout": settings.connect_timeout,
        "client_encoding": "utf8",
    }
    if settings.ssl:
        connect_args["sslmode"] = "require"
        if settings.ssl_ca:
            connect_args["sslrootcert"] = settings.ssl_ca
    return connect_args

###############################################################################
def build_postgres_url(settings: DatabaseSettings, database_name: str) -> str:
    port = settings.port or 5432
    engine_name = normalize_postgres_engine(settings.engine)
    url = sqlalchemy.engine.URL.create(
        engine_name,
        username=settings.username,
        password=settings.password or "",
        host=settings.host,
        port=port,
        database=database_name,
    )
    return url.render_as_string(hide_password=False)

###############################################################################
def clone_settings_with_database(
    settings: DatabaseSettings, database_name: str
) -> DatabaseSettings:
    return DatabaseSettings(
        backend="postgresql",
        engine=settings.engine,
        host=settings.host,
        port=settings.port,
        database_name=database_name,
        username=settings.username,
        password=settings.password,
        ssl=settings.ssl,
        ssl_ca=settings.ssl_ca,
        connect_timeout=settings.connect_timeout,
        insert_batch_size=settings.insert_batch_size,
    )

###############################################################################
def build_postgres_create_database_sql(database_name: str) -> TextClause:
    return sqlalchemy.text(_create_database_sql(database_name))

###############################################################################
def _database_label(settings: DatabaseSettings) -> str:
    if settings.backend == "sqlite":
        return str(settings.sqlite_path or "<unconfigured SQLite path>")
    return settings.database_name or "<unnamed PostgreSQL database>"

###############################################################################
def _advisory_key(name: str) -> int:
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=True)

###############################################################################
def _acquire_postgres_lock(
    connection: Connection,
    *,
    key: int,
    timeout: int,
    transaction: bool,
    name: str,
) -> None:
    function = "pg_try_advisory_xact_lock" if transaction else "pg_try_advisory_lock"
    deadline = time.monotonic() + timeout
    while True:
        acquired = bool(
            connection.execute(text(f"SELECT {function}(:key)"), {"key": key}).scalar()
        )
        if acquired:
            logger.debug("Acquired PostgreSQL %s lock", name)
            return
        if time.monotonic() >= deadline:
            raise DatabaseMigrationError(
                f"Timed out waiting for the PostgreSQL {name} lock after {timeout} seconds."
            )
        time.sleep(0.1)

###############################################################################
def _release_postgres_session_lock(connection: Connection, key: int) -> None:
    connection.execute(text("SELECT pg_advisory_unlock(:key)"), {"key": key})

###############################################################################
def _format_diffs(diffs: list[object]) -> str:
    if not diffs:
        return ""
    rendered = "; ".join(repr(diff) for diff in diffs[:12])
    if len(diffs) > 12:
        rendered += f"; ... ({len(diffs) - 12} more)"
    return rendered

###############################################################################
def _normalize_check_expression(expression: object) -> str:
    normalized = re.sub(r"\s+", " ", str(expression).strip().lower())
    normalized = re.sub(
        r"([<>=])\s*'(-?\d+(?:\.\d+)?)'::(?:numeric|integer|bigint|smallint|real|double precision)\b",
        r"\1\2",
        normalized,
    )
    normalized = re.sub(
        r"::(?:character varying|double precision|[a-z_][a-z0-9_]*)(?:\s*\[\])?",
        "",
        normalized,
    )
    normalized = re.sub(
        r"=\s*any\s*\(\s*array\s*\[([^]]*)\]\s*(?:\[\])?\s*\)",
        r"in (\1)",
        normalized,
    )
    normalized = re.sub(r"\s*([(),><=])\s*", r"\1", normalized)
    normalized = re.sub(r"\b(and|or)\(", r"\1 (", normalized)
    normalized = re.sub(
        r"\(([a-z_][a-z0-9_.]*\s+(?:is\s+(?:not\s+)?null|in\([^()]*\)))\)",
        r"\1",
        normalized,
    )
    while normalized.startswith("(") and normalized.endswith(")"):
        depth = 0
        balanced = True
        for index, character in enumerate(normalized):
            if character == "(":
                depth += 1
            elif character == ")":
                depth -= 1
                if depth == 0 and index != len(normalized) - 1:
                    balanced = False
                    break
        if not balanced:
            break
        normalized = normalized[1:-1]
    return normalized

###############################################################################
def _semantic_constraint_diffs(connection: Connection, metadata: MetaData) -> list[str]:
    inspector = inspect(connection)
    diffs: list[str] = []
    for table in metadata.tables.values():
        table_name = str(table.name)
        expected_primary_key = tuple(
            column.name for column in table.primary_key.columns
        )
        actual_primary_key = tuple(
            inspector.get_pk_constraint(table_name).get("constrained_columns") or ()
        )
        if actual_primary_key != expected_primary_key:
            diffs.append(
                f"table '{table_name}' primary key differs "
                f"(expected {expected_primary_key}, got {actual_primary_key})"
            )

        expected_unique = sorted(
            (
                constraint.name,
                tuple(column.name for column in constraint.columns),
            )
            for constraint in table.constraints
            if isinstance(constraint, UniqueConstraint)
        )
        actual_unique = sorted(
            (
                constraint.get("name"),
                tuple(constraint.get("column_names") or ()),
            )
            for constraint in inspector.get_unique_constraints(table_name)
        )
        if actual_unique != expected_unique:
            diffs.append(
                f"table '{table_name}' unique constraints differ "
                f"(expected {expected_unique}, got {actual_unique})"
            )

        expected_foreign_keys = sorted(
            (
                tuple(constraint.columns.keys()),
                str(constraint.referred_table.name),
                tuple(element.column.name for element in constraint.elements),
                tuple(
                    sorted(
                        (key, str(value).lower())
                        for key, value in {
                            "ondelete": constraint.ondelete,
                            "onupdate": constraint.onupdate,
                        }.items()
                        if value is not None
                    )
                ),
            )
            for constraint in table.foreign_key_constraints
        )
        actual_foreign_keys = sorted(
            (
                tuple(constraint.get("constrained_columns") or ()),
                str(constraint.get("referred_table")),
                tuple(constraint.get("referred_columns") or ()),
                tuple(
                    sorted(
                        (key, str(value).lower())
                        for key, value in (constraint.get("options") or {}).items()
                        if value is not None
                    )
                ),
            )
            for constraint in inspector.get_foreign_keys(table_name)
        )
        if actual_foreign_keys != expected_foreign_keys:
            diffs.append(
                f"table '{table_name}' foreign keys differ "
                f"(expected {expected_foreign_keys}, got {actual_foreign_keys})"
            )

        expected_indexes = sorted(
            (
                index.name,
                tuple(column.name for column in index.columns),
                bool(index.unique),
            )
            for index in table.indexes
        )
        actual_indexes = sorted(
            (
                index.get("name"),
                tuple(index.get("column_names") or ()),
                bool(index.get("unique")),
            )
            for index in inspector.get_indexes(table_name)
            if index.get("duplicates_constraint") is None
        )
        if actual_indexes != expected_indexes:
            diffs.append(
                f"table '{table_name}' indexes differ "
                f"(expected {expected_indexes}, got {actual_indexes})"
            )

        expected_checks = sorted(
            (constraint.name, _normalize_check_expression(constraint.sqltext))
            for constraint in table.constraints
            if isinstance(constraint, CheckConstraint)
        )
        actual_checks = sorted(
            (
                constraint.get("name"),
                _normalize_check_expression(constraint.get("sqltext", "")),
            )
            for constraint in inspector.get_check_constraints(table_name)
        )
        if actual_checks != expected_checks:
            diffs.append(
                f"table '{table_name}' check constraints differ "
                f"(expected {expected_checks}, got {actual_checks})"
            )
    return diffs

###############################################################################
def _validate_head_schema(connection: Connection, schema: ApplicationSchema) -> None:
    def include_object(
        _object: object,
        name: str,
        object_type: str,
        _reflected: bool,
        _compare_to: object,
    ) -> bool:
        return not (object_type == "table" and name == schema.version_table)

    migration_context = MigrationContext.configure(
        connection,
        opts={"compare_type": True, "include_object": include_object},
    )
    try:
        diffs = compare_metadata(migration_context, schema.metadata)
        diffs.extend(_semantic_constraint_diffs(connection, schema.metadata))
    except Exception as exc:
        raise DatabaseSchemaError(
            "Unable to verify the database schema against the current ORM metadata."
        ) from exc
    if diffs:
        raise DatabaseSchemaError(
            "Database is at the Alembic head but differs from the current ORM schema: "
            f"{_format_diffs(diffs)}"
        )

###############################################################################
def _build_alembic_config(
    schema: ApplicationSchema, connection: Connection | None = None
) -> AlembicConfig:
    if not schema.migrations_dir.is_dir():
        raise DatabaseMigrationError(
            f"Migration directory is missing: {schema.migrations_dir}"
        )
    config = AlembicConfig()
    config.set_main_option("script_location", str(schema.migrations_dir))
    if connection is not None:
        config.attributes["connection"] = connection
    return config

###############################################################################
def schema_head(schema: ApplicationSchema | None = None) -> str:
    """Return the single Alembic head of an application's migration stream."""
    resolved = schema or application_schema()
    script = ScriptDirectory.from_config(_build_alembic_config(resolved))
    heads = tuple(script.get_heads())
    if len(heads) != 1:
        raise DatabaseMigrationError(
            f"{resolved.title} requires exactly one Alembic head; discovered: "
            + (", ".join(heads) or "none")
        )
    return heads[0]

###############################################################################
def database_ownership_error(
    table_names: set[str], schema: ApplicationSchema
) -> str | None:
    """Explain why a database belongs to another schema, or return None."""
    for profile, version_table in VERSION_TABLES.items():
        if profile != schema.profile and version_table in table_names:
            return (
                f"This database belongs to {profile_title(profile)}. "
                f"Choose a separate database for {schema.title}."
            )
    if SHARED_VERSION_TABLE in table_names:
        return (
            "This database uses the retired shared XREPORT schema. "
            "Choose a new database or delete and recreate this one."
        )
    return None

###############################################################################
def _pending_revisions(
    script: ScriptDirectory,
    current: tuple[str, ...],
    head: str,
) -> list[tuple[str, str]]:
    lower = current[0] if current else None
    try:
        revisions = list(script.iterate_revisions(head, lower))
    except Exception as exc:
        current_label = ", ".join(current) or "base"
        raise DatabaseMigrationError(
            f"Migration history cannot reach head {head!r} from {current_label!r}."
        ) from exc
    revisions.reverse()
    return [
        (str(revision.revision), revision.doc or "(no migration message)")
        for revision in revisions
    ]

###############################################################################
def _current_revisions(
    connection: Connection, schema: ApplicationSchema
) -> tuple[str, ...]:
    return tuple(
        MigrationContext.configure(
            connection, opts={"version_table": schema.version_table}
        ).get_current_heads()
    )

###############################################################################
def _run_migrations_on_connection(
    connection: Connection,
    settings: DatabaseSettings,
    schema: ApplicationSchema,
) -> None:
    table_names = set(inspect(connection).get_table_names())
    ownership_error = database_ownership_error(table_names, schema)
    if ownership_error:
        raise DatabaseSchemaError(ownership_error)

    config = _build_alembic_config(schema, connection)
    script = ScriptDirectory.from_config(config)
    head = schema_head(schema)

    current = _current_revisions(connection, schema)
    if len(current) > 1:
        raise DatabaseMigrationError(
            "Database contains multiple Alembic revisions: " + ", ".join(current)
        )
    for revision in current:
        try:
            script.get_revision(revision)
        except Exception as exc:
            raise DatabaseMigrationError(
                f"Database revision {revision!r} is not present in this application. "
                "Recreate or replace the database before startup."
            ) from exc

    application_tables = table_names - {schema.version_table}
    if not current and application_tables:
        raise DatabaseSchemaError(
            "Non-empty database has no Alembic revision; refusing implicit schema "
            "adoption. Run an explicit migration or data import before startup."
        )

    pending = _pending_revisions(script, current, head)
    current_label = ", ".join(current) if current else "base"
    logger.info(
        "%s database migration check (backend=%s, database=%s, current=%s, head=%s)",
        schema.title,
        settings.backend,
        _database_label(settings),
        current_label,
        head,
    )
    if not pending:
        _validate_head_schema(connection, schema)
        logger.info("Database is already at Alembic head %s", head)
    else:
        for revision, message in pending:
            logger.info("Pending migration %s: %s", revision, message)
        command.upgrade(config, head)
        final_heads = _current_revisions(connection, schema)
        if final_heads != (head,):
            raise DatabaseMigrationError(
                "Alembic upgrade completed without reaching the expected head "
                f"{head}; current revisions: {', '.join(final_heads) or 'base'}"
            )
        _validate_head_schema(connection, schema)
        logger.info("Database migrations applied through Alembic head %s", head)

###############################################################################
@contextmanager
def _sqlite_migration_connection(engine: Engine) -> Iterator[Connection]:
    with engine.connect() as connection:
        try:
            connection.exec_driver_sql("BEGIN EXCLUSIVE")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise

###############################################################################
@contextmanager
def _postgres_migration_connection(
    engine: Engine,
    settings: DatabaseSettings,
    schema: ApplicationSchema,
) -> Iterator[Connection]:
    with engine.connect() as connection:
        with connection.begin():
            _acquire_postgres_lock(
                connection,
                key=_advisory_key(
                    f"xreport:{schema.profile}:migrate:{settings.host}:{settings.port}:{settings.database_name}"
                ),
                timeout=settings.connect_timeout,
                transaction=True,
                name="migration",
            )
            yield connection

###############################################################################
def _migrate_repository(
    repository: Database, settings: DatabaseSettings, schema: ApplicationSchema
) -> None:
    if settings.backend == "sqlite":
        with _sqlite_migration_connection(repository.engine) as connection:
            _run_migrations_on_connection(connection, settings, schema)
        return
    with _postgres_migration_connection(
        repository.engine, settings, schema
    ) as connection:
        _run_migrations_on_connection(connection, settings, schema)

###############################################################################
def _validate_postgres_engine(settings: DatabaseSettings) -> None:
    engine_name = normalize_postgres_engine(settings.engine).lower()
    if engine_name not in {
        "postgres",
        "postgresql",
        "postgresql+psycopg",
        "postgresql+psycopg2",
    }:
        raise ValueError(f"Unsupported database engine: {settings.engine}")

###############################################################################
def _ensure_postgres_database(settings: DatabaseSettings) -> str:
    if not settings.host:
        raise ValueError("Database host is required for PostgreSQL initialization.")
    if not settings.username:
        raise ValueError("Database username is required for PostgreSQL initialization.")
    if not settings.database_name:
        raise ValueError("Database name is required for PostgreSQL initialization.")

    target_database = settings.database_name
    admin_engine: Engine | None = None
    key = _advisory_key(
        f"xreport:create:{settings.host}:{settings.port}:{target_database}"
    )
    try:
        admin_engine = sqlalchemy.create_engine(
            build_postgres_url(settings, "postgres"),
            echo=False,
            future=True,
            connect_args=build_postgres_connect_args(settings),
            isolation_level="AUTOCOMMIT",
            pool_pre_ping=True,
        )
        try:
            with admin_engine.connect() as connection:
                _acquire_postgres_lock(
                    connection,
                    key=key,
                    timeout=settings.connect_timeout,
                    transaction=False,
                    name="database creation",
                )
                try:
                    exists = connection.execute(
                        text(_postgres_database_exists_sql()), {"name": target_database}
                    ).scalar()
                    if exists:
                        logger.info(
                            "PostgreSQL database %s already exists", target_database
                        )
                    else:
                        connection.execute(
                            build_postgres_create_database_sql(target_database)
                        )
                        logger.info("Created PostgreSQL database %s", target_database)
                finally:
                    _release_postgres_session_lock(connection, key)
        except SQLAlchemyError as exc:
            raise DatabaseMigrationError(
                "Unable to inspect or create the PostgreSQL database. "
                "Run database initialization with credentials that can access the "
                "administrative postgres database and have CREATEDB privilege."
            ) from exc
    finally:
        if admin_engine is not None:
            admin_engine.dispose()
    return target_database

###############################################################################
def initialize_sqlite_database(
    settings: DatabaseSettings, schema: ApplicationSchema
) -> None:
    repository = Database(settings)
    try:
        _migrate_repository(repository, settings, schema)
    finally:
        repository.engine.dispose()
    logger.info("SQLite database is synchronized at %s", repository.db_path)

###############################################################################
def initialize_postgres_database(
    settings: DatabaseSettings, schema: ApplicationSchema
) -> str:
    _validate_postgres_engine(settings)
    target_database = _ensure_postgres_database(settings)
    repository = Database(clone_settings_with_database(settings, target_database))
    try:
        _migrate_repository(repository, settings, schema)
    finally:
        repository.engine.dispose()
    logger.info("PostgreSQL database %s is synchronized", target_database)
    return target_database

###############################################################################
def verify_postgres_connection(settings: DatabaseSettings) -> None:
    repository = Database(settings)
    try:
        with repository.engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    finally:
        repository.engine.dispose()
    logger.info("Verified PostgreSQL connection to %s", settings.database_name)

###############################################################################
def run_database_initialization(
    settings: DatabaseSettings, schema: ApplicationSchema | None = None
) -> None:
    resolved_schema = schema or application_schema()
    if settings.backend == "sqlite":
        initialize_sqlite_database(settings, resolved_schema)
        return
    initialize_postgres_database(settings, resolved_schema)

###############################################################################
def _sanitize_error(message: str) -> str:
    sanitized = re.sub(
        r"(?i)(password\s*[=:]\s*)[^\s,;]+",
        r"\1<redacted>",
        message,
    )
    sanitized = re.sub(
        r"(?i)(://[^:/\s]+:)[^@\s]+@",
        r"\1<redacted>@",
        sanitized,
    )
    return sanitized.replace("\\", "/")

###############################################################################
def _run_database_action(
    action: str,
    settings: DatabaseSettings,
    schema: ApplicationSchema | None,
) -> None:
    try:
        run_database_initialization(settings, schema)
    except (DatabaseSchemaError, DatabaseMigrationError) as exc:
        message = _sanitize_error(str(exc))
        logger.error("%s failed: %s", action, message)
        raise RuntimeError(f"{action} failed: {message}") from exc
    except (CommandError, SQLAlchemyError, ValueError) as exc:
        message = _sanitize_error(str(exc))
        logger.error(
            "%s failed (%s): %s",
            action,
            type(exc).__name__,
            message,
        )
        raise RuntimeError(f"{action} failed: {message}") from exc
    except Exception as exc:
        message = _sanitize_error(str(exc))
        logger.error(
            "Unexpected error during %s (%s): %s",
            action.lower(),
            type(exc).__name__,
            message,
        )
        raise RuntimeError(
            f"Unexpected error during {action.lower()}: {message}"
        ) from exc

###############################################################################
def initialize_database(
    settings: DatabaseSettings | None = None,
    schema: ApplicationSchema | None = None,
) -> None:
    _run_database_action(
        "Database initialization",
        settings or get_database_settings(),
        schema,
    )

###############################################################################
def prepare_database_for_startup(
    settings: DatabaseSettings | None = None,
    schema: ApplicationSchema | None = None,
) -> None:
    _run_database_action(
        "Database startup migration",
        settings or get_database_settings(),
        schema,
    )

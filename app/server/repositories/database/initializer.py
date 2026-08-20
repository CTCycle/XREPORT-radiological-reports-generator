from __future__ import annotations

from collections.abc import Callable
import urllib.parse

import sqlalchemy
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.sql.elements import TextClause

from server.configurations import DatabaseSettings, get_server_settings
from server.common.path import DATABASE_FILE_PATH
from server.repositories.database.engine import Database
from server.repositories.database.utils import normalize_postgres_engine
from server.common.utils.logger import logger
from server.repositories.schemas import Base, SchemaMetadata

SCHEMA_METADATA_KEY = "xreport"
CURRENT_SCHEMA_VERSION = 1

###############################################################################
def _postgres_database_exists_sql() -> str:
    return "SELECT 1 FROM pg_database WHERE datname=:name"

###############################################################################
def _create_database_sql(database_name: str) -> str:
    safe_database = database_name.replace('"', '""')
    return f'CREATE DATABASE "{safe_database}" WITH ENCODING \'UTF8\' TEMPLATE template0'

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
    safe_username = urllib.parse.quote_plus(settings.username or "")
    safe_password = urllib.parse.quote_plus(settings.password or "")
    return (
        f"{engine_name}://{safe_username}:{safe_password}"
        f"@{settings.host}:{port}/{database_name}"
    )

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
def build_postgres_create_database_sql(
    database_name: str,
) -> TextClause:
    return sqlalchemy.text(_create_database_sql(database_name))

###############################################################################
class DatabaseSchemaError(RuntimeError):
    """Raised when a database cannot be safely used by the current release."""

###############################################################################
def _schema_mismatches(engine: sqlalchemy.Engine) -> list[str]:
    inspector = sqlalchemy.inspect(engine)
    mismatches: list[str] = []
    metadata_table_name = SchemaMetadata.__table__.name

    for table_name, table in Base.metadata.tables.items():
        if table_name == metadata_table_name and not inspector.has_table(table_name):
            continue
        if not inspector.has_table(table_name):
            mismatches.append(f"missing table '{table_name}'")
            continue
        actual_columns = {
            str(column["name"]) for column in inspector.get_columns(table_name)
        }
        missing_columns = sorted(
            str(column.name)
            for column in table.columns
            if str(column.name) not in actual_columns
        )
        if missing_columns:
            mismatches.append(
                f"table '{table_name}' is missing columns: {', '.join(missing_columns)}"
            )
    return mismatches

###############################################################################
def _write_schema_version(repository: Database) -> None:
    with repository.transaction() as session:
        metadata = session.get(SchemaMetadata, SCHEMA_METADATA_KEY)
        if metadata is None:
            session.add(
                SchemaMetadata(
                    schema_name=SCHEMA_METADATA_KEY,
                    schema_version=CURRENT_SCHEMA_VERSION,
                )
            )
            return
        metadata.schema_version = CURRENT_SCHEMA_VERSION

###############################################################################
def _validate_schema_compatibility(repository: Database) -> None:
    mismatches = _schema_mismatches(repository.engine)
    if mismatches:
        raise DatabaseSchemaError(
            f"Database schema is incompatible with XREPORT schema version "
            f"{CURRENT_SCHEMA_VERSION}: {'; '.join(mismatches)}. "
            "Recreate the database or apply a supported migration before restarting."
        )

    inspector = sqlalchemy.inspect(repository.engine)
    if not inspector.has_table(SchemaMetadata.__table__.name):
        SchemaMetadata.__table__.create(repository.engine)
        _write_schema_version(repository)
        logger.info(
            "Stamped existing database with XREPORT schema version %s",
            CURRENT_SCHEMA_VERSION,
        )
        return

    with repository.read_session() as session:
        metadata = session.get(SchemaMetadata, SCHEMA_METADATA_KEY)
    if metadata is None:
        raise DatabaseSchemaError(
            "Database schema metadata is missing the XREPORT version marker. "
            "Recreate the database or apply a supported migration before restarting."
        )
    if metadata.schema_version != CURRENT_SCHEMA_VERSION:
        raise DatabaseSchemaError(
            f"Database schema version {metadata.schema_version} is not supported; "
            f"this release requires version {CURRENT_SCHEMA_VERSION}. "
            "Recreate the database or apply a supported migration before restarting."
        )

###############################################################################
def verify_schema_compatibility(settings: DatabaseSettings) -> None:
    repository = Database(settings)
    try:
        _validate_schema_compatibility(repository)
    finally:
        repository.engine.dispose()

###############################################################################
def initialize_sqlite_database(settings: DatabaseSettings) -> None:
    if DATABASE_FILE_PATH.is_file():
        verify_schema_compatibility(settings)
        logger.info("Verified SQLite database at %s", DATABASE_FILE_PATH)
        return

    repository = Database(settings)
    try:
        Base.metadata.create_all(repository.engine)
        _write_schema_version(repository)
    finally:
        repository.engine.dispose()
    logger.info("Initialized SQLite database at %s", repository.db_path)

###############################################################################
def initialize_postgres_database(settings: DatabaseSettings) -> str:
    if not settings.host:
        raise ValueError("Database host is required for PostgreSQL initialization.")
    if not settings.username:
        raise ValueError("Database username is required for PostgreSQL initialization.")
    if not settings.database_name:
        raise ValueError("Database name is required for PostgreSQL initialization.")

    target_database = settings.database_name
    connect_args = build_postgres_connect_args(settings)

    admin_url = build_postgres_url(settings, "postgres")
    admin_engine = sqlalchemy.create_engine(
        admin_url,
        echo=False,
        future=True,
        connect_args=connect_args,
        isolation_level="AUTOCOMMIT",
        pool_pre_ping=True,
    )

    try:
        with admin_engine.connect() as conn:
            exists = conn.execute(
                sqlalchemy.text(_postgres_database_exists_sql()),
                {"name": target_database},
            ).scalar()
            if exists:
                logger.info("PostgreSQL database %s already exists", target_database)
            else:
                conn.execute(build_postgres_create_database_sql(target_database))
                logger.info("Created PostgreSQL database %s", target_database)
    finally:
        admin_engine.dispose()

    normalized_settings = clone_settings_with_database(settings, target_database)
    repository = Database(normalized_settings)
    try:
        Base.metadata.create_all(repository.engine)
        _write_schema_version(repository)
    finally:
        repository.engine.dispose()
    logger.info("Ensured PostgreSQL tables exist in %s", target_database)

    return target_database

###############################################################################
def verify_postgres_connection(settings: DatabaseSettings) -> None:
    repository = Database(settings)
    try:
        with repository.engine.connect() as connection:
            connection.execute(sqlalchemy.text("SELECT 1"))
    finally:
        repository.engine.dispose()
    logger.info("Verified PostgreSQL connection to %s", settings.database_name)

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
def run_database_initialization(settings: DatabaseSettings) -> None:
    if settings.backend == "sqlite":
        initialize_sqlite_database(settings)
        return

    _validate_postgres_engine(settings)
    initialize_postgres_database(settings)

###############################################################################
def _run_database_action(
    action: str,
    operation: Callable[[DatabaseSettings], object],
    settings: DatabaseSettings,
) -> None:
    try:
        operation(settings)
    except DatabaseSchemaError as exc:
        logger.error("%s failed: %s", action, exc)
        raise RuntimeError(f"{action} failed: {exc}") from exc
    except (SQLAlchemyError, ValueError) as exc:
        logger.error("%s failed: %s", action, exc)
        raise RuntimeError(f"{action} failed.") from exc
    except Exception as exc:
        logger.exception("Unexpected error during %s.", action.lower())
        raise RuntimeError(f"Unexpected error during {action.lower()}.") from exc

###############################################################################
def initialize_database(settings: DatabaseSettings | None = None) -> None:
    resolved_settings = settings or get_server_settings().database
    _run_database_action(
        "Database initialization",
        run_database_initialization,
        resolved_settings,
    )

###############################################################################
def _prepare_sqlite_startup(settings: DatabaseSettings) -> None:
    initialize_sqlite_database(settings)

###############################################################################
def _prepare_postgres_startup(settings: DatabaseSettings) -> None:
    _validate_postgres_engine(settings)
    verify_postgres_connection(settings)
    verify_schema_compatibility(settings)

###############################################################################
def prepare_database_for_startup(settings: DatabaseSettings | None = None) -> None:
    resolved_settings = settings or get_server_settings().database
    operation = (
        _prepare_sqlite_startup
        if resolved_settings.backend == "sqlite"
        else _prepare_postgres_startup
    )
    _run_database_action("Database startup check", operation, resolved_settings)

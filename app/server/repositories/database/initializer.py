from __future__ import annotations

import urllib.parse

import sqlalchemy
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.sql.elements import TextClause

from server.configurations import DatabaseSettings, get_server_settings
from server.repositories.database.engine import Database
from server.repositories.database.utils import normalize_postgres_engine
from server.common.utils.logger import logger
from server.repositories.schemas import Base


INFERENCE_RUN_COLUMNS = {
    "checkpoint_id",
    "provider",
    "model_ref",
    "model_revision",
    "generation_profile",
    "generation_config_json",
    "clinical_context",
    "request_id",
    "status",
    "execution_time_seconds",
    "executed_at",
}

###############################################################################
def validate_current_schema(repository: Database) -> None:
    inspector = sqlalchemy.inspect(repository.engine)
    if not inspector.has_table("inference_runs"):
        return
    columns = {column["name"] for column in inspector.get_columns("inference_runs")}
    missing = INFERENCE_RUN_COLUMNS - columns
    if missing:
        backend = repository.engine.dialect.name
        raise ValueError(
            "Database schema predates the inference-first branch. "
            f"Recreate the {backend} database before startup; create_all does not migrate "
            f"columns. Missing inference columns: {', '.join(sorted(missing))}"
        )

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
def initialize_sqlite_database(settings: DatabaseSettings) -> None:
    repository = Database(settings)
    Base.metadata.create_all(repository.engine)
    validate_current_schema(repository)
    repository.engine.dispose()
    logger.info("Initialized SQLite database at %s", repository.db_path)

###############################################################################
def ensure_postgres_database(settings: DatabaseSettings) -> str:
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
    Base.metadata.create_all(repository.engine)
    validate_current_schema(repository)
    repository.engine.dispose()
    logger.info("Ensured PostgreSQL tables exist in %s", target_database)

    return target_database

###############################################################################
def run_database_initialization(settings: DatabaseSettings) -> None:
    if settings.backend == "sqlite":
        initialize_sqlite_database(settings)
        return

    engine_name = normalize_postgres_engine(settings.engine).lower()
    if engine_name not in {
        "postgres",
        "postgresql",
        "postgresql+psycopg",
        "postgresql+psycopg2",
    }:
        raise ValueError(f"Unsupported database engine: {settings.engine}")

    ensure_postgres_database(settings)

###############################################################################
def initialize_database(settings: DatabaseSettings | None = None) -> None:
    resolved_settings = settings or get_server_settings().database
    try:
        run_database_initialization(resolved_settings)
    except (SQLAlchemyError, ValueError) as exc:
        logger.error("Database initialization failed: %s", exc)
        raise RuntimeError("Database initialization failed.") from exc
    except Exception as exc:
        logger.exception("Unexpected error during database initialization.")
        raise RuntimeError("Unexpected error during database initialization.") from exc

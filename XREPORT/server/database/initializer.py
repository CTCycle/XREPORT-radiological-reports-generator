from __future__ import annotations

import urllib.parse

import sqlalchemy
from sqlalchemy.exc import SQLAlchemyError

from APP.server.utils.configurations import DatabaseSettings, server_settings
from APP.server.database.postgres import PostgresRepository
from APP.server.database.schema import Base
from APP.server.database.sqlite import SQLiteRepository
from APP.server.database.utils import normalize_postgres_engine
from APP.server.utils.logger import logger


###############################################################################
def build_postgres_connect_args(settings: DatabaseSettings) -> dict[str, str | int]:
    connect_args: dict[str, str | int] = {"connect_timeout": settings.connect_timeout}
    if settings.ssl:
        connect_args["sslmode"] = "require"
        if settings.ssl_ca:
            connect_args["sslrootcert"] = settings.ssl_ca
    return connect_args


# -----------------------------------------------------------------------------
def build_postgres_url(settings: DatabaseSettings, database_name: str) -> str:
    port = settings.port or 5432
    engine_name = normalize_postgres_engine(settings.engine)
    safe_username = urllib.parse.quote_plus(settings.username or "")
    safe_password = urllib.parse.quote_plus(settings.password or "")
    return (
        f"{engine_name}://{safe_username}:{safe_password}"
        f"@{settings.host}:{port}/{database_name}"
    )


# -----------------------------------------------------------------------------
def clone_settings_with_database(
    settings: DatabaseSettings, database_name: str
) -> DatabaseSettings:
    return DatabaseSettings(
        embedded_database=False,
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


# -----------------------------------------------------------------------------
def initialize_sqlite_database(settings: DatabaseSettings) -> None:
    repository = SQLiteRepository(settings)
    Base.metadata.create_all(repository.engine)
    logger.info("Initialized SQLite database at %s", repository.db_path)


# -----------------------------------------------------------------------------
def ensure_postgres_database(settings: DatabaseSettings) -> str:
    if not settings.host:
        raise ValueError("Database host is required for PostgreSQL initialization.")
    if not settings.username:
        raise ValueError("Database username is required for PostgreSQL initialization.")
    if not settings.database_name:
        raise ValueError("Database name is required for PostgreSQL initialization.")

    target_database = settings.database_name
    safe_database = target_database.replace('"', '""')
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

    with admin_engine.connect() as conn:
        exists = conn.execute(
            sqlalchemy.text("SELECT 1 FROM pg_database WHERE datname=:name"),
            {"name": target_database},
        ).scalar()
        if exists:
            logger.info("PostgreSQL database %s already exists", target_database)
        else:
            conn.execute(sqlalchemy.text(f'CREATE DATABASE "{safe_database}"'))
            logger.info("Created PostgreSQL database %s", target_database)

    normalized_settings = clone_settings_with_database(settings, target_database)
    repository = PostgresRepository(normalized_settings)
    Base.metadata.create_all(repository.engine)
    logger.info("Ensured PostgreSQL tables exist in %s", target_database)

    return target_database


# -----------------------------------------------------------------------------
def run_database_initialization() -> None:
    settings = server_settings.database
    if settings.embedded_database:
        initialize_sqlite_database(settings)
        return

    engine_name = normalize_postgres_engine(settings.engine).lower()
    if engine_name not in {"postgres", "postgresql", "postgresql+psycopg", "postgresql+psycopg2"}:
        raise ValueError(f"Unsupported database engine: {settings.engine}")

    ensure_postgres_database(settings)


# -----------------------------------------------------------------------------
def initialize_database() -> None:
    try:
        run_database_initialization()
    except (SQLAlchemyError, ValueError) as exc:
        logger.error("Database initialization failed: %s", exc)
        raise SystemExit(1) from exc
    except Exception as exc:
        logger.exception("Unexpected error during database initialization.")
        raise SystemExit(1) from exc

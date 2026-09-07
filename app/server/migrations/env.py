"""Alembic environment wired to XREPORT's synchronous SQLAlchemy metadata."""

from __future__ import annotations

from alembic import context
from sqlalchemy import engine_from_config, pool
from sqlalchemy.engine import Connection, Engine

from server.configurations import get_server_settings
from server.repositories.database.engine import Database
from server.repositories.schemas import Base


config = context.config
target_metadata = Base.metadata

###############################################################################
def _configured_url() -> str | None:
    value = (config.get_main_option("sqlalchemy.url") or "").strip()
    return value or None

###############################################################################
def _database_engine() -> tuple[Engine, bool]:
    """Return a CLI engine and whether the caller owns its disposal."""
    configured_url = _configured_url()
    if configured_url:
        return (
            engine_from_config(
                config.get_section(config.config_ini_section, {}),
                prefix="sqlalchemy.",
                poolclass=pool.NullPool,
            ),
            True,
        )
    database = Database(get_server_settings().database)
    return database.engine, True

###############################################################################
def _configure(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        render_as_batch=connection.dialect.name == "sqlite",
        include_schemas=False,
    )

###############################################################################
def run_migrations_offline() -> None:
    url = _configured_url()
    if not url:
        database = Database(get_server_settings().database)
        try:
            url = str(database.engine.url)
        finally:
            database.engine.dispose()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        render_as_batch=url.startswith("sqlite"),
        include_schemas=False,
    )
    with context.begin_transaction():
        context.run_migrations()

###############################################################################
def run_migrations_online() -> None:
    supplied_connection: Connection | None = config.attributes.get("connection")
    if supplied_connection is not None:
        _configure(supplied_connection)
        with context.begin_transaction():
            context.run_migrations()
        return

    connectable, owned = _database_engine()
    try:
        with connectable.connect() as connection:
            _configure(connection)
            with context.begin_transaction():
                context.run_migrations()
    finally:
        if owned:
            connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

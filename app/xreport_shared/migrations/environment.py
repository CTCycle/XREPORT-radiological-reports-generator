"""Shared Alembic environment runner; each application stream supplies its schema."""

from __future__ import annotations

from alembic import context
from sqlalchemy import engine_from_config, pool
from sqlalchemy.engine import Connection, Engine

from xreport_shared.common.path import APP_PROFILE
from xreport_shared.configurations import get_database_settings
from xreport_shared.repositories.database.engine import Database
from xreport_shared.repositories.database.schema import ApplicationSchema

###############################################################################
def _configured_url() -> str | None:
    value = (context.config.get_main_option("sqlalchemy.url") or "").strip()
    return value or None

###############################################################################
def _profile_database(schema: ApplicationSchema) -> Database:
    """Open the active profile's database; a stream never targets the other app."""
    if schema.profile != APP_PROFILE:
        raise RuntimeError(
            f"The {schema.title} migration stream only targets its own database; "
            f"set XREPORT_APP={schema.profile} or pass sqlalchemy.url explicitly."
        )
    return Database(get_database_settings())

###############################################################################
def _database_engine(schema: ApplicationSchema) -> Engine:
    """Return a CLI engine owned and disposed by the caller."""
    config = context.config
    if _configured_url():
        return engine_from_config(
            config.get_section(config.config_ini_section, {}),
            prefix="sqlalchemy.",
            poolclass=pool.NullPool,
        )
    return _profile_database(schema).engine

###############################################################################
def _configure(connection: Connection, schema: ApplicationSchema) -> None:
    context.configure(
        connection=connection,
        target_metadata=schema.metadata,
        version_table=schema.version_table,
        compare_type=True,
        render_as_batch=connection.dialect.name == "sqlite",
        include_schemas=False,
    )

###############################################################################
def _run_offline(schema: ApplicationSchema) -> None:
    url = _configured_url()
    if not url:
        engine = _profile_database(schema).engine
        try:
            url = str(engine.url)
        finally:
            engine.dispose()
    context.configure(
        url=url,
        target_metadata=schema.metadata,
        version_table=schema.version_table,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        render_as_batch=url.startswith("sqlite"),
        include_schemas=False,
    )
    with context.begin_transaction():
        context.run_migrations()

###############################################################################
def _run_online(schema: ApplicationSchema) -> None:
    supplied_connection: Connection | None = context.config.attributes.get("connection")
    if supplied_connection is not None:
        _configure(supplied_connection, schema)
        with context.begin_transaction():
            context.run_migrations()
        return

    engine = _database_engine(schema)
    try:
        with engine.connect() as connection:
            _configure(connection, schema)
            with context.begin_transaction():
                context.run_migrations()
    finally:
        engine.dispose()

###############################################################################
def run(schema: ApplicationSchema) -> None:
    if context.is_offline_mode():
        _run_offline(schema)
    else:
        _run_online(schema)

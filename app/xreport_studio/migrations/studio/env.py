"""Alembic environment for the Studio application schema."""

import xreport_studio  # noqa: F401  (registers the application schema)
from xreport_shared.migrations.environment import run
from xreport_shared.repositories.database.schema import application_schema

run(application_schema("studio"))

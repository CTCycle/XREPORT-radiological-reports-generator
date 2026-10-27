"""Alembic environment for the Studio application schema."""

from server.migrations.environment import run
from server.repositories.database.schema import application_schema

run(application_schema("studio"))

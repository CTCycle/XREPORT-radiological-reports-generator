from __future__ import annotations


# -----------------------------------------------------------------------------
def create_database_sql(database_name: str) -> str:
    safe_database = database_name.replace('"', '""')
    return f'CREATE DATABASE "{safe_database}" WITH ENCODING \'UTF8\' TEMPLATE template0'


# -----------------------------------------------------------------------------
def postgres_database_exists_sql() -> str:
    return "SELECT 1 FROM pg_database WHERE datname=:name"

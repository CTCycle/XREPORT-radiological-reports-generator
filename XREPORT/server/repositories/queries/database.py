from __future__ import annotations


# -----------------------------------------------------------------------------
def create_database_sql(database_name: str) -> str:
    safe_database = database_name.replace('"', '""')
    return f'CREATE DATABASE "{safe_database}" WITH ENCODING \'UTF8\' TEMPLATE template0'


# -----------------------------------------------------------------------------
def postgres_database_exists_sql() -> str:
    return "SELECT 1 FROM pg_database WHERE datname=:name"


# -----------------------------------------------------------------------------
def delete_all_rows_sql(safe_table_name: str) -> str:
    return f'DELETE FROM "{safe_table_name}"'


# -----------------------------------------------------------------------------
def count_rows_sql(safe_table_name: str) -> str:
    return f'SELECT COUNT(*) FROM "{safe_table_name}"'


# -----------------------------------------------------------------------------
def enable_foreign_keys_sql() -> str:
    return "PRAGMA foreign_keys=ON"

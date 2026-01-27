import os
import sqlite3

import pytest
from playwright.sync_api import APIRequestContext


TABLE_NAME = "PAGINATION_TEST"


def get_sqlite_db_path() -> str:
    tests_dir = os.path.dirname(__file__)
    db_dir = os.path.abspath(os.path.join(tests_dir, "..", "XREPORT", "resources", "database"))
    return os.path.join(db_dir, "sqlite.db")


def seed_pagination_table(db_path: str) -> None:
    connection = sqlite3.connect(db_path)
    try:
        connection.execute(f"DROP TABLE IF EXISTS {TABLE_NAME}")
        connection.execute(f"CREATE TABLE {TABLE_NAME} (id INTEGER PRIMARY KEY, value TEXT)")
        rows = [(index, f"value_{index}") for index in range(1, 151)]
        connection.executemany(f"INSERT INTO {TABLE_NAME} (id, value) VALUES (?, ?)", rows)
        connection.commit()
    finally:
        connection.close()


def cleanup_pagination_table(db_path: str) -> None:
    connection = sqlite3.connect(db_path)
    try:
        connection.execute(f"DROP TABLE IF EXISTS {TABLE_NAME}")
        connection.commit()
    finally:
        connection.close()


@pytest.fixture(scope="module")
def pagination_table() -> str:
    db_path = get_sqlite_db_path()
    if not os.path.exists(db_path):
        pytest.skip("SQLite database file not found; pagination verification requires embedded database.")
    seed_pagination_table(db_path)
    yield TABLE_NAME
    cleanup_pagination_table(db_path)


def test_verify_pagination(api_context: APIRequestContext, pagination_table: str) -> None:
    limit = 50
    expected_total = 150

    for offset in (0, 50, 100):
        response = api_context.get(f"/data/browser/data/{pagination_table}?limit={limit}&offset={offset}")
        assert response.ok, f"Expected 200, got {response.status}"
        payload = response.json()
        assert payload["total_rows"] == expected_total
        assert payload["row_count"] == limit
        assert len(payload["data"]) == limit
        assert payload["status"] == "success"

    response = api_context.get(f"/data/browser/data/{pagination_table}?limit={limit}&offset=150")
    assert response.ok, f"Expected 200, got {response.status}"
    payload = response.json()
    assert payload["total_rows"] == expected_total
    assert payload["row_count"] == 0
    assert payload["data"] == []
    assert payload["status"] == "success"

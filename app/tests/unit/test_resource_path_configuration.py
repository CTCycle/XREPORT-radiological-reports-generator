from __future__ import annotations

from pathlib import Path

import server.common.path as application_path

###############################################################################
def test_resource_override_from_env_file_is_resolved_relative_to_root(
    tmp_path,
    monkeypatch,
) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("XREPORT_RESOURCES_DIR=runtime-data\n", encoding="utf-8")
    monkeypatch.setattr(application_path, "ENV_FILE_PATH", env_path)
    monkeypatch.delenv("XREPORT_RESOURCES_DIR", raising=False)

    assert application_path._configured_resources_dir() == "runtime-data"
    assert (
        application_path.ROOT_DIR / "runtime-data"
        == application_path._resolve_resources_dir()
    )

###############################################################################
def test_empty_resource_override_uses_default_directory(monkeypatch) -> None:
    monkeypatch.setattr(application_path, "ENV_FILE_PATH", Path("missing.env"))
    monkeypatch.delenv("XREPORT_RESOURCES_DIR", raising=False)

    assert application_path._resolve_resources_dir() == application_path.DEFAULT_RESOURCES_DIR

###############################################################################
def test_database_path_is_derived_from_resolved_resource_directory() -> None:
    assert application_path.DATABASE_FILE_PATH == (
        application_path.RESOURCES_DIR / "database.db"
    )

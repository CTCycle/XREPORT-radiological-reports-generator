from __future__ import annotations

import os

import pytest

import server.configurations.environment as environment


@pytest.fixture(autouse=True)
def reset_environment_state() -> None:
    environment._environment_state.cache_clear()
    yield
    environment._environment_state.cache_clear()


def _configure_environment_paths(tmp_path, monkeypatch: pytest.MonkeyPatch):
    settings_dir = tmp_path / "settings"
    settings_dir.mkdir()
    env_path = settings_dir / ".env"
    example_path = settings_dir / ".env.example"
    monkeypatch.setattr(environment, "ENV_FILE_PATH", env_path)
    monkeypatch.setattr(environment, "ENV_EXAMPLE_FILE_PATH", example_path)
    return env_path, example_path


def test_load_environment_creates_missing_env_from_template(tmp_path, monkeypatch) -> None:
    env_path, example_path = _configure_environment_paths(tmp_path, monkeypatch)
    contents = b"XREPORT_TEST_ENV=from-template\n"
    example_path.write_bytes(contents)
    monkeypatch.delenv("XREPORT_TEST_ENV", raising=False)

    loaded_path = environment.load_environment(force=True)

    assert loaded_path == env_path
    assert env_path.read_bytes() == contents
    assert os.environ["XREPORT_TEST_ENV"] == "from-template"


def test_load_environment_preserves_existing_env(tmp_path, monkeypatch) -> None:
    env_path, example_path = _configure_environment_paths(tmp_path, monkeypatch)
    existing_contents = b"XREPORT_TEST_ENV=existing\n"
    env_path.write_bytes(existing_contents)
    example_path.write_bytes(b"XREPORT_TEST_ENV=template\n")
    monkeypatch.delenv("XREPORT_TEST_ENV", raising=False)

    environment.load_environment(force=True)

    assert env_path.read_bytes() == existing_contents
    assert os.environ["XREPORT_TEST_ENV"] == "existing"


def test_load_environment_requires_template_when_env_is_missing(tmp_path, monkeypatch) -> None:
    env_path, _ = _configure_environment_paths(tmp_path, monkeypatch)

    with pytest.raises(FileNotFoundError, match="Environment template not found"):
        environment.load_environment(force=True)

    assert not env_path.exists()

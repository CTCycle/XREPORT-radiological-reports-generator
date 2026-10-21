from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from server.configurations.runtime_environment import (
    CorsSettings,
    RuntimeEnvironmentSettings,
    get_runtime_environment_settings,
)


def _clear_runtime_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "INFERENCE_HF_LOCAL_ONLY",
        "INFERENCE_DEVICE",
        "CORS_ALLOWED_ORIGINS",
        "CORS_ALLOW_CREDENTIALS",
        "CORS_ALLOW_METHODS",
        "CORS_ALLOW_HEADERS",
    ):
        monkeypatch.delenv(name, raising=False)


def test_runtime_environment_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_runtime_environment(monkeypatch)

    assert get_runtime_environment_settings() == RuntimeEnvironmentSettings(
        inference_hf_local_only=True,
        inference_device="auto",
        cors=CorsSettings(
            allowed_origins=(),
            allow_credentials=False,
            allow_methods=("GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"),
            allow_headers=(
                "Authorization",
                "Content-Type",
                "X-XREPORT-Desktop-Token",
            ),
        ),
    )


@pytest.mark.parametrize("value", ["1", "true", "YES", " on "])
def test_environment_boolean_accepts_true_spellings(
    value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_runtime_environment(monkeypatch)
    monkeypatch.setenv("INFERENCE_HF_LOCAL_ONLY", value)

    assert get_runtime_environment_settings().inference_hf_local_only is True


@pytest.mark.parametrize("value", ["0", "false", "NO", " off "])
def test_environment_boolean_accepts_false_spellings(
    value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_runtime_environment(monkeypatch)
    monkeypatch.setenv("INFERENCE_HF_LOCAL_ONLY", value)

    assert get_runtime_environment_settings().inference_hf_local_only is False


def test_runtime_environment_normalizes_device_and_cors_lists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_runtime_environment(monkeypatch)
    monkeypatch.setenv("INFERENCE_DEVICE", " CPU ")
    monkeypatch.setenv(
        "CORS_ALLOWED_ORIGINS", " https://ui.example, ,https://desktop.example "
    )
    monkeypatch.setenv("CORS_ALLOW_CREDENTIALS", "true")
    monkeypatch.setenv("CORS_ALLOW_METHODS", "GET, OPTIONS")
    monkeypatch.setenv("CORS_ALLOW_HEADERS", "Authorization, X-Test-Header")

    settings = get_runtime_environment_settings()

    assert settings.inference_device == "cpu"
    assert settings.cors == CorsSettings(
        allowed_origins=("https://ui.example", "https://desktop.example"),
        allow_credentials=True,
        allow_methods=("GET", "OPTIONS"),
        allow_headers=("Authorization", "X-Test-Header"),
    )


@pytest.mark.parametrize("value", ["maybe", "2", "enabled"])
def test_invalid_runtime_boolean_is_rejected(
    value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_runtime_environment(monkeypatch)
    monkeypatch.setenv("INFERENCE_HF_LOCAL_ONLY", value)

    with pytest.raises(ValueError, match="INFERENCE_HF_LOCAL_ONLY"):
        get_runtime_environment_settings()


@pytest.mark.parametrize("value", ["gpu", "mps"])
def test_invalid_device_is_rejected(
    value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_runtime_environment(monkeypatch)
    monkeypatch.setenv("INFERENCE_DEVICE", value)

    with pytest.raises(ValueError, match="INFERENCE_DEVICE"):
        get_runtime_environment_settings()


def test_blank_device_uses_auto_default(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_runtime_environment(monkeypatch)
    monkeypatch.setenv("INFERENCE_DEVICE", "   ")

    assert get_runtime_environment_settings().inference_device == "auto"


def test_wildcard_cors_origin_cannot_use_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_runtime_environment(monkeypatch)
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "https://ui.example, *")
    monkeypatch.setenv("CORS_ALLOW_CREDENTIALS", "true")

    with pytest.raises(ValueError, match="cannot include '\\*'"):
        get_runtime_environment_settings()


def test_inference_policy_is_composed_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from server.configurations.settings import (
        DEFAULT_APPLICATION_SETTINGS,
        DatabaseSettings,
        application_settings_to_server_settings,
    )

    _clear_runtime_environment(monkeypatch)
    monkeypatch.setenv("INFERENCE_HF_LOCAL_ONLY", "false")
    monkeypatch.setenv("INFERENCE_DEVICE", "cuda")
    database = DatabaseSettings(
        backend="sqlite",
        engine=None,
        host=None,
        port=None,
        database_name=None,
        username=None,
        password=None,
        ssl=False,
        ssl_ca=None,
        connect_timeout=3,
        insert_batch_size=1000,
    )

    settings = application_settings_to_server_settings(
        DEFAULT_APPLICATION_SETTINGS, database
    )

    assert settings.inference.hf_local_only is False
    assert settings.inference.device == "cuda"
    assert settings.inference.model_timeout == 600


def test_cors_is_disabled_when_origins_are_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_runtime_environment(monkeypatch)
    from server.app import create_app

    response = TestClient(create_app()).options(
        "/api/health",
        headers={
            "Origin": "https://ui.example",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert response.status_code == 405
    assert "access-control-allow-origin" not in response.headers


def test_cors_preflight_allows_configured_origin_and_rejects_other_origin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from server.app import create_app

    _clear_runtime_environment(monkeypatch)
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "https://ui.example")
    monkeypatch.setenv("CORS_ALLOW_CREDENTIALS", "true")
    monkeypatch.setenv("CORS_ALLOW_METHODS", "GET, OPTIONS")
    monkeypatch.setenv("CORS_ALLOW_HEADERS", "X-XREPORT-Desktop-Token")
    client = TestClient(create_app())
    preflight_headers = {
        "Access-Control-Request-Method": "GET",
        "Access-Control-Request-Headers": "X-XREPORT-Desktop-Token",
    }

    allowed = client.options(
        "/api/health",
        headers={"Origin": "https://ui.example", **preflight_headers},
    )
    denied = client.options(
        "/api/health",
        headers={"Origin": "https://untrusted.example", **preflight_headers},
    )

    assert allowed.status_code == 200
    assert allowed.headers["access-control-allow-origin"] == "https://ui.example"
    assert allowed.headers["access-control-allow-credentials"] == "true"
    assert "get" in allowed.headers["access-control-allow-methods"].lower()
    assert "x-xreport-desktop-token" in allowed.headers[
        "access-control-allow-headers"
    ].lower()
    assert denied.status_code == 400
    assert "access-control-allow-origin" not in denied.headers

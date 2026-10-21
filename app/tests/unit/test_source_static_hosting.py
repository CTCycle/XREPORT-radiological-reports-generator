from __future__ import annotations

from fastapi.testclient import TestClient
import pytest

from server import app as app_module


@pytest.fixture
def client_build(tmp_path, monkeypatch):
    index = tmp_path / "index.html"
    index.write_text("<html><body>source-spa</body></html>", encoding="utf-8")
    (tmp_path / "main.js").write_text("console.log('bundle')", encoding="utf-8")
    monkeypatch.setattr(app_module, "PACKAGED_MODE", False)
    monkeypatch.setattr(app_module, "CLIENT_DIST_DIR", tmp_path)
    return tmp_path


def test_source_static_mode_serves_root_assets_and_angular_routes(
    client_build, monkeypatch
):
    monkeypatch.setenv("XREPORT_SERVE_CLIENT", "1")
    monkeypatch.delenv("XREPORT_LAUNCH_TOKEN", raising=False)
    application = app_module.create_app()
    client = TestClient(application)

    root = client.get("/")
    asset = client.get("/main.js")
    route = client.get("/reports/42")

    assert root.status_code == 200
    assert "source-spa" in root.text
    assert asset.status_code == 200
    assert asset.text == "console.log('bundle')"
    assert route.status_code == 200
    assert "source-spa" in route.text


def test_source_static_mode_keeps_api_paths_out_of_spa_fallback(
    client_build, monkeypatch
):
    monkeypatch.setenv("XREPORT_SERVE_CLIENT", "true")
    monkeypatch.delenv("XREPORT_LAUNCH_TOKEN", raising=False)
    application = app_module.create_app()
    application.state.database_ready = True
    client = TestClient(application)

    health = client.get("/api/health")
    unknown = client.get("/api/not-a-route")
    unknown_api_root = client.get("/api")

    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert unknown.status_code == 404
    assert unknown.headers["content-type"].startswith("application/json")
    assert "source-spa" not in unknown.text
    assert unknown_api_root.status_code == 404
    assert "source-spa" not in unknown_api_root.text


def test_database_recovery_health_and_spa_remain_available(client_build, monkeypatch):
    monkeypatch.setenv("XREPORT_SERVE_CLIENT", "1")
    monkeypatch.delenv("XREPORT_LAUNCH_TOKEN", raising=False)
    application = app_module.create_app()
    client = TestClient(application)

    root = client.get("/")
    health = client.get("/api/health")

    assert root.status_code == 200
    assert "source-spa" in root.text
    assert health.status_code == 200
    assert health.json()["status"] == "database_recovery"


def test_source_static_mode_rejects_traversal(client_build, monkeypatch):
    monkeypatch.setenv("XREPORT_SERVE_CLIENT", "1")
    outside = client_build.parent / "secret.txt"
    outside.write_text("must not be served", encoding="utf-8")

    response = app_module.serve_source_frontend("..\\secret.txt")

    assert response.status_code == 404
    assert "must not be served" not in response.body.decode()


def test_default_source_root_still_redirects_to_docs(client_build, monkeypatch):
    monkeypatch.delenv("XREPORT_SERVE_CLIENT", raising=False)
    monkeypatch.delenv("XREPORT_LAUNCH_TOKEN", raising=False)
    application = app_module.create_app()
    client = TestClient(application)

    response = client.get("/", follow_redirects=False)

    assert response.status_code == 307
    assert response.headers["location"].endswith("/docs")


def test_source_shutdown_requires_launcher_token(client_build, monkeypatch):
    monkeypatch.delenv("XREPORT_SERVE_CLIENT", raising=False)
    monkeypatch.setenv("XREPORT_LAUNCH_TOKEN", "launcher-secret-token")
    application = app_module.create_app()
    client = TestClient(application)

    unauthorized = client.post("/__xreport/shutdown")
    authorized = client.post(
        "/__xreport/shutdown",
        headers={"X-XREPORT-Launch-Token": "launcher-secret-token"},
    )

    assert unauthorized.status_code == 403
    assert authorized.status_code == 200
    assert application.state.desktop_shutdown_event.is_set()


def test_source_static_mode_requires_a_built_index(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "PACKAGED_MODE", False)
    monkeypatch.setattr(app_module, "CLIENT_DIST_DIR", tmp_path)
    monkeypatch.setenv("XREPORT_SERVE_CLIENT", "1")

    with pytest.raises(FileNotFoundError, match="Built Angular frontend"):
        app_module.create_app()

from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from server.common.desktop_security import (
    DesktopSecurityMiddleware,
    PRIVATE_TOKEN_HEADER,
    SESSION_COOKIE,
)

###############################################################################
def _security_app() -> FastAPI:
    application = FastAPI()
    application.add_middleware(DesktopSecurityMiddleware)

    @application.get("/")
    async def root() -> dict[str, str]:
        return {"status": "ok"}

    @application.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @application.get("/__xreport/shutdown")
    async def shutdown() -> JSONResponse:
        return JSONResponse({"status": "accepted"})

    return application

###############################################################################
def test_browser_session_can_probe_health_but_not_shutdown(monkeypatch) -> None:
    token = "x" * 64
    monkeypatch.setenv("XREPORT_DESKTOP_TOKEN", token)

    with TestClient(_security_app()) as client:
        bootstrap = client.get(
            "/__xreport/bootstrap",
            params={"token": token},
            follow_redirects=False,
        )
        assert bootstrap.status_code == 303
        assert "samesite=lax" in bootstrap.headers["set-cookie"].lower()

        client.cookies.set(SESSION_COOKIE, token)
        assert client.get("/api/health").status_code == 200
        assert client.get("/").status_code == 200
        assert client.get("/__xreport/shutdown").status_code == 401
        assert client.get(
            "/__xreport/shutdown",
            headers={PRIVATE_TOKEN_HEADER: token},
        ).status_code == 200

"""Authentication and response hardening for the packaged loopback server."""

from __future__ import annotations

import asyncio
import hmac
import os
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse, RedirectResponse, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint


SESSION_COOKIE = "xreport_session"
PRIVATE_TOKEN_HEADER = "x-xreport-desktop-token"
BOOTSTRAP_PATH = "/__xreport/bootstrap"
SHUTDOWN_PATH = "/__xreport/shutdown"
HEALTH_PATH = "/api/health"

###############################################################################
def desktop_token() -> str:
    token = os.getenv("XREPORT_DESKTOP_TOKEN", "")
    if len(token) < 32:
        raise RuntimeError("Packaged XREPORT desktop token is missing or too short")
    return token

###############################################################################
def token_matches(candidate: str | None) -> bool:
    return bool(candidate) and hmac.compare_digest(candidate, desktop_token())

###############################################################################
def _security_headers(response: Response) -> Response:
    response.headers.setdefault(
        "Content-Security-Policy",
        "default-src 'self'; base-uri 'none'; object-src 'none'; frame-ancestors 'none'; "
        "form-action 'self'; connect-src 'self'; img-src 'self' data: blob:; "
        "style-src 'self' 'unsafe-inline'; script-src 'self'; font-src 'self'; "
        "media-src 'self' blob:; worker-src 'self' blob:",
    )
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault(
        "Permissions-Policy",
        "camera=(), microphone=(), geolocation=(), payment=(), usb=()",
    )
    response.headers.setdefault("Cross-Origin-Resource-Policy", "same-origin")
    response.headers.setdefault("Cross-Origin-Opener-Policy", "same-origin")
    response.headers.setdefault("Cache-Control", "no-store")
    return response

###############################################################################
class DesktopSecurityMiddleware(BaseHTTPMiddleware):
    """Require the per-launch cookie for every packaged UI/API request.

    The native shell may probe health and shutdown with the private header;
    the browser UI may probe health with its per-launch cookie.  The one-time
    bootstrap route is the only endpoint that accepts the token in a URL.
    """

    # -------------------------------------------------------------------------
    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        path = request.url.path
        if path == BOOTSTRAP_PATH:
            if request.method != "GET" or not token_matches(
                request.query_params.get("token")
            ):
                return _security_headers(
                    JSONResponse({"detail": "Invalid bootstrap token"}, status_code=403)
                )
            response = RedirectResponse(url="/", status_code=303)
            response.set_cookie(
                SESSION_COOKIE,
                desktop_token(),
                httponly=True,
                secure=False,
                # The packaged WebView starts at tauri.localhost and crosses
                # to loopback for this one top-level GET. Lax is required for
                # WebView2 to send the freshly established session cookie on
                # the immediate redirect to the protected root.
                samesite="lax",
                path="/",
            )
            return _security_headers(response)

        native_probe = token_matches(request.headers.get(PRIVATE_TOKEN_HEADER))
        session_probe = token_matches(request.cookies.get(SESSION_COOKIE))
        if path == HEALTH_PATH:
            if not native_probe and not session_probe:
                return _security_headers(
                    JSONResponse({"detail": "Unauthorized"}, status_code=401)
                )
        elif path == SHUTDOWN_PATH:
            if not native_probe:
                return _security_headers(
                    JSONResponse({"detail": "Unauthorized"}, status_code=401)
                )
        elif not session_probe:
            return _security_headers(
                JSONResponse({"detail": "Desktop session required"}, status_code=401)
            )

        response = await call_next(request)
        return _security_headers(response)

###############################################################################
def install_shutdown_event(application: Any) -> asyncio.Event:
    event_factory = getattr(asyncio, "Event")
    event = event_factory()
    application.state.desktop_shutdown_event = event
    return event

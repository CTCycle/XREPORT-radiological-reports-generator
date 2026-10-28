"""Uvicorn runner shared by the source-mode entry points of both applications."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
import os
import signal

import uvicorn
from fastapi import FastAPI


###############################################################################
class LauncherOwnedServer(uvicorn.Server):
    """Leave Ctrl+C to the launcher so shutdown uses the authenticated API."""

    # -------------------------------------------------------------------------
    @contextmanager
    def capture_signals(self) -> Generator[None, None, None]:
        yield


###############################################################################
def serve(app: FastAPI, port: int) -> None:
    host = os.getenv("FASTAPI_HOST", "127.0.0.1")
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    server = LauncherOwnedServer(
        uvicorn.Config(
            app,
            host=host,
            port=port,
            log_level="info",
            access_log=True,
            # Route uvicorn records through the central XREPORT logging config.
            log_config=None,
        )
    )
    app.state.desktop_server = server
    server.run()

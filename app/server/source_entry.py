"""Windows source-mode server entrypoint used by the XREPORT launcher."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
import os
import signal

import uvicorn

from server.app import app


class LauncherOwnedServer(uvicorn.Server):
    """Leave Ctrl+C to the launcher so shutdown uses the authenticated API."""

    @contextmanager
    def capture_signals(self) -> Generator[None, None, None]:
        yield


def main() -> None:
    host = os.getenv("FASTAPI_HOST", "127.0.0.1")
    port = int(os.getenv("FASTAPI_PORT", "5003"))
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    server = LauncherOwnedServer(
        uvicorn.Config(
            app,
            host=host,
            port=port,
            log_level="info",
            access_log=True,
        )
    )
    app.state.desktop_server = server
    server.run()


if __name__ == "__main__":
    main()

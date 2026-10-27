"""Windows source-mode server entrypoint used by the XREPORT launcher."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
import argparse
import os
import signal

import uvicorn


class LauncherOwnedServer(uvicorn.Server):
    """Leave Ctrl+C to the launcher so shutdown uses the authenticated API."""

    @contextmanager
    def capture_signals(self) -> Generator[None, None, None]:
        yield


def main() -> None:
    parser = argparse.ArgumentParser(description="XREPORT source-mode server")
    parser.add_argument("--app", choices=("xreport", "studio"), default="xreport")
    arguments = parser.parse_args()
    # The package selects data roots at import time, so the launcher must export
    # XREPORT_APP before starting this process; the flag only has to agree with it.
    if os.getenv("XREPORT_APP", "xreport").strip().lower() != arguments.app:
        raise SystemExit(
            f"XREPORT_APP must be '{arguments.app}' to start the {arguments.app} application"
        )
    studio = arguments.app == "studio"
    if studio:
        # Studio trains with Keras on PyTorch; set it before Keras is imported
        # so neither the server nor its spawned training workers fall back to
        # Keras' TensorFlow default when settings/.env is absent.
        os.environ["KERAS_BACKEND"] = "torch"
    from server.app import app

    host = os.getenv("FASTAPI_HOST", "127.0.0.1")
    port = int(
        os.getenv("STUDIO_FASTAPI_PORT", "5013")
        if studio
        else os.getenv("FASTAPI_PORT", "5003")
    )
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


if __name__ == "__main__":
    main()

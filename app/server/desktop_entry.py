"""Windowless, dynamically-portable backend entry point for Tauri builds."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from functools import partial
from importlib import import_module
import json
import os
from pathlib import Path
import secrets
import socket
from time import perf_counter
from typing import Any


_SAFE_INHERITED_ENVIRONMENT = {
    "COMSPEC",
    "CUDA_PATH",
    "CUDA_PATH_V13_0",
    "NUMBER_OF_PROCESSORS",
    "OS",
    "PATH",
    "PROCESSOR_ARCHITECTURE",
    "PROCESSOR_IDENTIFIER",
    "PROGRAMDATA",
    "SYSTEMDRIVE",
    "SYSTEMROOT",
    "TEMP",
    "TMP",
    "USERDOMAIN",
    "USERNAME",
    "USERPROFILE",
    "WINDIR",
}

_STARTUP_STARTED = perf_counter()


###############################################################################
def _startup_log(phase: str) -> None:
    print(
        f"[startup] phase={phase} elapsed_ms={(perf_counter() - _STARTUP_STARTED) * 1000:.0f}",
        flush=True,
    )

###############################################################################
def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)

###############################################################################
def _remove_file(path: Path | None) -> None:
    if path is not None:
            path.unlink(missing_ok=True)

###############################################################################
def _write_ready(
    ready_file: Path,
    session_file: Path,
    port: int,
    version: str,
    variant: str,
    token: str,
) -> None:
    payload = {
        "host": "127.0.0.1",
        "port": port,
        "pid": os.getpid(),
        "version": version,
        "variant": variant,
        "ready_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json(ready_file, payload)
    _atomic_json(
        session_file,
        {
            "version": version,
            "variant": variant,
            "pid": os.getpid(),
            "bootstrap_url": f"http://127.0.0.1:{port}/__xreport/bootstrap?token={token}",
        },
    )

###############################################################################
def _sanitize_inherited_environment() -> None:
    """Drop arbitrary parent variables before loading the packaged .env.

    The shell re-adds the XREPORT runtime contract immediately after this
    function.  Keeping this allowlist small prevents accidental secret/config
    inheritance from the process that launched the desktop app.
    """
    keep = {key for key in os.environ if key in _SAFE_INHERITED_ENVIRONMENT}
    keep.update(
        {
            "XREPORT_DESKTOP",
            "XREPORT_RUNTIME_ROOT",
            "XREPORT_DATA_ROOT",
            "XREPORT_RELEASE_VERSION",
            "XREPORT_RUNTIME_VARIANT",
            "XREPORT_DESKTOP_TOKEN",
            "XREPORT_CLIENT_DIST_DIR",
            "XREPORT_READY_FILE",
            "XREPORT_SESSION_FILE",
        }
    )
    for key in list(os.environ):
        if key not in keep:
            os.environ.pop(key, None)

###############################################################################
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="XREPORT packaged backend")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--ready-file", type=Path, required=True)
    parser.add_argument("--session-file", type=Path)
    parser.add_argument("--variant", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--version", required=True)
    return parser.parse_args()

###############################################################################
def _prepare_contract(arguments: argparse.Namespace) -> tuple[Path, Path]:
    _sanitize_inherited_environment()
    os.environ["XREPORT_DESKTOP"] = "true"
    os.environ["XREPORT_RUNTIME_VARIANT"] = arguments.variant
    os.environ["XREPORT_RELEASE_VERSION"] = arguments.version
    if len(os.environ.get("XREPORT_DESKTOP_TOKEN", "")) < 32:
        raise RuntimeError("XREPORT_DESKTOP_TOKEN must be supplied by the desktop shell")
    ready_file = arguments.ready_file.resolve()
    session_file = (arguments.session_file or ready_file.with_name("desktop-session.json")).resolve()
    os.environ["XREPORT_READY_FILE"] = str(ready_file)
    os.environ["XREPORT_SESSION_FILE"] = str(session_file)
    return ready_file, session_file

###############################################################################
def _validate_runtime(arguments: argparse.Namespace) -> None:
    layout_module = import_module("server.common.runtime_layout")
    layout = layout_module.runtime_layout_from_environment()
    if layout.variant != arguments.variant or layout.release_version != arguments.version:
        raise RuntimeError("Desktop entry arguments do not match the runtime contract")

###############################################################################
def _create_listener(arguments: argparse.Namespace) -> tuple[socket.socket, int]:
    if arguments.host not in {"127.0.0.1", "localhost"}:
        raise RuntimeError("Packaged XREPORT backend must bind to loopback")
    if arguments.port < 0 or arguments.port > 65535:
        raise RuntimeError("Invalid backend port")
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    exclusive_address_use = getattr(socket, "SO_EXCLUSIVEADDRUSE", None)
    if exclusive_address_use is not None:
        listener.setsockopt(socket.SOL_SOCKET, exclusive_address_use, 1)
    listener.bind(("127.0.0.1", arguments.port))
    listener.listen(128)
    listener.set_inheritable(True)
    return listener, int(listener.getsockname()[1])

###############################################################################
def _configure_server_environment(port: int) -> None:
    os.environ["FASTAPI_HOST"] = "127.0.0.1"
    os.environ["FASTAPI_PORT"] = str(port)
    os.environ["RELOAD"] = "false"
    os.environ["BACKEND_VISIBLE"] = "false"
    os.environ["KERAS_BACKEND"] = "torch"
    os.environ["MPLBACKEND"] = "Agg"
    data_root = Path(os.environ["XREPORT_DATA_ROOT"])
    # NLTK resolves its default download directory at import time.  Packaged
    # launches deliberately strip arbitrary user-environment variables, so a
    # deterministic user-data directory must be supplied explicitly.
    nltk_root = data_root / "nltk"
    nltk_root.mkdir(parents=True, exist_ok=True)
    os.environ["NLTK_DATA"] = str(nltk_root)
    os.environ["MPLCONFIGDIR"] = str(data_root / "caches" / "matplotlib")

###############################################################################
def _run_server(
    listener: socket.socket,
    port: int,
    ready_file: Path,
    session_file: Path,
    arguments: argparse.Namespace,
) -> None:
    _startup_log("server_app_import_start")
    app = import_module("server.app").app
    app.state.desktop_startup_started_at = _STARTUP_STARTED
    _startup_log("server_app_imported")
    uvicorn = import_module("uvicorn")
    _startup_log("uvicorn_imported")
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host="127.0.0.1",
            port=port,
            log_level="warning",
            access_log=False,
            lifespan="on",
            loop="asyncio",
        )
    )
    app.state.desktop_server = server
    app.state.desktop_ready_callback = partial(
        _write_ready,
        ready_file,
        session_file,
        port,
        arguments.version,
        arguments.variant,
        os.environ["XREPORT_DESKTOP_TOKEN"],
    )
    _startup_log("server_configured")
    server.run(sockets=[listener])

###############################################################################
def main() -> int:
    arguments = _parse_args()
    _startup_log("arguments_parsed")
    ready_file, session_file = _prepare_contract(arguments)
    _startup_log("contract_prepared")

    # Validate and seed before importing server.app.  Its module-level app is
    # intentionally created only after this contract has been established.
    _validate_runtime(arguments)
    _startup_log("runtime_validated")
    listener, port = _create_listener(arguments)
    _startup_log("listener_bound")
    _configure_server_environment(port)
    _startup_log("environment_configured")
    try:
        _run_server(listener, port, ready_file, session_file, arguments)
        return 0
    finally:
        listener.close()
        _remove_file(ready_file)
        _remove_file(session_file)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)

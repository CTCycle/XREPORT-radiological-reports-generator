from __future__ import annotations

import json
from pathlib import Path
import tomllib

from server.common.constants import FASTAPI_VERSION


ROOT_DIR = Path(__file__).resolve().parents[3]


###############################################################################
def _json_version(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return str(payload["version"])


###############################################################################
def test_release_version_has_one_canonical_source() -> None:
    with (ROOT_DIR / "app" / "server" / "pyproject.toml").open("rb") as pyproject_file:
        canonical = str(tomllib.load(pyproject_file)["project"]["version"])

    copies = [
        ROOT_DIR / "app" / "client" / "package.json",
        ROOT_DIR / "app" / "desktop" / "package.json",
        ROOT_DIR / "app" / "desktop" / "src-tauri" / "tauri.conf.json",
        ROOT_DIR / "app" / "desktop" / "src-tauri" / "tauri.cpu.conf.json",
        ROOT_DIR / "app" / "desktop" / "src-tauri" / "tauri.cuda.conf.json",
    ]

    assert FASTAPI_VERSION == canonical
    assert all(_json_version(path) == canonical for path in copies)

    openapi = json.loads(
        (ROOT_DIR / "app" / "shared" / "openapi.json").read_text(encoding="utf-8")
    )
    assert openapi["info"]["version"] == canonical

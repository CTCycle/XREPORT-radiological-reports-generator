from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

SERVER_ROOT = Path(__file__).resolve().parents[2]

DEV_PREFIXES = ("upload", "preparation", "training", "validation")

PROBE = """
import json, sys
from __MODULE__ import create_app
from xreport_shared.common.path import CHECKPOINTS_DIR, DATABASE_FILE_PATH, MODELS_DIR
application = create_app()
paths = sorted(application.openapi()["paths"])
print(json.dumps({
    "title": application.title,
    "areas": sorted({p.split("/")[2] for p in paths if p.startswith("/api/")}),
    "paths": paths,
    "loaded": [m for m in sys.modules if m.startswith(("xreport_studio", "xreport_inference"))],
    "database": str(DATABASE_FILE_PATH),
    "checkpoints": str(CHECKPOINTS_DIR),
    "models": str(MODELS_DIR),
}))
"""


APPLICATION_MODULES = {"studio": "xreport_studio.app"}


###############################################################################
def _probe(profile: str, resources: Path) -> dict:
    import json
    import os

    environment = {
        **os.environ,
        "XREPORT_APP": profile,
        "XREPORT_RESOURCES_DIR": str(resources),
        "PYTHONPATH": str(SERVER_ROOT),
    }
    result = subprocess.run(
        [sys.executable, "-c", PROBE.replace("__MODULE__", APPLICATION_MODULES.get(profile, "xreport_inference.app"))],
        capture_output=True,
        text=True,
        env=environment,
        cwd=SERVER_ROOT,
        timeout=120,
        check=True,
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


###############################################################################
def test_xreport_profile_has_no_development_endpoints(tmp_path: Path) -> None:
    probe = _probe("xreport", tmp_path)
    assert probe["title"] == "XREPORT Inference"
    assert not set(probe["areas"]) & set(DEV_PREFIXES)
    assert "/api/exchange/import" in probe["paths"]
    assert "/api/exchange/export" not in probe["paths"]
    assert not any(
        module.startswith(
            (
                "xreport_studio.api.upload",
                "xreport_studio.api.preparation",
                "xreport_studio.api.training",
                "xreport_studio.api.validation",
                "xreport_studio.services.training",
            )
        )
        for module in probe["loaded"]
    )
    assert probe["database"] == str(tmp_path.resolve() / "database.db")
    assert not [m for m in probe["loaded"] if m.startswith("xreport_studio")]


###############################################################################
def test_studio_profile_owns_development_endpoints_and_separate_data(
    tmp_path: Path,
) -> None:
    probe = _probe("studio", tmp_path)
    assert probe["title"] == "XREPORT Studio"
    assert set(DEV_PREFIXES) <= set(probe["areas"])
    assert "inference" not in probe["areas"]
    assert "/api/exchange/export" in probe["paths"]
    assert "/api/exchange/import" not in probe["paths"]
    studio_root = tmp_path.resolve() / "studio"
    assert probe["database"] == str(studio_root / "database.db")
    assert probe["checkpoints"] == str(studio_root / "checkpoints")
    # Pinned encoder/tokenizer resources stay shared and read-mostly.
    assert probe["models"] == str(tmp_path.resolve() / "models")
    assert not [m for m in probe["loaded"] if m.startswith("xreport_inference")]


###############################################################################
def test_unknown_profile_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(subprocess.CalledProcessError):
        _probe("bogus", tmp_path)

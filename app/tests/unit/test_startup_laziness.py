from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


###############################################################################
def test_server_app_import_does_not_load_heavy_runtime_modules() -> None:
    root = Path(__file__).parents[3]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "app")
    environment["KERAS_BACKEND"] = "torch"
    environment["MPLBACKEND"] = "Agg"
    environment.pop("XREPORT_DESKTOP", None)
    code = """
import json
import sys
import server.app

heavy_modules = (
    "torch",
    "transformers",
    "keras",
    "torchvision",
    "server.models.inference.providers.huggingface",
    "server.models.training.processing",
    "server.repositories.serialization.model",
    "server.services.dataset_processing",
    "server.services.inference_runtime",
)
print(json.dumps({
    "route_count": len(server.app.app.routes),
    "loaded_heavy_modules": [name for name in heavy_modules if name in sys.modules],
}))
"""

    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout.strip().splitlines()[-1])

    assert payload["route_count"] == 41
    assert payload["loaded_heavy_modules"] == []

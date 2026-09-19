from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

###############################################################################
def test_concurrent_service_initialization_keeps_ml_imports_lazy() -> None:
    root = Path(__file__).parents[3]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "app")
    environment["KERAS_BACKEND"] = "torch"
    environment["MPLBACKEND"] = "Agg"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment.pop("XREPORT_DESKTOP", None)
    code = """
import importlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import sys

module_names = (
    "server.services.preparation",
    "server.services.training",
    "server.services.validation_runs",
    "server.services.evaluation",
    "server.repositories.serialization.model",
)

def import_module(name):
    return importlib.import_module(name)

with ThreadPoolExecutor(max_workers=len(module_names)) as pool:
    list(pool.map(import_module, module_names))

from server.repositories.serialization.model import ModelSerializer
from server.services.preparation import get_preparation_service
from server.services.training import TrainingService, get_training_runtime, get_training_service
from server.services.validation_runs import get_validation_service

with ThreadPoolExecutor(max_workers=3) as pool:
    services = list(pool.map(lambda factory: factory(), (
        get_preparation_service,
        get_training_service,
        get_validation_service,
    )))

with TemporaryDirectory() as directory:
    checkpoint = Path(directory) / "checkpoint"
    configuration = checkpoint / "configuration"
    configuration.mkdir(parents=True)
    (configuration / "configuration.json").write_text("{}", encoding="utf-8")
    (configuration / "metadata.json").write_text("{}", encoding="utf-8")
    (configuration / "session_history.json").write_text(
        '{"epochs": 2, "history": {"loss": [1.0], "val_loss": [1.1]}}',
        encoding="utf-8",
    )

    record = SimpleNamespace(name="checkpoint", path=checkpoint, artifact_complete=True)
    class CheckpointRepositoryStub:
        def list_checkpoints(self):
            return [record]

    listing = TrainingService(
        job_manager=object(),
        training_runtime=get_training_runtime(),
        checkpoint_repository=CheckpointRepositoryStub(),
    ).get_checkpoints()

heavy_modules = (
    "torch",
    "torch.utils",
    "keras",
    "torchvision",
    "transformers",
    "server.models.training.model",
    "server.models.training.dataloader",
    "server.models.inference.generator",
)
print(json.dumps({
    "factory_types": [type(service).__name__ for service in services],
    "checkpoint_epochs": listing.checkpoints[0].epochs,
    "heavy_modules": [name for name in heavy_modules if name in sys.modules],
    "serializer_loaded": "server.repositories.serialization.model" in sys.modules,
    "serializer_type": ModelSerializer.__name__,
}))
"""

    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=90,
    )
    payload = json.loads(completed.stdout.strip().splitlines()[-1])

    assert payload["factory_types"] == [
        "PreparationService",
        "TrainingService",
        "ValidationService",
    ]
    assert payload["checkpoint_epochs"] == 2
    assert payload["heavy_modules"] == []
    assert payload["serializer_loaded"] is True
    assert payload["serializer_type"] == "ModelSerializer"

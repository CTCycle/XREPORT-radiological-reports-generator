"""Studio trains and exports a checkpoint; an independent Inference process runs it.

Both applications run as separate uvicorn processes with their own data roots.
The test is slow (a real one-epoch training run) and needs the pinned BEiT
encoder and the distilbert tokenizer cache under `data/models`; it is skipped
when they are missing.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
APP_DIR = REPO_ROOT / "app"
MODELS = REPO_ROOT / "data" / "models"
FIXTURES = Path("G:/Database/datasets/Synthetic_Fixtures/XREPORT")
TOKENIZER = "distilbert-base-uncased"

pytestmark = pytest.mark.skipif(
    not (MODELS / "XRAYEncoder").is_dir()
    or not (MODELS / "tokenizers" / TOKENIZER).is_dir()
    or not (FIXTURES / "reports" / "xreport_reports_6.csv").is_file(),
    reason="pinned encoder, cached tokenizer or synthetic fixtures are not available",
)


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _start(application: str, profile: str, resources: Path, port: int) -> subprocess.Popen:
    (resources / "models").mkdir(parents=True)
    for name in ("XRAYEncoder", "tokenizers"):
        shutil.copytree(MODELS / name, resources / "models" / name, symlinks=False, ignore=shutil.ignore_patterns("blobs"))
    environment = {
        **os.environ,
        "XREPORT_APP": profile,
        "XREPORT_RESOURCES_DIR": str(resources),
        "KERAS_BACKEND": "torch",
        "PYTHONPATH": str(APP_DIR),
    }
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", application, "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"],
        cwd=APP_DIR,
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _wait_ready(client: httpx.Client) -> None:
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        try:
            if client.get("/api/health").json().get("status") == "ok":
                return
        except httpx.HTTPError:
            pass
        time.sleep(1)
    raise AssertionError("application did not become ready")


def _finish(client: httpx.Client, job_id: str, timeout: float = 600) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in {"completed", "failed", "cancelled"}:
            assert job["status"] == "completed", job.get("error") or job
            return job
        time.sleep(2)
    raise AssertionError(f"job {job_id} did not finish")


@pytest.fixture(scope="module")
def applications(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[httpx.Client, httpx.Client]]:
    root = tmp_path_factory.mktemp("exchange")
    studio_port, inference_port = _free_port(), _free_port()
    processes = [
        _start("xreport_studio.app:app", "studio", root / "studio", studio_port),
        _start("xreport_inference.app:app", "xreport", root / "inference", inference_port),
    ]
    studio = httpx.Client(base_url=f"http://127.0.0.1:{studio_port}", timeout=60)
    inference = httpx.Client(base_url=f"http://127.0.0.1:{inference_port}", timeout=60)
    try:
        _wait_ready(studio)
        _wait_ready(inference)
        yield studio, inference
    finally:
        studio.close()
        inference.close()
        for process in processes:
            process.terminate()
        for process in processes:
            process.wait(timeout=30)


###############################################################################
def test_studio_checkpoint_runs_in_the_independent_inference_application(
    applications: tuple[httpx.Client, httpx.Client], tmp_path: Path
) -> None:
    studio, inference = applications

    # Each application serves only its own API surface.
    assert studio.get("/api/tokenizers").status_code == 200
    assert inference.get("/api/tokenizers").status_code == 404
    assert inference.get("/api/inference/models").status_code == 200
    assert studio.get("/api/inference/models").status_code == 404

    with (FIXTURES / "reports" / "xreport_reports_6.csv").open("rb") as report_file:
        upload = studio.post("/api/upload/dataset", files={"file": report_file}).json()
    started = studio.post(
        "/api/preparation/dataset/load",
        json={"upload_id": upload["upload_id"], "image_folder_path": str(FIXTURES / "images"), "sample_size": 1.0, "confirm_unmatched": False},
    ).json()
    assert _finish(studio, started["job_id"])["result"]["matched_records"] == 6

    validation = studio.post(
        "/api/validation/run",
        json={"dataset_name": "xreport_reports_6", "metrics": ["text_statistics", "image_statistics", "pixels_distribution"], "sample_size": 1.0},
    ).json()
    _finish(studio, validation["job_id"])
    runs = studio.get("/api/validation/runs/xreport_reports_6").json()["runs"]
    assert runs[0]["records_evaluated"] == 6

    processing = studio.post(
        "/api/preparation/dataset/process",
        json={"dataset_name": "xreport_reports_6", "custom_name": "exchange_set", "sample_size": 1.0, "validation_size": 0.2, "tokenizer": TOKENIZER, "max_report_size": 60},
    ).json()
    _finish(studio, processing["job_id"])

    training = studio.post(
        "/api/training/start",
        json={
            "dataset_name": "exchange_set", "epochs": 1, "batch_size": 2, "gradient_accumulation_steps": 1,
            "architecture_preset": "compact", "use_img_augmentation": False, "save_checkpoints": True,
            "checkpoint_id": "exchange_ckpt", "use_device_GPU": False, "device_ID": 0, "use_mixed_precision": False,
            "dataloader_workers": 0, "prefetch_factor": 2, "pin_memory": False, "persistent_workers": False,
            "plot_training_metrics": False, "use_scheduler": False, "target_LR": 0.0001, "warmup_steps": 10,
        },
    ).json()
    _finish(studio, training["job_id"])

    export = studio.post("/api/exchange/export", json={"checkpoint": "exchange_ckpt", "include_encoder": False}).json()
    filename = _finish(studio, export["job_id"])["result"]["export"]["filename"]
    archive = tmp_path / filename
    archive.write_bytes(studio.get(f"/api/exchange/exports/{filename}", timeout=300).content)

    with archive.open("rb") as handle:
        imported = inference.post("/api/exchange/import", files={"file": handle}, timeout=300).json()
    outcome = _finish(inference, imported["job_id"])["result"]["import"]
    assert outcome["status"] == "imported"

    models = {item["model_ref"]: item for item in inference.get("/api/inference/models").json()["models"]}
    assert models["xreport:exchange_ckpt"]["status"] == "ready"

    with (FIXTURES / "images" / "xr_syn_001.jpg").open("rb") as image:
        generation = inference.post(
            "/api/inference/generate",
            data={"model_ref": "xreport:exchange_ckpt", "generation_profile": "deterministic"},
            files={"images": ("xr_syn_001.jpg", image, "image/jpeg")},
            timeout=300,
        ).json()
    reports = _finish(inference, generation["job_id"])["result"]["reports"]
    assert reports["xr_syn_001.jpg"].strip()

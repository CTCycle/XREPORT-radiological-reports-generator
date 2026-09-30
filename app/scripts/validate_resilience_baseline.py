"""Run the bounded S51/S53 resilience and baseline workload matrix.

The harness is intentionally limited to technical observations.  It does not
assign a campaign status, make a clinical claim, or replace the official
launcher.  ``--plan-only`` is the safe way to inspect the deterministic
scenario and receipt contract without contacting a running backend.

The live path keeps scenario definitions, HTTP operations, job polling,
resource sampling, cleanup, and receipt serialization separate so that a
future scale or runtime lane can be added without changing the API contract.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterable


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_FIXTURE_ROOT = (
    ROOT_DIR
    / "assets"
    / "QA"
    / "validation_campaign"
    / "s27"
    / "fixtures"
)

SCHEMA_VERSION = "s51-s53-resilience-baseline-v2"
TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})
VALID_GROUPS = frozenset({"ALL", "S51", "S53"})
EXECUTION_LANES = frozenset({"cuda", "cpu", "unavailable-gpu"})
START_MODES = frozenset({"clean", "warm"})
GPU_QUERY = (
    "nvidia-smi",
    "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
    "--format=csv,noheader,nounits",
)
MEASUREMENT_FIELDS = (
    "submitted_at_utc",
    "submit_elapsed_seconds",
    "polls",
    "terminal_at_utc",
    "job_wall_time_seconds",
    "system_samples",
    "gpu_memory_used_mib",
    "gpu_utilization_percent",
    "cpu_percent",
    "system_memory_used_bytes",
    "system_memory_available_bytes",
    "xreport_python_working_set_bytes",
    "xreport_python_processes",
    "device_log_matches",
    "worker_pid",
    "worker_pids",
    "worker_phase_transitions",
    "worker_cleanup",
    "resource_attribution",
    "first_progress_latency_seconds",
    "first_batch_latency_seconds",
    "last_worker_phase",
)


def utc_now() -> str:
    """Return a stable ISO-8601 representation for receipt timestamps."""

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def normalize_group(value: str) -> str:
    """Normalize and validate the public S51/S53 group selector."""

    group = value.strip().upper()
    if group not in VALID_GROUPS:
        choices = ", ".join(sorted(VALID_GROUPS))
        raise ValueError(f"Unknown group {value!r}; expected one of: {choices}")
    return group


def normalize_execution_lane(value: str) -> str:
    """Normalize the device lane used by the generated workload matrix."""

    lane = value.strip().lower()
    if lane not in EXECUTION_LANES:
        choices = ", ".join(sorted(EXECUTION_LANES))
        raise ValueError(f"Unknown execution lane {value!r}; expected one of: {choices}")
    return lane


def normalize_start_mode(value: str) -> str:
    """Normalize the externally controlled launcher lifecycle declaration."""

    mode = value.strip().lower()
    if mode not in START_MODES:
        choices = ", ".join(sorted(START_MODES))
        raise ValueError(f"Unknown start mode {value!r}; expected one of: {choices}")
    return mode


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class FixtureDefinition:
    """Names and provenance for the pinned synthetic technical fixture."""

    dataset: str = "s27_technical_fixture"
    processed_dataset: str = "s28_release_20260925"
    checkpoint: str = "XREPORT_20260925T141533"
    root: Path = DEFAULT_FIXTURE_ROOT
    scale: int = 1
    source_rows: int = 8
    image_dimensions: tuple[int, int] = (64, 64)
    provenance: str = "assets/QA/validation_campaign/s27/fixtures"
    de_identification: str = "non-clinical generated synthetic fixture"

    def __post_init__(self) -> None:
        if self.scale < 1:
            raise ValueError("Fixture scale must be at least 1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "processed_dataset": self.processed_dataset,
            "checkpoint": self.checkpoint,
            "root": str(self.root.resolve()),
            "scale": self.scale,
            "source_rows": self.source_rows,
            "expected_rows": self.source_rows * self.scale,
            "image_dimensions": list(self.image_dimensions),
            "provenance": self.provenance,
            "de_identification": self.de_identification,
        }


@dataclass(frozen=True)
class ResourceRef:
    """A task-created resource that may be cleaned after a run."""

    kind: str
    name: str

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "name": self.name}


@dataclass(frozen=True)
class MultipartPart:
    """A deterministic multipart field or file sent by an API operation."""

    field_name: str
    content: bytes
    filename: str | None = None
    content_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "field_name": self.field_name,
            "filename": self.filename,
            "content_type": self.content_type,
            "size_bytes": len(self.content),
            "sha256": hashlib.sha256(self.content).hexdigest(),
        }


@dataclass(frozen=True)
class OperationSpec:
    """One API operation in a deterministic scenario definition."""

    operation_id: str
    method: str
    path: str
    payload: dict[str, Any]
    job_type: str
    expected_statuses: tuple[int, ...] = (202,)
    expected_terminal_statuses: tuple[str, ...] = ("completed",)
    creates: tuple[ResourceRef, ...] = ()
    multipart: tuple[MultipartPart, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation_id": self.operation_id,
            "method": self.method,
            "path": self.path,
            "payload": self.payload,
            "job_type": self.job_type,
            "expected_statuses": list(self.expected_statuses),
            "expected_terminal_statuses": list(self.expected_terminal_statuses),
            "creates": [resource.to_dict() for resource in self.creates],
            "multipart": [part.to_dict() for part in self.multipart],
        }


@dataclass(frozen=True)
class ScenarioDefinition:
    """One independently receipted concurrent workload batch."""

    scenario_id: str
    group: str
    description: str
    operations: tuple[OperationSpec, ...]
    timeout_seconds: int
    requested_device: str | None = None
    concurrent: bool = True
    poll_interval_seconds: float = 1.0
    sample_interval_seconds: float = 1.0
    cancel_after_seconds: float | None = None
    tags: tuple[str, ...] = ()
    execution_lane: str = "cuda"

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "group": self.group,
            "description": self.description,
            "execution": {
                "concurrent": self.concurrent,
                "worker_count": len(self.operations),
                "timeout_seconds": self.timeout_seconds,
                "poll_interval_seconds": self.poll_interval_seconds,
                "sample_interval_seconds": self.sample_interval_seconds,
                "requested_device": self.requested_device,
                "cancel_after_seconds": self.cancel_after_seconds,
                "execution_lane": self.execution_lane,
            },
            "tags": list(self.tags),
            "measurements": list(MEASUREMENT_FIELDS),
            "operations": [operation.to_dict() for operation in self.operations],
        }


def training_payload(
    fixture: FixtureDefinition,
    checkpoint_id: str,
    *,
    use_gpu: bool,
) -> dict[str, Any]:
    """Return the exact lightweight one-epoch training contract."""

    return {
        "dataset_name": fixture.processed_dataset,
        "epochs": 1,
        "batch_size": 1,
        "num_encoders": 1,
        "num_decoders": 1,
        "embedding_dims": 64,
        "attention_heads": 1,
        "train_temp": 1.0,
        "freeze_img_encoder": True,
        "use_img_augmentation": False,
        "shuffle_with_buffer": False,
        "shuffle_size": 256,
        "save_checkpoints": True,
        "checkpoint_id": checkpoint_id,
        "use_device_GPU": use_gpu,
        "device_ID": 0,
        "jit_compile": False,
        "jit_backend": "inductor",
        "use_mixed_precision": False,
        "dataloader_workers": 0,
        "prefetch_factor": 1,
        "pin_memory": False,
        "persistent_workers": False,
        "plot_training_metrics": False,
        "use_scheduler": False,
        "target_LR": 0.001,
        "warmup_steps": 1000,
    }


def validation_payload(fixture: FixtureDefinition) -> dict[str, Any]:
    return {
        "dataset_name": fixture.dataset,
        "metrics": ["text_statistics", "image_statistics", "pixels_distribution"],
        "sample_size": 1.0,
        "seed": 42,
    }


def evaluation_payload(fixture: FixtureDefinition) -> dict[str, Any]:
    return {
        "checkpoint": fixture.checkpoint,
        "metrics": ["evaluation_report"],
        "num_samples": 10,
        "seed": 42,
    }


def inference_payload(fixture: FixtureDefinition) -> dict[str, Any]:
    return {
        "model_ref": f"xreport:{fixture.checkpoint}",
        "generation_profile": "deterministic",
        "clinical_context": "",
    }


def inference_multipart(fixture: FixtureDefinition) -> tuple[MultipartPart, ...]:
    image_paths = sorted(
        path
        for path in (fixture.root / "images").iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if not image_paths:
        raise FileNotFoundError(f"No supported fixture image found under {fixture.root / 'images'}")
    image_path = image_paths[0]
    content_type = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
    }[image_path.suffix.lower()]
    return (
        MultipartPart("images", image_path.read_bytes(), image_path.name, content_type),
    )


def processing_payload(
    fixture: FixtureDefinition, custom_name: str
) -> dict[str, Any]:
    return {
        "dataset_name": fixture.dataset,
        "custom_name": custom_name,
        "sample_size": 1.0,
        "validation_size": 0.25,
        "tokenizer": "distilbert-base-uncased",
        "max_report_size": 200,
    }


def _operation(
    operation_id: str,
    method: str,
    path: str,
    payload: dict[str, Any],
    job_type: str,
    *,
    expected_statuses: tuple[int, ...] = (202,),
    expected_terminal_statuses: tuple[str, ...] = ("completed",),
    creates: Iterable[ResourceRef] = (),
    multipart: Iterable[MultipartPart] = (),
) -> OperationSpec:
    return OperationSpec(
        operation_id=operation_id,
        method=method,
        path=path,
        payload=payload,
        job_type=job_type,
        expected_statuses=expected_statuses,
        expected_terminal_statuses=expected_terminal_statuses,
        creates=tuple(creates),
        multipart=tuple(multipart),
    )


def _lane_uses_gpu(execution_lane: str) -> bool:
    """Return the request value for a GPU-backed training operation.

    The unavailable-GPU lane deliberately keeps GPU requested.  The backend
    must then prove its documented CPU fallback instead of the harness
    silently converting the request to a CPU run.
    """

    return normalize_execution_lane(execution_lane) in {"cuda", "unavailable-gpu"}


def _lane_device_label(execution_lane: str) -> str:
    lane = normalize_execution_lane(execution_lane)
    return {
        "cuda": "cuda",
        "cpu": "cpu",
        "unavailable-gpu": "cpu-fallback",
    }[lane]


def build_scenarios(
    group: str = "ALL",
    fixture: FixtureDefinition | None = None,
    *,
    poll_interval_seconds: float = 1.0,
    sample_interval_seconds: float = 1.0,
    execution_lane: str = "cuda",
) -> tuple[ScenarioDefinition, ...]:
    """Build the stable S51/S53 matrix in historical execution order."""

    selected_group = normalize_group(group)
    selected_fixture = fixture or FixtureDefinition()
    selected_lane = normalize_execution_lane(execution_lane)
    use_gpu = _lane_uses_gpu(selected_lane)
    device_label = _lane_device_label(selected_lane)
    scenarios: list[ScenarioDefinition] = []

    if selected_group in {"ALL", "S51"}:
        for index in range(1, 4):
            checkpoint_name = f"s51_r{index}_cuda"
            scenarios.append(
                ScenarioDefinition(
                    scenario_id=f"s51_triple_cuda_{index}",
                    group="S51",
                    description=(
                        "Concurrent dataset validation, checkpoint evaluation, and "
                        "one-epoch CUDA training."
                    ),
                    operations=(
                        _operation(
                            "validation",
                            "POST",
                            "/api/validation/run",
                            validation_payload(selected_fixture),
                            "validation",
                        ),
                        _operation(
                            "checkpoint_evaluation",
                            "POST",
                            "/api/validation/checkpoint",
                            evaluation_payload(selected_fixture),
                            "checkpoint_evaluation",
                        ),
                        _operation(
                            "training",
                            "POST",
                            "/api/training/start",
                            training_payload(
                                selected_fixture, checkpoint_name, use_gpu=use_gpu
                            ),
                            "training",
                            creates=(ResourceRef("checkpoint", checkpoint_name),),
                        ),
                    ),
                    timeout_seconds=360,
                    requested_device=device_label,
                    execution_lane=selected_lane,
                    tags=("contention", "triple", "synthetic"),
                )
            )

        evaluation_dataset = "s51_pair_eval_processed"
        scenarios.append(
            ScenarioDefinition(
                scenario_id="s51_processing_and_evaluation",
                group="S51",
                description="Concurrent dataset processing and checkpoint evaluation.",
                operations=(
                    _operation(
                        "dataset_processing",
                        "POST",
                        "/api/preparation/dataset/process",
                        processing_payload(selected_fixture, evaluation_dataset),
                        "dataset_processing",
                        creates=(ResourceRef("processed_dataset", evaluation_dataset),),
                    ),
                    _operation(
                        "checkpoint_evaluation",
                        "POST",
                        "/api/validation/checkpoint",
                        evaluation_payload(selected_fixture),
                        "checkpoint_evaluation",
                    ),
                ),
                timeout_seconds=360,
                requested_device="mixed",
                execution_lane=selected_lane,
                tags=("contention", "pair", "synthetic"),
            )
        )

        training_dataset = "s51_pair_training_processed"
        training_checkpoint = "s51_pair_training_cuda"
        scenarios.append(
            ScenarioDefinition(
                scenario_id="s51_processing_and_training",
                group="S51",
                description="Concurrent dataset processing and one-epoch CUDA training.",
                operations=(
                    _operation(
                        "dataset_processing",
                        "POST",
                        "/api/preparation/dataset/process",
                        processing_payload(selected_fixture, training_dataset),
                        "dataset_processing",
                        creates=(ResourceRef("processed_dataset", training_dataset),),
                    ),
                    _operation(
                        "training",
                        "POST",
                        "/api/training/start",
                        training_payload(
                            selected_fixture, training_checkpoint, use_gpu=use_gpu
                        ),
                        "training",
                        creates=(ResourceRef("checkpoint", training_checkpoint),),
                    ),
                ),
                timeout_seconds=600,
                requested_device=device_label,
                execution_lane=selected_lane,
                tags=("contention", "pair", "synthetic"),
            )
        )

        mixed_checkpoint = "s51_mixed_training_cuda"
        scenarios.append(
            ScenarioDefinition(
                scenario_id="s51_four_way_mixed_contention",
                group="S51",
                description=(
                    "Concurrent validation, checkpoint evaluation, inference, and "
                    "one-epoch CUDA training."
                ),
                operations=(
                    _operation(
                        "validation",
                        "POST",
                        "/api/validation/run",
                        validation_payload(selected_fixture),
                        "validation",
                    ),
                    _operation(
                        "checkpoint_evaluation",
                        "POST",
                        "/api/validation/checkpoint",
                        evaluation_payload(selected_fixture),
                        "checkpoint_evaluation",
                    ),
                    _operation(
                        "inference",
                        "POST",
                        "/api/inference/generate",
                        inference_payload(selected_fixture),
                        "inference",
                        multipart=inference_multipart(selected_fixture),
                    ),
                    _operation(
                        "training",
                        "POST",
                        "/api/training/start",
                        training_payload(
                            selected_fixture, mixed_checkpoint, use_gpu=use_gpu
                        ),
                        "training",
                        creates=(ResourceRef("checkpoint", mixed_checkpoint),),
                    ),
                ),
                timeout_seconds=600,
                requested_device="mixed",
                execution_lane=selected_lane,
                tags=("contention", "four-way", "inference", "synthetic"),
            )
        )

        scenarios.append(
            ScenarioDefinition(
                scenario_id="s51_same_type_validation_race",
                group="S51",
                description="Two identical validation API submissions raced at the same time.",
                operations=(
                    _operation(
                        "validation_a",
                        "POST",
                        "/api/validation/run",
                        validation_payload(selected_fixture),
                        "validation",
                        expected_statuses=(202, 409),
                    ),
                    _operation(
                        "validation_b",
                        "POST",
                        "/api/validation/run",
                        validation_payload(selected_fixture),
                        "validation",
                        expected_statuses=(202, 409),
                    ),
                ),
                timeout_seconds=360,
                requested_device="cpu",
                execution_lane=selected_lane,
                tags=("contention", "same-type", "race", "synthetic"),
            )
        )

        cancelled_checkpoint = "s51_cancelled_training"
        cancelled_payload = training_payload(
            selected_fixture, cancelled_checkpoint, use_gpu=use_gpu
        )
        cancelled_payload["epochs"] = 3
        scenarios.append(
            ScenarioDefinition(
                scenario_id="s51_training_cancellation",
                group="S51",
                description="A long-running training request is cancelled and polled to terminal state.",
                operations=(
                    _operation(
                        "training",
                        "POST",
                        "/api/training/start",
                        cancelled_payload,
                        "training",
                        creates=(ResourceRef("checkpoint", cancelled_checkpoint),),
                        expected_terminal_statuses=("cancelled",),
                    ),
                ),
                timeout_seconds=360,
                requested_device=device_label,
                execution_lane=selected_lane,
                concurrent=False,
                cancel_after_seconds=1.0,
                tags=("cancellation", "isolation", "synthetic"),
            )
        )

    if selected_group in {"ALL", "S53"}:
        if selected_lane == "cuda":
            training_lanes = (("cuda", True, "cuda"), ("cpu", False, "cpu"))
        elif selected_lane == "cpu":
            training_lanes = (("cpu", False, "cpu"),)
        else:
            training_lanes = (("fallback", True, "cpu-fallback"),)
        for label, lane_use_gpu, device in training_lanes:
            for repeat in range(1, 4):
                checkpoint_name = f"s53_{label}_baseline_{repeat}"
                scenarios.append(
                    ScenarioDefinition(
                        scenario_id=f"s53_{label}_baseline_{repeat}",
                        group="S53",
                        description=(
                            f"Matched one-epoch {device.upper()} training baseline "
                            f"repeat {repeat}/3."
                        ),
                        operations=(
                            _operation(
                                "training",
                                "POST",
                                "/api/training/start",
                                training_payload(
                                    selected_fixture,
                                    checkpoint_name,
                                    use_gpu=lane_use_gpu,
                                ),
                                "training",
                                creates=(ResourceRef("checkpoint", checkpoint_name),),
                            ),
                        ),
                        timeout_seconds=600,
                        requested_device=device,
                        concurrent=False,
                        execution_lane=selected_lane,
                        tags=("baseline", "matched", "repeat", device, "synthetic"),
                    )
                )

        long_checkpoint = "s53_cuda_long_training"
        long_payload = training_payload(
            selected_fixture, long_checkpoint, use_gpu=use_gpu
        )
        long_payload["epochs"] = 3
        scenarios.append(
            ScenarioDefinition(
                scenario_id="s53_cuda_long_training",
                group="S53",
                description=(
                    f"Three-epoch {device_label.upper()} training stability observation."
                ),
                operations=(
                    _operation(
                        "training",
                        "POST",
                        "/api/training/start",
                        long_payload,
                        "training",
                        creates=(ResourceRef("checkpoint", long_checkpoint),),
                    ),
                ),
                timeout_seconds=1200,
                requested_device=device_label,
                execution_lane=selected_lane,
                concurrent=False,
                tags=("long-operation", "training", "cuda", "synthetic"),
            )
        )

        for repeat in range(1, 4):
            scenarios.append(
                ScenarioDefinition(
                    scenario_id=f"s53_inference_repeat_{repeat}",
                    group="S53",
                    description=f"Single-image inference baseline repeat {repeat}/3.",
                    operations=(
                        _operation(
                            "inference",
                            "POST",
                            "/api/inference/generate",
                            inference_payload(selected_fixture),
                            "inference",
                            multipart=inference_multipart(selected_fixture),
                        ),
                    ),
                    timeout_seconds=600,
                    requested_device="mixed",
                    execution_lane=selected_lane,
                    concurrent=False,
                    tags=("baseline", "inference", "repeat", "synthetic"),
                )
            )

    if poll_interval_seconds <= 0 or sample_interval_seconds <= 0:
        raise ValueError("Polling and sampling intervals must be positive")
    return tuple(
        replace(
            scenario,
            poll_interval_seconds=poll_interval_seconds,
            sample_interval_seconds=sample_interval_seconds,
        )
        for scenario in scenarios
    )


def build_plan(
    group: str = "ALL",
    fixture: FixtureDefinition | None = None,
    *,
    poll_interval_seconds: float = 1.0,
    sample_interval_seconds: float = 1.0,
    request_timeout_seconds: float = 30.0,
    cold_start_count: int = 10,
    execution_lane: str = "cuda",
    start_mode: str = "clean",
) -> dict[str, Any]:
    """Return the serializable plan and receipt-field contract."""

    selected_group = normalize_group(group)
    selected_fixture = fixture or FixtureDefinition()
    selected_lane = normalize_execution_lane(execution_lane)
    selected_start_mode = normalize_start_mode(start_mode)
    scenarios = build_scenarios(
        selected_group,
        selected_fixture,
        poll_interval_seconds=poll_interval_seconds,
        sample_interval_seconds=sample_interval_seconds,
        execution_lane=selected_lane,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "selected_group": selected_group,
        "execution": {
            "lane": selected_lane,
            "start_mode": selected_start_mode,
            "lane_contract": {
                "cuda": "GPU requested and CUDA provenance must be observed.",
                "cpu": "CPU requested with no GPU dependency.",
                "unavailable-gpu": (
                    "GPU requested while CUDA is unavailable; backend CPU fallback "
                    "and false CUDA provenance must be observed."
                ),
            }[selected_lane],
        },
        "fixture": selected_fixture.to_dict(),
        "scenarios": [scenario.to_dict() for scenario in scenarios],
        "polling": {
            "terminal_statuses": sorted(TERMINAL_STATUSES),
            "default_interval_seconds": poll_interval_seconds,
            "request_timeout_seconds": request_timeout_seconds,
            "job_timeout_is_per_scenario": True,
        },
        "resource_sampling": {
            "interval_seconds": sample_interval_seconds,
            "gpu_command": list(GPU_QUERY),
            "process_selector": "python processes whose cwd or command line contains the repository root",
            "fields": [
                "captured_at_utc",
                "gpu_memory_used_mib",
                "gpu_utilization_percent",
                "gpus",
                "cpu_percent",
                "system_memory_used_bytes",
                "system_memory_available_bytes",
                "xreport_python_working_set_bytes",
                "xreport_python_processes",
            "xreport_python_process_details",
                "device_log_matches",
            ],
        },
        "cold_start_probes": {
            "requested_count": cold_start_count if selected_group in {"ALL", "S51"} else 0,
            "method": "fresh Python process imports server.app with the same resource root",
            "timeout_seconds": 45,
        },
        "cleanup": {
            "tracked_resource_types": ["checkpoint", "processed_dataset"],
            "delete_order": ["checkpoint", "processed_dataset"],
            "preexisting_resources_are_not_deleted": True,
            "checkpoint_endpoint": "DELETE /api/training/checkpoints/{name}",
            "processed_dataset_endpoint": "DELETE /api/preparation/dataset/{name}",
        },
        "receipt_fields": [
            "schema_version",
            "captured_at_utc",
            "repository",
            "host",
            "base_url",
            "execution",
            "resource_root",
            "selected_group",
            "fixture",
            "plan",
            "initial",
            "initial_database",
            "cold_start_probes",
            "scenarios",
            "cleanup",
            "final",
            "final_database",
            "limitations",
        ],
        "limitations": [
            "Technical synthetic observations only; no clinical or release claim.",
            "The unavailable-GPU lane requires the backend to run with CUDA unavailable; setting a client-side environment variable does not prove that condition.",
            "This harness does not establish packaged/native WebView behavior or representative clinical scale.",
            "Cold-start probes import the backend app in fresh processes; they do not replace packaged launcher or native WebView startup validation.",
            "The harness records observations and execution errors but does not assign S51/S53 campaign status.",
        ],
    }


class ApiFailure(RuntimeError):
    """Raised when an API response is outside an operation's contract."""

    def __init__(self, method: str, path: str, status: int, body: Any) -> None:
        super().__init__(f"{method} {path} returned HTTP {status}: {body}")
        self.method = method
        self.path = path
        self.status = status
        self.body = body


class ApiClient:
    """Small standard-library JSON transport used by the live harness."""

    def __init__(self, base_url: str, *, timeout_seconds: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        expected_statuses: Iterable[int] | None = None,
        multipart: Iterable[MultipartPart] = (),
    ) -> dict[str, Any]:
        parts = tuple(multipart)
        data = None
        headers = {"Accept": "application/json"}
        if parts:
            boundary = f"----XREPORT-{hashlib.sha256(os.urandom(16)).hexdigest()}"
            chunks: list[bytes] = []
            for key, value in (payload or {}).items():
                chunks.extend(
                    [
                        f"--{boundary}\r\n".encode(),
                        f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode(),
                        str(value).encode("utf-8"),
                        b"\r\n",
                    ]
                )
            for part in parts:
                disposition = f'Content-Disposition: form-data; name="{part.field_name}"'
                if part.filename is not None:
                    disposition += f'; filename="{part.filename}"'
                chunks.extend([f"--{boundary}\r\n".encode(), f"{disposition}\r\n".encode()])
                if part.content_type:
                    chunks.append(f"Content-Type: {part.content_type}\r\n".encode())
                chunks.extend([b"\r\n", part.content, b"\r\n"])
            chunks.append(f"--{boundary}--\r\n".encode())
            data = b"".join(chunks)
            headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
        elif payload is not None:
            data = json.dumps(payload, sort_keys=True).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                status = response.status
                raw = response.read()
        except urllib.error.HTTPError as exc:
            status = exc.code
            raw = exc.read()
        except urllib.error.URLError as exc:
            raise RuntimeError(f"{method} {path} failed: {exc}") from exc

        text = raw.decode("utf-8", errors="replace")
        try:
            body: Any = json.loads(text) if text else {}
        except json.JSONDecodeError:
            body = text

        accepted = set(expected_statuses) if expected_statuses is not None else None
        if accepted is not None and status not in accepted:
            raise ApiFailure(method, path, status, body)
        if accepted is None and status >= 400:
            raise ApiFailure(method, path, status, body)
        return {"status": status, "body": body}


class ApiOperations:
    """Named application operations kept separate from transport mechanics."""

    def __init__(self, client: ApiClient) -> None:
        self.client = client

    def submit(self, operation: OperationSpec) -> dict[str, Any]:
        return self.client.request(
            operation.method,
            operation.path,
            operation.payload,
            expected_statuses=operation.expected_statuses,
            multipart=operation.multipart,
        )

    def job_status(self, job_id: str) -> dict[str, Any]:
        return self.client.request("GET", f"/api/jobs/{job_id}")

    def running_jobs(self) -> dict[str, Any]:
        return self.client.request("GET", "/api/jobs?status=running")

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        return self.client.request("DELETE", f"/api/jobs/{urllib.parse.quote(job_id, safe='')}")

    def health(self) -> dict[str, Any]:
        return self.client.request("GET", "/api/health")

    def checkpoints(self) -> dict[str, Any]:
        return self.client.request("GET", "/api/training/checkpoints")

    def processed_datasets(self) -> dict[str, Any]:
        return self.client.request(
            "GET", "/api/preparation/dataset/processed/names"
        )

    def delete_checkpoint(self, name: str) -> dict[str, Any]:
        return self.client.request(
            "DELETE",
            f"/api/training/checkpoints/{urllib.parse.quote(name, safe='')}",
            expected_statuses={200, 404},
        )

    def delete_processed_dataset(self, name: str) -> dict[str, Any]:
        return self.client.request(
            "DELETE",
            f"/api/preparation/dataset/{urllib.parse.quote(name, safe='')}",
            expected_statuses={200, 404},
        )

    def capture_state(
        self,
        fixture: FixtureDefinition,
        *,
        include_reports: bool,
    ) -> dict[str, Any]:
        state: dict[str, Any] = {
            "health": self.health(),
            "running_jobs": self.running_jobs(),
            "checkpoints": self.checkpoints(),
            "processed_datasets": self.processed_datasets(),
        }
        if include_reports:
            dataset = urllib.parse.quote(fixture.dataset, safe="")
            checkpoint = urllib.parse.quote(fixture.checkpoint, safe="")
            state["validation_report"] = self.client.request(
                "GET",
                f"/api/validation/reports/{dataset}",
                expected_statuses={200, 404},
            )
            state["checkpoint_evaluation_report"] = self.client.request(
                "GET",
                f"/api/validation/checkpoint/reports/{checkpoint}",
                expected_statuses={200, 404},
            )
        return state


class ResourceSampler:
    """Best-effort host/GPU sampler with no application-side instrumentation."""

    def __init__(self, repository_root: Path, resource_root: Path) -> None:
        self.repository_root = repository_root.resolve()
        self.resource_root = resource_root.resolve()

    def sample(self) -> dict[str, Any]:
        sample: dict[str, Any] = {
            "captured_at_utc": utc_now(),
            "gpu_memory_used_mib": None,
            "gpu_utilization_percent": None,
            "gpus": [],
            "cpu_percent": None,
            "system_memory_used_bytes": None,
            "system_memory_available_bytes": None,
            "xreport_python_working_set_bytes": None,
            "xreport_python_processes": 0,
            "xreport_python_process_details": [],
            "device_log_matches": [],
        }
        self._sample_gpu(sample)
        self._sample_processes(sample)
        self._sample_device_logs(sample)
        return sample

    @staticmethod
    def _sample_gpu(sample: dict[str, Any]) -> None:
        try:
            completed = subprocess.run(
                list(GPU_QUERY),
                capture_output=True,
                check=False,
                text=True,
                timeout=3,
            )
        except (FileNotFoundError, OSError, subprocess.SubprocessError) as exc:
            sample["gpu_query_error"] = type(exc).__name__
            return
        if completed.returncode != 0:
            sample["gpu_query_error"] = f"exit_{completed.returncode}"
            return

        gpus: list[dict[str, Any]] = []
        for line in completed.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 5:
                continue
            try:
                gpu = {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "memory_total_mib": int(parts[2]),
                    "memory_used_mib": int(parts[3]),
                    "utilization_percent": int(parts[4]),
                }
            except ValueError:
                continue
            gpus.append(gpu)
        sample["gpus"] = gpus
        if gpus:
            sample["gpu_memory_used_mib"] = gpus[0]["memory_used_mib"]
            sample["gpu_utilization_percent"] = gpus[0]["utilization_percent"]
        else:
            sample["gpu_query_error"] = "unparseable_output"

    def _sample_processes(self, sample: dict[str, Any]) -> None:
        try:
            import psutil
        except ImportError:
            sample["process_sample_error"] = "psutil_unavailable"
            return

        sample["cpu_percent"] = psutil.cpu_percent(interval=None)
        virtual_memory = psutil.virtual_memory()
        sample["system_memory_used_bytes"] = int(virtual_memory.used)
        sample["system_memory_available_bytes"] = int(virtual_memory.available)

        repository_text = str(self.repository_root).casefold()
        working_set = 0
        process_count = 0
        process_details: list[dict[str, Any]] = []
        for process in psutil.process_iter(["name", "cwd", "cmdline", "memory_info"]):
            try:
                info = process.info
                name = str(info.get("name") or "").casefold()
                cwd = str(info.get("cwd") or "").casefold()
                command_line = " ".join(info.get("cmdline") or []).casefold()
                if "python" not in name:
                    continue
                if repository_text not in cwd and repository_text not in command_line:
                    continue
                memory_info = info.get("memory_info")
                if memory_info is not None:
                    working_set += int(memory_info.rss)
                    process_count += 1
                    process_details.append(
                        {
                            "pid": process.pid,
                            "name": info.get("name"),
                            "rss_bytes": int(memory_info.rss),
                            "cpu_percent": process.cpu_percent(interval=None),
                        }
                    )
            except (psutil.Error, OSError, TypeError, ValueError):
                continue
        sample["xreport_python_working_set_bytes"] = working_set
        sample["xreport_python_processes"] = process_count
        sample["xreport_python_process_details"] = process_details

    def _sample_device_logs(self, sample: dict[str, Any]) -> None:
        log_dir = self.resource_root / "logs"
        if not log_dir.is_dir():
            return
        matches: list[str] = []
        log_paths = sorted(
            log_dir.glob("*.log"),
            key=lambda path: (path.stat().st_mtime, path.name),
        )
        for log_path in log_paths:
            try:
                lines = log_path.read_text(
                    encoding="utf-8", errors="replace"
                ).splitlines()
            except OSError:
                continue
            for line in lines:
                if (
                    "GPU (cuda:" in line
                    or "CPU is set as the active device" in line
                    or "No GPU found" in line
                ):
                    matches.append(line.strip())
        sample["device_log_matches"] = matches[-12:]


class ResourceMonitor:
    """Capture resource samples until a scenario's jobs reach terminal state."""

    def __init__(self, sampler: ResourceSampler, interval_seconds: float) -> None:
        if interval_seconds <= 0:
            raise ValueError("Resource sample interval must be positive")
        self.sampler = sampler
        self.interval_seconds = interval_seconds
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.samples.append(self.sampler.sample())
        self._thread = threading.Thread(
            target=self._run,
            name="s51-s53-resource-monitor",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(5.0, self.interval_seconds + 1.0))

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self.samples.append(self.sampler.sample())


def _worker_pid_from_status(status: Any) -> int | None:
    """Extract a worker PID from either the job result or diagnostics fields."""

    if not isinstance(status, dict):
        return None
    candidates: list[Any] = [status.get("worker_pid")]
    result = status.get("result")
    if isinstance(result, dict):
        candidates.append(result.get("worker_pid"))
        diagnostics = result.get("worker_diagnostics")
        if isinstance(diagnostics, dict):
            candidates.append(diagnostics.get("pid"))
    diagnostics = status.get("worker_diagnostics")
    if isinstance(diagnostics, dict):
        candidates.append(diagnostics.get("pid"))
    for candidate in candidates:
        if isinstance(candidate, int) and not isinstance(candidate, bool) and candidate > 0:
            return candidate
    return None


class JobPoller:
    """Poll submitted jobs with bounded, receipt-visible terminal handling."""

    def __init__(self, api: ApiOperations, default_interval_seconds: float) -> None:
        if default_interval_seconds <= 0:
            raise ValueError("Poll interval must be positive")
        self.api = api
        self.default_interval_seconds = default_interval_seconds

    def poll(
        self,
        jobs: dict[str, dict[str, Any]],
        errors: list[dict[str, Any]],
        *,
        timeout_seconds: int,
        started_monotonic: float,
        cancel_after_seconds: float | None = None,
    ) -> None:
        deadline = time.monotonic() + timeout_seconds
        cancel_at = (
            started_monotonic + cancel_after_seconds
            if cancel_after_seconds is not None
            else None
        )
        cancellation_sent = False
        while jobs and time.monotonic() < deadline:
            if cancel_at is not None and not cancellation_sent and time.monotonic() >= cancel_at:
                for job_id, job in jobs.items():
                    if job.get("terminal"):
                        continue
                    try:
                        job["cancellation"] = self.api.cancel_job(job_id)
                    except Exception as exc:  # noqa: BLE001
                        errors.append(
                            {
                                "job_id": job_id,
                                "phase": "cancellation",
                                "error": str(exc),
                            }
                        )
                cancellation_sent = True
            all_terminal = True
            for job_id, job in jobs.items():
                if job.get("terminal"):
                    continue
                all_terminal = False
                try:
                    response = self.api.job_status(job_id)
                except Exception as exc:  # noqa: BLE001
                    errors.append({
                        "job_id": job_id,
                        "phase": "poll",
                        "error": str(exc),
                    })
                    job.setdefault("poll_errors", []).append(str(exc))
                    continue

                captured_at = utc_now()
                status = response["body"]
                job.setdefault("polls", []).append(
                    {
                        "captured_at_utc": captured_at,
                        "elapsed_seconds": round(
                            time.monotonic() - started_monotonic, 3
                        ),
                        "response": response,
                        "worker_pid": _worker_pid_from_status(status),
                    }
                )
                job["last_status"] = status
                status_name = status.get("status") if isinstance(status, dict) else None
                worker_pid = _worker_pid_from_status(status)
                if worker_pid is not None:
                    job["worker_pid"] = worker_pid
                    worker_pids = job.setdefault("worker_pids", [])
                    if worker_pid not in worker_pids:
                        worker_pids.append(worker_pid)
                if status_name in TERMINAL_STATUSES:
                    job["terminal"] = True
                    job["terminal_status"] = status_name
                    job["terminal_at_utc"] = captured_at
                    job["job_wall_time_seconds"] = round(
                        time.monotonic() - started_monotonic, 3
                    )
                    expected = set(job.get("expected_terminal_statuses", ("completed",)))
                    if status_name not in expected:
                        errors.append(
                            {
                                "job_id": job_id,
                                "phase": "terminal_status",
                                "error": (
                                    f"Expected terminal status {sorted(expected)}, "
                                    f"observed {status_name}."
                                ),
                            }
                        )

            if all_terminal:
                break
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(min(self.default_interval_seconds, remaining))

        for job_id, job in jobs.items():
            if not job.get("terminal"):
                errors.append(
                    {
                        "job_id": job_id,
                        "phase": "poll",
                        "error": "poll timeout",
                    }
                )


def _body(state: dict[str, Any], key: str) -> Any:
    response = state.get(key, {})
    if isinstance(response, dict) and "body" in response:
        return response["body"]
    return response


def _names(body: Any, key: str) -> set[str]:
    if not isinstance(body, dict):
        return set()
    values = body.get(key, [])
    if not isinstance(values, list):
        return set()
    result: set[str] = set()
    for value in values:
        if isinstance(value, str):
            result.add(value)
        elif isinstance(value, dict) and isinstance(value.get("name"), str):
            result.add(value["name"])
    return result


@dataclass
class ResourceTracker:
    """Track only resources absent from the pre-run inventory."""

    initial_checkpoints: set[str]
    initial_processed_datasets: set[str]
    created_checkpoints: list[str] = field(default_factory=list)
    created_processed_datasets: list[str] = field(default_factory=list)
    preexisting: list[ResourceRef] = field(default_factory=list)

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> ResourceTracker:
        return cls(
            initial_checkpoints=_names(_body(state, "checkpoints"), "checkpoints"),
            initial_processed_datasets=_names(
                _body(state, "processed_datasets"), "datasets"
            ),
        )

    def register(self, resource: ResourceRef) -> None:
        if resource.kind == "checkpoint":
            existing = self.initial_checkpoints
            target = self.created_checkpoints
        elif resource.kind == "processed_dataset":
            existing = self.initial_processed_datasets
            target = self.created_processed_datasets
        else:
            raise ValueError(f"Unsupported cleanup resource kind: {resource.kind}")

        if resource.name in existing:
            if resource not in self.preexisting:
                self.preexisting.append(resource)
            return
        if resource.name not in target:
            target.append(resource.name)

    def register_checkpoint_path(self, checkpoint_path: Any) -> None:
        if isinstance(checkpoint_path, str) and checkpoint_path.strip():
            self.register(ResourceRef("checkpoint", Path(checkpoint_path).name))

    def to_dict(self) -> dict[str, Any]:
        return {
            "initial_checkpoints": sorted(self.initial_checkpoints),
            "initial_processed_datasets": sorted(self.initial_processed_datasets),
            "created_checkpoints": list(self.created_checkpoints),
            "created_processed_datasets": list(self.created_processed_datasets),
            "preexisting": [resource.to_dict() for resource in self.preexisting],
        }


class CleanupManager:
    """Remove task-created records and verify the post-run inventory."""

    def __init__(self, api: ApiOperations) -> None:
        self.api = api

    def cleanup(self, tracker: ResourceTracker) -> dict[str, Any]:
        result: dict[str, Any] = {
            "tracker": tracker.to_dict(),
            "checkpoints": [],
            "processed_datasets": [],
            "errors": [],
        }
        for name in tracker.created_checkpoints:
            try:
                result["checkpoints"].append(
                    {"name": name, "response": self.api.delete_checkpoint(name)}
                )
            except Exception as exc:  # noqa: BLE001
                result["errors"].append(
                    {"kind": "checkpoint", "name": name, "error": str(exc)}
                )
        for name in tracker.created_processed_datasets:
            try:
                result["processed_datasets"].append(
                    {
                        "name": name,
                        "response": self.api.delete_processed_dataset(name),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                result["errors"].append(
                    {"kind": "processed_dataset", "name": name, "error": str(exc)}
                )

        try:
            result["running_jobs"] = self.api.running_jobs()
            result["checkpoints_after"] = self.api.checkpoints()
            result["processed_datasets_after"] = self.api.processed_datasets()
        except Exception as exc:  # noqa: BLE001
            result["errors"].append({"phase": "post_cleanup_inventory", "error": str(exc)})
        return result


def _training_checkpoint_from_job(job: dict[str, Any]) -> str | None:
    status = job.get("last_status")
    if not isinstance(status, dict):
        return None
    result = status.get("result")
    if not isinstance(result, dict):
        return None
    checkpoint_path = result.get("checkpoint_path")
    if not isinstance(checkpoint_path, str):
        return None
    return Path(checkpoint_path).name


def _record_training_observations(job: dict[str, Any]) -> None:
    """Summarize phase and first-progress evidence from the poll transcript."""

    transitions: list[dict[str, Any]] = []
    first_progress_latency: float | None = None
    first_batch_latency: float | None = None
    last_phase: str | None = None
    worker_pids: list[int] = [
        pid
        for pid in job.get("worker_pids", [])
        if isinstance(pid, int) and not isinstance(pid, bool) and pid > 0
    ]
    for poll in job.get("polls", []):
        if not isinstance(poll, dict):
            continue
        response = poll.get("response")
        body = response.get("body") if isinstance(response, dict) else None
        if not isinstance(body, dict):
            continue
        worker_pid = _worker_pid_from_status(body)
        if worker_pid is not None and worker_pid not in worker_pids:
            worker_pids.append(worker_pid)
        result = body.get("result")
        if not isinstance(result, dict):
            continue
        phase = result.get("worker_phase")
        if isinstance(phase, str) and phase and phase != last_phase:
            transitions.append(
                {
                    "captured_at_utc": poll.get("captured_at_utc"),
                    "elapsed_seconds": poll.get("elapsed_seconds"),
                    "phase": phase,
                    "status": result.get("worker_phase_status"),
                    "phase_elapsed_seconds": result.get(
                        "worker_phase_elapsed_seconds"
                    ),
                }
            )
            last_phase = phase
        progress = result.get("progress_percent", body.get("progress", 0))
        if (
            first_progress_latency is None
            and isinstance(progress, (int, float))
            and progress > 0
        ):
            first_progress_latency = poll.get("elapsed_seconds")
        if (
            first_batch_latency is None
            and phase in {"first_batch_completed", "batch_completed"}
        ):
            first_batch_latency = poll.get("elapsed_seconds")

    job["worker_phase_transitions"] = transitions
    job["first_progress_latency_seconds"] = first_progress_latency
    job["first_batch_latency_seconds"] = first_batch_latency
    job["last_worker_phase"] = last_phase
    job["worker_pids"] = sorted(worker_pids)
    if worker_pids and not isinstance(job.get("worker_pid"), int):
        job["worker_pid"] = worker_pids[-1]


def _record_worker_cleanup(job: dict[str, Any]) -> None:
    """Record whether observed worker processes remain after terminal status."""

    pids = job.get("worker_pids", [])
    if not isinstance(pids, list) or not pids:
        job["worker_cleanup"] = {"status": "no_pid_observed", "processes": []}
        return
    try:
        import psutil
    except ImportError:
        job["worker_cleanup"] = {
            "status": "unmeasurable",
            "reason": "psutil_unavailable",
            "processes": [],
        }
        return

    processes: list[dict[str, Any]] = []
    for pid in pids:
        if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
            continue
        try:
            process = psutil.Process(pid)
            processes.append({"pid": pid, "alive": process.is_running()})
        except psutil.NoSuchProcess:
            processes.append({"pid": pid, "alive": False})
        except (psutil.AccessDenied, psutil.Error) as exc:
            processes.append({"pid": pid, "alive": None, "error": type(exc).__name__})
    alive = [item for item in processes if item.get("alive") is True]
    job["worker_cleanup"] = {
        "status": "orphaned" if alive else "clean",
        "processes": processes,
    }


def _attribute_job_resources(
    job: dict[str, Any], samples: list[dict[str, Any]]
) -> None:
    """Attach host samples belonging to a job's observed worker PID(s)."""

    pids = set(job.get("worker_pids", []))
    attributed: list[dict[str, Any]] = []
    for sample in samples:
        details = sample.get("xreport_python_process_details", [])
        if not isinstance(details, list):
            continue
        matches = [
            detail
            for detail in details
            if isinstance(detail, dict) and detail.get("pid") in pids
        ]
        if matches:
            attributed.append(
                {
                    "captured_at_utc": sample.get("captured_at_utc"),
                    "processes": matches,
                    "gpus": sample.get("gpus", []),
                }
            )
    max_rss = max(
        (
            int(process.get("rss_bytes", 0))
            for sample in attributed
            for process in sample["processes"]
            if isinstance(process.get("rss_bytes"), int)
        ),
        default=None,
    )
    max_cpu = max(
        (
            float(process.get("cpu_percent"))
            for sample in attributed
            for process in sample["processes"]
            if isinstance(process.get("cpu_percent"), (int, float))
        ),
        default=None,
    )
    job["resource_attribution"] = {
        "worker_pids": sorted(pids),
        "sample_count": len(attributed),
        "max_worker_rss_bytes": max_rss,
        "max_worker_cpu_percent": max_cpu,
        "samples": attributed,
        "gpu_attribution": "host-level samples; per-job GPU attribution unavailable",
    }


def _running_job_entries(response: Any) -> list[dict[str, Any]]:
    """Return normalized running-job entries from the inventory response."""

    body = response.get("body") if isinstance(response, dict) else response
    jobs = body.get("jobs", []) if isinstance(body, dict) else []
    if not isinstance(jobs, list):
        return []
    return [job for job in jobs if isinstance(job, dict)]


def _job_id_from_entry(entry: dict[str, Any]) -> str | None:
    for key in ("job_id", "id"):
        value = entry.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _assert_device_lane_observed(
    batch: dict[str, Any], scenario: ScenarioDefinition
) -> None:
    """Require truthful device-log evidence for scenarios that start training."""

    if not any(operation.job_type == "training" for operation in scenario.operations):
        return
    matches = [
        line
        for sample in batch.get("system_samples", [])
        for line in sample.get("device_log_matches", [])
        if isinstance(line, str)
    ]
    lane = normalize_execution_lane(scenario.execution_lane)
    if lane == "cuda":
        observed = any("GPU (cuda:" in line for line in matches)
        expected = "a GPU (cuda:N) activation log"
    elif lane == "cpu":
        observed = any("CPU is set as the active device" in line for line in matches)
        expected = "a CPU activation log"
    else:
        observed = (
            any("No GPU found" in line for line in matches)
            and any("CPU is set as the active device" in line for line in matches)
        )
        expected = "both the no-GPU fallback and CPU activation logs"
    if not observed:
        batch["errors"].append(
            {
                "phase": "device_provenance",
                "error": f"Expected {expected} for execution lane {lane}.",
                "observed_device_log_matches": matches[-12:],
            }
        )


def run_scenario(
    api: ApiOperations,
    scenario: ScenarioDefinition,
    sampler: ResourceSampler,
    tracker: ResourceTracker,
) -> dict[str, Any]:
    """Submit, poll, and sample one scenario without mixing cleanup logic."""

    started_at = utc_now()
    started_monotonic = time.monotonic()
    batch: dict[str, Any] = {
        "scenario_id": scenario.scenario_id,
        "group": scenario.group,
        "description": scenario.description,
        "plan": scenario.to_dict(),
        "started_at_utc": started_at,
        "submissions": [],
        "jobs": {},
        "system_samples": [],
        "errors": [],
    }
    for operation in scenario.operations:
        for resource in operation.creates:
            tracker.register(resource)

    monitor = ResourceMonitor(sampler, scenario.sample_interval_seconds)
    monitor.start()
    barrier = (
        threading.Barrier(len(scenario.operations))
        if scenario.concurrent and len(scenario.operations) > 1
        else None
    )

    def submit(operation: OperationSpec) -> dict[str, Any]:
        if barrier is not None:
            barrier.wait(timeout=15)
        submitted_at = utc_now()
        submit_started = time.monotonic()
        try:
            response = api.submit(operation)
            return {
                "operation_id": operation.operation_id,
                "job_type": operation.job_type,
                "submitted_at_utc": submitted_at,
                "submit_elapsed_seconds": round(
                    time.monotonic() - submit_started, 3
                ),
                "response": response,
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "operation_id": operation.operation_id,
                "job_type": operation.job_type,
                "submitted_at_utc": submitted_at,
                "submit_elapsed_seconds": round(
                    time.monotonic() - submit_started, 3
                ),
                "error": str(exc),
            }

    try:
        if scenario.concurrent and len(scenario.operations) > 1:
            with ThreadPoolExecutor(max_workers=len(scenario.operations)) as executor:
                futures = [executor.submit(submit, operation) for operation in scenario.operations]
                submissions = [future.result(timeout=45) for future in futures]
        else:
            submissions = [submit(operation) for operation in scenario.operations]
        batch["submissions"] = submissions

        if scenario.scenario_id == "s51_same_type_validation_race":
            statuses = sorted(
                submission["response"]["status"]
                for submission in submissions
                if isinstance(submission.get("response"), dict)
                and isinstance(submission["response"].get("status"), int)
            )
            if statuses != [202, 409]:
                batch["errors"].append(
                    {
                        "phase": "submission",
                        "error": (
                            "Same-type validation race must produce exactly one "
                            f"202 and one 409; observed statuses: {statuses}"
                        ),
                    }
                )

        for submission in submissions:
            response = submission.get("response")
            operation = next(
                (
                    candidate
                    for candidate in scenario.operations
                    if candidate.operation_id == submission.get("operation_id")
                ),
                None,
            )
            response_status = (
                response.get("status") if isinstance(response, dict) else None
            )
            if (
                operation is not None
                and isinstance(response_status, int)
                and response_status in operation.expected_statuses
                and response_status != 202
            ):
                continue
            body = response.get("body") if isinstance(response, dict) else None
            job_id = body.get("job_id") if isinstance(body, dict) else None
            if not isinstance(job_id, str) or not job_id:
                batch["errors"].append(
                    {
                        "operation_id": submission.get("operation_id"),
                        "phase": "submission",
                        "error": submission.get("error", "response did not contain job_id"),
                    }
                )
                continue
            batch["jobs"][job_id] = {
                "operation_id": submission["operation_id"],
                "job_type": submission["job_type"],
                "job_id": job_id,
                "submitted_at_utc": submission["submitted_at_utc"],
                "expected_terminal_statuses": list(
                    operation.expected_terminal_statuses
                    if operation is not None
                    else ("completed",)
                ),
                "poll_interval_seconds": body.get("poll_interval", scenario.poll_interval_seconds),
                "polls": [],
                "terminal": False,
            }

        poller = JobPoller(api, scenario.poll_interval_seconds)
        poller.poll(
            batch["jobs"],
            batch["errors"],
            timeout_seconds=scenario.timeout_seconds,
            started_monotonic=started_monotonic,
            cancel_after_seconds=scenario.cancel_after_seconds,
        )
    except Exception as exc:  # noqa: BLE001
        batch["errors"].append({"phase": "scenario", "error": str(exc)})
    finally:
        monitor.stop()

    batch["system_samples"] = monitor.samples
    _assert_device_lane_observed(batch, scenario)
    for job in batch["jobs"].values():
        _record_training_observations(job)
        _record_worker_cleanup(job)
        _attribute_job_resources(job, batch["system_samples"])
        if job.get("worker_cleanup", {}).get("status") == "orphaned":
            batch["errors"].append(
                {
                    "job_id": job.get("job_id"),
                    "phase": "worker_cleanup",
                    "error": "A worker process remained alive after terminal status.",
                }
            )
        checkpoint_name = _training_checkpoint_from_job(job)
        if checkpoint_name is not None:
            tracker.register_checkpoint_path(checkpoint_name)
    try:
        batch["running_jobs_after"] = api.running_jobs()
        batch["nonterminal_jobs_after"] = _running_job_entries(batch["running_jobs_after"])
        if batch["nonterminal_jobs_after"]:
            batch["errors"].append(
                {
                    "phase": "post_scenario_inventory",
                    "error": "Nonterminal jobs remained after scenario completion.",
                    "jobs": batch["nonterminal_jobs_after"],
                }
            )
            cancellations: list[dict[str, Any]] = []
            for entry in batch["nonterminal_jobs_after"]:
                job_id = _job_id_from_entry(entry)
                if job_id is None:
                    cancellations.append(
                        {"entry": entry, "error": "Running job has no identifiable job_id."}
                    )
                    continue
                try:
                    cancellations.append(
                        {"job_id": job_id, "response": api.cancel_job(job_id)}
                    )
                except Exception as exc:  # noqa: BLE001
                    cancellations.append({"job_id": job_id, "error": str(exc)})
            batch["stranded_job_cancellations"] = cancellations
            for _ in range(20):
                time.sleep(0.25)
                batch["running_jobs_after_cleanup"] = api.running_jobs()
                if not _running_job_entries(batch["running_jobs_after_cleanup"]):
                    break
            if _running_job_entries(batch.get("running_jobs_after_cleanup")):
                batch["errors"].append(
                    {
                        "phase": "post_scenario_cleanup",
                        "error": "Stranded jobs remained after bounded cancellation cleanup.",
                        "jobs": _running_job_entries(batch["running_jobs_after_cleanup"]),
                    }
                )
    except Exception as exc:  # noqa: BLE001
        batch["errors"].append(
            {"phase": "post_scenario_inventory", "error": str(exc)}
        )
    batch["reconciliation"] = {
        "terminal_jobs": [
            {
                "job_id": job.get("job_id"),
                "status": job.get("terminal_status"),
                "expected_statuses": job.get("expected_terminal_statuses", []),
                "worker_pids": job.get("worker_pids", []),
                "worker_cleanup": job.get("worker_cleanup"),
            }
            for job in batch["jobs"].values()
        ],
        "nonterminal_jobs_after": batch.get("nonterminal_jobs_after", []),
        "created_resources": tracker.to_dict(),
    }
    batch["finished_at_utc"] = utc_now()
    batch["wall_time_seconds"] = round(time.monotonic() - started_monotonic, 3)
    batch["execution_state"] = (
        "completed_with_errors" if batch["errors"] else "completed"
    )
    return batch


class ReceiptSerializer:
    """Write stable, human-readable JSON receipts without status inference."""

    @staticmethod
    def write(path: Path, receipt: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(receipt, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )


def _repository_metadata() -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "head": os.environ.get("XREPORT_VALIDATION_HEAD"),
        "dirty": None,
    }
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT_DIR,
            capture_output=True,
            check=False,
            text=True,
            timeout=5,
        )
        if head.returncode == 0:
            metadata["head"] = head.stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=ROOT_DIR,
            capture_output=True,
            check=False,
            text=True,
            timeout=5,
        )
        if status.returncode == 0:
            metadata["dirty"] = bool(status.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        metadata["error"] = "git metadata unavailable"
    return metadata


def _host_metadata() -> dict[str, Any]:
    return {
        "os": platform.platform(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }


def database_integrity(resource_root: Path) -> dict[str, Any]:
    """Read SQLite integrity and table metadata without mutating the database."""

    database_path = resource_root.resolve() / "database.db"
    result: dict[str, Any] = {
        "path": str(database_path),
        "exists": database_path.is_file(),
        "integrity_check": None,
        "tables": [],
    }
    if not result["exists"]:
        return result
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(
            f"file:{database_path.as_posix()}?mode=ro",
            uri=True,
            timeout=5,
        )
        result["integrity_check"] = connection.execute(
            "PRAGMA integrity_check"
        ).fetchone()[0]
        result["tables"] = [
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'table' ORDER BY name"
            ).fetchall()
            if isinstance(row[0], str)
        ]
    except (OSError, sqlite3.Error) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if connection is not None:
            connection.close()
    return result


def run_cold_start_probes(
    resource_root: Path,
    count: int,
) -> list[dict[str, Any]]:
    """Import the backend app in independent fresh Python processes."""

    if count <= 0:
        return []
    probe_code = "from server.app import app; print(app.title)"
    environment = os.environ.copy()
    environment["XREPORT_RESOURCES_DIR"] = str(resource_root.resolve())
    server_root = ROOT_DIR / "app" / "server"
    package_root = server_root.parent
    existing_python_path = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(package_root), existing_python_path) if value
    )
    probes: list[dict[str, Any]] = []
    for index in range(1, count + 1):
        started = time.monotonic()
        process: subprocess.Popen[str] | None = None
        try:
            process = subprocess.Popen(
                [sys.executable, "-c", probe_code],
                cwd=server_root,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            stdout, stderr = process.communicate(timeout=45)
            probe = {
                "probe": index,
                "pid": process.pid,
                "returncode": process.returncode,
                "duration_seconds": round(time.monotonic() - started, 3),
                "stdout": stdout.strip()[-500:],
                "stderr": stderr.strip()[-500:],
            }
        except subprocess.TimeoutExpired as exc:
            if process is not None:
                process.kill()
                stdout, stderr = process.communicate()
            else:
                stdout, stderr = "", ""
            probe = {
                "probe": index,
                "pid": process.pid if process is not None else None,
                "returncode": process.returncode if process is not None else None,
                "duration_seconds": round(time.monotonic() - started, 3),
                "timeout": True,
                "stdout": str(stdout or exc.stdout or "")[-500:],
                "stderr": str(stderr or exc.stderr or "")[-500:],
            }
        probes.append(probe)
    return probes


def generate_scaled_fixture(
    source_root: Path,
    output_root: Path,
    scale: int,
) -> dict[str, Any]:
    """Copy the S27 CSV/image convention into a new deterministic directory.

    The generator never edits the source fixture and refuses to write into an
    existing or overlapping directory.  It is a preparation utility only; it
    does not upload or import the generated fixture into the application.
    """

    if scale < 1:
        raise ValueError("Fixture scale must be at least 1")
    if scale > 1000:
        raise ValueError("Fixture scale is capped at 1000 for an explicit local run")
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    if (
        source_root == output_root
        or source_root in output_root.parents
        or output_root in source_root.parents
    ):
        raise ValueError("Scaled fixture output must be outside the source fixture")
    if output_root.exists():
        raise FileExistsError(f"Scaled fixture output already exists: {output_root}")

    source_tables = sorted(source_root.glob("*.csv"))
    if len(source_tables) != 1:
        raise ValueError("Expected exactly one CSV table in the S27 fixture root")
    source_table = source_tables[0]
    source_images = source_root / "images"
    if not source_images.is_dir():
        raise FileNotFoundError(f"Missing S27 image directory: {source_images}")

    with source_table.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        fieldnames = list(reader.fieldnames or [])
        if fieldnames != ["text", "image_name"]:
            raise ValueError("S27 fixture CSV must have text,image_name columns")
        source_rows = list(reader)
    if not source_rows:
        raise ValueError("S27 fixture CSV is empty")

    output_images = output_root / "images"
    output_images.mkdir(parents=True, exist_ok=False)
    output_rows: list[dict[str, str]] = []
    image_hashes: dict[str, str] = {}
    row_number = 0
    for repetition in range(scale):
        for source_row in source_rows:
            row_number += 1
            source_image = source_images / source_row["image_name"]
            if not source_image.is_file():
                raise FileNotFoundError(f"Missing S27 image: {source_image}")
            output_name = f"image_{row_number:04d}{source_image.suffix.lower()}"
            output_image = output_images / output_name
            shutil.copyfile(source_image, output_image)
            image_hashes[output_name] = _sha256(output_image)
            output_rows.append(
                {"text": source_row["text"], "image_name": output_name}
            )

    output_table = output_root / f"{source_table.stem}_scale{scale}.csv"
    with output_table.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    manifest = {
        "fixture": "S27 generated non-clinical synthetic dataset",
        "source_root": str(source_root),
        "scale": scale,
        "rows": len(output_rows),
        "image_dimensions": [64, 64],
        "table": {"name": output_table.name, "sha256": _sha256(output_table)},
        "images": image_hashes,
    }
    manifest_path = output_root / "fixture-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "root": str(output_root),
        "table": str(output_table),
        "manifest": str(manifest_path),
        "rows": len(output_rows),
        "scale": scale,
    }


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--group",
        type=normalize_group,
        choices=sorted(VALID_GROUPS),
        default="ALL",
        help="Run or print only one group: S51, S53, or ALL.",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("XREPORT_BASE_URL", "http://127.0.0.1:8003"),
    )
    parser.add_argument("--resource-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--fixture-root", type=Path, default=DEFAULT_FIXTURE_ROOT)
    parser.add_argument("--fixture-scale", type=int, default=1)
    parser.add_argument("--dataset-name", default="s27_technical_fixture")
    parser.add_argument("--processed-dataset", default="s28_release_20260925")
    parser.add_argument("--checkpoint", default="XREPORT_20260925T141533")
    parser.add_argument(
        "--execution-lane",
        type=normalize_execution_lane,
        choices=sorted(EXECUTION_LANES),
        default="cuda",
        help="Device contract: cuda, cpu, or unavailable-gpu fallback.",
    )
    parser.add_argument(
        "--start-mode",
        type=normalize_start_mode,
        choices=sorted(START_MODES),
        default="clean",
        help="Declare whether the backend was started clean or already warm.",
    )
    parser.add_argument(
        "--generate-scaled-fixture",
        action="store_true",
        help="Copy the S27 fixture convention to --scaled-fixture-output and exit.",
    )
    parser.add_argument("--scaled-fixture-output", type=Path)
    parser.add_argument("--request-timeout", type=float, default=30.0)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--sample-interval", type=float, default=1.0)
    parser.add_argument(
        "--cold-start-count",
        type=int,
        default=10,
        help="Fresh backend-app import probes for S51/ALL (default: 10).",
    )
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    if args.generate_scaled_fixture:
        if args.scaled_fixture_output is None:
            raise SystemExit("--scaled-fixture-output is required with --generate-scaled-fixture")
        result = generate_scaled_fixture(
            args.fixture_root,
            args.scaled_fixture_output,
            args.fixture_scale,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.poll_interval <= 0 or args.sample_interval <= 0:
        raise SystemExit("--poll-interval and --sample-interval must be positive")
    if args.request_timeout <= 0:
        raise SystemExit("--request-timeout must be positive")
    if args.cold_start_count < 0:
        raise SystemExit("--cold-start-count must be non-negative")
    fixture = FixtureDefinition(
        dataset=args.dataset_name,
        processed_dataset=args.processed_dataset,
        checkpoint=args.checkpoint,
        root=args.fixture_root,
        scale=args.fixture_scale,
    )
    plan = build_plan(
        args.group,
        fixture,
        poll_interval_seconds=args.poll_interval,
        sample_interval_seconds=args.sample_interval,
        request_timeout_seconds=args.request_timeout,
        cold_start_count=args.cold_start_count,
        execution_lane=args.execution_lane,
        start_mode=args.start_mode,
    )

    if args.plan_only:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    if args.resource_root is None:
        raise SystemExit("--resource-root is required for a live run")
    if args.output is None:
        raise SystemExit("--output is required for a live run")
    scenarios = build_scenarios(
        args.group,
        fixture,
        poll_interval_seconds=args.poll_interval,
        sample_interval_seconds=args.sample_interval,
        execution_lane=args.execution_lane,
    )
    client = ApiClient(args.base_url, timeout_seconds=args.request_timeout)
    api = ApiOperations(client)
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "captured_at_utc": utc_now(),
        "repository": _repository_metadata(),
        "host": _host_metadata(),
        "base_url": args.base_url,
        "execution": plan["execution"],
        "resource_root": str(args.resource_root.resolve()),
        "selected_group": args.group,
        "fixture": fixture.to_dict(),
        "plan": plan,
        "initial": None,
        "initial_database": None,
        "cold_start_probes": [],
        "scenarios": [],
        "cleanup": None,
        "final": None,
        "final_database": None,
        "limitations": plan["limitations"],
    }
    tracker: ResourceTracker | None = None
    execution_errors = 0
    try:
        initial = api.capture_state(fixture, include_reports=False)
        receipt["initial"] = initial
        receipt["initial_database"] = database_integrity(args.resource_root)
        tracker = ResourceTracker.from_state(initial)
        sampler = ResourceSampler(ROOT_DIR, args.resource_root)
        if args.group in {"ALL", "S51"} and args.cold_start_count:
            receipt["cold_start_probes"] = run_cold_start_probes(
                args.resource_root,
                args.cold_start_count,
            )
        for scenario in scenarios:
            batch = run_scenario(api, scenario, sampler, tracker)
            receipt["scenarios"].append(batch)
            execution_errors += len(batch["errors"])
    except Exception as exc:  # noqa: BLE001
        receipt["fatal_error"] = str(exc)
        execution_errors += 1
    finally:
        if tracker is not None:
            receipt["cleanup"] = CleanupManager(api).cleanup(tracker)
            execution_errors += len(receipt["cleanup"].get("errors", []))
        try:
            receipt["final"] = api.capture_state(fixture, include_reports=True)
            receipt["final_database"] = database_integrity(args.resource_root)
        except Exception as exc:  # noqa: BLE001
            receipt["final"] = {"error": str(exc)}
            execution_errors += 1
        ReceiptSerializer.write(args.output, receipt)

    print(
        json.dumps(
            {
                "output": str(args.output),
                "group": args.group,
                "scenarios": len(receipt["scenarios"]),
                "execution_errors": execution_errors,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 1 if execution_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())

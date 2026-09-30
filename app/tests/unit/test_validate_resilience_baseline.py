from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import time


SCRIPT_PATH = Path(__file__).parents[2] / "scripts" / "validate_resilience_baseline.py"
_SPEC = importlib.util.spec_from_file_location("validate_resilience_baseline", SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
resilience = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = resilience
_SPEC.loader.exec_module(resilience)


def test_execution_lane_and_scale_are_visible_in_the_scenario_contract() -> None:
    fixture = resilience.FixtureDefinition(scale=8)
    fallback = resilience.build_scenarios(
        "S51",
        fixture,
        execution_lane="unavailable-gpu",
    )
    cpu = resilience.build_scenarios("S51", fixture, execution_lane="cpu")

    fallback_training = [
        operation
        for scenario in fallback
        for operation in scenario.operations
        if operation.job_type == "training"
    ]
    cpu_training = [
        operation
        for scenario in cpu
        for operation in scenario.operations
        if operation.job_type == "training"
    ]
    assert fixture.to_dict()["expected_rows"] == 64
    assert fallback[0].execution_lane == "unavailable-gpu"
    assert all(operation.payload["use_device_GPU"] for operation in fallback_training)
    assert all(not operation.payload["use_device_GPU"] for operation in cpu_training)
    cancellation = next(
        operation
        for scenario in fallback
        for operation in scenario.operations
        if scenario.scenario_id == "s51_training_cancellation"
    )
    assert cancellation.expected_terminal_statuses == ("cancelled",)


def test_training_observation_records_pid_phase_latency_and_cleanup() -> None:
    job = {
        "worker_pids": [],
        "polls": [
            {
                "captured_at_utc": "2026-09-30T12:00:00Z",
                "elapsed_seconds": 1.5,
                "response": {
                    "body": {
                        "status": "running",
                        "result": {
                            "worker_pid": 4321,
                            "worker_phase": "first_batch_completed",
                            "progress_percent": 25,
                        },
                    }
                },
            }
        ],
    }

    resilience._record_training_observations(job)
    resilience._record_worker_cleanup(job)

    assert job["worker_pid"] == 4321
    assert job["worker_pids"] == [4321]
    assert job["first_progress_latency_seconds"] == 1.5
    assert job["first_batch_latency_seconds"] == 1.5
    assert job["last_worker_phase"] == "first_batch_completed"
    assert job["worker_cleanup"]["status"] in {"clean", "orphaned", "unmeasurable"}


def test_device_lane_assertion_fails_when_provenance_is_missing() -> None:
    scenario = resilience.build_scenarios(
        "S51",
        resilience.FixtureDefinition(),
        execution_lane="unavailable-gpu",
    )[0]
    batch = {"system_samples": [], "errors": []}

    resilience._assert_device_lane_observed(batch, scenario)

    assert batch["errors"]
    assert batch["errors"][0]["phase"] == "device_provenance"


def test_job_poller_rejects_an_unexpected_terminal_status() -> None:
    class FakeApi:
        def job_status(self, job_id: str) -> dict[str, object]:
            assert job_id == "job-1"
            return {
                "status": 200,
                "body": {
                    "status": "completed",
                    "result": {"worker_pid": 1234},
                },
            }

    jobs = {
        "job-1": {
            "job_id": "job-1",
            "expected_terminal_statuses": ["cancelled"],
            "polls": [],
            "terminal": False,
        }
    }
    errors: list[dict[str, object]] = []

    resilience.JobPoller(FakeApi(), 0.001).poll(
        jobs,
        errors,
        timeout_seconds=1,
        started_monotonic=time.monotonic(),
    )

    assert jobs["job-1"]["terminal_status"] == "completed"
    assert any(error["phase"] == "terminal_status" for error in errors)

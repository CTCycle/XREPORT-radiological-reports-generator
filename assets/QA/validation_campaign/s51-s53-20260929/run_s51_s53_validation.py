from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import platform
import subprocess
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TERMINAL_STATUSES = {"completed", "failed", "cancelled"}
DATASET = "s27_technical_fixture"
PROCESSED_DATASET = "s28_release_20260925"
CHECKPOINT = "XREPORT_20260925T141533"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class ApiFailure(RuntimeError):
    def __init__(self, method: str, path: str, status: int, body: Any) -> None:
        super().__init__(f"{method} {path} returned HTTP {status}: {body}")
        self.method = method
        self.path = path
        self.status = status
        self.body = body


class ApiClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        expected: set[int] | None = None,
    ) -> dict[str, Any]:
        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
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
        if expected is not None and status not in expected:
            raise ApiFailure(method, path, status, body)
        if expected is None and status >= 400:
            raise ApiFailure(method, path, status, body)
        return {"status": status, "body": body}


def training_payload(checkpoint_id: str, use_gpu: bool) -> dict[str, Any]:
    return {
        "dataset_name": PROCESSED_DATASET,
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


def validation_payload() -> dict[str, Any]:
    return {
        "dataset_name": DATASET,
        "metrics": ["text_statistics", "image_statistics", "pixels_distribution"],
        "sample_size": 1.0,
        "seed": 42,
    }


def evaluation_payload() -> dict[str, Any]:
    return {
        "checkpoint": CHECKPOINT,
        "metrics": ["evaluation_report"],
        "num_samples": 10,
        "seed": 42,
    }


def processing_payload(custom_name: str) -> dict[str, Any]:
    return {
        "dataset_name": DATASET,
        "custom_name": custom_name,
        "sample_size": 1.0,
        "validation_size": 0.25,
        "tokenizer": "distilbert-base-uncased",
        "max_report_size": 200,
    }


def sample_system(resource_root: Path) -> dict[str, Any]:
    sample: dict[str, Any] = {"captured_at_utc": utc_now()}
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            check=False,
            text=True,
            timeout=3,
        )
        if completed.returncode == 0:
            values = completed.stdout.strip().split(",")
            if len(values) >= 2:
                sample["gpu_memory_used_mib"] = int(values[0].strip())
                sample["gpu_utilization_percent"] = int(values[1].strip())
    except (FileNotFoundError, subprocess.SubprocessError, ValueError):
        pass

    try:
        import psutil

        repo_text = str(Path.cwd()).casefold()
        working_set = 0
        process_count = 0
        for process in psutil.process_iter(["name", "cwd", "cmdline", "memory_info"]):
            try:
                info = process.info
                name = str(info.get("name") or "").casefold()
                cwd = str(info.get("cwd") or "").casefold()
                command_line = " ".join(info.get("cmdline") or []).casefold()
                if "python" not in name:
                    continue
                if repo_text not in cwd and repo_text not in command_line:
                    continue
                memory_info = info.get("memory_info")
                if memory_info is not None:
                    working_set += int(memory_info.rss)
                    process_count += 1
            except (psutil.Error, OSError):
                continue
        sample["xreport_python_working_set_bytes"] = working_set
        sample["xreport_python_processes"] = process_count
    except ImportError:
        pass

    log_dir = resource_root / "logs"
    if log_dir.is_dir():
        device_lines: list[str] = []
        for log_path in sorted(log_dir.glob("*.log"), key=lambda item: item.stat().st_mtime):
            try:
                for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
                    if "GPU (cuda:" in line or "CPU is set as the active device" in line or "No GPU found" in line:
                        device_lines.append(line.strip())
            except OSError:
                continue
        if device_lines:
            sample["device_log_matches"] = device_lines[-12:]
    return sample


def start_and_poll(
    client: ApiClient,
    name: str,
    specs: list[tuple[str, str, dict[str, Any]]],
    resource_root: Path,
    *,
    timeout_seconds: int = 360,
) -> dict[str, Any]:
    batch: dict[str, Any] = {
        "name": name,
        "started_at_utc": utc_now(),
        "specs": [
            {"name": spec_name, "path": path, "payload": payload}
            for spec_name, path, payload in specs
        ],
        "submissions": [],
        "jobs": {},
        "system_samples": [],
        "errors": [],
    }
    stop_monitor = threading.Event()

    def monitor() -> None:
        while not stop_monitor.is_set():
            batch["system_samples"].append(sample_system(resource_root))
            stop_monitor.wait(0.5)

    monitor_thread = threading.Thread(target=monitor, name=f"monitor-{name}", daemon=True)
    monitor_thread.start()
    barrier = threading.Barrier(len(specs))

    def submit(spec: tuple[str, str, dict[str, Any]]) -> dict[str, Any]:
        spec_name, path, payload = spec
        barrier.wait(timeout=15)
        submitted_at = utc_now()
        monotonic_start = time.monotonic()
        try:
            response = client.request("POST", path, payload, expected={202})
            return {
                "name": spec_name,
                "submitted_at_utc": submitted_at,
                "submit_elapsed_seconds": round(time.monotonic() - monotonic_start, 3),
                "response": response,
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "name": spec_name,
                "submitted_at_utc": submitted_at,
                "submit_elapsed_seconds": round(time.monotonic() - monotonic_start, 3),
                "error": str(exc),
            }

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(specs)) as executor:
            futures = [executor.submit(submit, spec) for spec in specs]
            submissions = [future.result(timeout=45) for future in futures]
        batch["submissions"] = submissions
        for submission in submissions:
            body = submission.get("response", {}).get("body", {})
            job_id = body.get("job_id") if isinstance(body, dict) else None
            if job_id:
                batch["jobs"][job_id] = {
                    "name": submission["name"],
                    "job_id": job_id,
                    "submitted_at_utc": submission["submitted_at_utc"],
                    "polls": [],
                }
            else:
                batch["errors"].append(submission)

        deadline = time.monotonic() + timeout_seconds
        while batch["jobs"] and time.monotonic() < deadline:
            all_terminal = True
            for job_id, job in batch["jobs"].items():
                if job.get("terminal"):
                    continue
                response = client.request("GET", f"/api/jobs/{job_id}")
                status = response["body"]
                poll = {"captured_at_utc": utc_now(), "response": status}
                job["polls"].append(poll)
                job["last_status"] = status
                if isinstance(status, dict) and status.get("status") in TERMINAL_STATUSES:
                    job["terminal"] = True
                    job["terminal_at_utc"] = poll["captured_at_utc"]
                else:
                    all_terminal = False
            if all_terminal:
                break
            time.sleep(0.5)
        for job in batch["jobs"].values():
            if not job.get("terminal"):
                batch["errors"].append({"job_id": job["job_id"], "error": "poll timeout"})
    except Exception as exc:  # noqa: BLE001
        batch["errors"].append({"error": str(exc)})
    finally:
        stop_monitor.set()
        monitor_thread.join(timeout=5)

    batch["finished_at_utc"] = utc_now()
    batch["running_jobs_after"] = client.request(
        "GET", "/api/jobs?status=running"
    )["body"]
    return batch


def extract_training_checkpoint(batch: dict[str, Any]) -> str | None:
    for job in batch.get("jobs", {}).values():
        if job.get("name") != "training":
            continue
        status = job.get("last_status") or {}
        result = status.get("result") if isinstance(status, dict) else None
        if not isinstance(result, dict):
            continue
        checkpoint_path = result.get("checkpoint_path")
        if isinstance(checkpoint_path, str):
            return Path(checkpoint_path).name
    return None


def cleanup(
    client: ApiClient,
    checkpoint_names: list[str],
    processed_names: list[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {"checkpoints": [], "processed_datasets": []}
    for name in dict.fromkeys(checkpoint_names):
        try:
            response = client.request("DELETE", f"/api/training/checkpoints/{name}", expected={200, 404})
            result["checkpoints"].append({"name": name, "response": response})
        except Exception as exc:  # noqa: BLE001
            result["checkpoints"].append({"name": name, "error": str(exc)})
    for name in dict.fromkeys(processed_names):
        try:
            response = client.request("DELETE", f"/api/preparation/dataset/{name}", expected={200, 404})
            result["processed_datasets"].append({"name": name, "response": response})
        except Exception as exc:  # noqa: BLE001
            result["processed_datasets"].append({"name": name, "error": str(exc)})
    result["running_jobs"] = client.request("GET", "/api/jobs?status=running")["body"]
    result["checkpoints_after"] = client.request("GET", "/api/training/checkpoints")["body"]
    result["processed_datasets_after"] = client.request(
        "GET", "/api/preparation/dataset/processed/names"
    )["body"]
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=os.environ.get("XREPORT_BASE_URL", "http://127.0.0.1:8003"))
    parser.add_argument("--resource-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    client = ApiClient(args.base_url)
    output: dict[str, Any] = {
        "captured_at_utc": utc_now(),
        "repository_head": os.environ.get("XREPORT_VALIDATION_HEAD"),
        "host": {
            "os": platform.platform(),
            "python": platform.python_version(),
            "machine": platform.machine(),
        },
        "base_url": args.base_url,
        "fixture": {
            "dataset": DATASET,
            "processed_dataset": PROCESSED_DATASET,
            "checkpoint": CHECKPOINT,
        },
        "initial": {
            "health": client.request("GET", "/api/health")["body"],
            "running_jobs": client.request("GET", "/api/jobs?status=running")["body"],
            "checkpoints": client.request("GET", "/api/training/checkpoints")["body"],
            "processed_datasets": client.request(
                "GET", "/api/preparation/dataset/processed/names"
            )["body"],
        },
        "batches": [],
        "cleanup": {"checkpoints": [], "processed_datasets": []},
    }

    created_checkpoints: list[str] = []
    created_processed_names: list[str] = []

    for index in range(1, 4):
        checkpoint_name = f"s51_r{index}_cuda"
        batch = start_and_poll(
            client,
            f"triple_cuda_{index}",
            [
                ("validation", "/api/validation/run", validation_payload()),
                (
                    "checkpoint_evaluation",
                    "/api/validation/checkpoint",
                    evaluation_payload(),
                ),
                ("training", "/api/training/start", training_payload(checkpoint_name, True)),
            ],
            args.resource_root,
        )
        output["batches"].append(batch)
        created_checkpoints.append(checkpoint_name)
        extracted = extract_training_checkpoint(batch)
        if extracted and extracted not in created_checkpoints:
            created_checkpoints.append(extracted)

    processed_name = "s51_pair_eval_processed"
    created_processed_names.append(processed_name)
    output["batches"].append(
        start_and_poll(
            client,
            "processing_and_evaluation",
            [
                (
                    "dataset_processing",
                    "/api/preparation/dataset/process",
                    processing_payload(processed_name),
                ),
                (
                    "checkpoint_evaluation",
                    "/api/validation/checkpoint",
                    evaluation_payload(),
                ),
            ],
            args.resource_root,
        )
    )

    processed_name = "s51_pair_training_processed"
    checkpoint_name = "s51_pair_training_cuda"
    created_processed_names.append(processed_name)
    created_checkpoints.append(checkpoint_name)
    batch = start_and_poll(
        client,
        "processing_and_training",
        [
            (
                "dataset_processing",
                "/api/preparation/dataset/process",
                processing_payload(processed_name),
            ),
            ("training", "/api/training/start", training_payload(checkpoint_name, True)),
        ],
        args.resource_root,
    )
    output["batches"].append(batch)
    extracted = extract_training_checkpoint(batch)
    if extracted and extracted not in created_checkpoints:
        created_checkpoints.append(extracted)

    for label, use_gpu in (("cuda_baseline", True), ("cpu_baseline", False)):
        checkpoint_name = f"s53_{label}"
        created_checkpoints.append(checkpoint_name)
        batch = start_and_poll(
            client,
            label,
            [("training", "/api/training/start", training_payload(checkpoint_name, use_gpu))],
            args.resource_root,
            timeout_seconds=600,
        )
        batch["requested_device"] = "cuda" if use_gpu else "cpu"
        output["batches"].append(batch)
        extracted = extract_training_checkpoint(batch)
        if extracted and extracted not in created_checkpoints:
            created_checkpoints.append(extracted)

    output["cleanup"] = cleanup(client, created_checkpoints, created_processed_names)
    output["final"] = {
        "running_jobs": client.request("GET", "/api/jobs?status=running")["body"],
        "health": client.request("GET", "/api/health")["body"],
        "validation_report": client.request(
            "GET", f"/api/validation/reports/{DATASET}"
        )["body"],
        "checkpoint_evaluation_report": client.request(
            "GET", f"/api/validation/checkpoint/reports/{CHECKPOINT}"
        )["body"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "batches": len(output["batches"]), "cleanup": output["cleanup"]}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

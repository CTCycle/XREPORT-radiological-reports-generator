from __future__ import annotations

import ctypes
import gc
import os
import shutil
import time
import tracemalloc
import uuid
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
import pytest
import sqlalchemy

from XREPORT.server.routes.training import training_endpoint, training_state
from XREPORT.server.schemas.training import StartTrainingRequest
from XREPORT.server.utils.constants import (
    CHECKPOINT_PATH,
    ENCODERS_PATH,
    PROCESSING_METADATA_TABLE,
    TRAINING_DATASET_TABLE,
)
from XREPORT.server.utils.jobs import job_manager
from XREPORT.server.utils.repository.serializer import DataSerializer
from XREPORT.server.database.database import database


###############################################################################
class ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("page_fault_count", ctypes.c_ulong),
        ("peak_working_set_size", ctypes.c_size_t),
        ("working_set_size", ctypes.c_size_t),
        ("quota_peak_paged_pool_usage", ctypes.c_size_t),
        ("quota_paged_pool_usage", ctypes.c_size_t),
        ("quota_peak_nonpaged_pool_usage", ctypes.c_size_t),
        ("quota_nonpaged_pool_usage", ctypes.c_size_t),
        ("pagefile_usage", ctypes.c_size_t),
        ("peak_pagefile_usage", ctypes.c_size_t),
    ]


###############################################################################
def get_env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


# -----------------------------------------------------------------------------
def get_env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


###############################################################################
def ensure_offline_mode() -> None:
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


###############################################################################
def is_encoder_cached() -> bool:
    encoder_root = os.path.join(
        ENCODERS_PATH, "models--microsoft--beit-base-patch16-224"
    )
    if not os.path.isdir(encoder_root):
        return False
    for walk_entry in os.walk(encoder_root):
        file_names = walk_entry[2]
        if file_names:
            return True
    return False


###############################################################################
def build_synthetic_training_data(
    image_root: str,
    sample_count: int,
    sequence_length: int,
    vocabulary_size: int,
    validation_size: float,
    seed: int,
    class_count: int,
    class_probabilities: list[float] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if sample_count < 2:
        raise ValueError("sample_count must be >= 2 to allow train/validation split")
    if sequence_length < 2:
        raise ValueError("sequence_length must be >= 2")
    if vocabulary_size < 4:
        raise ValueError("vocabulary_size must be >= 4")
    if class_count < 2:
        raise ValueError("class_count must be >= 2")
    if class_count >= vocabulary_size:
        raise ValueError("class_count must be smaller than vocabulary_size")

    rng = np.random.default_rng(seed)
    os.makedirs(image_root, exist_ok=True)

    base_image = rng.integers(0, 255, size=(224, 224, 3), dtype=np.uint8)
    token_values = np.arange(1, class_count + 1, dtype=np.int64)
    if class_probabilities is not None:
        probabilities = np.asarray(class_probabilities, dtype=np.float64)
        if probabilities.shape[0] != token_values.shape[0]:
            raise ValueError("class_probabilities length must match class_count")
        probabilities = probabilities / probabilities.sum()
    else:
        probabilities = None

    token_length = sequence_length + 1
    tokens = rng.choice(
        token_values,
        size=(sample_count, token_length),
        replace=True,
        p=probabilities,
    )

    ids = list(range(1, sample_count + 1))
    filenames = [f"synthetic_{item_id}.png" for item_id in ids]
    paths = [os.path.join(image_root, name) for name in filenames]

    for index, path in enumerate(paths):
        image_array = (base_image.astype(np.uint16) + index) % 255
        image = Image.fromarray(image_array.astype(np.uint8))
        image.save(path, format="PNG")

    indices = np.arange(sample_count)
    rng.shuffle(indices)
    train_count = max(1, int(sample_count * (1.0 - validation_size)))
    split_flags = ["train"] * sample_count
    for idx in indices[train_count:]:
        split_flags[idx] = "validation"

    dataset = pd.DataFrame(
        {
            "id": ids,
            "image": filenames,
            "tokens": tokens.tolist(),
            "split": split_flags,
            "path": paths,
        }
    )

    metadata = {
        "sample_size": 1.0,
        "validation_size": validation_size,
        "seed": seed,
        "vocabulary_size": vocabulary_size,
        "max_report_size": sequence_length,
        "tokenizer": None,
    }

    return dataset, metadata


###############################################################################
def write_training_data(
    dataset_name: str,
    dataset: pd.DataFrame,
    metadata: dict[str, Any],
) -> str:
    hashcode = DataSerializer.generate_hashcode(
        {
            "dataset_name": dataset_name,
            "sample_size": metadata["sample_size"],
            "validation_size": metadata["validation_size"],
            "seed": metadata["seed"],
            "vocabulary_size": metadata["vocabulary_size"],
            "max_report_size": metadata["max_report_size"],
            "tokenizer": metadata.get("tokenizer"),
        }
    )
    serializer = DataSerializer()
    dataset_copy = dataset.copy()
    dataset_copy["dataset_name"] = dataset_name
    dataset_copy["hashcode"] = hashcode
    serializer.upsert_table(dataset_copy, TRAINING_DATASET_TABLE)

    metadata_record = {
        "dataset_name": dataset_name,
        "date": time.strftime("%Y-%m-%d"),
        "seed": metadata["seed"],
        "sample_size": metadata["sample_size"],
        "validation_size": metadata["validation_size"],
        "vocabulary_size": metadata["vocabulary_size"],
        "max_report_size": metadata["max_report_size"],
        "tokenizer": metadata.get("tokenizer"),
        "hashcode": hashcode,
        "source_dataset": None,
    }
    metadata_df = pd.DataFrame([metadata_record])
    serializer.upsert_table(metadata_df, PROCESSING_METADATA_TABLE)
    return hashcode


###############################################################################
def cleanup_training_data(dataset_name: str) -> None:
    with database.backend.engine.begin() as conn:
        conn.execute(
            sqlalchemy.text(
                'DELETE FROM "TRAINING_DATASET" WHERE dataset_name = :dataset_name'
            ),
            {"dataset_name": dataset_name},
        )
        conn.execute(
            sqlalchemy.text(
                'DELETE FROM "PROCESSING_METADATA" WHERE dataset_name = :dataset_name'
            ),
            {"dataset_name": dataset_name},
        )


###############################################################################
def get_training_timeout_seconds() -> float:
    return float(get_env_int("PERF_TEST_TIMEOUT_SECONDS", 120))


# -----------------------------------------------------------------------------
def get_perf_limits(strict: bool) -> tuple[int, int, int]:
    if strict:
        runtime_limit = get_env_int("PERF_TEST_RUNTIME_LIMIT_SECONDS", 90)
        memory_limit_mb = get_env_int("PERF_TEST_MEMORY_LIMIT_MB", 2048)
        growth_limit_mb = get_env_int("PERF_TEST_MEMORY_GROWTH_MB", 256)
    else:
        runtime_limit = get_env_int("PERF_TEST_RUNTIME_LIMIT_SECONDS", 240)
        memory_limit_mb = get_env_int("PERF_TEST_MEMORY_LIMIT_MB", 4096)
        growth_limit_mb = get_env_int("PERF_TEST_MEMORY_GROWTH_MB", 2048)
    return runtime_limit, memory_limit_mb, growth_limit_mb


###############################################################################
def get_scenario_matrix() -> list[dict[str, Any]]:
    scenario_mode = os.environ.get("PERF_TEST_MATRIX", "default").lower()

    scenarios = [
        {
            "name": "small_fast",
            "samples": 8,
            "validation_size": 0.25,
            "batch_size": 8,
            "embedding_dims": 64,
            "sequence_length": 32,
            "epochs": 1,
            "attention_heads": 1,
            "num_encoders": 1,
            "num_decoders": 1,
        },
        {
            "name": "small_large_batch",
            "samples": 8,
            "validation_size": 0.25,
            "batch_size": 16,
            "embedding_dims": 64,
            "sequence_length": 32,
            "epochs": 1,
            "attention_heads": 1,
            "num_encoders": 1,
            "num_decoders": 1,
        },
        {
            "name": "medium_sequence",
            "samples": 16,
            "validation_size": 0.25,
            "batch_size": 8,
            "embedding_dims": 128,
            "sequence_length": 64,
            "epochs": 1,
            "attention_heads": 1,
            "num_encoders": 1,
            "num_decoders": 1,
        },
    ]

    if scenario_mode == "fast":
        return scenarios[:1]
    return scenarios


###############################################################################
def wait_for_worker_pid(timeout_seconds: float) -> int | None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        worker = training_state.worker
        if worker is not None and worker.process is not None and worker.process.pid:
            return int(worker.process.pid)
        time.sleep(0.05)
    return None


###############################################################################
def sample_process_rss_bytes(process_id: int) -> int:
    if os.name != "nt":
        return 0
    process_query = 0x1000
    process_vm_read = 0x0010
    process_handle = ctypes.windll.kernel32.OpenProcess(
        process_query | process_vm_read, False, process_id
    )
    if not process_handle:
        return 0

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(ProcessMemoryCounters)
    result = ctypes.windll.psapi.GetProcessMemoryInfo(
        process_handle, ctypes.byref(counters), counters.cb
    )
    ctypes.windll.kernel32.CloseHandle(process_handle)
    if not result:
        return 0
    return int(counters.working_set_size)


###############################################################################
def start_training_job(request: StartTrainingRequest) -> str:
    job_response = training_endpoint.start_training(request)
    return job_response.job_id


###############################################################################
@pytest.mark.parametrize("scenario", get_scenario_matrix())
def test_training_pipeline_performance_matrix(tmp_path, scenario: dict[str, Any]) -> None:
    ensure_offline_mode()

    if not is_encoder_cached():
        pytest.skip("BEiT encoder cache not available for offline training")

    if job_manager.is_job_running("training"):
        pytest.skip("Training already running; perf test skipped")

    strict_mode = get_env_bool("PERF_TEST_STRICT", False)
    runtime_limit, memory_limit_mb, growth_limit_mb = get_perf_limits(strict_mode)
    timeout_seconds = get_training_timeout_seconds()

    temp_root = str(tmp_path)
    dataset_name = f"perf_{scenario['name']}_{uuid.uuid4().hex[:8]}"
    checkpoint_id = f"perf_{scenario['name']}_{uuid.uuid4().hex[:8]}"
    image_dir = os.path.join(temp_root, dataset_name)

    dataset, metadata = build_synthetic_training_data(
        image_root=image_dir,
        sample_count=scenario["samples"],
        sequence_length=scenario["sequence_length"],
        vocabulary_size=256,
        validation_size=scenario["validation_size"],
        seed=42,
        class_count=32,
        class_probabilities=None,
    )
    write_training_data(dataset_name, dataset, metadata)

    request = StartTrainingRequest(
        epochs=scenario["epochs"],
        batch_size=scenario["batch_size"],
        num_encoders=scenario["num_encoders"],
        num_decoders=scenario["num_decoders"],
        embedding_dims=scenario["embedding_dims"],
        attention_heads=scenario["attention_heads"],
        train_temp=1.0,
        freeze_img_encoder=True,
        use_img_augmentation=False,
        shuffle_with_buffer=False,
        shuffle_size=1,
        save_checkpoints=False,
        checkpoint_id=checkpoint_id,
        use_device_GPU=False,
        device_ID=0,
        plot_training_metrics=False,
        use_scheduler=False,
        target_LR=0.0001,
        warmup_steps=0,
    )

    job_id = ""
    start_time = time.perf_counter()
    timeout_deadline = time.monotonic() + timeout_seconds
    peak_rss_bytes = 0
    baseline_rss_bytes = 0
    worker_pid = None
    job_status = None
    hit_timeout = False

    tracemalloc.start()
    try:
        job_id = start_training_job(request)
        worker_pid = wait_for_worker_pid(timeout_seconds=10.0)
        if worker_pid:
            baseline_rss_bytes = sample_process_rss_bytes(worker_pid)

        while True:
            job_status = job_manager.get_job_status(job_id)
            if job_status is None:
                raise AssertionError("Training job status not found")

            status_value = job_status.get("status")
            if status_value in {"completed", "failed", "cancelled"}:
                break

            if worker_pid is None:
                worker = training_state.worker
                if worker is not None and worker.process is not None and worker.process.pid:
                    worker_pid = int(worker.process.pid)
                    baseline_rss_bytes = sample_process_rss_bytes(worker_pid)

            if worker_pid:
                rss_value = sample_process_rss_bytes(worker_pid)
                if rss_value > peak_rss_bytes:
                    peak_rss_bytes = rss_value

            if time.monotonic() >= timeout_deadline:
                hit_timeout = True
                training_endpoint.cancel_training_job(job_id)
                break

            time.sleep(0.25)

        if hit_timeout:
            raise AssertionError(
                f"Training scenario '{scenario['name']}' exceeded timeout of "
                f"{timeout_seconds:.1f}s and was cancelled."
            )

        if job_status and job_status.get("status") == "failed":
            raise AssertionError(
                f"Training scenario '{scenario['name']}' failed: "
                f"{job_status.get('error')}"
            )
    finally:
        tracemalloc.stop()
        if hit_timeout and training_state.worker is not None:
            if training_state.worker.is_alive():
                training_state.worker.terminate()
                training_state.worker.join(timeout=5)
        cleanup_training_data(dataset_name)
        if os.path.isdir(image_dir):
            shutil.rmtree(image_dir, ignore_errors=True)
        checkpoint_path = os.path.join(CHECKPOINT_PATH, checkpoint_id)
        if os.path.isdir(checkpoint_path):
            shutil.rmtree(checkpoint_path, ignore_errors=True)
        gc.collect()

    duration_seconds = time.perf_counter() - start_time
    current_rss_bytes = sample_process_rss_bytes(worker_pid) if worker_pid else 0
    if peak_rss_bytes < current_rss_bytes:
        peak_rss_bytes = current_rss_bytes

    growth_bytes = max(0, peak_rss_bytes - baseline_rss_bytes)
    peak_rss_mb = peak_rss_bytes / (1024 * 1024)
    growth_mb = growth_bytes / (1024 * 1024)

    assert duration_seconds <= runtime_limit, (
        f"Training scenario '{scenario['name']}' runtime "
        f"{duration_seconds:.2f}s exceeded limit {runtime_limit}s. "
        f"samples={scenario['samples']} batch={scenario['batch_size']} "
        f"seq={scenario['sequence_length']} embed={scenario['embedding_dims']}"
    )
    assert peak_rss_mb <= memory_limit_mb, (
        f"Training scenario '{scenario['name']}' peak RSS "
        f"{peak_rss_mb:.1f} MB exceeded limit {memory_limit_mb} MB. "
        f"samples={scenario['samples']} batch={scenario['batch_size']} "
        f"seq={scenario['sequence_length']} embed={scenario['embedding_dims']}"
    )
    assert growth_mb <= growth_limit_mb, (
        f"Training scenario '{scenario['name']}' memory growth "
        f"{growth_mb:.1f} MB exceeded limit {growth_limit_mb} MB. "
        f"samples={scenario['samples']} batch={scenario['batch_size']} "
        f"seq={scenario['sequence_length']} embed={scenario['embedding_dims']}"
    )

    print(
        "Training perf metrics:",
        {
            "scenario": scenario["name"],
            "duration_seconds": round(duration_seconds, 2),
            "peak_rss_mb": round(peak_rss_mb, 1),
            "growth_mb": round(growth_mb, 1),
            "samples": scenario["samples"],
            "batch_size": scenario["batch_size"],
            "sequence_length": scenario["sequence_length"],
            "embedding_dims": scenario["embedding_dims"],
        },
    )

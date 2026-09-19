"""Run a cache-only real-inference validation for one configured model.

This command never downloads weights, changes the model manifest, or accepts
gated access terms. It requires an already cached executable model in the
unvalidated or ready state and an explicit public/de-identified fixture.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import mimetypes
from pathlib import Path
import sys
import time
from typing import Any

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from server.common.path import ROOT_DIR  # noqa: E402
from server.configurations.inference_models import (  # noqa: E402
    embedded_inference_models,
)
from server.configurations.startup import get_server_settings  # noqa: E402
from server.domain.inference import InferenceImage  # noqa: E402
from server.models.inference.providers.huggingface import HuggingFaceProvider  # noqa: E402
from server.services.inference import (  # noqa: E402
    InferenceImageStore,
    run_inference_job,
)
import server.services.inference as inference_service  # noqa: E402
from server.services.inference_catalog import (  # noqa: E402
    InferenceModelCatalog,
    validation_contract_hash,
)
from server.services.inference_runtime import InferenceRuntimeCoordinator  # noqa: E402
from server.services.jobs import JobManager, JobState  # noqa: E402
from server.services.model_installation import ModelInstallationManager  # noqa: E402


RUN_LOG_DIR = ROOT_DIR / "assets" / "QA" / "inference_validation_runs"
RECEIPT_DIR = ROOT_DIR / "assets" / "QA" / "inference_validation"

###############################################################################
def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-ref", required=True, help="Configured model reference to validate"
    )
    parser.add_argument(
        "--image",
        type=Path,
        action="append",
        required=True,
        help="Public/de-identified fixture image; repeat for a multi-image study.",
    )
    parser.add_argument(
        "--profile",
        choices=("deterministic", "concise", "detailed"),
        default="deterministic",
    )
    parser.add_argument("--clinical-context", default="")
    parser.add_argument(
        "--fixture-provenance",
        default="",
        help="Public dataset accession, release, or URL; required for real validation.",
    )
    parser.add_argument(
        "--fixture-deidentification",
        default="",
        help="Explicit de-identification statement; required for real validation.",
    )
    parser.add_argument(
        "--fixture-sha256",
        action="append",
        required=True,
        help="Expected SHA-256; repeat in the same order as --image.",
    )
    return parser.parse_args()

###############################################################################
def _slug(model_ref: str) -> str:
    return model_ref.removeprefix("huggingface:").replace("/", "__")

###############################################################################
def _write_run_log(model_ref: str, payload: dict[str, object]) -> Path:
    RUN_LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = RUN_LOG_DIR / f"{_slug(model_ref)}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path

###############################################################################
def _fixture_metadata(
    image_path: Path,
    data: bytes,
    *,
    provenance: str,
    deidentification: str,
    expected_sha256: str,
) -> dict[str, str]:
    fixture_provenance = provenance.strip()
    fixture_deidentification = deidentification.strip()
    actual_sha256 = hashlib.sha256(data).hexdigest()
    if not fixture_provenance:
        raise ValueError(
            "Fixture provenance must identify a public source or dataset accession."
        )
    if not fixture_deidentification:
        raise ValueError(
            "Fixture de-identification provenance must be stated explicitly."
        )
    if expected_sha256.strip().lower() != actual_sha256:
        raise ValueError(
            "Fixture SHA-256 does not match the supplied image bytes: "
            f"expected {expected_sha256}, computed {actual_sha256}"
        )
    return {
        "filename": image_path.name,
        "provenance": fixture_provenance,
        "de_identification": fixture_deidentification,
        "sha256": actual_sha256,
    }

###############################################################################
def _deferred_payload(
    *,
    model_ref: str,
    revision: str | None,
    catalog_status: str,
    reason: str,
) -> dict[str, object]:
    return {
        "status": "deferred",
        "model_ref": model_ref,
        "revision": revision,
        "catalog_status": catalog_status,
        "reason": reason,
        "weights_downloaded": False,
    }

###############################################################################
def _expected_hashes(value: str | list[str], image_count: int) -> list[str]:
    hashes = [value] if isinstance(value, str) else list(value)
    if len(hashes) != image_count:
        raise ValueError("Provide exactly one --fixture-sha256 value per --image")
    return hashes

###############################################################################
def validate_cached_model(
    *,
    model_ref: str,
    image_paths: list[Path],
    profile: str,
    clinical_context: str,
    fixture_provenance: str,
    fixture_deidentification: str,
    fixture_sha256: str | list[str],
) -> dict[str, object]:
    """Validate one locally installed public model and return its evidence."""
    settings = get_server_settings().inference
    catalog = InferenceModelCatalog(settings).list_models()
    selected = next(
        (model for model in catalog.models if model.model_ref == model_ref), None
    )
    if selected is None:
        return _deferred_payload(
            model_ref=model_ref,
            revision=None,
            catalog_status="not_configured",
            reason=f"Model is not configured: {model_ref}",
        )
    if selected.provider != "huggingface" or selected.status not in {
        "ready",
        "unvalidated",
    }:
        payload = _deferred_payload(
            model_ref=selected.model_ref,
            revision=selected.model_revision,
            catalog_status=selected.status,
            reason=selected.status_message
            or "The cache-only validator supports Hugging Face catalogue entries only.",
        )
        path = _write_run_log(model_ref, payload)
        return {**payload, "log": str(path.relative_to(ROOT_DIR))}

    manifest_entry = next(
        (
            entry
            for entry in embedded_inference_models()
            if entry.model_ref == selected.model_ref
        ),
        None,
    )
    if manifest_entry is None:
        payload = _deferred_payload(
            model_ref=selected.model_ref,
            revision=selected.model_revision,
            catalog_status=selected.status,
            reason=(
                "The selected Hugging Face catalogue entry is absent from the "
                "configured manifest."
            ),
        )
        path = _write_run_log(model_ref, payload)
        return {**payload, "log": str(path.relative_to(ROOT_DIR))}

    if not image_paths:
        raise ValueError("At least one fixture image is required")
    hashes = _expected_hashes(fixture_sha256, len(image_paths))
    fixtures: list[dict[str, str]] = []
    images: list[InferenceImage] = []
    for image_path, expected_hash in zip(image_paths, hashes, strict=True):
        resolved_path = image_path.resolve()
        if not resolved_path.is_file():
            raise FileNotFoundError(resolved_path)
        data = resolved_path.read_bytes()
        fixtures.append(
            _fixture_metadata(
                resolved_path,
                data,
                provenance=fixture_provenance,
                deidentification=fixture_deidentification,
                expected_sha256=expected_hash,
            )
        )
        images.append(
            InferenceImage(
                filename=resolved_path.name,
                content_type=mimetypes.guess_type(resolved_path.name)[0]
                or "application/octet-stream",
                data=data,
                size_bytes=len(data),
            )
        )

    manifest = selected.model_dump(mode="json")
    manifest["revision"] = selected.model_revision

    ###############################################################################
    class RecordingRepository:
        saved_reports: list[dict[str, str]] = []
        generation_config: dict[str, object] = {}

        # -------------------------------------------------------------------------
        def save_generated_reports(
            self, reports: list[dict[str, str]], **kwargs: object
        ) -> None:
            self.saved_reports = list(reports)
            generation_config = kwargs.get("generation_config")
            self.generation_config = (
                dict(generation_config) if isinstance(generation_config, dict) else {}
            )

    job_manager = JobManager()
    image_store = InferenceImageStore()
    provider = HuggingFaceProvider(settings)
    runtime = InferenceRuntimeCoordinator(
        huggingface_provider=provider,
        installation_manager=ModelInstallationManager(),
    )
    recorder = RecordingRepository()
    request_id = "validation_fixture"
    image_store.store(request_id, images)
    original_get_manager = inference_service.get_job_manager
    original_get_store = inference_service.get_inference_image_store
    original_get_provider = inference_service.get_huggingface_provider
    original_repository = inference_service.InferenceRepository
    inference_service.get_job_manager = lambda: job_manager
    inference_service.get_inference_image_store = lambda: image_store
    inference_service.get_huggingface_provider = lambda: provider
    inference_service.InferenceRepository = lambda: recorder  # type: ignore[assignment]
    job_id = "validation-job"
    job_manager.jobs[job_id] = JobState(
        job_id=job_id,
        job_type="inference",
        status="running",
    )
    image_store.link_job(job_id, request_id)

    timings: dict[str, float] = {}
    peak_cuda_enabled = False
    original_load = provider._load
    original_generate_study = provider._generate_study

    def timed_load(load_manifest: dict[str, Any]):
        nonlocal peak_cuda_enabled
        started = time.perf_counter()
        loaded = original_load(load_manifest)
        timings["load_seconds"] = time.perf_counter() - started
        runtime_metadata = provider._runtime_metadata(loaded[0])
        if runtime_metadata["cuda_used"] and runtime_metadata["cuda_available"]:
            import torch

            torch.cuda.reset_peak_memory_stats()
            peak_cuda_enabled = True
        return loaded

    def timed_generate_study(**kwargs: Any):
        started = time.perf_counter()
        try:
            return original_generate_study(**kwargs)
        finally:
            timings["generation_seconds"] = time.perf_counter() - started

    provider._load = timed_load  # type: ignore[method-assign]
    provider._generate_study = timed_generate_study  # type: ignore[method-assign]
    total_started = time.perf_counter()
    try:
        result = run_inference_job(
            model_ref=model_ref,
            model_revision=selected.model_revision,
            model_manifest=manifest,
            generation_profile=profile,
            clinical_context=clinical_context,
            request_id=request_id,
            job_id=job_id,
            job_manager=job_manager,
            inference_image_store=image_store,
            runtime=runtime,
            repository=recorder,
        )
        previous_result = job_manager.jobs[job_id].result or {}
        job_manager.jobs[job_id].update(
            status="completed",
            result={**previous_result, **result},
        )
    finally:
        timings["total_seconds"] = time.perf_counter() - total_started
        inference_service.get_job_manager = original_get_manager
        inference_service.get_inference_image_store = original_get_store
        inference_service.get_huggingface_provider = original_get_provider
        inference_service.InferenceRepository = original_repository

    job_status = job_manager.get_job_status(job_id) or {}
    if job_status.get("status") != "completed":
        raise RuntimeError(
            "Validation job did not complete: "
            f"{job_status.get('error', job_status.get('status'))}"
        )
    api_result = job_status.get("result") or {}
    reports = api_result.get("reports")
    display_sections = api_result.get("display_sections")
    provenance = api_result.get("provenance")
    if not isinstance(reports, dict) or not reports:
        raise RuntimeError("The provider returned no reports")
    if not isinstance(display_sections, dict) or not isinstance(provenance, dict):
        raise RuntimeError("The job result omitted display sections or provenance")
    if (
        api_result.get("count") != len(reports)
        or api_result.get("report_filenames") != list(reports)
        or api_result.get("reports_ordered") != list(reports.values())
    ):
        raise RuntimeError("The job result is not API-compatible")
    declared_sections = set(selected.output_sections)
    for filename, report in reports.items():
        sections = display_sections.get(filename)
        if not isinstance(sections, dict) or set(sections) != declared_sections:
            raise RuntimeError(
                f"Output sections do not match the declared contract for {filename}"
            )
        if any(
            not isinstance(value, str) or not value.strip()
            for value in sections.values()
        ):
            raise RuntimeError(f"Output sections contain an empty value for {filename}")
        if "raw_report" in declared_sections and sections["raw_report"] != report:
            raise RuntimeError("Raw report text changed before the display contract")
    if recorder.saved_reports != [
        {"image": filename, "report": report} for filename, report in reports.items()
    ]:
        raise RuntimeError("Raw report text changed at the persistence boundary")
    if not {"display_sections", "provenance"}.issubset(recorder.generation_config):
        raise RuntimeError("Persistence metadata omitted display sections or provenance")

    runtime_metadata = provenance.get("runtime")
    if not isinstance(runtime_metadata, dict):
        runtime_metadata = {}
    input_metadata = api_result.get("metadata")
    if not isinstance(input_metadata, list):
        input_metadata = []
    payload: dict[str, object] = {
        "status": "passed",
        "real_inference": True,
        "model": manifest_entry.repository_id,
        "model_ref": selected.model_ref,
        "revision": selected.model_revision,
        "adapter": selected.adapter,
        "generation_profile": profile,
        "requested_device": runtime_metadata.get("requested_device"),
        "resolved_runtime_devices": runtime_metadata.get("resolved_devices", []),
        "resolved_device": runtime_metadata.get("resolved_device"),
        "dtype": runtime_metadata.get("model_dtype"),
        "cuda_available": runtime_metadata.get("cuda_available"),
        "cuda_used": runtime_metadata.get("cuda_used"),
        "image_count": len(images),
        "input_tensor_dimensions": [
            item.get("processed_tensor_dimensions")
            for item in input_metadata
            if isinstance(item, dict)
        ],
        "load_seconds": timings.get("load_seconds"),
        "generation_seconds": timings.get("generation_seconds"),
        "total_seconds": timings.get("total_seconds"),
        "peak_cuda_memory_bytes": None,
        "report_character_counts": {
            filename: len(report) for filename, report in reports.items()
        },
        "output_sections": {
            filename: sorted(sections)
            for filename, sections in display_sections.items()
            if isinstance(sections, dict)
        },
        "contract_hash": validation_contract_hash(manifest_entry),
        "fixtures": fixtures,
        "reports": reports,
        "display_sections": display_sections,
        "provenance": provenance,
        "api_result": api_result,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "weights_downloaded": False,
        "manifest_promoted": False,
    }
    if len(fixtures) == 1:
        payload["fixture"] = fixtures[0]
    if peak_cuda_enabled:
        import torch

        payload["peak_cuda_memory_bytes"] = int(torch.cuda.max_memory_allocated())

    RECEIPT_DIR.mkdir(parents=True, exist_ok=True)
    receipt = RECEIPT_DIR / f"{_slug(model_ref)}-{selected.model_revision}.json"
    receipt.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return {**payload, "receipt": str(receipt.relative_to(ROOT_DIR))}

###############################################################################
def main() -> int:
    args = _arguments()
    try:
        payload = validate_cached_model(
            model_ref=args.model_ref,
            image_paths=list(args.image),
            profile=args.profile,
            clinical_context=args.clinical_context,
            fixture_provenance=args.fixture_provenance,
            fixture_deidentification=args.fixture_deidentification,
            fixture_sha256=list(args.fixture_sha256),
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload.get("status") == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())

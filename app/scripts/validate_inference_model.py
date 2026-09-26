"""Run a cache-only real-inference validation for one configured model.

This command never downloads weights, changes the model manifest, or accepts
gated access terms. It requires an already cached executable model in the
unvalidated or ready state and an explicit public/de-identified fixture.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import mimetypes
import os
from pathlib import Path
import sys
import time
from typing import Any, cast

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from server.common.path import ROOT_DIR  # noqa: E402
from server.configurations.inference_models import (  # noqa: E402
    embedded_inference_models,
)
from server.configurations.startup import get_server_settings  # noqa: E402
from server.domain.inference import GenerationProfile, InferenceImage  # noqa: E402
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
from server.repositories.serialization.inference import (  # noqa: E402
    InferenceRepository,
)


RUN_LOG_DIR = ROOT_DIR / "assets" / "QA" / "inference_validation_runs"
RECEIPT_DIR = ROOT_DIR / "assets" / "QA" / "inference_validation"


@dataclass(frozen=True)
class ValidationCase:
    """One independent image-to-report validation case."""

    case_id: str
    image_path: Path
    profile: GenerationProfile
    clinical_context: str
    expected_sha256: str

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
        help="Public/de-identified fixture image; repeat for a multi-image study.",
    )
    parser.add_argument(
        "--case-manifest",
        type=Path,
        help=(
            "JSON manifest for independent cases; mutually exclusive with the "
            "legacy --image/--profile/--clinical-context/--fixture-sha256 path."
        ),
    )
    parser.add_argument(
        "--receipt-path",
        type=Path,
        help=(
            "Optional aggregate receipt path for independent-case mode. When omitted, "
            "the legacy canonical receipt location is used."
        ),
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


def _configure_validation_cache() -> Path | None:
    """Redirect Transformers' dynamic-module cache for isolated validation runs."""
    configured_root = os.environ.get("XREPORT_VALIDATION_CACHE_ROOT", "").strip()
    if not configured_root:
        return None
    cache_root = Path(configured_root).expanduser().resolve()
    modules_cache = cache_root / "huggingface" / "modules"
    modules_cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_MODULES_CACHE"] = str(modules_cache)

    import transformers.utils.hub as transformers_hub  # pyright: ignore[reportMissingImports]

    transformers_hub.HF_MODULES_CACHE = str(modules_cache)
    from server.models.inference.providers import adapters as adapters_module
    from server.models.inference.providers import huggingface as huggingface_module

    adapters_module.HF_MODULES_CACHE = str(modules_cache)
    huggingface_module.HF_MODULES_CACHE = str(modules_cache)
    return modules_cache


def load_case_manifest(path: Path) -> list[ValidationCase]:
    """Load and validate independent-case definitions from a JSON file."""
    manifest_path = path.resolve()
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Invalid case manifest: {manifest_path}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("cases"), list):
        raise ValueError("Case manifest must contain a 'cases' list")
    if not payload["cases"]:
        raise ValueError("Case manifest must contain at least one case")

    cases: list[ValidationCase] = []
    case_ids: set[str] = set()
    filenames: set[str] = set()
    valid_profiles = {"deterministic", "concise", "detailed"}
    for index, raw_case in enumerate(payload["cases"], start=1):
        if not isinstance(raw_case, dict):
            raise ValueError(f"Case {index} must be an object")
        case_id = str(raw_case.get("id", "")).strip()
        image_value = str(raw_case.get("image", "")).strip()
        profile = str(raw_case.get("profile", "")).strip()
        clinical_context = str(raw_case.get("clinical_context", ""))
        expected_sha256 = str(raw_case.get("sha256", "")).strip().lower()
        if not case_id or not image_value or not expected_sha256:
            raise ValueError(
                f"Case {index} requires non-empty id, image, and sha256 values"
            )
        if case_id in case_ids:
            raise ValueError(f"Duplicate case id: {case_id}")
        if profile not in valid_profiles:
            raise ValueError(
                f"Case {case_id} has unsupported profile: {profile or '<empty>'}"
            )
        if len(expected_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in expected_sha256
        ):
            raise ValueError(f"Case {case_id} has an invalid SHA-256 value")

        image_path = Path(image_value)
        if not image_path.is_absolute():
            image_path = manifest_path.parent / image_path
        image_path = image_path.resolve()
        filename = image_path.name
        if filename in filenames:
            raise ValueError(f"Duplicate case image filename: {filename}")
        case_ids.add(case_id)
        filenames.add(filename)
        cases.append(
            ValidationCase(
                case_id=case_id,
                image_path=image_path,
                profile=profile,  # type: ignore[arg-type]
                clinical_context=clinical_context,
                expected_sha256=expected_sha256,
            )
        )
    return cases

###############################################################################
def validate_cached_model(
    *,
    model_ref: str,
    image_paths: list[Path],
    profile: GenerationProfile,
    clinical_context: str,
    fixture_provenance: str,
    fixture_deidentification: str,
    fixture_sha256: str | list[str],
) -> dict[str, object]:
    """Validate one locally installed public model and return its evidence."""
    _configure_validation_cache()
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
            import torch  # pyright: ignore[reportMissingImports]

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
            repository=cast(InferenceRepository, recorder),
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
        import torch  # pyright: ignore[reportMissingImports]

        payload["peak_cuda_memory_bytes"] = int(torch.cuda.max_memory_allocated())

    RECEIPT_DIR.mkdir(parents=True, exist_ok=True)
    receipt = RECEIPT_DIR / f"{_slug(model_ref)}-{selected.model_revision}.json"
    receipt.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return {**payload, "receipt": str(receipt.relative_to(ROOT_DIR))}


def validate_cached_cases(
    *,
    model_ref: str,
    cases: list[ValidationCase],
    fixture_provenance: str,
    fixture_deidentification: str,
    repeat_case_id: str | None = None,
    require_distinct_reports: bool = False,
    write_receipt: bool = True,
    receipt_path: Path | None = None,
) -> dict[str, object]:
    """Run independent one-image cases while reusing one loaded model.

    This is intentionally separate from ``validate_cached_model``: repeated
    ``--image`` values in the legacy command represent one multi-view study for
    study-level adapters, whereas this function gives every case its own job,
    request, context, profile, and persisted technical result.
    """
    if not cases:
        raise ValueError("At least one independent validation case is required")

    validation_cache = _configure_validation_cache()
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
            or "The independent-case validator supports runnable Hugging Face entries only.",
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
        raise ValueError(f"The selected model is absent from the manifest: {model_ref}")

    fixture_provenance = fixture_provenance.strip()
    fixture_deidentification = fixture_deidentification.strip()
    if not fixture_provenance or not fixture_deidentification:
        raise ValueError(
            "Fixture provenance and de-identification statements are required"
        )

    prepared_cases: list[tuple[ValidationCase, bytes, dict[str, str]]] = []
    fixtures: list[dict[str, str]] = []
    for case in cases:
        resolved_path = case.image_path.resolve()
        if not resolved_path.is_file():
            raise FileNotFoundError(resolved_path)
        data = resolved_path.read_bytes()
        fixture = _fixture_metadata(
            resolved_path,
            data,
            provenance=fixture_provenance,
            deidentification=fixture_deidentification,
            expected_sha256=case.expected_sha256,
        )
        prepared_cases.append((case, data, fixture))
        fixtures.append(fixture)

    manifest = selected.model_dump(mode="json")
    manifest["revision"] = selected.model_revision

    class RecordingRepository:
        saved_reports: list[dict[str, str]] = []
        generation_config: dict[str, object] = {}

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
    original_get_manager = inference_service.get_job_manager
    original_get_store = inference_service.get_inference_image_store
    original_get_provider = inference_service.get_huggingface_provider
    original_repository = inference_service.InferenceRepository
    original_load = provider._load
    original_generate_study = provider._generate_study

    load_count = 0
    load_seconds: list[float] = []
    generation_seconds: list[float] = []
    peak_cuda_enabled = False
    peak_by_case: dict[str, int | None] = {}

    def timed_load(load_manifest: dict[str, Any]):
        nonlocal load_count, peak_cuda_enabled
        started = time.perf_counter()
        loaded = original_load(load_manifest)
        load_count += 1
        load_seconds.append(time.perf_counter() - started)
        runtime_metadata = provider._runtime_metadata(loaded[0])
        if runtime_metadata["cuda_used"] and runtime_metadata["cuda_available"]:
            import torch  # pyright: ignore[reportMissingImports]

            torch.cuda.reset_peak_memory_stats()
            peak_cuda_enabled = True
        return loaded

    def timed_generate_study(**kwargs: Any):
        started = time.perf_counter()
        try:
            return original_generate_study(**kwargs)
        finally:
            generation_seconds.append(time.perf_counter() - started)

    provider._load = timed_load  # type: ignore[method-assign]
    provider._generate_study = timed_generate_study  # type: ignore[method-assign]
    inference_service.get_job_manager = lambda: job_manager
    inference_service.get_inference_image_store = lambda: image_store
    inference_service.get_huggingface_provider = lambda: provider
    inference_service.InferenceRepository = lambda: recorder  # type: ignore[assignment]

    def _reset_peak_memory() -> None:
        if peak_cuda_enabled:
            import torch  # pyright: ignore[reportMissingImports]

            torch.cuda.reset_peak_memory_stats()

    def _peak_memory() -> int | None:
        if not peak_cuda_enabled:
            return None
        import torch  # pyright: ignore[reportMissingImports]

        return int(torch.cuda.max_memory_allocated())

    def _run_case(
        case: ValidationCase,
        data: bytes,
        fixture: dict[str, str],
        *,
        run_id: str,
    ) -> dict[str, object]:
        request_id = f"validation_case_{run_id}"
        job_id = f"validation-job-{run_id}"
        image = InferenceImage(
            filename=fixture["filename"],
            content_type=mimetypes.guess_type(fixture["filename"])[0]
            or "application/octet-stream",
            data=data,
            size_bytes=len(data),
        )
        image_store.store(request_id, [image])
        image_store.link_job(job_id, request_id)
        job_manager.jobs[job_id] = JobState(
            job_id=job_id,
            job_type="inference",
            status="running",
        )
        recorder.saved_reports = []
        recorder.generation_config = {}
        _reset_peak_memory()
        started = time.perf_counter()
        generation_count_before = len(generation_seconds)
        try:
            result = run_inference_job(
                model_ref=model_ref,
                model_revision=selected.model_revision,
                model_manifest=manifest,
                generation_profile=case.profile,
                clinical_context=case.clinical_context,
                request_id=request_id,
                job_id=job_id,
                job_manager=job_manager,
                inference_image_store=image_store,
                runtime=runtime,
                repository=cast(InferenceRepository, recorder),
            )
            previous_result = job_manager.jobs[job_id].result or {}
            job_manager.jobs[job_id].update(
                status="completed",
                result={**previous_result, **result},
            )
        except Exception as exc:  # noqa: BLE001
            job_manager.jobs[job_id].update(status="failed", error=str(exc))
            return {
                "case_id": case.case_id,
                "filename": fixture["filename"],
                "sha256": fixture["sha256"],
                "clinical_context": case.clinical_context,
                "generation_profile": case.profile,
                "error": str(exc),
                "total_seconds": time.perf_counter() - started,
                "generation_seconds": sum(
                    generation_seconds[generation_count_before:]
                ),
                "peak_cuda_memory_bytes": _peak_memory(),
            }

        total_seconds = time.perf_counter() - started
        job_status = job_manager.get_job_status(job_id) or {}
        if job_status.get("status") != "completed":
            return {
                "case_id": case.case_id,
                "filename": fixture["filename"],
                "sha256": fixture["sha256"],
                "clinical_context": case.clinical_context,
                "generation_profile": case.profile,
                "error": str(
                    job_status.get("error")
                    or f"Validation job ended as {job_status.get('status')}"
                ),
                "total_seconds": total_seconds,
                "generation_seconds": sum(
                    generation_seconds[generation_count_before:]
                ),
                "peak_cuda_memory_bytes": _peak_memory(),
            }

        api_result = job_status.get("result") or {}
        reports = api_result.get("reports")
        display_sections = api_result.get("display_sections")
        provenance = api_result.get("provenance")
        input_metadata = api_result.get("inference_metadata")
        report = reports.get(fixture["filename"]) if isinstance(reports, dict) else None
        sections = (
            display_sections.get(fixture["filename"])
            if isinstance(display_sections, dict)
            else None
        )
        declared_sections = set(selected.output_sections)
        errors: list[str] = []
        if not isinstance(reports, dict) or set(reports) != {fixture["filename"]}:
            errors.append("The independent case returned an unexpected report map")
        if not isinstance(report, str) or not report.strip():
            errors.append("The independent case returned no non-empty report")
        if not isinstance(display_sections, dict) or not isinstance(sections, dict):
            errors.append("The independent case omitted display sections")
        elif set(sections) != declared_sections or any(
            not isinstance(value, str) or not value.strip()
            for value in sections.values()
        ):
            errors.append("The independent case violated the declared output contract")
        if not isinstance(provenance, dict):
            errors.append("The independent case omitted provenance")
        if not isinstance(input_metadata, list) or len(input_metadata) != 1:
            errors.append("The independent case omitted one-image input metadata")
        else:
            metadata = input_metadata[0]
            if not isinstance(metadata, dict):
                errors.append("The independent case metadata is malformed")
            elif (
                metadata.get("filename") != fixture["filename"]
                or metadata.get("generation_profile") != case.profile
                or metadata.get("clinical_context") != case.clinical_context
            ):
                errors.append("The independent case input contract was not preserved")
        if recorder.saved_reports != [
            {"image": fixture["filename"], "report": report}
        ]:
            errors.append("Raw report text changed at the persistence boundary")
        if not {"display_sections", "provenance"}.issubset(
            recorder.generation_config
        ):
            errors.append("Persistence metadata omitted display sections or provenance")

        case_result: dict[str, object] = {
            "case_id": case.case_id,
            "filename": fixture["filename"],
            "sha256": fixture["sha256"],
            "clinical_context": case.clinical_context,
            "generation_profile": case.profile,
            "report": report,
            "display_sections": sections,
            "provenance": provenance,
            "input_metadata": input_metadata,
            "total_seconds": total_seconds,
            "generation_seconds": sum(
                generation_seconds[generation_count_before:]
            ),
            "peak_cuda_memory_bytes": _peak_memory(),
        }
        if errors:
            case_result["errors"] = errors
        peak_by_case[case.case_id] = _peak_memory()
        return case_result

    case_results: list[dict[str, object]] = []
    errors: list[str] = []
    repeat_result: dict[str, object] | None = None
    try:
        for case, data, fixture in prepared_cases:
            result = _run_case(case, data, fixture, run_id=case.case_id)
            case_results.append(result)
            result_errors = result.get("errors")
            if isinstance(result_errors, list):
                for error in result_errors:
                    errors.append(f"{case.case_id}: {error}")
            if result.get("error"):
                errors.append(f"{case.case_id}: {result['error']}")

        if repeat_case_id is not None:
            repeat_case = next(
                (
                    prepared
                    for prepared in prepared_cases
                    if prepared[0].case_id == repeat_case_id
                ),
                None,
            )
            if repeat_case is None:
                errors.append(
                    f"The requested repeat case was not found: {repeat_case_id}"
                )
            elif not errors:
                provider.unload()
                repeat_case_result = _run_case(
                    repeat_case[0],
                    repeat_case[1],
                    repeat_case[2],
                    run_id=f"{repeat_case[0].case_id}-reload",
                )
                repeat_result = repeat_case_result
                initial = next(
                    item
                    for item in case_results
                    if item["case_id"] == repeat_case[0].case_id
                )
                repeat_errors = repeat_case_result.get("errors")
                repeat_errors = (
                    repeat_errors if isinstance(repeat_errors, list) else []
                )
                report_reused = repeat_case_result.get("report") == initial.get(
                    "report"
                )
                sections_reused = (
                    repeat_case_result.get("display_sections")
                    == initial.get("display_sections")
                )
                repeat_ok = (
                    not repeat_case_result.get("error")
                    and not repeat_errors
                    and report_reused
                    and sections_reused
                    and load_count >= 2
                )
                repeat_result["report_reused"] = report_reused
                repeat_result["sections_reused"] = sections_reused
                repeat_result["load_count"] = load_count
                repeat_result["reuse_ok"] = repeat_ok
                if not repeat_ok:
                    errors.append(
                        f"{repeat_case[0].case_id}: reload/reuse sentinel failed"
                    )
    finally:
        provider.unload()
        inference_service.get_job_manager = original_get_manager
        inference_service.get_inference_image_store = original_get_store
        inference_service.get_huggingface_provider = original_get_provider
        inference_service.InferenceRepository = original_repository
        provider._load = original_load  # type: ignore[method-assign]
        provider._generate_study = original_generate_study  # type: ignore[method-assign]

    completed: list[dict[str, object]] = []
    for result in case_results:
        report = result.get("report")
        if isinstance(report, str) and report.strip():
            completed.append(result)
    reports = {
        str(result["filename"]): cast(str, result["report"])
        for result in completed
    }
    display_sections = {
        str(result["filename"]): result["display_sections"]
        for result in completed
        if isinstance(result.get("display_sections"), dict)
    }
    input_contract_ok = True
    output_contract_ok = True
    for result in completed:
        input_metadata = result.get("input_metadata")
        if not isinstance(input_metadata, list) or len(input_metadata) != 1:
            input_contract_ok = False
        else:
            metadata = input_metadata[0]
            input_contract_ok = input_contract_ok and isinstance(metadata, dict)
            if isinstance(metadata, dict):
                input_contract_ok = input_contract_ok and (
                    metadata.get("filename") == result["filename"]
                    and metadata.get("generation_profile")
                    == result["generation_profile"]
                    and metadata.get("clinical_context") == result["clinical_context"]
                )

        sections = result.get("display_sections")
        if not isinstance(sections, dict):
            output_contract_ok = False
        else:
            output_contract_ok = output_contract_ok and (
                set(sections) == set(selected.output_sections)
                and all(
                    isinstance(value, str) and bool(value.strip())
                    for value in sections.values()
                )
            )
    unique_report_count = len(set(reports.values()))
    repeat_ok = (
        repeat_case_id is None
        or (repeat_result is not None and repeat_result.get("reuse_ok") is True)
    )
    technical_passed = (
        not errors
        and len(completed) == len(cases)
        and input_contract_ok
        and output_contract_ok
        and repeat_ok
        and (not require_distinct_reports or unique_report_count == len(cases))
    )
    runtime_metadata = {}
    first_provenance = completed[0].get("provenance") if completed else None
    if isinstance(first_provenance, dict):
        candidate_runtime = first_provenance.get("runtime")
        if isinstance(candidate_runtime, dict):
            runtime_metadata = candidate_runtime
    technical_checks = {
        "completed_cases": len(completed),
        "expected_cases": len(cases),
        "input_contract_ok": input_contract_ok,
        "output_contract_ok": output_contract_ok,
        "unique_report_count": unique_report_count,
        "reports_all_distinct": unique_report_count == len(cases),
        "reload_reuse_requested": repeat_case_id is not None,
        "reload_reuse_ok": repeat_ok,
        "errors": errors,
        "technical_passed": technical_passed,
    }
    resource = {
        "model_load_count": load_count,
        "model_load_seconds": load_seconds,
        "generation_seconds": generation_seconds,
        "peak_cuda_memory_bytes": max(
            (value for value in peak_by_case.values() if value is not None),
            default=None,
        ),
        "peak_cuda_memory_by_case": peak_by_case,
        "runtime": runtime_metadata,
    }
    payload: dict[str, object] = {
        "status": "passed" if technical_passed else "failed",
        "real_inference": bool(completed),
        "model": manifest_entry.repository_id,
        "model_ref": selected.model_ref,
        "revision": selected.model_revision,
        "adapter": selected.adapter,
        "case_count": len(cases),
        "image_count": len(cases),
        "fixtures": fixtures,
        "cases": case_results,
        "reports": reports,
        "display_sections": display_sections,
        "output_sections": {
            filename: sorted(sections)
            for filename, sections in display_sections.items()
            if isinstance(sections, dict)
        },
        "provenance_by_case": {
            str(result["case_id"]): result.get("provenance")
            for result in case_results
            if "provenance" in result
        },
        "provenance": completed[0].get("provenance") if completed else {},
        "contract_hash": validation_contract_hash(manifest_entry),
        "checks": technical_checks,
        "resource": resource,
        "validation_cache_root": str(validation_cache) if validation_cache else None,
        "repeat_case": repeat_result,
        "quality_review": {
            "status": "pending",
            "scope": "manual conservative case-responsiveness review",
        },
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "weights_downloaded": False,
        "manifest_promoted": False,
        "api_result": {
            "count": len(reports),
            "reports": reports,
            "reports_ordered": list(reports.values()),
            "report_filenames": list(reports),
            "display_sections": display_sections,
            "inference_metadata": [
                metadata
                for result in case_results
                if isinstance(result.get("input_metadata"), list)
                for metadata in cast(list[object], result["input_metadata"])
                if isinstance(metadata, dict)
            ],
        },
    }
    payload["fixture_provenance"] = fixture_provenance
    payload["fixture_deidentification"] = fixture_deidentification
    if write_receipt:
        if receipt_path is None:
            RECEIPT_DIR.mkdir(parents=True, exist_ok=True)
        receipt = (
            receipt_path
            if receipt_path is not None
            else RECEIPT_DIR / f"{_slug(model_ref)}-{selected.model_revision}.json"
        ).resolve()
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        try:
            payload["receipt"] = str(receipt.relative_to(ROOT_DIR))
        except ValueError:
            payload["receipt"] = str(receipt)
    return payload

###############################################################################
def main() -> int:
    args = _arguments()
    try:
        if args.case_manifest is not None:
            if args.image or args.fixture_sha256:
                raise ValueError(
                    "--case-manifest cannot be combined with --image or --fixture-sha256"
                )
            cases = load_case_manifest(args.case_manifest)
            payload = validate_cached_cases(
                model_ref=args.model_ref,
                cases=cases,
                fixture_provenance=args.fixture_provenance,
                fixture_deidentification=args.fixture_deidentification,
                repeat_case_id=cases[0].case_id,
                receipt_path=args.receipt_path,
            )
        else:
            if not args.image or not args.fixture_sha256:
                raise ValueError(
                    "Provide --case-manifest or at least one --image and --fixture-sha256"
                )
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

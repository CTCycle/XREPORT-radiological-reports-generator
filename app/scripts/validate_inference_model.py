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

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from server.common.path import ROOT_DIR  # noqa: E402
from server.configurations.startup import get_server_settings  # noqa: E402
from server.domain.inference import InferenceImage, InferenceManifest  # noqa: E402
from server.domain.jobs import JobState  # noqa: E402
from server.models.inference.providers.huggingface import HuggingFaceProvider  # noqa: E402
from server.services.jobs import JobManager  # noqa: E402
from server.services.inference_catalog import (  # noqa: E402
    CATALOG_PATH,
    InferenceModelCatalog,
    validation_contract_hash,
)
from server.services.inference import (  # noqa: E402
    InferenceImageStore,
    run_inference_job,
)
import server.services.inference as inference_service  # noqa: E402


RUN_LOG_DIR = ROOT_DIR / "assets" / "QA" / "inference_validation_runs"
RECEIPT_DIR = ROOT_DIR / "assets" / "QA" / "inference_validation"

###############################################################################
def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-ref", required=True, help="Configured model reference to validate")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument(
        "--profile", choices=("deterministic", "concise", "detailed"), default="deterministic"
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
        default="",
        help="Expected SHA-256 hash of the supplied bytes; required for real validation.",
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
        raise SystemExit("Fixture provenance must identify a public source or dataset accession.")
    if not fixture_deidentification:
        raise SystemExit("Fixture de-identification provenance must be stated explicitly.")
    if expected_sha256.strip().lower() != actual_sha256:
        raise SystemExit(
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
def main() -> int:
    args = _arguments()
    settings = get_server_settings().inference
    catalog = InferenceModelCatalog(settings).list_models()
    selected = next((model for model in catalog.models if model.model_ref == args.model_ref), None)
    if selected is None:
        payload = {"status": "deferred", "reason": f"Model is not configured: {args.model_ref}"}
        print(json.dumps(payload, indent=2))
        return 2
    if selected.provider != "huggingface" or selected.status not in {"ready", "unvalidated"}:
        payload = {
            "status": "deferred",
            "model_ref": selected.model_ref,
            "revision": selected.model_revision,
            "catalog_status": selected.status,
            "reason": selected.status_message or "The cache-only validator supports Hugging Face catalogue entries only.",
            "weights_downloaded": False,
        }
        path = _write_run_log(args.model_ref, payload)
        print(json.dumps({**payload, "log": str(path.relative_to(ROOT_DIR))}, indent=2))
        return 2
    manifest = selected.model_dump(mode="json")
    manifest["revision"] = selected.model_revision
    configured_manifest = InferenceManifest.model_validate(
        json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    )
    manifest_entry = next(
        (entry for entry in configured_manifest.models if entry.model_ref == selected.model_ref),
        None,
    )
    if manifest_entry is None:
        payload = {
            "status": "deferred",
            "model_ref": selected.model_ref,
            "revision": selected.model_revision,
            "catalog_status": selected.status,
            "reason": "The selected Hugging Face catalogue entry is absent from the configured manifest.",
            "weights_downloaded": False,
        }
        path = _write_run_log(args.model_ref, payload)
        print(json.dumps({**payload, "log": str(path.relative_to(ROOT_DIR))}, indent=2))
        return 2
    if not args.image.is_file():
        raise SystemExit(f"Fixture does not exist: {args.image}")

    data = args.image.read_bytes()
    fixture = _fixture_metadata(
        args.image,
        data,
        provenance=args.fixture_provenance,
        deidentification=args.fixture_deidentification,
        expected_sha256=args.fixture_sha256,
    )
    content_type = mimetypes.guess_type(args.image.name)[0] or "application/octet-stream"
    image = InferenceImage(
        filename=args.image.name,
        content_type=content_type,
        data=data,
        size_bytes=len(data),
    )

    ###############################################################################
    class RecordingRepository:
        saved_reports: list[dict[str, str]] = []
        generation_config: dict[str, object] = {}

        # -------------------------------------------------------------------------
        def save_generated_reports(self, reports: list[dict[str, str]], **kwargs: object) -> None:
            self.saved_reports = list(reports)
            generation_config = kwargs.get("generation_config")
            self.generation_config = (
                dict(generation_config) if isinstance(generation_config, dict) else {}
            )

    job_manager = JobManager()
    image_store = InferenceImageStore()
    provider = HuggingFaceProvider(settings)
    recorder = RecordingRepository()
    request_id = "validation_fixture"
    image_store.store(request_id, [image])
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
    try:
        result = run_inference_job(
            model_ref=args.model_ref,
            model_revision=selected.model_revision,
            model_manifest=manifest,
            generation_profile=args.profile,
            clinical_context=args.clinical_context,
            request_id=request_id,
            job_id=job_id,
        )
        previous_result = job_manager.jobs[job_id].result or {}
        job_manager.jobs[job_id].update(
            status="completed",
            result={**previous_result, **result},
        )
    finally:
        inference_service.get_job_manager = original_get_manager
        inference_service.get_inference_image_store = original_get_store
        inference_service.get_huggingface_provider = original_get_provider
        inference_service.InferenceRepository = original_repository

    job_status = job_manager.get_job_status(job_id) or {}
    if job_status.get("status") != "completed":
        raise RuntimeError(f"Validation job did not complete: {job_status.get('error', job_status.get('status'))}")
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
            raise RuntimeError(f"Output sections do not match the declared contract for {filename}")
        if any(not isinstance(value, str) or not value.strip() for value in sections.values()):
            raise RuntimeError(f"Output sections contain an empty value for {filename}")
        if "raw_report" in declared_sections and sections["raw_report"] != report:
            raise RuntimeError("Raw report text changed before the display contract")
    if recorder.saved_reports != [
        {"image": filename, "report": report}
        for filename, report in reports.items()
    ]:
        raise RuntimeError("Raw report text changed at the persistence boundary")
    if not {
        "display_sections",
        "provenance",
    }.issubset(recorder.generation_config):
        raise RuntimeError("Persistence metadata omitted display sections or provenance")

    payload = {
        "status": "passed",
        "real_inference": True,
        "model_ref": selected.model_ref,
        "revision": selected.model_revision,
        "contract_hash": validation_contract_hash(manifest_entry),
        "fixture": fixture,
        "reports": reports,
        "display_sections": display_sections,
        "provenance": provenance,
        "api_result": api_result,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "weights_downloaded": False,
        "manifest_promoted": False,
    }
    RECEIPT_DIR.mkdir(parents=True, exist_ok=True)
    receipt = RECEIPT_DIR / f"{_slug(args.model_ref)}-{selected.model_revision}.json"
    receipt.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({**payload, "receipt": str(receipt.relative_to(ROOT_DIR))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

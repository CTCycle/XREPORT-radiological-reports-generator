"""Validate every locally runnable public inference model sequentially.

The command is cache-only. Models that are not installed, gated without local
access, or unavailable for the current runtime are recorded as deferred rather
than being counted as successful validation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from server.common.path import ROOT_DIR  # noqa: E402
from server.configurations.inference_models import (  # noqa: E402
    embedded_inference_models,
)
from server.configurations.startup import get_server_settings  # noqa: E402
from server.services.inference_catalog import InferenceModelCatalog  # noqa: E402
from scripts.validate_inference_model import validate_cached_model  # noqa: E402


SUMMARY_DIR = ROOT_DIR / "assets" / "QA" / "inference_validation_runs"

###############################################################################
def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, action="append", required=True)
    parser.add_argument("--profile", choices=("deterministic", "concise", "detailed"), default="deterministic")
    parser.add_argument("--clinical-context", default="")
    parser.add_argument("--fixture-provenance", required=True)
    parser.add_argument("--fixture-deidentification", required=True)
    parser.add_argument("--fixture-sha256", type=str, action="append", required=True)
    return parser.parse_args()

###############################################################################
def _deferred_state(model: object) -> str:
    if bool(getattr(model, "gated", False)):
        return "deferred_access_required"
    if getattr(model, "status", None) in {"not_installed", "downloading", "staged"}:
        return "deferred_not_installed"
    return "deferred_resource_unavailable"

###############################################################################
def _summary_path() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return SUMMARY_DIR / f"public-inference-models-{stamp}.json"

###############################################################################
def main() -> int:
    args = _arguments()
    settings = get_server_settings().inference
    catalog = InferenceModelCatalog(settings).list_models()
    configured_entries = embedded_inference_models()
    models: list[dict[str, object]] = []
    failed = False
    deferred = False

    for entry in configured_entries:
        selected = next(
            (model for model in catalog.models if model.model_ref == entry.model_ref),
            None,
        )
        if selected is None:
            failed = True
            models.append(
                {
                    "model": entry.repository_id,
                    "model_ref": entry.model_ref,
                    "revision": entry.revision,
                    "adapter": entry.adapter,
                    "state": "failed",
                    "error": "The embedded model is missing from the runtime catalogue.",
                }
            )
            continue

        if selected.status not in {"ready", "unvalidated"}:
            deferred = True
            models.append(
                {
                    "model": entry.repository_id,
                    "model_ref": entry.model_ref,
                    "revision": entry.revision,
                    "adapter": entry.adapter,
                    "state": _deferred_state(selected),
                    "catalog_status": selected.status,
                    "reason": selected.status_message,
                }
            )
            continue

        try:
            result = validate_cached_model(
                model_ref=entry.model_ref,
                image_paths=list(args.image),
                profile=args.profile,
                clinical_context=args.clinical_context,
                fixture_provenance=args.fixture_provenance,
                fixture_deidentification=args.fixture_deidentification,
                fixture_sha256=list(args.fixture_sha256),
            )
        except Exception as exc:  # noqa: BLE001
            failed = True
            models.append(
                {
                    "model": entry.repository_id,
                    "model_ref": entry.model_ref,
                    "revision": entry.revision,
                    "adapter": entry.adapter,
                    "state": "failed",
                    "error": str(exc),
                }
            )
            continue

        if result.get("status") == "passed":
            models.append(
                {
                    "model": entry.repository_id,
                    "model_ref": entry.model_ref,
                    "revision": result.get("revision", entry.revision),
                    "adapter": result.get("adapter", entry.adapter),
                    "state": "passed",
                    "profile": result.get("generation_profile"),
                    "requested_device": result.get("requested_device"),
                    "resolved_devices": result.get("resolved_runtime_devices"),
                    "dtype": result.get("dtype"),
                    "cuda_available": result.get("cuda_available"),
                    "cuda_used": result.get("cuda_used"),
                    "image_count": result.get("image_count"),
                    "input_tensor_dimensions": result.get("input_tensor_dimensions"),
                    "load_seconds": result.get("load_seconds"),
                    "generation_seconds": result.get("generation_seconds"),
                    "total_seconds": result.get("total_seconds"),
                    "peak_cuda_memory_bytes": result.get("peak_cuda_memory_bytes"),
                    "output_sections": result.get("output_sections"),
                    "report_character_counts": result.get("report_character_counts"),
                    "receipt": result.get("receipt"),
                }
            )
        else:
            failed = True
            models.append(
                {
                    "model": entry.repository_id,
                    "model_ref": entry.model_ref,
                    "revision": entry.revision,
                    "adapter": entry.adapter,
                    "state": "failed",
                    "error": result.get("reason", "Validation did not pass."),
                }
            )

    payload = {
        "status": "failed" if failed else "deferred" if deferred else "passed",
        "real_inference": any(model["state"] == "passed" for model in models),
        "model_count": len(models),
        "models": models,
        "fixture_count": len(args.image),
        "captured_at": datetime.now(timezone.utc).isoformat(),
    }
    path = _summary_path()
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({**payload, "summary": str(path.relative_to(ROOT_DIR))}, indent=2))
    if failed:
        return 1
    return 2 if deferred else 0


if __name__ == "__main__":
    raise SystemExit(main())

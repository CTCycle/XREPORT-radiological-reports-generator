"""Run a cache-only CXRMate-ED image/context/profile sensitivity canary."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from server.common.path import ROOT_DIR  # noqa: E402
from server.configurations.startup import get_server_settings  # noqa: E402
from server.domain.inference import InferenceImage  # noqa: E402
from server.models.inference.providers.huggingface import HuggingFaceProvider  # noqa: E402
from server.services.inference_catalog import (  # noqa: E402
    CATALOG_PATH,
    InferenceModelCatalog,
)
from server.services.inference_runtime import InferenceRuntimeCoordinator  # noqa: E402
from server.services.model_installation import ModelInstallationManager  # noqa: E402


RUN_LOG_DIR = ROOT_DIR / "assets" / "QA" / "inference_validation_runs"
DEFAULT_IMAGE_DIR = (
    ROOT_DIR / "assets" / "QA" / "full-e2e-20260821-101310" / "inputs" / "images"
)
CASES = (
    ("qa_pa.png", "cough", "detailed"),
    ("qa_lateral.png", "dyspnea", "concise"),
    ("qa_normal.png", "screening", "deterministic"),
)


###############################################################################
def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument(
        "--fixture-provenance",
        required=True,
        help="Public/de-identified source or QA fixture provenance statement.",
    )
    parser.add_argument(
        "--fixture-deidentification",
        required=True,
        help="Explicit de-identification statement for the supplied fixtures.",
    )
    return parser.parse_args()


###############################################################################
def _write_log(payload: dict[str, Any]) -> Path:
    RUN_LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = RUN_LOG_DIR / f"cxrmate-ed-sensitivity-{stamp}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


###############################################################################
def main() -> int:
    args = _arguments()
    settings = get_server_settings().inference
    catalog = InferenceModelCatalog(settings).list_models()
    model_ref = "huggingface:aehrc/cxrmate-ed"
    selected = next(
        (model for model in catalog.models if model.model_ref == model_ref), None
    )
    if selected is None:
        raise SystemExit(f"Model is not configured: {model_ref}")
    if selected.provider != "huggingface" or selected.adapter != "cxrmate_ed":
        raise SystemExit(f"Unexpected CXRMate-ED catalog adapter: {selected.adapter}")
    if selected.status not in {"ready", "unvalidated"}:
        payload = {
            "status": "deferred",
            "model_ref": model_ref,
            "catalog_status": selected.status,
            "reason": selected.status_message or "The cached model is not runnable.",
            "captured_at": datetime.now(timezone.utc).isoformat(),
        }
        path = _write_log(payload)
        print(json.dumps({**payload, "log": str(path.relative_to(ROOT_DIR))}, indent=2))
        return 2

    manifest_payload = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    manifest_entry = next(
        entry for entry in manifest_payload["models"] if entry["model_ref"] == model_ref
    )
    manifest = selected.model_dump(mode="json")
    manifest.update(
        {
            "repository_id": manifest_entry["repository_id"],
            "revision": selected.model_revision,
        }
    )
    provider = HuggingFaceProvider(settings)
    runtime = InferenceRuntimeCoordinator(
        huggingface_provider=provider,
        installation_manager=ModelInstallationManager(),
    )
    cases: list[dict[str, Any]] = []
    reports: list[str] = []
    errors: list[str] = []
    for filename, context, profile in CASES:
        image_path = (args.image_dir / filename).resolve()
        case: dict[str, Any] = {
            "filename": filename,
            "clinical_context": context,
            "generation_profile": profile,
        }
        try:
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            data = image_path.read_bytes()
            case["sha256"] = hashlib.sha256(data).hexdigest()
            result = runtime.generate(
                model_ref=model_ref,
                model_revision=selected.model_revision,
                model_manifest=manifest,
                generation_profile=profile,
                clinical_context=context,
                images=[
                    InferenceImage(
                        filename=filename,
                        content_type="image/png",
                        data=data,
                        size_bytes=len(data),
                    )
                ],
                should_stop=lambda: False,
                report_progress=lambda *_args: None,
                report_lifecycle=lambda _payload: None,
            )
            report = result.reports.get(filename)
            if not isinstance(report, str) or not report.strip():
                raise RuntimeError("The model returned no non-empty report")
            case["report"] = report
            case["provenance"] = result.provenance
            case["input_metadata"] = result.metadata
            reports.append(report)
        except Exception as exc:  # noqa: BLE001
            case["error"] = str(exc)
            errors.append(f"{filename}: {exc}")
        cases.append(case)

    completed = [case for case in cases if "report" in case]
    input_contract_ok = all(
        case.get("provenance", {}).get("generation_profile")
        == case["generation_profile"]
        and case.get("provenance", {}).get("clinical_context")
        == case["clinical_context"]
        and case.get("input_metadata", [{}])[0].get("filename") == case["filename"]
        for case in completed
    )
    unique_reports = len(set(reports))
    sensitivity_passed = (
        not errors
        and len(completed) == len(CASES)
        and input_contract_ok
        and unique_reports == len(CASES)
    )
    payload = {
        "status": "passed" if sensitivity_passed else "failed",
        "real_inference": True,
        "model_ref": model_ref,
        "revision": selected.model_revision,
        "catalog_validation_status": selected.validation_status,
        "fixture_provenance": args.fixture_provenance.strip(),
        "fixture_deidentification": args.fixture_deidentification.strip(),
        "cases": cases,
        "checks": {
            "completed_cases": len(completed),
            "expected_cases": len(CASES),
            "unique_report_count": unique_reports,
            "input_contract_ok": input_contract_ok,
            "reports_all_distinct": unique_reports == len(CASES),
            "errors": errors,
        },
        "captured_at": datetime.now(timezone.utc).isoformat(),
    }
    path = _write_log(payload)
    print(json.dumps({**payload, "log": str(path.relative_to(ROOT_DIR))}, indent=2))
    return 0 if sensitivity_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Run a cache-only CXRMate-ED image/context/profile sensitivity canary."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from server.common.path import ROOT_DIR  # noqa: E402
from scripts.validate_inference_model import (  # noqa: E402
    ValidationCase,
    validate_cached_cases,
)


RUN_LOG_DIR = ROOT_DIR / "assets" / "QA" / "inference_validation_runs"
DEFAULT_IMAGE_DIR = (
    ROOT_DIR / "assets" / "QA" / "inference_validation_runs"
)
CASES = (
    (
        "qa-pa",
        "qa_pa.png",
        "f30ed78a4c18d162dde6e5305116daa1c8cf9cc653eb032495341ba40889c8e8",
        "cough",
        "detailed",
    ),
    (
        "qa-lateral",
        "qa_lateral.png",
        "8897f8d5349f3367ecf1292c639cc27da502e619c762da5cfb13bcc9091f879a",
        "dyspnea",
        "concise",
    ),
    (
        "qa-normal",
        "qa_normal.png",
        "a0c3c331aa2c5d94a3a7cc7765f11356262a4c4700fae3e361025deb7a26b6f6",
        "screening",
        "deterministic",
    ),
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
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output path for the run receipt; defaults to the legacy QA log directory.",
    )
    return parser.parse_args()

###############################################################################
def _write_log(payload: dict[str, Any], output: Path | None = None) -> Path:
    if output is not None:
        path = output.resolve()
    else:
        RUN_LOG_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = RUN_LOG_DIR / f"cxrmate-ed-sensitivity-{stamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path

###############################################################################
def main() -> int:
    args = _arguments()
    model_ref = "huggingface:aehrc/cxrmate-ed"
    cases = [
        ValidationCase(
            case_id=case_id,
            image_path=(args.image_dir / filename).resolve(),
            profile=profile,  # type: ignore[arg-type]
            clinical_context=context,
            expected_sha256=sha256,
        )
        for case_id, filename, sha256, context, profile in CASES
    ]
    payload = validate_cached_cases(
        model_ref=model_ref,
        cases=cases,
        fixture_provenance=args.fixture_provenance,
        fixture_deidentification=args.fixture_deidentification,
        require_distinct_reports=True,
        repeat_case_id=None,
        write_receipt=False,
    )
    provenance = payload.get("provenance")
    payload.update(
        {
            "catalog_validation_status": (
                provenance.get("validation_status")
                if isinstance(provenance, dict)
                else None
            ),
            "fixture_provenance": args.fixture_provenance.strip(),
            "fixture_deidentification": args.fixture_deidentification.strip(),
        }
    )
    path = _write_log(payload, args.output)
    try:
        log_path = str(path.relative_to(ROOT_DIR))
    except ValueError:
        log_path = str(path)
    print(json.dumps({**payload, "log": log_path}, indent=2))
    if payload.get("status") == "passed":
        return 0
    if payload.get("status") == "deferred":
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

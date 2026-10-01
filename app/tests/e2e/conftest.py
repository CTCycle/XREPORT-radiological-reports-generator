"""Disposable persisted inference history fixtures for API and UI E2E tests."""

from datetime import datetime, timezone
import uuid

import pytest

###############################################################################
@pytest.fixture
def seeded_history() -> str:
    from server.repositories.serialization.inference import InferenceRepository

    repository = InferenceRepository()
    request_id = f"e2e_history_{uuid.uuid4().hex}"
    repository.save_generated_reports(
        [
            {"image": "e2e-first.png", "report": "Findings\nClear lungs.\n\nImpression\nNo acute disease."},
            {"image": "e2e-second.png", "report": "Findings\nMild bibasal opacity.\n\nImpression\nFollow-up recommended."},
        ],
        provider="huggingface",
        model_ref="huggingface:e2e/reports",
        model_revision="e2e-revision",
        generation_profile="deterministic",
        generation_config={
            "profile": "deterministic",
            "display_sections": {
                "e2e-first.png": {
                    "findings": "Clear lungs.",
                    "impression": "No acute disease.",
                },
                "e2e-second.png": {
                    "findings": "Mild bibasal opacity.",
                    "impression": "Follow-up recommended.",
                },
            },
            "provenance": {"source": "e2e-test"},
        },
        clinical_context="E2E fixture",
        request_id=request_id,
        status="succeeded",
        execution_time_seconds=0.4,
        executed_at=datetime(2026, 9, 22, 12, 0, tzinfo=timezone.utc),
    )
    try:
        yield request_id
    finally:
        repository.delete_inference_history(request_id)

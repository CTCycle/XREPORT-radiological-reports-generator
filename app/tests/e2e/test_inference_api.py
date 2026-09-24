"""Cross-layer contract check for the catalog-backed inference API."""

from contextlib import ExitStack
from pathlib import Path
from typing import Any

import httpx
from playwright.sync_api import APIRequestContext


###############################################################################
def test_inference_catalog_is_reachable_and_unknown_models_are_rejected(
    api_context: APIRequestContext,
) -> None:
    catalog_response = api_context.get("/api/inference/models")
    assert catalog_response.ok
    catalog = catalog_response.json()
    public_models = [
        model for model in catalog["models"] if model["origin"] == "public"
    ]
    custom_models = [
        model for model in catalog["models"] if model["origin"] == "custom"
    ]
    assert len(public_models) == 5
    assert all(model["provider"] == "huggingface" for model in public_models)
    assert all(model["model_ref"].startswith("huggingface:") for model in public_models)
    assert all(model["status"] for model in public_models)
    assert all(model["installation_state"] for model in public_models)
    assert all(model["integrity_status"] for model in public_models)
    assert all(model["validation_status"] for model in public_models)
    assert all(model["validation_receipt_status"] for model in public_models)
    assert all(model["model_revision"] for model in public_models)
    assert all(model["output_sections"] for model in public_models)
    assert all(model["model_ref"].startswith("xreport:") for model in custom_models)
    expected_revisions = {
        "huggingface:aehrc/cxrmate-multi-tf": "330721b9aa5bba201a3eb88eba4dd9a6607f3e7a",
        "huggingface:aehrc/cxrmate-ed": "68251c7605067ddbea330413aade032713fd2192",
        "huggingface:StanfordAIMI/CheXOne": "0c350e6852ea08f9d9baf3b7595c1a10d4849927",
        "huggingface:aehrc/cxrmate-2": "aa8e2d16470e20671acf049687b4707c9bf2f2b5",
        "huggingface:google/medgemma-1.5-4b-it": "91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b",
    }
    assert {
        model["model_ref"]: model["model_revision"] for model in public_models
    } == expected_revisions
    cxrmate_ed = next(
        model
        for model in public_models
        if model["model_ref"] == "huggingface:aehrc/cxrmate-ed"
    )
    assert cxrmate_ed["validation_status"] == "degraded"
    assert "sensitivity canary failed" in cxrmate_ed["validation_message"].lower()
    medgemma = next(
        model
        for model in public_models
        if model["model_ref"] == "huggingface:google/medgemma-1.5-4b-it"
    )
    assert medgemma["access_policy"] == "gated"
    assert medgemma["access_url"]
    assert set(catalog["providers"]) == {"huggingface", "xreport"}

    response = api_context.post(
        "/api/inference/models/maintenance",
        data={
            "model_ref": "xreport:not-in-the-catalog",
            "action": "delete_local",
        },
    )

    assert response.status == 404
    assert "catalog" in response.json()["detail"]


###############################################################################
def _image_upload(
    filename: str = "scan.png",
    content_type: str = "image/png",
    data: bytes = b"image-fixture",
) -> dict[str, Any]:
    return {"name": filename, "mimeType": content_type, "buffer": data}


###############################################################################
def test_inference_generate_enforces_model_context_profile_and_image_limits(
    api_context: APIRequestContext,
    api_base_url: str,
) -> None:
    catalog = api_context.get("/api/inference/models").json()
    cxrmate_multi = next(
        model
        for model in catalog["models"]
        if model["model_ref"] == "huggingface:aehrc/cxrmate-multi-tf"
    )
    maximum_images = cxrmate_multi["max_current_images"]

    for profile in ("deterministic", "concise", "detailed"):
        response = api_context.post(
            "/api/inference/generate",
            multipart={
                "model_ref": cxrmate_multi["model_ref"],
                "generation_profile": profile,
                "clinical_context": "Follow-up for cough",
                "images": _image_upload(),
            },
        )
        assert response.status == 400
        assert response.json() == {
            "detail": "Selected model does not support clinical context"
        }

        supported_profile = api_context.post(
            "/api/inference/generate",
            multipart={
                "model_ref": cxrmate_multi["model_ref"],
                "generation_profile": profile,
                "clinical_context": "",
                "images": _image_upload("scan.gif", "image/gif", b"GIF89a"),
            },
        )
        assert supported_profile.status == 400
        assert supported_profile.json() == {
            "detail": "Unsupported image type: image/gif"
        }

    with httpx.Client(base_url=api_base_url) as client:
        over_limit = client.post(
            "/api/inference/generate",
            data={
                "model_ref": cxrmate_multi["model_ref"],
                "generation_profile": "deterministic",
                "clinical_context": "",
            },
            files=[
                (
                    "images",
                    (f"scan-{index}.png", b"image-fixture", "image/png"),
                )
                for index in range(maximum_images + 1)
            ],
        )
    assert over_limit.status_code == 400, over_limit.json()
    assert over_limit.json() == {
        "detail": f"Selected model accepts at most {maximum_images} current image(s)"
    }

    invalid_type = api_context.post(
        "/api/inference/generate",
        multipart={
            "model_ref": "huggingface:aehrc/cxrmate-ed",
            "generation_profile": "concise",
            "clinical_context": "Cough",
            "images": _image_upload("scan.gif", "image/gif", b"GIF89a"),
        },
    )
    assert invalid_type.status == 400
    assert invalid_type.json() == {"detail": "Unsupported image type: image/gif"}

    empty_image = api_context.post(
        "/api/inference/generate",
        multipart={
            "model_ref": "huggingface:aehrc/cxrmate-ed",
            "generation_profile": "detailed",
            "clinical_context": "Dyspnea",
            "images": _image_upload(data=b""),
        },
    )
    assert empty_image.status == 400
    assert empty_image.json() == {"detail": "Empty image payload: scan.png"}

    invalid_profile = api_context.post(
        "/api/inference/generate",
        multipart={
            "model_ref": cxrmate_multi["model_ref"],
            "generation_profile": "unrecognized",
            "clinical_context": "",
            "images": _image_upload(),
        },
    )
    assert invalid_profile.status == 422
    assert set(invalid_profile.json()) == {"detail"}


###############################################################################
def test_inference_generate_rejects_image_payload_over_total_limit(
    api_base_url: str,
    tmp_path: Path,
) -> None:
    image_paths = [tmp_path / "scan-1.png", tmp_path / "scan-2.png"]
    for image_path, size in zip(
        image_paths,
        (32 * 1024 * 1024, 32 * 1024 * 1024 + 1),
        strict=True,
    ):
        with image_path.open("wb") as image:
            image.truncate(size)

    with ExitStack() as stack:
        files = [
            (
                "images",
                (
                    image_path.name,
                    stack.enter_context(image_path.open("rb")),
                    "image/png",
                ),
            )
            for image_path in image_paths
        ]
        client = stack.enter_context(httpx.Client(base_url=api_base_url, timeout=120))
        response = client.post(
            "/api/inference/generate",
            data={
                "model_ref": "huggingface:aehrc/cxrmate-ed",
                "generation_profile": "deterministic",
                "clinical_context": "",
            },
            files=files,
        )

    assert response.status_code == 413
    assert response.json() == {"detail": "Total image payload exceeds 64 MB limit"}


###############################################################################
def test_inference_catalog_exposes_chexone_findings_only_contract(
    api_context: APIRequestContext,
) -> None:
    response = api_context.get("/api/inference/models")

    assert response.ok
    chexone = next(
        model
        for model in response.json()["models"]
        if model["model_ref"] == "huggingface:StanfordAIMI/CheXOne"
    )

    assert chexone["output_sections"] == ["findings"]
    assert chexone["capabilities"]["findings"] is True
    assert chexone["capabilities"]["impression"] is False


###############################################################################
def test_inference_history_contract_filters_and_missing_entries(
    api_context: APIRequestContext,
) -> None:
    response = api_context.get(
        "/api/inference/history",
        params={"limit": 1, "offset": 0, "sort": "oldest"},
    )

    assert response.ok
    payload = response.json()
    assert set(payload) == {"items", "total", "limit", "offset"}
    assert payload["limit"] == 1
    assert payload["offset"] == 0
    assert isinstance(payload["items"], list)

    invalid_limit = api_context.get("/api/inference/history", params={"limit": 0})
    assert invalid_limit.status == 400
    invalid_status = api_context.get(
        "/api/inference/history", params={"status": "not-a-status"}
    )
    assert invalid_status.status == 400

    missing = "history-entry-that-does-not-exist"
    assert api_context.get(f"/api/inference/history/{missing}").status == 404
    assert (
        api_context.patch(
            f"/api/inference/history/{missing}",
            data={"reports": [{"image_index": 0, "edited_report": "draft"}]},
        ).status
        == 404
    )
    assert api_context.delete(f"/api/inference/history/{missing}").status == 404


###############################################################################
def test_inference_history_crud_preserves_original_output(
    api_context: APIRequestContext,
    seeded_history: str,
) -> None:
    listing = api_context.get(
        "/api/inference/history",
        params={
            "model_ref": "huggingface:e2e/reports",
            "status": "succeeded",
            "limit": 10,
        },
    )
    assert listing.ok
    item = next(
        item for item in listing.json()["items"] if item["request_id"] == seeded_history
    )
    assert item["report_count"] == 2
    assert item["reports"][0]["input_image_name"] == "e2e-first.png"
    assert item["provenance_available"] is True

    detail = api_context.get(f"/api/inference/history/{seeded_history}")
    assert detail.ok
    original = detail.json()
    assert [report["image_index"] for report in original["reports"]] == [0, 1]
    assert original["reports"][0]["generated_report"].startswith("Findings")
    assert original["reports"][0]["edited_report"] is None

    duplicate_indexes = api_context.patch(
        f"/api/inference/history/{seeded_history}",
        data={
            "reports": [
                {"image_index": 0, "edited_report": "one"},
                {"image_index": 0, "edited_report": "two"},
            ]
        },
    )
    assert duplicate_indexes.status == 400

    updated = api_context.patch(
        f"/api/inference/history/{seeded_history}",
        data={
            "reports": [
                {
                    "image_index": 1,
                    "edited_report": "Impression\nEdited follow-up recommendation.",
                }
            ]
        },
    )
    assert updated.ok
    updated_report = next(
        report for report in updated.json()["reports"] if report["image_index"] == 1
    )
    assert updated_report["generated_report"].endswith("Follow-up recommended.")
    assert (
        updated_report["edited_report"]
        == "Impression\nEdited follow-up recommendation."
    )
    assert updated_report["effective_report"] == updated_report["edited_report"]
    assert updated_report["edited_at"]

    cleared = api_context.patch(
        f"/api/inference/history/{seeded_history}",
        data={
            "reports": [
                {
                    "image_index": 1,
                    "edited_report": original["reports"][1]["generated_report"],
                }
            ]
        },
    )
    assert cleared.ok
    cleared_report = next(
        report for report in cleared.json()["reports"] if report["image_index"] == 1
    )
    assert cleared_report["edited_report"] is None
    assert cleared_report["edited_at"] is None

    deleted = api_context.delete(f"/api/inference/history/{seeded_history}")
    assert deleted.ok
    assert deleted.json()["success"] is True
    assert api_context.get(f"/api/inference/history/{seeded_history}").status == 404

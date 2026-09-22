"""Cross-layer contract check for the catalog-backed inference API."""

from playwright.sync_api import APIRequestContext

###############################################################################
def test_inference_catalog_is_reachable_and_unknown_models_are_rejected(
    api_context: APIRequestContext,
) -> None:
    catalog_response = api_context.get("/api/inference/models")
    assert catalog_response.ok
    catalog = catalog_response.json()
    assert catalog["models"]
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
    assert api_context.patch(
        f"/api/inference/history/{missing}",
        data={"reports": [{"image_index": 0, "edited_report": "draft"}]},
    ).status == 404
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
    assert updated_report["edited_report"] == "Impression\nEdited follow-up recommendation."
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

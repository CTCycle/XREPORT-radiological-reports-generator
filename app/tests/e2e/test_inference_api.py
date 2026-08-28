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

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

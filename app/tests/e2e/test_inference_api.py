"""E2E tests for the catalog-backed inference API."""

from playwright.sync_api import APIRequestContext


def _minimal_png() -> bytes:
    return bytes.fromhex(
        "89504e470d0a1a0a0000000d4948445200000001000000010802000000907753de"
        "0000000c4944415408d763f8ffff3f0005fe02fedccc59e70000000049454e44ae426082"
    )


class TestInferenceEndpoints:
    def test_get_models_returns_catalog(self, api_context: APIRequestContext) -> None:
        response = api_context.get("/api/inference/models")

        assert response.ok, f"Expected 200, got {response.status}"
        payload = response.json()
        assert isinstance(payload["models"], list)
        assert isinstance(payload["providers"], dict)
        for model in payload["models"]:
            assert model["model_ref"]
            assert model["status"]
            assert model["research_only"] is True

    def test_legacy_checkpoints_endpoint_is_removed(
        self, api_context: APIRequestContext
    ) -> None:
        response = api_context.get("/api/inference/checkpoints")

        assert response.status == 404

    def test_generate_requires_catalog_request_fields(
        self, api_context: APIRequestContext
    ) -> None:
        response = api_context.post(
            "/api/inference/generate",
            multipart={
                "images": {
                    "name": "test.png",
                    "mimeType": "image/png",
                    "buffer": _minimal_png(),
                },
            },
        )

        assert response.status == 422

    def test_generate_rejects_legacy_request_fields(
        self, api_context: APIRequestContext
    ) -> None:
        response = api_context.post(
            "/api/inference/generate",
            multipart={
                "checkpoint": "test_checkpoint",
                "generation_mode": "greedy_search",
                "images": {
                    "name": "test.png",
                    "mimeType": "image/png",
                    "buffer": _minimal_png(),
                },
            },
        )

        assert response.status == 422

    def test_generate_rejects_unknown_model_ref(
        self, api_context: APIRequestContext
    ) -> None:
        response = api_context.post(
            "/api/inference/generate",
            multipart={
                "model_ref": "xreport:not-in-the-catalog",
                "generation_profile": "deterministic",
                "clinical_context": "",
                "images": {
                    "name": "test.png",
                    "mimeType": "image/png",
                    "buffer": _minimal_png(),
                },
            },
        )

        assert response.status == 404
        assert "catalog" in response.json()["detail"]

    def test_generate_rejects_invalid_generation_profile(
        self, api_context: APIRequestContext
    ) -> None:
        response = api_context.post(
            "/api/inference/generate",
            multipart={
                "model_ref": "xreport:not-in-the-catalog",
                "generation_profile": "legacy-mode",
                "clinical_context": "",
                "images": {
                    "name": "test.png",
                    "mimeType": "image/png",
                    "buffer": _minimal_png(),
                },
            },
        )

        assert response.status == 422

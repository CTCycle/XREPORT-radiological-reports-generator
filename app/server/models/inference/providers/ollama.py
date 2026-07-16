from __future__ import annotations

import base64
from collections.abc import Callable

import httpx

from server.configurations import InferenceSettings
from server.domain.inference import GenerationProfile, InferenceImage


###############################################################################
class OllamaProvider:
    """Local Ollama discovery and generation; never pulls models."""

    # -------------------------------------------------------------------------
    def __init__(self, settings: InferenceSettings) -> None:
        self.settings = settings

    # -------------------------------------------------------------------------
    def installed_models(self) -> set[str] | None:
        try:
            response = httpx.get(
                f"{self.settings.ollama_base_url.rstrip('/')}/api/tags",
                timeout=httpx.Timeout(5.0),
            )
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError):
            return None
        models = payload.get("models", []) if isinstance(payload, dict) else []
        return {str(item.get("name")) for item in models if isinstance(item, dict) and item.get("name")}

    # -------------------------------------------------------------------------
    def generate(
        self,
        *,
        model: str,
        profile: GenerationProfile,
        clinical_context: str,
        images: list[InferenceImage],
        should_stop: Callable[[], bool],
        report_progress: Callable[[int, int, dict[str, str]], None],
    ) -> dict[str, str]:
        installed = self.installed_models()
        if installed is None:
            raise RuntimeError("Ollama runtime is unavailable at the configured local address")
        if model not in installed:
            raise RuntimeError(f"Ollama model is not installed: {model}")
        if len(images) != 1:
            raise ValueError("Selected Ollama model accepts exactly one image")
        if should_stop():
            return {}

        prompt = self._prompt(profile, clinical_context)
        payload = {
            "model": model,
            "stream": False,
            "keep_alive": self.settings.ollama_keep_alive,
            "messages": [{
                "role": "user",
                "content": prompt,
                "images": [base64.b64encode(images[0].data).decode("ascii")],
            }],
        }
        timeout = httpx.Timeout(
            connect=5.0,
            read=float(self.settings.model_timeout),
            write=30.0,
            pool=5.0,
        )
        try:
            response = httpx.post(
                f"{self.settings.ollama_base_url.rstrip('/')}/api/chat",
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
            body = response.json()
        except httpx.ConnectError as exc:
            raise RuntimeError("Ollama runtime became unavailable during generation") from exc
        except httpx.TimeoutException as exc:
            raise RuntimeError("Ollama generation timed out") from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise RuntimeError(f"Ollama generation failed: {exc}") from exc

        message = body.get("message") if isinstance(body, dict) else None
        report = message.get("content") if isinstance(message, dict) else None
        if not isinstance(report, str) or not report.strip():
            raise RuntimeError("Ollama returned an empty report")
        if should_stop():
            return {}
        reports = {images[0].filename: report.strip()}
        report_progress(1, 1, reports)
        return reports

    # -------------------------------------------------------------------------
    @staticmethod
    def _prompt(profile: GenerationProfile, clinical_context: str) -> str:
        profile_instruction = {
            "deterministic": "Use a consistent, conservative report structure.",
            "concise": "Keep the draft concise and focused on salient findings.",
            "detailed": "Provide detailed Findings and Impression sections.",
        }[profile]
        context = clinical_context.strip() or "No clinical context supplied."
        return (
            "Draft a radiology report for research use only. Do not claim clinical approval. "
            f"{profile_instruction}\nClinical context: {context}"
        )

    # -------------------------------------------------------------------------
    def unload(self) -> None:
        """Ollama owns its process-level model lifecycle."""

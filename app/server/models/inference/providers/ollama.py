from __future__ import annotations

from urllib.error import URLError
from urllib.request import urlopen
import json

from server.configurations import InferenceSettings


class OllamaProvider:
    """Read-only local Ollama discovery; it never pulls or downloads models."""

    def __init__(self, settings: InferenceSettings) -> None:
        self.settings = settings

    def installed_models(self) -> set[str] | None:
        try:
            with urlopen(f"{self.settings.ollama_base_url.rstrip('/')}/api/tags", timeout=5) as response:
                payload = json.load(response)
        except (OSError, URLError, ValueError):
            return None
        models = payload.get("models", []) if isinstance(payload, dict) else []
        return {str(item.get("name")) for item in models if isinstance(item, dict) and item.get("name")}

    def unload(self) -> None:
        """Ollama owns its process-level model lifecycle."""

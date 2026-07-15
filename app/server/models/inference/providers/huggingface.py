from __future__ import annotations

from pathlib import Path

from server.configurations import InferenceSettings


class HuggingFaceProvider:
    """Offline model-cache discovery; generation is added after service generalization."""

    def __init__(self, settings: InferenceSettings) -> None:
        self.settings = settings

    def is_cached(self, repository_id: str) -> bool:
        if not self.settings.hf_cache_dir:
            return False
        cache_path = Path(self.settings.hf_cache_dir)
        repository_path = cache_path / f"models--{repository_id.replace('/', '--')}"
        return repository_path.is_dir() and any(repository_path.rglob("config.json"))

    def unload(self) -> None:
        """No runtime is loaded while the provider performs catalog discovery."""

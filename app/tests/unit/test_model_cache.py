from __future__ import annotations

import os

from server.common import model_cache


def test_model_cache_bootstrap_overwrites_hostile_global_cache_values(monkeypatch) -> None:
    monkeypatch.setenv("HF_CACHE_DIR", r"C:\Users\Public\global-hf")
    monkeypatch.setenv("HF_HOME", r"C:\Users\Public\global-hf-home")
    monkeypatch.setenv("HF_HUB_CACHE", r"C:\Users\Public\global-hub")

    model_cache.configure_model_cache()

    assert os.environ["HF_HOME"] == str(model_cache.HUGGINGFACE_MODELS_DIR)
    assert os.environ["HF_HUB_CACHE"] == str(model_cache.HF_HUB_CACHE_DIR)
    assert os.environ["TRANSFORMERS_CACHE"] == str(model_cache.HF_HUB_CACHE_DIR)
    assert os.environ["TORCH_HOME"] == str(model_cache.TORCH_CACHE_DIR)
    assert os.environ["KERAS_HOME"] == str(model_cache.KERAS_CACHE_DIR)
    assert os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "1"
    assert "HF_CACHE_DIR" not in os.environ
    for path in (
        model_cache.HF_HUB_CACHE_DIR,
        model_cache.HF_INSTALLED_DIR,
        model_cache.HF_STAGING_DIR,
        model_cache.HF_ROLLBACK_DIR,
        model_cache.HF_METADATA_DIR,
    ):
        assert path.is_dir()

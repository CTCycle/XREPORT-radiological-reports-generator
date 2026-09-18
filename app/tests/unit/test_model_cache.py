from __future__ import annotations

import os

from server.common import model_cache

###############################################################################
def test_model_cache_bootstrap_overwrites_hostile_global_cache_values(
    monkeypatch,
) -> None:
    monkeypatch.setenv("HF_CACHE_DIR", r"C:\Users\Public\global-hf")
    monkeypatch.setenv("HF_HOME", r"C:\Users\Public\global-hf-home")
    monkeypatch.setenv("HF_HUB_CACHE", r"C:\Users\Public\global-hub")
    monkeypatch.setenv("TRANSFORMERS_CACHE", r"C:\Users\Public\external-transformers")

    model_cache.configure_model_cache()

    assert os.environ["XREPORT_CACHE_ROOT"] == str(model_cache.RUNTIME_CACHE_DIR)
    assert os.environ["XDG_CACHE_HOME"] == str(model_cache.RUNTIME_CACHE_DIR)
    assert os.environ["UV_CACHE_DIR"] == str(model_cache.UV_CACHE_DIR)
    assert os.environ["PIP_CACHE_DIR"] == str(model_cache.PIP_CACHE_DIR)
    assert os.environ["NPM_CONFIG_CACHE"] == str(model_cache.NPM_CACHE_DIR)
    assert os.environ["npm_config_cache"] == str(model_cache.NPM_CACHE_DIR)
    assert os.environ["PLAYWRIGHT_BROWSERS_PATH"] == str(
        model_cache.PLAYWRIGHT_BROWSERS_CACHE_DIR
    )
    assert os.environ["PYTEST_CACHE_DIR"] == str(model_cache.PYTEST_CACHE_DIR)
    assert os.environ["PYTEST_BASETEMP"] == str(model_cache.PYTEST_BASETEMP_DIR)
    assert os.environ["RUFF_CACHE_DIR"] == str(model_cache.RUFF_CACHE_DIR)
    assert os.environ["MYPY_CACHE_DIR"] == str(model_cache.MYPY_CACHE_DIR)
    assert os.environ["COVERAGE_FILE"] == str(
        model_cache.COVERAGE_CACHE_DIR / ".coverage"
    )
    assert os.environ["HF_HOME"] == str(model_cache.HUGGINGFACE_CACHE_DIR)
    assert os.environ["HF_HUB_CACHE"] == str(model_cache.HF_HUB_CACHE_DIR)
    assert os.environ["HF_MODULES_CACHE"] == str(model_cache.HF_MODULES_CACHE_DIR)
    assert os.environ["HF_DATASETS_CACHE"] == str(model_cache.HF_DATASETS_CACHE_DIR)
    assert "TRANSFORMERS_CACHE" not in os.environ
    assert os.environ["TORCH_HOME"] == str(model_cache.TORCH_CACHE_DIR)
    assert os.environ["KERAS_HOME"] == str(model_cache.KERAS_CACHE_DIR)
    assert os.environ["MPLCONFIGDIR"] == str(model_cache.MATPLOTLIB_CACHE_DIR)
    assert os.environ["PYTHONPYCACHEPREFIX"] == str(model_cache.PYTHON_CACHE_DIR)
    assert os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "1"
    assert "HF_CACHE_DIR" not in os.environ
    for path in (
        model_cache.HUGGINGFACE_CACHE_DIR,
        model_cache.HF_HUB_CACHE_DIR,
        model_cache.HF_MODULES_CACHE_DIR,
        model_cache.HF_DATASETS_CACHE_DIR,
        model_cache.TORCH_CACHE_DIR,
        model_cache.KERAS_CACHE_DIR,
        model_cache.MATPLOTLIB_CACHE_DIR,
    ):
        assert path.is_relative_to(model_cache.RUNTIME_CACHE_DIR)
        assert path.is_dir()
    for path in (
        model_cache.HF_INSTALLED_DIR,
        model_cache.HF_STAGING_DIR,
        model_cache.HF_ROLLBACK_DIR,
        model_cache.HF_METADATA_DIR,
    ):
        assert path.is_dir()
        assert not path.is_relative_to(model_cache.RUNTIME_CACHE_DIR)

from __future__ import annotations

from pathlib import Path

from maira2_worker.app import WorkerRuntime


###############################################################################
def test_readiness_requires_exact_cached_revision(monkeypatch) -> None:
    monkeypatch.setenv("XREPORT_MAIRA2_CACHE_DIR", "C:/offline-cache")
    monkeypatch.setenv("XREPORT_MAIRA2_REVISION", "main")
    assert WorkerRuntime().readiness()[0] == "incompatible"

    revision = "b" * 40
    monkeypatch.setenv("XREPORT_MAIRA2_REVISION", revision)
    monkeypatch.setattr(Path, "is_file", lambda self: self.name == "config.json")

    assert WorkerRuntime().readiness() == ("ready", None)


###############################################################################
def test_worker_source_enforces_offline_local_loading() -> None:
    source = Path(__file__).parents[1] / "maira2_worker" / "app.py"
    content = source.read_text(encoding="utf-8")

    assert 'os.environ["HF_HUB_OFFLINE"] = "1"' in content
    assert 'os.environ["TRANSFORMERS_OFFLINE"] = "1"' in content
    assert '"local_files_only": True' in content
    assert "snapshot_download" not in content

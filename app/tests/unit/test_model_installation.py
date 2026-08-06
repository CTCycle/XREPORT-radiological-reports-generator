from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from server.services import model_installation as installation_module
from server.services.model_installation import (
    InstallationCancelled,
    InstallationError,
    ModelInstallationManager,
)


REVISION = "a" * 40
NEXT_REVISION = "b" * 40

###############################################################################
def _manifest(revision: str = REVISION) -> dict[str, object]:
    return {
        "repository_id": "example/report-model",
        "revision": revision,
        "required_files": ["config.json", "tokenizer.json"],
        "weight_file_sets": [["model.safetensors"]],
    }

###############################################################################
class FakeApi:

    # -------------------------------------------------------------------------
    def __init__(self, revision: str = REVISION) -> None:
        self.revision = revision
        self.calls: list[dict[str, object]] = []

    # -------------------------------------------------------------------------
    def model_info(self, repository_id: str, **kwargs: object) -> SimpleNamespace:
        self.calls.append({"repository_id": repository_id, **kwargs})
        siblings = [
            SimpleNamespace(rfilename="config.json", size=2, lfs=None),
            SimpleNamespace(rfilename="tokenizer.json", size=2, lfs=None),
            SimpleNamespace(
                rfilename="model.safetensors",
                size=2,
                lfs=SimpleNamespace(sha256=""),
            ),
        ]
        return SimpleNamespace(sha=self.revision, siblings=siblings)

###############################################################################
class CompleteResponse:
    status_code = 200

    # -------------------------------------------------------------------------
    def __init__(self, body: bytes) -> None:
        self.body = body

    # -------------------------------------------------------------------------
    def raise_for_status(self) -> None:
        return None

    # -------------------------------------------------------------------------
    def iter_content(self, chunk_size: int) -> list[bytes]:
        del chunk_size
        return [self.body]

    # -------------------------------------------------------------------------
    def close(self) -> None:
        return None

###############################################################################
def complete_get(url: str, **_kwargs: object) -> CompleteResponse:
    filename = url.rsplit("/", 1)[-1].split("?", 1)[0]
    return CompleteResponse(b"ok" if filename == "model.safetensors" else b"{}")

###############################################################################
@pytest.fixture
def manager_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    root = tmp_path / "portable"
    resources = root / "app" / "resources"
    monkeypatch.setattr(installation_module, "ROOT_DIR", root)
    monkeypatch.setattr(installation_module, "HF_STAGING_DIR", resources / "models" / "huggingface" / "staging")
    monkeypatch.setattr(installation_module, "HF_INSTALLED_DIR", resources / "models" / "huggingface" / "installed")
    monkeypatch.setattr(installation_module, "HF_ROLLBACK_DIR", resources / "models" / "huggingface" / "rollback")
    monkeypatch.setattr(installation_module, "HF_METADATA_DIR", resources / "models" / "huggingface" / "metadata")
    for path in (resources, installation_module.HF_STAGING_DIR, installation_module.HF_INSTALLED_DIR):
        path.mkdir(parents=True, exist_ok=True)
    return root

###############################################################################
def test_stage_downloads_only_approved_files_and_writes_verified_metadata(
    manager_paths: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = FakeApi()
    calls: list[str] = []

    def fake_get(url: str, **kwargs: object) -> CompleteResponse:
        calls.append(url)
        return complete_get(url, **kwargs)

    monkeypatch.setattr(installation_module.requests, "get", fake_get)
    target = ModelInstallationManager(api=api).stage(
        manifest=_manifest(),
        revision=REVISION,
        should_stop=lambda: False,
        report_progress=lambda _payload: None,
    )

    assert target.candidate is True
    assert target.path.is_dir()
    assert all(REVISION in url for url in calls)
    assert {url.rsplit("/", 1)[-1].split("?", 1)[0] for url in calls} == {
        "config.json",
        "tokenizer.json",
        "model.safetensors",
    }
    metadata = ModelInstallationManager(api=api).read_metadata("example/report-model")
    assert metadata["state"] == "staged"
    assert metadata["integrity"] == "verified"
    assert metadata["candidate"]["revision"] == REVISION
    assert metadata["candidate"]["relative_path"].startswith("app/resources/models/huggingface/staging/")

###############################################################################
def test_stage_reuses_partial_staging_directory_after_interruption(
    manager_paths: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = ModelInstallationManager(api=FakeApi())
    partial = installation_module.HF_STAGING_DIR / "resume-op" / "example__report-model" / REVISION
    partial.mkdir(parents=True)
    (partial / "config.json").write_text("{}", encoding="utf-8")
    seen: list[Path] = []

    def fake_get(url: str, **kwargs: object) -> CompleteResponse:
        del kwargs
        seen.append(partial.parent)
        return complete_get(url)

    monkeypatch.setattr(installation_module.requests, "get", fake_get)
    target = manager.stage(
        manifest=_manifest(),
        revision=REVISION,
        should_stop=lambda: False,
        report_progress=lambda _payload: None,
    )
    assert seen
    assert target.path == partial

###############################################################################
def test_http_downloader_resumes_partial_files_and_reports_progress(
    manager_paths: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = ModelInstallationManager(api=FakeApi())
    partial = (
        installation_module.HF_STAGING_DIR
        / "resume-http"
        / "example__report-model"
        / REVISION
        / "model.safetensors.incomplete"
    )
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"o")
    calls: list[dict[str, object]] = []

    ###############################################################################
    class FakeResponse:
        status_code = 206

        # -------------------------------------------------------------------------
        def __init__(self, body: bytes) -> None:
            self.body = body

        # -------------------------------------------------------------------------
        def raise_for_status(self) -> None:
            return None

        # -------------------------------------------------------------------------
        def iter_content(self, chunk_size: int) -> list[bytes]:
            return [self.body]

        # -------------------------------------------------------------------------
        def close(self) -> None:
            return None

    def fake_get(url: str, **kwargs: object) -> FakeResponse:
        calls.append(kwargs)
        filename = url.rsplit("/", 1)[-1].split("?", 1)[0]
        return FakeResponse(b"k" if filename == "model.safetensors" else b"{}")

    monkeypatch.setattr(installation_module.requests, "get", fake_get)
    progress: list[dict[str, object]] = []
    target = manager.stage(
        manifest=_manifest(),
        revision=REVISION,
        should_stop=lambda: False,
        report_progress=progress.append,
    )

    assert target.path == partial.parent
    assert (partial.parent / "model.safetensors").read_bytes() == b"ok"
    assert any(call.get("headers") == {"Range": "bytes=1-"} for call in calls)
    assert any(item.get("downloaded_bytes") == 2 for item in progress)

###############################################################################
def test_activation_promotes_candidate_and_keeps_previous_active_for_rollback(
    manager_paths: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = ModelInstallationManager(api=FakeApi())

    monkeypatch.setattr(installation_module.requests, "get", complete_get)
    first = manager.stage(manifest=_manifest(), revision=REVISION, should_stop=lambda: False, report_progress=lambda _payload: None)
    manager.activate(manifest=_manifest(), target=first)
    second = manager.stage(manifest=_manifest(NEXT_REVISION), revision=NEXT_REVISION, should_stop=lambda: False, report_progress=lambda _payload: None)
    metadata = manager.activate(manifest=_manifest(NEXT_REVISION), target=second)

    assert metadata["state"] == "active"
    assert metadata["active_revision"] == NEXT_REVISION
    assert metadata["rollback"]["revision"] == REVISION
    assert (installation_module.HF_ROLLBACK_DIR / "example__report-model").exists()

###############################################################################
def test_corrupt_active_snapshot_is_rejected(manager_paths: Path) -> None:
    manager = ModelInstallationManager(api=FakeApi())
    installed = installation_module.HF_INSTALLED_DIR / "example__report-model" / REVISION
    installed.mkdir(parents=True)
    for filename in ("config.json", "tokenizer.json", "model.safetensors"):
        (installed / filename).write_text("ok", encoding="utf-8")
    metadata = {
        "repository_id": "example/report-model",
        "state": "active",
        "active_revision": REVISION,
        "active_relative_path": manager.relative_path(installed),
        "file_manifest": {"model.safetensors": {"size": 999, "sha256": "bad"}},
    }
    manager._write_metadata("example/report-model", metadata)
    with pytest.raises(InstallationError, match="integrity mismatch"):
        manager.active_target(_manifest())

###############################################################################
def test_cancellation_is_distinguished_from_download_failure() -> None:
    assert InstallationCancelled.__name__ != InstallationError.__name__

###############################################################################
def test_failed_maintenance_preserves_working_active_revision(manager_paths: Path) -> None:
    manager = ModelInstallationManager(api=FakeApi())
    installed = installation_module.HF_INSTALLED_DIR / "example__report-model" / REVISION
    installed.mkdir(parents=True)
    for filename in ("config.json", "tokenizer.json", "model.safetensors"):
        (installed / filename).write_text("ok", encoding="utf-8")
    manager._write_metadata(
        "example/report-model",
        {
            "repository_id": "example/report-model",
            "state": "active",
            "integrity": "verified",
            "active_revision": REVISION,
            "active_relative_path": manager.relative_path(installed),
        },
    )
    partial = installation_module.HF_STAGING_DIR / "cancelled-op" / "example__report-model" / REVISION
    partial.mkdir(parents=True)
    (partial / "model.safetensors.incomplete").write_bytes(b"partial")

    manager.record_error("example/report-model", "cancelled", interrupted=True)
    metadata = manager.read_metadata("example/report-model")
    assert metadata["state"] == "active"
    assert metadata["active_revision"] == REVISION
    assert metadata["interruption"]["resumable"] is True
    assert not partial.exists()

###############################################################################
def test_failed_first_use_keeps_resumable_staging(manager_paths: Path) -> None:
    manager = ModelInstallationManager(api=FakeApi())
    partial = installation_module.HF_STAGING_DIR / "first-use-op" / "example__report-model" / REVISION
    partial.mkdir(parents=True)
    (partial / "model.safetensors.incomplete").write_bytes(b"partial")

    manager.record_error("example/report-model", "cancelled", interrupted=True)

    assert partial.exists()

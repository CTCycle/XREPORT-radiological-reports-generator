from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import os
import shutil
import threading
import uuid
from collections.abc import Callable, Mapping
from typing import Any

from huggingface_hub import HfApi, snapshot_download
import requests

from server.common.path import (
    HF_HUB_CACHE_DIR,
    HF_INSTALLED_DIR,
    HF_METADATA_DIR,
    HF_ROLLBACK_DIR,
    HF_STAGING_DIR,
    ROOT_DIR,
)


REVISION_PATTERN = r"^[0-9a-f]{40}$"
ProgressCallback = Callable[[dict[str, Any]], None]
ORIGINAL_SNAPSHOT_DOWNLOAD = snapshot_download


class InstallationCancelled(RuntimeError):
    """Raised when the user cancels a download or maintenance operation."""


class InstallationError(RuntimeError):
    """Raised when a model cannot be installed or verified safely."""


@dataclass(frozen=True)
class InstallationTarget:
    repository_id: str
    revision: str
    path: Path
    candidate: bool
    operation_id: str | None = None


def _slug(repository_id: str) -> str:
    return repository_id.replace("/", "__").replace("\\", "__")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ModelInstallationManager:
    """Owns project-local Hugging Face snapshots and their lifecycle metadata."""

    _locks: dict[str, threading.RLock] = {}
    _locks_guard = threading.Lock()

    def __init__(self, *, api: HfApi | None = None) -> None:
        self.api = api or HfApi(endpoint="https://huggingface.co")

    @classmethod
    def _lock_for(cls, repository_id: str) -> threading.RLock:
        with cls._locks_guard:
            return cls._locks.setdefault(repository_id, threading.RLock())

    @staticmethod
    def _metadata_path(repository_id: str) -> Path:
        return HF_METADATA_DIR / f"{_slug(repository_id)}.json"

    @staticmethod
    def _relative(path: Path) -> str:
        return path.resolve().relative_to(ROOT_DIR.resolve()).as_posix()

    @classmethod
    def relative_path(cls, path: Path) -> str:
        """Return a portable path suitable for API responses and metadata."""
        return cls._relative(path)

    @staticmethod
    def _absolute(relative_path: str | None) -> Path | None:
        if not relative_path:
            return None
        path = (ROOT_DIR / relative_path).resolve()
        try:
            path.relative_to(ROOT_DIR.resolve())
        except ValueError as exc:
            raise InstallationError("Model metadata points outside the application root") from exc
        return path

    def read_metadata(self, repository_id: str) -> dict[str, Any]:
        path = self._metadata_path(repository_id)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {
                "schema_version": 1,
                "repository_id": repository_id,
                "state": "not_installed",
                "active_revision": None,
                "active_relative_path": None,
                "candidate": None,
                "rollback": None,
            }
        if not isinstance(payload, dict) or payload.get("repository_id") != repository_id:
            return {
                "schema_version": 1,
                "repository_id": repository_id,
                "state": "corrupt",
                "active_revision": None,
                "active_relative_path": None,
                "candidate": None,
                "rollback": None,
                "last_error": "Installation metadata is invalid.",
            }
        return payload

    def _write_metadata(self, repository_id: str, payload: Mapping[str, Any]) -> None:
        target = self._metadata_path(repository_id)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + f".{uuid.uuid4().hex}.tmp")
        temporary.write_text(
            json.dumps(dict(payload), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary, target)

    def inspect(self, manifest: Mapping[str, Any]) -> dict[str, Any]:
        repository_id = str(manifest["repository_id"])
        metadata = self.read_metadata(repository_id)
        active_path = self._absolute(metadata.get("active_relative_path"))
        active_revision = metadata.get("active_revision")
        active_present = bool(
            active_path
            and active_revision
            and active_path.is_dir()
            and self._has_required_files(active_path, manifest)
        )
        if metadata.get("state") == "active" and not active_present:
            metadata = {
                **metadata,
                "state": "corrupt",
                "integrity": "failed",
                "last_error": "The active model installation is incomplete or missing.",
            }
            self._write_metadata(repository_id, metadata)
        candidate = metadata.get("candidate")
        candidate_path = self._absolute(candidate.get("relative_path")) if isinstance(candidate, dict) else None
        candidate_present = bool(
            candidate_path
            and candidate_path.is_dir()
            and self._has_required_files(candidate_path, manifest)
        )
        return {
            "metadata": metadata,
            "state": metadata.get("state", "not_installed"),
            "integrity": metadata.get("integrity", "unknown"),
            "active_revision": active_revision if active_present else None,
            "active_path": active_path if active_present else None,
            "candidate": candidate if candidate_present else None,
            "candidate_path": candidate_path if candidate_present else None,
            "candidate_revision": (
                candidate.get("revision")
                if candidate_present and isinstance(candidate, dict)
                else None
            ),
        }

    def active_target(self, manifest: Mapping[str, Any]) -> InstallationTarget | None:
        inspected = self.inspect(manifest)
        path = inspected["active_path"]
        revision = inspected["active_revision"]
        if not path or not revision:
            return None
        self.verify_snapshot(path, manifest, inspected["metadata"].get("file_manifest"))
        return InstallationTarget(str(manifest["repository_id"]), str(revision), path, False)

    def candidate_target(self, manifest: Mapping[str, Any]) -> InstallationTarget | None:
        inspected = self.inspect(manifest)
        path = inspected["candidate_path"]
        revision = inspected["candidate_revision"]
        candidate = inspected["candidate"]
        if not path or not revision or not isinstance(candidate, dict):
            return None
        self.verify_snapshot(path, manifest, candidate.get("file_manifest"))
        return InstallationTarget(
            str(manifest["repository_id"]),
            str(revision),
            path,
            True,
            str(candidate.get("operation_id")) if candidate.get("operation_id") else None,
        )

    @staticmethod
    def _required_files(manifest: Mapping[str, Any]) -> list[str]:
        return [str(path) for path in manifest.get("required_files", [])]

    def _has_required_files(self, path: Path, manifest: Mapping[str, Any]) -> bool:
        required = self._required_files(manifest)
        if any(not (path / item).is_file() for item in required):
            return False
        weight_sets = manifest.get("weight_file_sets", [])
        return any(
            all((path / str(item)).is_file() for item in group)
            for group in weight_sets
        )

    def verify_snapshot(
        self,
        snapshot: Path,
        manifest: Mapping[str, Any],
        recorded_files: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not snapshot.is_dir() or not self._has_required_files(snapshot, manifest):
            raise InstallationError("The model snapshot is incomplete.")
        files: dict[str, dict[str, Any]] = {}
        total_bytes = 0
        for path in sorted(snapshot.rglob("*")):
            if not path.is_file() or ".cache" in path.parts:
                continue
            relative = path.relative_to(snapshot).as_posix()
            size = path.stat().st_size
            digest = _sha256(path)
            files[relative] = {"size": size, "sha256": digest}
            total_bytes += size
            if recorded_files and relative in recorded_files:
                recorded = recorded_files[relative]
                if recorded.get("size") != size or recorded.get("sha256") != digest:
                    raise InstallationError(f"Checkpoint integrity mismatch: {relative}")
        return {"files": files, "total_bytes": total_bytes}

    def _remote_metadata(self, repository_id: str, revision: str) -> dict[str, Any]:
        try:
            info = self.api.model_info(
                repository_id,
                revision=revision,
                timeout=30,
                files_metadata=True,
                token=False,
            )
            files: dict[str, dict[str, Any]] = {}
            for item in getattr(info, "siblings", []) or []:
                name = getattr(item, "rfilename", None)
                if not name:
                    continue
                lfs = getattr(item, "lfs", None)
                files[str(name)] = {
                    "size": getattr(item, "size", None),
                    "sha256": getattr(lfs, "sha256", None) if lfs else None,
                }
            return {"status": "available", "files": files}
        except Exception as exc:  # noqa: BLE001
            return {"status": "unavailable", "error": str(exc)[:200], "files": {}}

    @staticmethod
    def _allow_patterns(manifest: Mapping[str, Any]) -> list[str]:
        patterns = {str(item) for item in manifest.get("required_files", [])}
        for group in manifest.get("weight_file_sets", []):
            for item in group:
                item = str(item)
                if item.endswith(".safetensors") or item.endswith(".index.json"):
                    patterns.add(item)
        # Keep generation config when a repository supplies it, but never pull
        # the duplicate PyTorch .bin weight from generate-cxr.
        patterns.add("generation_config.json")
        return sorted(patterns)

    @staticmethod
    def _resumable_target(repository_id: str, revision: str) -> Path | None:
        slug = _slug(repository_id)
        candidates = [
            path
            for path in HF_STAGING_DIR.glob(f"*/{slug}/{revision}")
            if path.is_dir() and any(item.is_file() for item in path.rglob("*"))
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda path: path.stat().st_mtime)

    def _download(
        self,
        *,
        repository_id: str,
        revision: str,
        target: Path,
        manifest: Mapping[str, Any],
        should_stop: Callable[[], bool],
        report_progress: ProgressCallback,
        force_download: bool = False,
    ) -> None:
        target.mkdir(parents=True, exist_ok=True)

        def callback(payload: dict[str, Any]) -> None:
            if should_stop():
                raise InstallationCancelled("Model download cancelled")
            report_progress(payload)

        try:
            if should_stop():
                raise InstallationCancelled("Model download cancelled")
            # Keep the snapshot_download seam for deterministic unit tests, but
            # use an application-owned HTTP downloader in production.  The Hub
            # snapshot helper can block indefinitely on large Xet files before
            # its progress callback runs, which makes cancellation unusable.
            if snapshot_download is not ORIGINAL_SNAPSHOT_DOWNLOAD:
                snapshot_download(
                    repo_id=repository_id,
                    revision=revision,
                    local_dir=str(target),
                    cache_dir=str(HF_HUB_CACHE_DIR),
                    allow_patterns=self._allow_patterns(manifest),
                    local_files_only=False,
                    token=False,
                    max_workers=4,
                    force_download=force_download,
                )
                return

            remote = self._remote_metadata(repository_id, revision)
            if remote.get("status") != "available":
                raise InstallationError(
                    f"Unable to resolve files for {repository_id}: "
                    f"{remote.get('error', 'Hub metadata unavailable')}"
                )
            approved = set(self._allow_patterns(manifest))
            files = {
                name: details
                for name, details in remote.get("files", {}).items()
                if name in approved
            }
            if not files:
                raise InstallationError("The model repository contains no approved files.")

            # Recover any Hub-managed partial weight from a previous interrupted
            # attempt, then remove sidecars and unrelated files.  Only approved
            # artifacts and resumable partials are allowed in staging.
            for cached_partial in target.rglob("*.incomplete"):
                if ".cache" not in cached_partial.parts:
                    continue
                for name, details in files.items():
                    expected_hash = str(details.get("sha256") or "")
                    if expected_hash and expected_hash in cached_partial.name:
                        destination_partial = target / (name + ".incomplete")
                        if not destination_partial.exists():
                            destination_partial.parent.mkdir(parents=True, exist_ok=True)
                            os.replace(cached_partial, destination_partial)
                        break
            for existing in target.rglob("*"):
                if not existing.is_file():
                    continue
                relative = existing.relative_to(target).as_posix()
                if relative in files or relative in {f"{name}.incomplete" for name in files}:
                    continue
                existing.unlink()
            cache_dir = target / ".cache"
            if cache_dir.exists():
                shutil.rmtree(cache_dir, ignore_errors=True)

            total_bytes = sum(
                int(details["size"])
                for details in files.values()
                if details.get("size") is not None
            )
            completed_files = 0
            downloaded_total = 0
            for index, (name, details) in enumerate(sorted(files.items()), start=1):
                callback({
                    "phase": "downloading",
                    "message": f"Downloading {name}",
                    "current_file": name,
                    "files_completed": completed_files,
                    "total_files": len(files),
                    "downloaded_bytes": downloaded_total,
                    "total_bytes": total_bytes,
                })
                destination = target / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                partial = destination.with_name(destination.name + ".incomplete")
                expected_size = details.get("size")
                if force_download:
                    destination.unlink(missing_ok=True)
                    partial.unlink(missing_ok=True)
                if destination.is_file() and (
                    expected_size is None or destination.stat().st_size == expected_size
                ):
                    downloaded_total += destination.stat().st_size
                    completed_files += 1
                    continue
                if destination.exists():
                    destination.unlink()
                start = partial.stat().st_size if partial.is_file() else 0
                if expected_size is not None and start > expected_size:
                    partial.unlink()
                    start = 0
                url = (
                    f"https://huggingface.co/{repository_id}/resolve/"
                    f"{revision}/{name}?download=true"
                )
                headers = {"Range": f"bytes={start}-"} if start else {}
                try:
                    response = requests.get(
                        url,
                        headers=headers,
                        stream=True,
                        allow_redirects=True,
                        timeout=(30, 60),
                    )
                    response.raise_for_status()
                    if start and response.status_code == 200:
                        # The server ignored the range. Restart safely rather
                        # than appending a duplicate full response.
                        start = 0
                        partial.unlink(missing_ok=True)
                    if start and response.status_code != 206:
                        raise InstallationError(
                            f"Hub did not honour resume for {name} (HTTP {response.status_code})"
                        )
                    mode = "ab" if start else "wb"
                    with partial.open(mode) as stream:
                        downloaded_file = start
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if not chunk:
                                continue
                            if should_stop():
                                raise InstallationCancelled("Model download cancelled")
                            stream.write(chunk)
                            downloaded_file += len(chunk)
                            callback({
                                "phase": "downloading",
                                "message": f"Downloading {name}",
                                "current_file": name,
                                "file_index": index,
                                "files_completed": completed_files,
                                "total_files": len(files),
                                "downloaded_bytes": downloaded_total + downloaded_file,
                                "total_bytes": total_bytes,
                            })
                    response.close()
                except InstallationCancelled:
                    raise
                except requests.RequestException as exc:
                    raise InstallationError(f"Model download failed for {name}: {exc}") from exc
                if expected_size is not None and partial.stat().st_size != expected_size:
                    raise InstallationError(
                        f"Incomplete download for {name}: {partial.stat().st_size} of {expected_size} bytes"
                    )
                os.replace(partial, destination)
                downloaded_total += destination.stat().st_size
                completed_files += 1
        except InstallationCancelled:
            raise
        except InstallationError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise InstallationError(f"Model download failed: {exc}") from exc

    def stage(
        self,
        *,
        manifest: Mapping[str, Any],
        revision: str,
        should_stop: Callable[[], bool],
        report_progress: ProgressCallback,
        operation_id: str | None = None,
        force_download: bool = False,
    ) -> InstallationTarget:
        repository_id = str(manifest["repository_id"])
        requested_operation_id = operation_id
        operation_id = operation_id or uuid.uuid4().hex
        target = HF_STAGING_DIR / operation_id / _slug(repository_id) / revision
        lock = self._lock_for(repository_id)
        with lock:
            metadata = self.read_metadata(repository_id)
            candidate = metadata.get("candidate")
            if (
                isinstance(candidate, dict)
                and candidate.get("revision") == revision
                and candidate.get("relative_path")
                and requested_operation_id is None
            ):
                operation_id = str(candidate.get("operation_id"))
                target = self._absolute(str(candidate["relative_path"])) or target
            elif requested_operation_id is None:
                resumable = self._resumable_target(repository_id, revision)
                if resumable is not None:
                    target = resumable
                    operation_id = target.parts[-3]
            report_progress({
                "phase": "checking",
                "message": (
                    f"Resuming {repository_id} revision {revision[:12]}"
                    if target.exists() and any(target.iterdir())
                    else f"Preparing {repository_id} revision {revision[:12]}"
                ),
                "revision": revision,
            })
            if should_stop():
                raise InstallationCancelled("Model download cancelled")
            self._download(
                repository_id=repository_id,
                revision=revision,
                target=target,
                manifest=manifest,
                should_stop=should_stop,
                report_progress=report_progress,
                force_download=force_download,
            )
            report_progress({"phase": "verifying", "message": "Verifying downloaded model files"})
            verification = self.verify_snapshot(target, manifest)
            remote = self._remote_metadata(repository_id, revision)
            for name, remote_file in remote.get("files", {}).items():
                local_file = verification["files"].get(name)
                if not local_file:
                    continue
                expected_size = remote_file.get("size")
                expected_hash = remote_file.get("sha256")
                if expected_size is not None and expected_size != local_file["size"]:
                    raise InstallationError(f"Downloaded file size mismatch: {name}")
                if expected_hash and expected_hash != local_file["sha256"]:
                    raise InstallationError(f"Downloaded file hash mismatch: {name}")
            metadata = {
                **metadata,
                "schema_version": 1,
                "repository_id": repository_id,
                "source": f"https://huggingface.co/{repository_id}",
                "state": "staged",
                "integrity": "verified",
                "candidate": {
                    "operation_id": operation_id,
                    "revision": revision,
                    "relative_path": self._relative(target),
                    "file_manifest": verification["files"],
                    "total_bytes": verification["total_bytes"],
                    "remote_metadata": remote,
                },
                "last_error": None,
            }
            self._write_metadata(repository_id, metadata)
            report_progress({
                "phase": "verified",
                "message": "Model files verified",
                "files_completed": len(verification["files"]),
                "total_files": len(verification["files"]),
                "downloaded_bytes": verification["total_bytes"],
                "total_bytes": verification["total_bytes"],
            })
            return InstallationTarget(repository_id, revision, target, True, operation_id)

    def activate(
        self,
        *,
        manifest: Mapping[str, Any],
        target: InstallationTarget,
    ) -> dict[str, Any]:
        repository_id = target.repository_id
        if not target.candidate or not target.operation_id:
            raise InstallationError("Only a staged candidate can be activated")
        lock = self._lock_for(repository_id)
        with lock:
            metadata = self.read_metadata(repository_id)
            candidate = metadata.get("candidate") or {}
            candidate_path = self._absolute(candidate.get("relative_path"))
            if candidate_path != target.path or not candidate_path or not candidate_path.is_dir():
                raise InstallationError("The staged model candidate is missing")
            installed_target = HF_INSTALLED_DIR / _slug(repository_id) / target.revision
            installed_target.parent.mkdir(parents=True, exist_ok=True)
            old_active = self._absolute(metadata.get("active_relative_path"))
            rollback: dict[str, Any] | None = None
            if old_active and old_active.is_dir() and old_active != installed_target:
                rollback_target = (
                    HF_ROLLBACK_DIR
                    / _slug(repository_id)
                    / f"{metadata.get('active_revision') or 'active'}-{uuid.uuid4().hex[:8]}"
                )
                rollback_target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(old_active, rollback_target)
                rollback = {
                    "revision": metadata.get("active_revision"),
                    "relative_path": self._relative(rollback_target),
                }
            if installed_target.exists() and installed_target != candidate_path:
                rollback_target = (
                    HF_ROLLBACK_DIR
                    / _slug(repository_id)
                    / f"{target.revision}-{uuid.uuid4().hex[:8]}"
                )
                rollback_target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(installed_target, rollback_target)
                rollback = {
                    "revision": target.revision,
                    "relative_path": self._relative(rollback_target),
                }
            if candidate_path != installed_target:
                os.replace(candidate_path, installed_target)
            metadata = {
                **metadata,
                "state": "active",
                "integrity": "verified",
                "active_revision": target.revision,
                "active_relative_path": self._relative(installed_target),
                "file_manifest": candidate.get("file_manifest", {}),
                "total_bytes": candidate.get("total_bytes", 0),
                "candidate": None,
                "rollback": rollback,
                "activated_at": datetime.now(timezone.utc).isoformat(),
                "interruption": None,
                "last_error": None,
            }
            self._write_metadata(repository_id, metadata)
            return metadata

    def record_success(
        self,
        repository_id: str,
        *,
        inference: bool = False,
    ) -> None:
        """Persist successful local initialization/inference evidence."""
        metadata = self.read_metadata(repository_id)
        now = datetime.now(timezone.utc).isoformat()
        updates: dict[str, Any] = {
            "last_successful_initialization": now,
            "interruption": None,
            "last_error": None,
        }
        if inference:
            updates["last_successful_inference"] = now
        self._write_metadata(repository_id, {**metadata, **updates})

    @staticmethod
    def is_resumable_error(error: str) -> bool:
        """Return whether an installation failure can safely resume staging."""
        message = error.lower()
        if any(marker in message for marker in ("hash mismatch", "size mismatch", "integrity mismatch")):
            return False
        return any(marker in message for marker in ("cancelled", "download failed", "incomplete download"))

    def record_error(
        self,
        repository_id: str,
        error: str,
        *,
        state: str = "failed",
        interrupted: bool = False,
    ) -> None:
        metadata = self.read_metadata(repository_id)
        active_path = self._absolute(metadata.get("active_relative_path"))
        preserves_active = bool(
            metadata.get("active_revision")
            and active_path
            and active_path.is_dir()
        )
        self._write_metadata(
            repository_id,
            {
                **metadata,
                # A failed repair/reinstall must never hide or replace a
                # working active revision.
                "state": "active" if preserves_active else state,
                "integrity": "verified" if preserves_active else metadata.get("integrity", "unknown"),
                "last_error": error[:500],
                "interruption": {
                    "at": datetime.now(timezone.utc).isoformat(),
                    "message": error[:500],
                    "resumable": interrupted,
                },
            },
        )

    def check_update(self, repository_id: str) -> dict[str, Any]:
        metadata = self.read_metadata(repository_id)
        installed_revision = metadata.get("active_revision")
        try:
            info = self.api.model_info(repository_id, token=False)
            latest_revision = str(getattr(info, "sha", ""))
            available = bool(latest_revision and latest_revision != installed_revision)
            result = {
                "model_ref": f"huggingface:{repository_id}",
                "repository_id": repository_id,
                "installed_revision": installed_revision,
                "latest_revision": latest_revision or None,
                "update_available": available,
                "source": f"https://huggingface.co/{repository_id}",
                "checked_at": datetime.now(timezone.utc).isoformat(),
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001
            result = {
                "model_ref": f"huggingface:{repository_id}",
                "repository_id": repository_id,
                "installed_revision": installed_revision,
                "latest_revision": None,
                "update_available": False,
                "source": f"https://huggingface.co/{repository_id}",
                "checked_at": datetime.now(timezone.utc).isoformat(),
                "error": str(exc)[:200],
            }
        metadata = {**metadata, "update_check": result}
        self._write_metadata(repository_id, metadata)
        return result

    def assess_cloud(self, repository_id: str) -> dict[str, Any]:
        """Record whether a model has a qualifying free cloud route.

        Hugging Face provider mappings are metered/account-backed rather than an
        anonymous, unmetered image-to-report service, so they never replace the
        local path for this application.
        """
        mapping: Any = None
        error: str | None = None
        try:
            info = self.api.model_info(
                repository_id,
                expand=["inferenceProviderMapping"],
                token=False,
            )
            mapping = getattr(info, "inferenceProviderMapping", None)
            if mapping is None:
                mapping = getattr(info, "inference_provider_mapping", None)
        except Exception as exc:  # noqa: BLE001
            error = str(exc)[:200]
        if mapping is not None:
            try:
                json.dumps(mapping)
            except TypeError:
                mapping = str(mapping)
        assessment = {
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "source": f"https://huggingface.co/{repository_id}",
            "free_cloud_available": False,
            "provider_mapping": mapping,
            "reason": (
                "No deployed provider mapping was returned. Any general Hugging Face "
                "Inference Provider route is metered or account-backed and is not a "
                "qualifying free cloud service for chest-X-ray uploads."
            ),
            "error": error,
        }
        metadata = self.read_metadata(repository_id)
        self._write_metadata(repository_id, {**metadata, "cloud_assessment": assessment})
        return assessment

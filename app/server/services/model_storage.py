from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import TYPE_CHECKING, Any

from server.common.path import (
    HF_HUB_CACHE_DIR,
    HF_INSTALLED_DIR,
    HF_METADATA_DIR,
    HF_ROLLBACK_DIR,
    HF_STAGING_DIR,
    ROOT_DIR,
)
from server.services.model_installation import InstallationError

if TYPE_CHECKING:
    from server.services.model_installation import ModelInstallationManager


def _slug(repository_id: str) -> str:
    return repository_id.replace("/", "__").replace("\\", "__")


class ModelStorageLifecycle:
    """Deletion and retired-snapshot discovery for public model storage."""

    LEGACY_RETIRED_REPOSITORIES = {"nathansutton/generate-cxr"}

    def __init__(
        self,
        manager: ModelInstallationManager,
        *,
        root_dir: Path = ROOT_DIR,
        installed_dir: Path = HF_INSTALLED_DIR,
        hub_cache_dir: Path = HF_HUB_CACHE_DIR,
        metadata_dir: Path = HF_METADATA_DIR,
        rollback_dir: Path = HF_ROLLBACK_DIR,
        staging_dir: Path = HF_STAGING_DIR,
    ) -> None:
        self.manager = manager
        self.root_dir = root_dir
        self.installed_dir = installed_dir
        self.hub_cache_dir = hub_cache_dir
        self.metadata_dir = metadata_dir
        self.rollback_dir = rollback_dir
        self.staging_dir = staging_dir

    @staticmethod
    def _tree_size(path: Path) -> int:
        if not path.exists():
            return 0
        return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())

    @staticmethod
    def _remove_tree(path: Path, *, root_dir: Path = ROOT_DIR) -> int:
        resolved = path.resolve()
        try:
            resolved.relative_to(root_dir.resolve())
        except ValueError as exc:
            raise InstallationError("Refusing to delete a model path outside the project") from exc
        size = ModelStorageLifecycle._tree_size(resolved)
        if resolved.is_dir() and not resolved.is_symlink():
            shutil.rmtree(resolved)
        elif resolved.is_file() or resolved.is_symlink():
            resolved.unlink(missing_ok=True)
        return size

    def delete_local(  # noqa: C901
        self,
        repository_id: str,
        *,
        processor_repository_id: str | None = None,
        processor_revision: str | None = None,
    ) -> dict[str, Any]:
        """Delete only project-local files owned by one public repository."""
        lock = self.manager._lock_for(repository_id)
        with lock:
            slug = _slug(repository_id)
            roots = [
                self.installed_dir / slug,
                self.rollback_dir / slug,
                self.staging_dir,
                self.hub_cache_dir / f"models--{repository_id.replace('/', '--')}",
                self.hub_cache_dir / f"models--{slug}",
            ]
            metadata = self.manager.read_metadata(repository_id)
            revisions = {
                str(value)
                for value in (
                    metadata.get("active_revision"),
                    (metadata.get("candidate") or {}).get("revision")
                    if isinstance(metadata.get("candidate"), dict)
                    else None,
                    processor_revision,
                )
                if value
            }
            modules_root = self.hub_cache_dir.parent / "modules" / "transformers_modules"
            roots.extend(modules_root / f"_{revision}" for revision in revisions)
            if processor_repository_id:
                processor_slug = processor_repository_id.replace("/", "--")
                roots.append(self.hub_cache_dir / f"models--{processor_slug}")
                processor_parts = processor_repository_id.split("/")
                if processor_revision and len(processor_parts) == 2:
                    roots.append(
                        modules_root
                        / processor_parts[0]
                        / processor_parts[1]
                        / processor_revision
                    )
            bytes_freed = 0
            deleted_paths: list[str] = []
            for root in roots:
                if root == self.staging_dir:
                    if not root.is_dir():
                        continue
                    for operation_root in list(root.iterdir()):
                        model_root = operation_root / slug
                        if model_root.exists():
                            bytes_freed += self._remove_tree(model_root, root_dir=self.root_dir)
                            deleted_paths.append(self.manager._relative(model_root))
                            if operation_root.exists() and not any(operation_root.iterdir()):
                                operation_root.rmdir()
                    continue
                if root.exists():
                    bytes_freed += self._remove_tree(root, root_dir=self.root_dir)
                    deleted_paths.append(self.manager._relative(root))
            metadata_path = self.manager._metadata_path(repository_id)
            if metadata_path.exists():
                bytes_freed += metadata_path.stat().st_size
                metadata_path.unlink()
                deleted_paths.append(self.manager._relative(metadata_path))
            return {
                "repository_id": repository_id,
                "state": "not_installed",
                "bytes_freed": bytes_freed,
                "deleted_paths": deleted_paths,
            }

    def discover_legacy_local_models(
        self,
        configured_repository_ids: set[str],
    ) -> list[dict[str, Any]]:
        """Find retained, allowlisted snapshots from the retired catalogue."""
        legacy: list[dict[str, Any]] = []
        for metadata_path in sorted(self.metadata_dir.glob("*.json")):
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not isinstance(metadata, dict):
                continue
            repository_id = metadata.get("repository_id")
            if not isinstance(repository_id, str):
                continue
            if repository_id in configured_repository_ids:
                continue
            if repository_id not in self.LEGACY_RETIRED_REPOSITORIES:
                continue
            slug = _slug(repository_id)
            owned_paths = [
                self.installed_dir / slug,
                self.rollback_dir / slug,
                self.hub_cache_dir / f"models--{repository_id.replace('/', '--')}",
                self.hub_cache_dir / f"models--{slug}",
                metadata_path,
            ]
            owned_paths.extend(operation_root / slug for operation_root in self.staging_dir.glob("*"))
            existing = [path for path in owned_paths if path.exists()]
            if not existing:
                continue
            bytes_reclaimable = sum(self._tree_size(path) for path in existing)
            legacy.append({
                "model_ref": f"legacy:huggingface:{repository_id}",
                "repository_id": repository_id,
                "display_name": "Retired generate-cxr local snapshot",
                "bytes_reclaimable": bytes_reclaimable,
            })
        return legacy

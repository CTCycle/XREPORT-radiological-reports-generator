from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any

from server.common.path import (
    HF_HUB_CACHE_DIR,
    HF_INSTALLED_DIR,
    HF_ROLLBACK_DIR,
    HF_STAGING_DIR,
    ROOT_DIR,
    is_within_allowed_roots,
)
from server.services.model_installation import (
    InstallationError,
    ModelInstallationManager,
)


###############################################################################
def _slug(repository_id: str) -> str:
    return repository_id.replace("/", "__").replace("\\", "__")


###############################################################################
class ModelStorageLifecycle:
    """Owns deletion and reclaim accounting for public model storage."""

    # -------------------------------------------------------------------------
    def __init__(
        self,
        manager: ModelInstallationManager,
        *,
        root_dir: Path = ROOT_DIR,
        installed_dir: Path = HF_INSTALLED_DIR,
        hub_cache_dir: Path = HF_HUB_CACHE_DIR,
        rollback_dir: Path = HF_ROLLBACK_DIR,
        staging_dir: Path = HF_STAGING_DIR,
    ) -> None:
        self.manager = manager
        self.root_dir = root_dir
        self.installed_dir = installed_dir
        self.hub_cache_dir = hub_cache_dir
        self.rollback_dir = rollback_dir
        self.staging_dir = staging_dir

    # -------------------------------------------------------------------------
    @staticmethod
    def _tree_size(path: Path) -> int:
        if not path.exists():
            return 0
        return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())

    # -------------------------------------------------------------------------
    @staticmethod
    def _remove_tree(path: Path, *, root_dir: Path = ROOT_DIR) -> int:
        resolved = path.resolve()
        if not is_within_allowed_roots(resolved):
            raise InstallationError(
                "Refusing to delete a model path outside application storage"
            )
        size = ModelStorageLifecycle._tree_size(resolved)
        if resolved.is_dir() and not resolved.is_symlink():
            shutil.rmtree(resolved)
        elif resolved.is_file() or resolved.is_symlink():
            resolved.unlink(missing_ok=True)
        return size

    # -------------------------------------------------------------------------
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
            modules_root = (
                self.hub_cache_dir.parent / "modules" / "transformers_modules"
            )
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
                            bytes_freed += self._remove_tree(
                                model_root, root_dir=self.root_dir
                            )
                            deleted_paths.append(self.manager._relative(model_root))
                            if operation_root.exists() and not any(
                                operation_root.iterdir()
                            ):
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

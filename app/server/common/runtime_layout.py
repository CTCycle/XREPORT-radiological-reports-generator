"""Runtime/data layout helpers shared by source and packaged launches.

The packaged shell sets the ``XREPORT_*`` values before importing FastAPI.  All
mutable state is then created below the user data root; the extracted runtime
is treated as immutable and can be replaced atomically on an upgrade.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import tempfile


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class RuntimeLayout:
    mode: str
    runtime_root: Path
    data_root: Path
    release_version: str | None
    variant: str | None

    @property
    def packaged(self) -> bool:
        return self.mode == "packaged"

    @property
    def settings_template(self) -> Path:
        return self.runtime_root / "settings" / ".env.example"

    @property
    def configuration_template(self) -> Path:
        return self.runtime_root / "settings" / "configurations.json"

    @property
    def mutable_settings_dir(self) -> Path:
        return self.data_root / "settings"

    @classmethod
    def from_environment(cls) -> "RuntimeLayout":
        packaged = _truthy(os.getenv("XREPORT_DESKTOP"))
        if not packaged:
            # This value is only descriptive in source mode; path.py remains
            # responsible for resolving the historical repository root.
            source_root = Path(__file__).resolve().parents[3]
            return cls("source", source_root, source_root / "app" / "resources", None, None)

        runtime_value = os.getenv("XREPORT_RUNTIME_ROOT", "").strip()
        data_value = os.getenv("XREPORT_DATA_ROOT", "").strip()
        version = os.getenv("XREPORT_RELEASE_VERSION", "").strip()
        variant = os.getenv("XREPORT_RUNTIME_VARIANT", "").strip().lower()
        if not runtime_value or not data_value or not version or variant not in {"cpu", "cuda"}:
            raise RuntimeError(
                "Packaged XREPORT requires runtime root, data root, version, and cpu/cuda variant"
            )
        runtime_root = Path(runtime_value).expanduser().resolve()
        data_root = Path(data_value).expanduser().resolve()
        if not runtime_root.is_dir():
            raise RuntimeError(f"Packaged runtime root does not exist: {runtime_root}")
        return cls("packaged", runtime_root, data_root, version, variant)


def _atomic_copy_if_missing(source: Path, destination: Path) -> bool:
    if destination.exists():
        return False
    if not source.is_file():
        raise FileNotFoundError(f"Packaged runtime template is missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=destination.parent, prefix=f".{destination.name}.", delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
        with source.open("rb") as source_stream:
            shutil.copyfileobj(source_stream, temporary)
    try:
        os.replace(temporary_path, destination)
    except FileExistsError:
        temporary_path.unlink(missing_ok=True)
        return False
    return True


def ensure_packaged_data(layout: RuntimeLayout) -> None:
    """Create the data tree and seed first-run files without overwriting edits."""
    if not layout.packaged:
        return
    layout.data_root.mkdir(parents=True, exist_ok=True)
    layout.mutable_settings_dir.mkdir(parents=True, exist_ok=True)
    _atomic_copy_if_missing(
        layout.settings_template,
        layout.data_root / ".env",
    )
    _atomic_copy_if_missing(
        layout.configuration_template,
        layout.mutable_settings_dir / "configurations.json",
    )
    for name in (
        "logs",
        "models",
        "checkpoints",
        "templates",
        "state",
        "validation_receipts",
    ):
        (layout.data_root / name).mkdir(parents=True, exist_ok=True)


def validate_runtime_manifest(layout: RuntimeLayout) -> dict[str, object]:
    """Validate the extracted runtime manifest before importing application code."""
    if not layout.packaged:
        return {}
    manifest_path = layout.runtime_root / "runtime-manifest.json"
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Invalid packaged runtime manifest: {manifest_path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("Packaged runtime manifest must be a JSON object")
    if payload.get("version") != layout.release_version:
        raise RuntimeError("Packaged runtime version does not match the shell")
    if payload.get("variant") != layout.variant:
        raise RuntimeError("Packaged runtime variant does not match the shell")
    expected = str(payload.get("payload_sha256", "")).lower()
    if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
        raise RuntimeError("Packaged runtime manifest has an invalid payload hash")
    return payload


def runtime_layout_from_environment() -> RuntimeLayout:
    layout = RuntimeLayout.from_environment()
    if layout.packaged:
        validate_runtime_manifest(layout)
        ensure_packaged_data(layout)
    return layout

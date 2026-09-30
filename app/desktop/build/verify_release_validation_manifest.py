"""Verify the human-approved release-validation manifest and package hashes.

The release workflow invokes this script only after package artifacts have been
downloaded.  The manifest is intentionally a small, version-controlled
receipt: it binds the release decision to one source commit and every asset
that the publishing job would upload.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
from typing import Any


SCHEMA_VERSION = "xreport-release-validation-v1"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")


class ManifestError(ValueError):
    """Raised when an approval manifest does not bind the expected release."""


def expected_artifacts(version: str) -> tuple[str, ...]:
    prefix = f"XREPORT-v{version}-windows-x64"
    return (
        f"{prefix}-cpu-portable.exe",
        f"{prefix}-cpu.msi",
        f"{prefix}-cpu.sha256",
        f"{prefix}-cpu-build.json",
        f"{prefix}-cuda-portable.exe",
        f"{prefix}-cuda.msi",
        f"{prefix}-cuda.sha256",
        f"{prefix}-cuda-build.json",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_string(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"Manifest field {key!r} must be a non-empty string.")
    return value.strip()


def _validate_approval(manifest: dict[str, Any]) -> dict[str, str]:
    approval = manifest.get("approval")
    if not isinstance(approval, dict):
        raise ManifestError("Manifest must contain an approval object.")
    decision = _require_string(approval, "decision").lower()
    if decision != "approved":
        raise ManifestError(f"Manifest approval decision is {decision!r}, not 'approved'.")
    approved_by = _require_string(approval, "approved_by")
    approved_at = _require_string(approval, "approved_at_utc")
    try:
        parsed = datetime.fromisoformat(approved_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ManifestError("approval.approved_at_utc must be ISO-8601.") from exc
    if parsed.tzinfo is None:
        raise ManifestError("approval.approved_at_utc must include a timezone.")
    return {
        "decision": decision,
        "approved_by": approved_by,
        "approved_at_utc": approved_at,
    }


def verify_manifest(
    manifest_path: Path,
    release_root: Path,
    *,
    source_commit: str,
    version: str,
    repository_root: Path | None = None,
) -> dict[str, Any]:
    """Verify and return a compact receipt for an approved release manifest."""

    if not COMMIT_PATTERN.fullmatch(source_commit):
        raise ManifestError("source_commit must be a 40-character lowercase SHA.")
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise ManifestError(f"Invalid release version: {version}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ManifestError(f"Approval manifest not found: {manifest_path}") from exc
    except json.JSONDecodeError as exc:
        raise ManifestError(f"Approval manifest is not valid JSON: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ManifestError("Approval manifest root must be an object.")
    if _require_string(manifest, "schema_version") != SCHEMA_VERSION:
        raise ManifestError(f"Manifest schema_version must be {SCHEMA_VERSION!r}.")
    if _require_string(manifest, "status").lower() != "approved":
        raise ManifestError("Manifest status must be 'approved'.")
    if _require_string(manifest, "source_commit") != source_commit:
        raise ManifestError("Manifest source_commit does not match the workflow SHA.")
    if _require_string(manifest, "version") != version:
        raise ManifestError("Manifest version does not match the release version.")
    approval = _validate_approval(manifest)

    validation_record = _require_string(manifest, "validation_record")
    if repository_root is not None:
        record_path = (repository_root / validation_record).resolve()
        root = repository_root.resolve()
        if root not in record_path.parents:
            raise ManifestError("validation_record must remain inside the repository.")
        if not record_path.is_file():
            raise ManifestError(f"Validation record not found: {validation_record}")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ManifestError("Manifest artifacts must be an object of filename to SHA-256.")
    expected = expected_artifacts(version)
    if set(artifacts) != set(expected):
        missing = sorted(set(expected) - set(artifacts))
        unexpected = sorted(set(artifacts) - set(expected))
        raise ManifestError(
            f"Manifest artifact set mismatch; missing={missing}, unexpected={unexpected}."
        )

    root = release_root.resolve()
    verified: list[dict[str, Any]] = []
    for name in expected:
        digest = artifacts.get(name)
        if not isinstance(digest, str) or not SHA256_PATTERN.fullmatch(digest):
            raise ManifestError(f"Artifact hash for {name} is not a lowercase SHA-256.")
        artifact_path = (root / name).resolve()
        if root not in artifact_path.parents:
            raise ManifestError(f"Artifact path escapes release root: {name}")
        if not artifact_path.is_file() or artifact_path.stat().st_size == 0:
            raise ManifestError(f"Missing or empty release artifact: {name}")
        observed = _sha256(artifact_path)
        if observed != digest:
            raise ManifestError(
                f"SHA-256 mismatch for {name}: expected {digest}, observed {observed}."
            )
        verified.append({"name": name, "sha256": observed, "size_bytes": artifact_path.stat().st_size})

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "approved",
        "source_commit": source_commit,
        "version": version,
        "approval": approval,
        "validation_record": validation_record,
        "artifacts": verified,
    }


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--version", required=True)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    try:
        receipt = verify_manifest(
            args.manifest,
            args.release_root,
            source_commit=args.source_commit,
            version=args.version,
            repository_root=args.repository_root,
        )
    except (ManifestError, OSError) as exc:
        print(f"Release validation manifest rejected: {exc}")
        return 1
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Create and audit a deterministic streamed runtime ZIP for one variant."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import re
import stat
import zipfile


CHUNK_SIZE = 1024 * 1024
FORBIDDEN_NAMES = {".env", ".git", ".gitignore", "database.db"}
FORBIDDEN_PARTS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "node_modules",
    "tests",
    "test",
    "logs",
    "caches",
}
FORBIDDEN_SUFFIXES = {".db", ".sqlite", ".sqlite3", ".log", ".pyc", ".pyo"}
REQUIRED_MEMBERS = {
    "client/index.html",
    "client/error.html",
    "backend/XREPORT-backend.exe",
    "settings/.env.example",
    "settings/configurations.json",
    "settings/inference_models.json",
}
COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40}$")


def iter_files(root: Path) -> list[tuple[Path, str]]:
    if not root.is_dir():
        raise FileNotFoundError(f"runtime staging root does not exist: {root}")
    entries: list[tuple[Path, str]] = []
    seen: set[str] = set()
    for source in sorted(root.rglob("*"), key=lambda path: path.as_posix().lower()):
        relative = source.relative_to(root)
        parts = relative.parts
        normalized = relative.as_posix()
        if source.is_symlink():
            raise ValueError(f"symlink is not allowed in runtime staging: {normalized}")
        if source.is_dir():
            continue
        if not source.is_file():
            raise ValueError(f"unsupported runtime staging entry: {normalized}")
        lowered = normalized.lower()
        if (
            ":" in normalized
            or any(part.lower() in FORBIDDEN_PARTS for part in parts)
            or lowered.startswith(("models/", "checkpoints/", "logs/", "resources/"))
            or lowered in FORBIDDEN_NAMES
            or Path(lowered).name in FORBIDDEN_NAMES
            or Path(lowered).suffix in FORBIDDEN_SUFFIXES
        ):
            raise ValueError(f"forbidden runtime staging entry: {normalized}")
        normalized_key = normalized.lower()
        if normalized_key in seen:
            raise ValueError(f"duplicate runtime archive member: {normalized}")
        seen.add(normalized_key)
        entries.append((source, normalized))
    return entries


def digest_entries(
    entries: list[tuple[Path, str]],
) -> tuple[str, int, list[dict[str, object]]]:
    digest = hashlib.sha256()
    total = 0
    sizes: list[dict[str, object]] = []
    for source, name in entries:
        size = source.stat().st_size
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        with source.open("rb") as stream:
            while chunk := stream.read(CHUNK_SIZE):
                digest.update(chunk)
        total += size
        sizes.append({"path": name, "bytes": size})
    sizes.sort(key=lambda item: (-int(item["bytes"]), str(item["path"])))
    return digest.hexdigest(), total, sizes[:20]


def zip_entry(zf: zipfile.ZipFile, source: Path, name: str) -> None:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 0
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    with (
        source.open("rb") as stream,
        zf.open(info, "w", force_zip64=True) as destination,
    ):
        while chunk := stream.read(CHUNK_SIZE):
            destination.write(chunk)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--variant", choices=("cpu", "cuda"), required=True)
    parser.add_argument(
        "--architecture", choices=("windows-x64",), default="windows-x64"
    )
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--dirty", action="store_true")
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    if not COMMIT_RE.fullmatch(args.source_commit):
        raise ValueError("source commit must be a full 40-character hexadecimal SHA")

    entries = iter_files(args.staging)
    entry_names = {name for _, name in entries}
    missing = sorted(REQUIRED_MEMBERS.difference(entry_names))
    if missing:
        raise ValueError(
            f"runtime staging is missing required files: {', '.join(missing)}"
        )
    payload_sha256, payload_bytes, largest = digest_entries(entries)
    manifest = {
        "format": 2,
        "application": "XREPORT",
        "version": args.version,
        "variant": args.variant,
        "architecture": args.architecture,
        "backend_executable": "backend/XREPORT-backend.exe",
        "payload_sha256": payload_sha256,
        "file_count": len(entries),
        "payload_bytes": payload_bytes,
        "source_commit": args.source_commit,
        "dirty_tree": bool(args.dirty),
        "created_utc": datetime.now(UTC)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "largest_entries": largest,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with zipfile.ZipFile(
        temporary,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        allowZip64=True,
    ) as archive:
        for source, name in entries:
            zip_entry(archive, source, name)
        manifest_info = zipfile.ZipInfo(
            "runtime-manifest.json", date_time=(1980, 1, 1, 0, 0, 0)
        )
        manifest_info.compress_type = zipfile.ZIP_DEFLATED
        manifest_info.create_system = 0
        manifest_info.external_attr = (stat.S_IFREG | 0o644) << 16
        archive.writestr(
            manifest_info,
            json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8"),
        )
    temporary.replace(args.output)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"archive": str(args.output), **manifest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

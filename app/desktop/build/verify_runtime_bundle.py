"""Validate one XREPORT runtime archive and its metadata contract."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import re
import stat
import struct
import zipfile


ARCHITECTURE = "windows-x64"
FORMAT = 2
REQUIRED_MEMBERS = {
    "client/index.html",
    "client/error.html",
    "backend/XREPORT-backend.exe",
    "settings/.env.example",
    "settings/configurations.json",
    "settings/inference_models.json",
    "runtime-manifest.json",
}
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40}$")
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


def _manifest_timestamp(value: object) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError("runtime manifest has no created_utc timestamp")
    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(
            "runtime manifest has an invalid created_utc timestamp"
        ) from error
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("runtime manifest created_utc must include a timezone")


def _validate_member_name(name: str) -> None:
    if (
        not name
        or "\x00" in name
        or "\\" in name
        or ":" in name
        or name.startswith("/")
        or re.match(r"^[A-Za-z]:", name)
        or any(part in {"", ".", ".."} for part in name.split("/"))
    ):
        raise ValueError(f"runtime archive contains an unsafe member path: {name}")


def _verify_zip(
    archive: zipfile.ZipFile,
    *,
    archive_label: str,
    expected_version: str,
    expected_variant: str,
    expected_source_commit: str | None = None,
    expected_architecture: str = ARCHITECTURE,
) -> dict[str, object]:
    """Open, inspect, and hash an archive without extracting it."""

    infos = archive.infolist()
    names = [info.filename for info in infos]
    if len(names) != len(set(names)):
        raise ValueError("runtime archive contains duplicate members")
    if len(names) != len({name.lower() for name in names}):
        raise ValueError("runtime archive contains case-insensitive duplicate members")
    if any(info.is_dir() for info in infos):
        raise ValueError("runtime archive contains directory members")
    for info in infos:
        _validate_member_name(info.filename)
        mode = (info.external_attr >> 16) & 0o170000
        if mode == stat.S_IFLNK:
            raise ValueError(
                f"runtime archive contains a symlink member: {info.filename}"
            )
    for name in names:
        parts = Path(name).parts
        lowered = name.lower()
        if (
            any(part.lower() in FORBIDDEN_PARTS for part in parts)
            or lowered.startswith(("models/", "checkpoints/", "logs/", "resources/"))
            or lowered in FORBIDDEN_NAMES
            or Path(lowered).name in FORBIDDEN_NAMES
            or Path(lowered).suffix in FORBIDDEN_SUFFIXES
        ):
            raise ValueError(f"runtime archive contains forbidden member: {name}")
    if not REQUIRED_MEMBERS.issubset(names):
        missing = sorted(REQUIRED_MEMBERS.difference(names))
        raise ValueError(
            f"runtime archive is missing required members: {', '.join(missing)}"
        )

    try:
        manifest = json.loads(archive.read("runtime-manifest.json"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("runtime manifest is not valid JSON") from error
    if not isinstance(manifest, dict):
        raise ValueError("runtime manifest must be a JSON object")

    if manifest.get("format") != FORMAT:
        raise ValueError(
            f"unsupported runtime manifest format: {manifest.get('format')!r}"
        )
    if manifest.get("application") != "XREPORT":
        raise ValueError("runtime manifest application is not XREPORT")
    if manifest.get("version") != expected_version:
        raise ValueError("runtime manifest version does not match the desktop build")
    if manifest.get("variant") != expected_variant:
        raise ValueError("runtime manifest variant does not match the desktop build")
    if manifest.get("architecture") != expected_architecture:
        raise ValueError(
            "runtime manifest architecture does not match the desktop build"
        )
    source_commit = manifest.get("source_commit")
    if not isinstance(source_commit, str) or not COMMIT_RE.fullmatch(source_commit):
        raise ValueError("runtime manifest source_commit is not a full commit SHA")
    if expected_source_commit is not None and source_commit != expected_source_commit:
        raise ValueError("runtime manifest source_commit does not match HEAD")
    _manifest_timestamp(manifest.get("created_utc"))
    backend = manifest.get("backend_executable")
    if backend != "backend/XREPORT-backend.exe":
        raise ValueError(
            "runtime manifest backend path is not the expected frozen executable"
        )
    payload_sha256 = manifest.get("payload_sha256")
    if not isinstance(payload_sha256, str) or not SHA256_RE.fullmatch(payload_sha256):
        raise ValueError("runtime manifest payload_sha256 is invalid")
    file_count = manifest.get("file_count")
    if file_count != len(names) - 1:
        raise ValueError("runtime manifest file_count does not match the archive")
    payload_bytes_manifest = manifest.get("payload_bytes")
    if not isinstance(payload_bytes_manifest, int) or payload_bytes_manifest < 0:
        raise ValueError("runtime manifest payload_bytes is invalid")

    digest = hashlib.sha256()
    payload_bytes = 0
    for info in infos:
        if info.filename == "runtime-manifest.json":
            continue
        digest.update(info.filename.encode("utf-8"))
        digest.update(b"\0")
        with archive.open(info, "r") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
                payload_bytes += len(chunk)
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != payload_sha256.lower():
        raise ValueError("runtime payload hash does not match its manifest")
    if payload_bytes != payload_bytes_manifest:
        raise ValueError("runtime manifest payload_bytes does not match the archive")

    return {
        "archive": archive_label,
        "format": FORMAT,
        "version": expected_version,
        "variant": expected_variant,
        "architecture": expected_architecture,
        "source_commit": source_commit,
        "payload_sha256": actual_sha256,
        "file_count": file_count,
        "payload_bytes": payload_bytes,
    }


def verify_archive(
    archive_path: Path,
    *,
    expected_version: str,
    expected_variant: str,
    expected_source_commit: str | None = None,
    expected_architecture: str = ARCHITECTURE,
) -> dict[str, object]:
    """Open, inspect, and hash an archive without extracting it."""

    with zipfile.ZipFile(archive_path, "r") as archive:
        return _verify_zip(
            archive,
            archive_label=str(archive_path),
            expected_version=expected_version,
            expected_variant=expected_variant,
            expected_source_commit=expected_source_commit,
            expected_architecture=expected_architecture,
        )


class _BoundedFile(io.BufferedIOBase):
    def __init__(
        self, file: io.BufferedRandom | io.BufferedReader, start: int, length: int
    ) -> None:
        self._file = file
        self._start = start
        self._length = length
        self._position = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            target = offset
        elif whence == io.SEEK_CUR:
            target = self._position + offset
        elif whence == io.SEEK_END:
            target = self._length + offset
        else:
            raise ValueError(f"unsupported seek mode: {whence}")
        if target < 0 or target > self._length:
            raise ValueError("portable runtime overlay seek outside its bounded region")
        self._position = target
        return target

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = self._length - self._position
        size = min(size, self._length - self._position)
        self._file.seek(self._start + self._position)
        result = self._file.read(size)
        self._position += len(result)
        return result


def verify_portable(
    portable_path: Path,
    *,
    expected_version: str,
    expected_variant: str,
    expected_source_commit: str | None = None,
    expected_architecture: str = ARCHITECTURE,
) -> dict[str, object]:
    """Validate the ZIP64 archive appended to a portable PE executable."""

    footer_length = 24
    with portable_path.open("rb") as file:
        if file.read(2) != b"MZ":
            raise ValueError("portable artifact is not a Windows PE executable")
        file.seek(0, io.SEEK_END)
        file_length = file.tell()
        if file_length < footer_length:
            raise ValueError("portable executable has no runtime overlay footer")
        file.seek(file_length - footer_length)
        footer = file.read(footer_length)
        if footer[:8] != b"XRPZIP01":
            raise ValueError("portable executable has no XREPORT runtime overlay")
        offset, length = struct.unpack("<QQ", footer[8:])
        if (
            offset > file_length - footer_length
            or length > file_length - footer_length - offset
        ):
            raise ValueError("portable runtime overlay bounds are invalid")
        bounded = _BoundedFile(file, offset, length)
        with zipfile.ZipFile(bounded, "r") as archive:
            return _verify_zip(
                archive,
                archive_label=str(portable_path),
                expected_version=expected_version,
                expected_variant=expected_variant,
                expected_source_commit=expected_source_commit,
                expected_architecture=expected_architecture,
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--archive", type=Path)
    source.add_argument("--portable", type=Path)
    parser.add_argument("--version", required=True)
    parser.add_argument("--variant", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--source-commit")
    parser.add_argument("--architecture", default=ARCHITECTURE)
    args = parser.parse_args()
    verifier = verify_archive if args.archive else verify_portable
    result = verifier(
        args.archive or args.portable,
        expected_version=args.version,
        expected_variant=args.variant,
        expected_source_commit=args.source_commit,
        expected_architecture=args.architecture,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

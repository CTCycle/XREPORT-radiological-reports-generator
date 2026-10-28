from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
import hashlib
from pathlib import Path, PurePosixPath
import re
import stat
import zipfile

from xreport_shared.domain.checkpoint_exchange import ManifestFile

MANIFEST_NAME = "manifest.json"
MAX_ENTRIES = 128
MAX_MANIFEST_BYTES = 1024 * 1024
MAX_ENTRY_BYTES = 4 * 1024**3
MAX_TOTAL_BYTES = 6 * 1024**3
# Zip metadata is attacker controlled; this bounds the declared expansion factor.
MAX_COMPRESSION_RATIO = 1000
COPY_CHUNK_BYTES = 1024 * 1024

CORE_FILES = (
    "saved_model.keras",
    "configuration/configuration.json",
    "configuration/metadata.json",
    "configuration/session_history.json",
)
_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_ALLOWED_PREFIXES = ("tokenizer", "encoder")

ProgressCallback = Callable[[float], None]
StopCallback = Callable[[], bool]


###############################################################################
class ArchiveError(ValueError):
    """Raised when an exchange archive is malformed, unsafe or inconsistent."""


###############################################################################
class ArchiveCancelled(RuntimeError):
    """Raised when the caller requested cancellation during archive processing."""


###############################################################################
@dataclass(frozen=True)
class ArchiveEntry:
    name: str
    size: int


###############################################################################
def is_allowed_member(name: str) -> bool:
    """Return whether an archive member name is part of the exchange contract."""
    if name == MANIFEST_NAME or name in CORE_FILES:
        return True
    parts = name.split("/")
    if parts[0] not in _ALLOWED_PREFIXES or not 2 <= len(parts) <= 6:
        return False
    return all(_SEGMENT.fullmatch(part) for part in parts[1:])


###############################################################################
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(COPY_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


###############################################################################
def content_digest(files: Iterable[ManifestFile]) -> str:
    """Identity of an archive payload, independent of zip layout and timestamps."""
    digest = hashlib.sha256()
    for item in sorted(files, key=lambda entry: entry.path):
        digest.update(f"{item.path}\0{item.size}\0{item.sha256}\n".encode())
    return digest.hexdigest()


###############################################################################
def _check_member(info: zipfile.ZipInfo) -> None:
    name = info.filename
    if info.is_dir():
        raise ArchiveError(f"Archive contains an unexpected directory: {name}")
    if (
        "\\" in name
        or name.startswith("/")
        or ":" in name
        or any(part in {"", ".", ".."} for part in PurePosixPath(name).parts)
        or name != str(PurePosixPath(name))
    ):
        raise ArchiveError(f"Archive contains an unsafe path: {name!r}")
    if not is_allowed_member(name):
        raise ArchiveError(f"Archive contains an unsupported file: {name}")
    file_type = stat.S_IFMT(info.external_attr >> 16)
    if file_type and file_type != stat.S_IFREG:
        raise ArchiveError(f"Archive contains a non-regular entry: {name}")
    if info.flag_bits & 0x1:
        raise ArchiveError("Encrypted archives are not supported")
    if info.file_size > MAX_ENTRY_BYTES:
        raise ArchiveError(f"Archive entry is too large: {name}")
    if info.file_size > 1024 * 1024 and (
        info.compress_size == 0
        or info.file_size / info.compress_size > MAX_COMPRESSION_RATIO
    ):
        raise ArchiveError(f"Archive entry has a suspicious compression ratio: {name}")


###############################################################################
def inspect_archive(archive: zipfile.ZipFile) -> dict[str, ArchiveEntry]:
    """Validate every member name and size before anything is extracted."""
    infos = archive.infolist()
    if len(infos) > MAX_ENTRIES:
        raise ArchiveError(f"Archive has more than {MAX_ENTRIES} entries")
    entries: dict[str, ArchiveEntry] = {}
    total = 0
    for info in infos:
        _check_member(info)
        if info.filename in entries:
            raise ArchiveError(f"Archive contains a duplicate entry: {info.filename}")
        total += info.file_size
        if total > MAX_TOTAL_BYTES:
            raise ArchiveError("Archive expands beyond the supported size")
        entries[info.filename] = ArchiveEntry(info.filename, info.file_size)
    return entries


###############################################################################
def extract_member(
    archive: zipfile.ZipFile,
    name: str,
    destination: Path,
    expected_size: int,
    should_stop: StopCallback | None = None,
    on_bytes: Callable[[int], None] | None = None,
) -> str:
    """Stream one member to ``destination`` and return its sha256.

    The byte count is enforced while streaming so a lying header cannot expand
    beyond the size the manifest declared.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    written = 0
    with archive.open(name) as source, destination.open("xb") as target:
        for chunk in iter(lambda: source.read(COPY_CHUNK_BYTES), b""):
            written += len(chunk)
            if written > expected_size:
                raise ArchiveError(f"Archive entry exceeds its declared size: {name}")
            digest.update(chunk)
            target.write(chunk)
            if on_bytes is not None:
                on_bytes(len(chunk))
            if should_stop is not None and should_stop():
                raise ArchiveCancelled("Import was cancelled")
    if written != expected_size:
        raise ArchiveError(f"Archive entry is shorter than declared: {name}")
    return digest.hexdigest()


###############################################################################
def write_archive(
    destination: Path,
    members: list[tuple[str, Path]],
    manifest_json: str,
    should_stop: StopCallback | None = None,
    on_progress: ProgressCallback | None = None,
) -> None:
    """Write members then the manifest to a temporary file and publish atomically."""
    temporary = destination.with_name(f".{destination.name}.partial")
    temporary.unlink(missing_ok=True)
    total = sum(path.stat().st_size for _, path in members) or 1
    done = 0
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, path in members:
                with path.open("rb") as source, archive.open(
                    name, "w", force_zip64=True
                ) as target:
                    for chunk in iter(lambda: source.read(COPY_CHUNK_BYTES), b""):
                        target.write(chunk)
                        done += len(chunk)
                        if on_progress is not None:
                            on_progress(done / total)
                        if should_stop is not None and should_stop():
                            raise ArchiveCancelled("Export was cancelled")
            archive.writestr(MANIFEST_NAME, manifest_json)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)

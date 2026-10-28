from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any
import uuid
import zipfile

from xreport_shared.common.path import (
    CHECKPOINTS_DIR,
    ENCODERS_DIR,
    EXPORTS_DIR,
    TOKENIZERS_DIR,
)
from xreport_shared.common.constants import FASTAPI_VERSION
from xreport_shared.common.utils.logger import logger
from xreport_shared.common.utils.security import validate_checkpoint_name
from xreport_shared.domain.checkpoint_exchange import (
    CUSTOM_OBJECTS_API,
    ENCODER_REPOSITORY,
    EXCHANGE_ARCHITECTURE,
    EXCHANGE_FILE_EXTENSION,
    EXCHANGE_FORMAT_VERSION,
    ExchangeManifest,
    ExportResult,
    ImportOutcome,
    ManifestEncoder,
    ManifestFile,
    ManifestProducer,
    ManifestProvenance,
    ManifestTokenizer,
)
from xreport_shared.domain.jobs import JobStartResponse
from xreport_shared.repositories.checkpoint_archive import (
    CORE_FILES,
    MANIFEST_NAME,
    MAX_ENTRIES,
    MAX_ENTRY_BYTES,
    MAX_MANIFEST_BYTES,
    MAX_TOTAL_BYTES,
    ArchiveCancelled,
    ArchiveError,
    content_digest,
    extract_member,
    inspect_archive,
    is_allowed_member,
    sha256_file,
    write_archive,
)
from xreport_shared.repositories.checkpoints import (
    CheckpointRegistryError,
    CheckpointRepository,
)
from xreport_shared.services.errors import BadRequestError, ConflictError, NotFoundError
from xreport_shared.services.jobs import JobAlreadyRunningError, JobExecutionError, JobManager, get_job_manager

EXPORT_JOB_TYPE = "checkpoint_export"
IMPORT_JOB_TYPE = "checkpoint_import"
STAGING_PREFIX = ".import-"
EXCHANGE_RECORD = "exchange.json"
ENCODER_CACHE_DIRNAME = "models--microsoft--beit-base-patch16-224"
_TOKENIZER_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*(/[A-Za-z0-9][A-Za-z0-9._-]*)?$")

###############################################################################
def tokenizer_exchange_dir(name: str) -> Path:
    """Directory holding a tokenizer installed from an exchange archive."""
    if not _TOKENIZER_NAME.fullmatch(name):
        raise ValueError(f"Unsupported tokenizer name: {name!r}")
    return TOKENIZERS_DIR / "_exchange" / name.replace("/", "--")

###############################################################################
def encoder_snapshot_dir(revision: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Encoder revision must be a 40-character commit hash")
    return ENCODERS_DIR / ENCODER_CACHE_DIRNAME / "snapshots" / revision

###############################################################################
def encoder_installed(revision: str) -> bool:
    snapshot = encoder_snapshot_dir(revision)
    return (snapshot / "config.json").is_file() and any(
        (snapshot / weights).is_file()
        for weights in ("model.safetensors", "pytorch_model.bin")
    )

###############################################################################
def pinned_encoder_revision() -> str:
    from xreport_shared.models.training.encoder import BeitXRayImageEncoder

    return BeitXRayImageEncoder.ENCODER_REVISION

###############################################################################
def sweep_stale_staging() -> int:
    """Remove interrupted import staging left by a crash; known-good folders stay."""
    removed = 0
    if not CHECKPOINTS_DIR.is_dir():
        return removed
    for entry in CHECKPOINTS_DIR.iterdir():
        if not entry.name.startswith(STAGING_PREFIX):
            continue
        if entry.is_dir():
            shutil.rmtree(entry, ignore_errors=True)
        else:
            entry.unlink(missing_ok=True)
        removed += 1
    return removed

###############################################################################
@dataclass(frozen=True)
class ImportPlan:
    manifest: ExchangeManifest
    name: str
    renamed_from: str | None

###############################################################################
class CheckpointExchangeService:

    # -------------------------------------------------------------------------
    def __init__(
        self,
        checkpoint_repository: CheckpointRepository | None = None,
        job_manager: JobManager | None = None,
    ) -> None:
        self._checkpoints = checkpoint_repository
        self._jobs = job_manager

    # -------------------------------------------------------------------------
    @property
    def checkpoints(self) -> CheckpointRepository:
        if self._checkpoints is None:
            self._checkpoints = CheckpointRepository()
        return self._checkpoints

    # -------------------------------------------------------------------------
    @property
    def jobs(self) -> JobManager:
        if self._jobs is None:
            self._jobs = get_job_manager()
        return self._jobs

    ###########################################################################
    # Export
    ###########################################################################
    def start_export(self, checkpoint: str, include_encoder: bool) -> JobStartResponse:
        try:
            name = validate_checkpoint_name(checkpoint)
        except ValueError as exc:
            raise BadRequestError(str(exc)) from exc
        record = self.checkpoints.get_checkpoint(name)
        if record is None:
            raise NotFoundError(f"Checkpoint is not registered: {name}")
        if not record.artifact_complete:
            raise BadRequestError(f"Checkpoint artifact is incomplete: {name}")
        if include_encoder and not encoder_installed(pinned_encoder_revision()):
            raise BadRequestError(
                "The pinned BEiT encoder is not installed locally, so it cannot be bundled"
            )
        try:
            job_id = self.jobs.start_job(
                EXPORT_JOB_TYPE,
                self.run_export,
                kwargs={"name": name, "include_encoder": include_encoder},
                require_idle=True,
            )
        except JobAlreadyRunningError as exc:
            raise ConflictError(str(exc)) from exc
        return JobStartResponse(
            job_id=job_id,
            job_type=EXPORT_JOB_TYPE,
            status="running",
            message=f"Exporting checkpoint {name}",
            poll_interval=1.0,
        )

    # -------------------------------------------------------------------------
    def run_export(
        self, name: str, include_encoder: bool, job_id: str | None = None
    ) -> dict[str, Any]:
        record = self.checkpoints.verify_artifact(name)
        if not record.artifact_complete:
            raise JobExecutionError(
                f"Checkpoint artifact is incomplete: {name}", code="artifact_incomplete"
            )
        should_stop = self._stop_callback(job_id)
        EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
        workdir = Path(tempfile.mkdtemp(prefix=".export-", dir=EXPORTS_DIR))
        try:
            members: list[tuple[str, Path]] = [
                (relative, record.path / relative) for relative in CORE_FILES
            ]
            metadata = json.loads(
                (record.path / "configuration" / "metadata.json").read_text("utf-8")
            )
            history = json.loads(
                (record.path / "configuration" / "session_history.json").read_text("utf-8")
            )
            tokenizer_name = metadata.get("tokenizer")
            if not isinstance(tokenizer_name, str) or not tokenizer_name:
                raise JobExecutionError(
                    "Checkpoint metadata has no tokenizer", code="tokenizer_missing"
                )
            missing = [
                key
                for key in ("pad_token_id", "start_token_id", "end_token_id", "vocabulary_size")
                if metadata.get(key) is None
            ]
            if missing:
                raise JobExecutionError(
                    f"Checkpoint metadata does not record {', '.join(missing)}; "
                    "retrain the model to export it.",
                    code="metadata_invalid",
                )
            members += self._stage_tokenizer(metadata, workdir / "tokenizer")
            revision = pinned_encoder_revision()
            if include_encoder:
                snapshot = encoder_snapshot_dir(revision)
                for source in sorted(p for p in snapshot.rglob("*") if p.is_file()):
                    relative = source.relative_to(snapshot).as_posix()
                    members.append((f"encoder/{relative}", source))
            self._progress(job_id, 5)

            files = [
                ManifestFile(
                    path=relative,
                    size=path.stat().st_size,
                    sha256=sha256_file(path),
                )
                for relative, path in members
            ]
            self._assert_exportable(files)
            self._progress(job_id, 30)
            manifest = ExchangeManifest(
                format_version=EXCHANGE_FORMAT_VERSION,
                architecture=EXCHANGE_ARCHITECTURE,
                custom_objects_api=CUSTOM_OBJECTS_API,
                checkpoint_name=name,
                content_sha256=content_digest(files),
                producer=self._producer(),
                tokenizer=ManifestTokenizer(
                    name=tokenizer_name,
                    bundled=True,
                    revision=metadata.get("tokenizer_revision"),
                    vocabulary_size=metadata.get("vocabulary_size"),
                    pad_token_id=metadata.get("pad_token_id"),
                    start_token_id=metadata.get("start_token_id"),
                    end_token_id=metadata.get("end_token_id"),
                ),
                encoder=ManifestEncoder(
                    repository=ENCODER_REPOSITORY,
                    revision=revision,
                    bundled=include_encoder,
                ),
                provenance=self._provenance(history),
                files=files,
            )
            destination = EXPORTS_DIR / f"{name}{EXCHANGE_FILE_EXTENSION}"
            write_archive(
                destination,
                members,
                manifest.model_dump_json(indent=2),
                should_stop=should_stop,
                on_progress=lambda fraction: self._progress(job_id, 30 + fraction * 69),
            )
        except ArchiveCancelled:
            return {}
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
        result = ExportResult(
            filename=destination.name,
            size=destination.stat().st_size,
            content_sha256=manifest.content_sha256,
        )
        return {"export": result.model_dump()}

    # -------------------------------------------------------------------------
    @staticmethod
    def _assert_exportable(files: list[ManifestFile]) -> None:
        """Refuse to build an archive that the import limits would reject."""
        rejected = next((item.path for item in files if not is_allowed_member(item.path)), None)
        if rejected is not None:
            raise JobExecutionError(
                f"'{rejected}' cannot be exported in a checkpoint archive.",
                code="export_member_unsupported",
            )
        if (
            len(files) + 1 > MAX_ENTRIES
            or max(item.size for item in files) > MAX_ENTRY_BYTES
            or sum(item.size for item in files) > MAX_TOTAL_BYTES
        ):
            raise JobExecutionError(
                "The checkpoint is too large to export as a single archive; "
                "export it without the encoder or reduce its size.",
                code="export_too_large",
            )

    # -------------------------------------------------------------------------
    def resolve_export_path(self, filename: str) -> Path:
        if (
            filename != Path(filename).name
            or not filename.endswith(EXCHANGE_FILE_EXTENSION)
            or filename.startswith(".")
        ):
            raise BadRequestError("Invalid export file name")
        path = EXPORTS_DIR / filename
        if not path.is_file():
            raise NotFoundError("Export file was not found")
        return path

    # -------------------------------------------------------------------------
    @staticmethod
    def _stage_tokenizer(
        metadata: dict[str, Any], directory: Path
    ) -> list[tuple[str, Path]]:
        from xreport_shared.models.training.processing import TokenizerHandler

        try:
            handler = TokenizerHandler(metadata)
            if handler.tokenizer is None:
                raise RuntimeError("tokenizer unavailable")
            directory.mkdir(parents=True, exist_ok=True)
            handler.tokenizer.save_pretrained(directory)
        except Exception as exc:  # noqa: BLE001
            raise JobExecutionError(
                "The tokenizer is not available locally; install it before exporting",
                code="tokenizer_missing",
            ) from exc
        return [
            (f"tokenizer/{path.name}", path)
            for path in sorted(directory.iterdir())
            if path.is_file()
        ]

    # -------------------------------------------------------------------------
    @staticmethod
    def _producer() -> ManifestProducer:
        try:
            import keras
            import torch

            keras_version, torch_version = str(keras.__version__), str(torch.__version__)
        except ImportError:
            keras_version = torch_version = None
        return ManifestProducer(
            application="XREPORT",
            version=FASTAPI_VERSION,
            keras=keras_version,
            torch=torch_version,
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def _provenance(history: dict[str, Any]) -> ManifestProvenance:
        def last(values: Any) -> float | None:
            if isinstance(values, list) and values and isinstance(values[-1], int | float):
                return float(values[-1])
            return None

        series = history.get("history")
        series = series if isinstance(series, dict) else {}
        epochs = history.get("epochs")
        return ManifestProvenance(
            created_utc=datetime.now(timezone.utc).isoformat(),
            epochs=epochs if isinstance(epochs, int) else None,
            final_loss=last(series.get("loss")),
            final_val_loss=last(series.get("val_loss")),
        )

    ###########################################################################
    # Import
    ###########################################################################
    def assess_archive(
        self, archive_path: Path, rename_to: str | None = None
    ) -> ImportPlan:
        """Cheap synchronous validation: structure, compatibility and name conflicts."""
        try:
            with zipfile.ZipFile(archive_path) as archive:
                entries = inspect_archive(archive)
                if MANIFEST_NAME not in entries:
                    raise BadRequestError(
                        f"Checkpoint archive is missing required {MANIFEST_NAME}"
                    )
                if entries[MANIFEST_NAME].size > MAX_MANIFEST_BYTES:
                    raise ArchiveError("Archive manifest is too large")
                manifest = ExchangeManifest.model_validate_json(
                    archive.read(MANIFEST_NAME)
                )
        except zipfile.BadZipFile as exc:
            raise BadRequestError(
                "The file is not a valid XREPORT checkpoint archive. A lone .keras "
                "file is not enough: a checkpoint needs saved_model.keras plus "
                "configuration/configuration.json, metadata.json and session_history.json."
            ) from exc
        except (ArchiveError, ValueError) as exc:
            raise BadRequestError(f"Invalid checkpoint archive: {exc}") from exc

        self._assert_compatible(manifest, entries)
        return self._plan_name(manifest, rename_to)

    # -------------------------------------------------------------------------
    @staticmethod
    def _assert_versions(manifest: ExchangeManifest) -> None:
        if manifest.format_version != EXCHANGE_FORMAT_VERSION:
            raise BadRequestError(
                f"Unsupported checkpoint format version {manifest.format_version}; "
                f"this XREPORT requires version {EXCHANGE_FORMAT_VERSION}."
            )
        if manifest.architecture != EXCHANGE_ARCHITECTURE:
            raise BadRequestError(f"Unsupported architecture: {manifest.architecture}")
        if manifest.custom_objects_api != CUSTOM_OBJECTS_API:
            raise BadRequestError(
                "Checkpoint was built with an incompatible custom-layer version "
                f"({manifest.custom_objects_api}); this XREPORT expects {CUSTOM_OBJECTS_API}."
            )
        if (
            manifest.encoder.repository != ENCODER_REPOSITORY
            or manifest.encoder.revision != pinned_encoder_revision()
        ):
            raise BadRequestError(
                "Checkpoint was built with a different image encoder "
                f"({manifest.encoder.repository}@{manifest.encoder.revision[:8]}); this "
                f"XREPORT requires {ENCODER_REPOSITORY}@{pinned_encoder_revision()[:8]}."
            )
        try:
            validate_checkpoint_name(manifest.checkpoint_name)
            encoder_snapshot_dir(manifest.encoder.revision)
            tokenizer_exchange_dir(manifest.tokenizer.name)
        except ValueError as exc:
            raise BadRequestError(f"Invalid checkpoint archive: {exc}") from exc

    # -------------------------------------------------------------------------
    @staticmethod
    def _assert_payload(manifest: ExchangeManifest, entries: dict[str, Any]) -> None:
        declared = {item.path: item for item in manifest.files}
        payload = {name for name in entries if name != MANIFEST_NAME}
        if len(declared) != len(manifest.files) or set(declared) != payload:
            raise BadRequestError(
                "Invalid checkpoint archive: manifest does not match archive contents"
            )
        for path, item in declared.items():
            if entries[path].size != item.size:
                raise BadRequestError(f"Invalid checkpoint archive: size mismatch for {path}")
        missing = [path for path in CORE_FILES if path not in declared]
        if missing:
            raise BadRequestError(
                f"Invalid checkpoint archive: missing {', '.join(missing)}"
            )
        if content_digest(manifest.files) != manifest.content_sha256:
            raise BadRequestError("Invalid checkpoint archive: content hash mismatch")
        if not manifest.tokenizer.bundled or not any(
            path.startswith("tokenizer/") for path in declared
        ):
            raise BadRequestError("Invalid checkpoint archive: tokenizer files are missing")
        has_encoder = any(path.startswith("encoder/") for path in declared)
        if manifest.encoder.bundled != has_encoder:
            raise BadRequestError("Invalid checkpoint archive: encoder flag mismatch")

    # -------------------------------------------------------------------------
    @classmethod
    def _assert_compatible(
        cls, manifest: ExchangeManifest, entries: dict[str, Any]
    ) -> None:
        cls._assert_versions(manifest)
        cls._assert_payload(manifest, entries)
        if not manifest.encoder.bundled and not encoder_installed(
            manifest.encoder.revision
        ):
            raise BadRequestError(
                "The pinned BEiT encoder "
                f"({manifest.encoder.repository}@{manifest.encoder.revision[:8]}) is not "
                "installed on this machine and the archive does not bundle it. Install it "
                "locally or re-export the checkpoint with the encoder included."
            )

    # -------------------------------------------------------------------------
    def _plan_name(
        self,
        manifest: ExchangeManifest,
        rename_to: str | None,
    ) -> ImportPlan:
        original = manifest.checkpoint_name
        try:
            name = validate_checkpoint_name(rename_to or original)
        except ValueError as exc:
            raise BadRequestError(str(exc)) from exc
        existing = self.checkpoints.get_checkpoint(name)
        on_disk = (CHECKPOINTS_DIR / name).exists()
        if existing is not None or on_disk:
            if (
                existing is not None
                and self._installed_digest(existing.path) == manifest.content_sha256
            ):
                return ImportPlan(manifest, name, None)
            suggestion = self._suggest_name(name)
            raise ConflictError(
                f"A different checkpoint named '{name}' already exists. "
                f"Import it under another name, for example '{suggestion}'."
            )
        renamed = original if rename_to and rename_to != original else None
        return ImportPlan(manifest, name, renamed)

    # -------------------------------------------------------------------------
    def _suggest_name(self, name: str) -> str:
        for index in range(2, 1000):
            candidate = f"{name[:120]}_{index}"
            if (
                self.checkpoints.get_checkpoint(candidate) is None
                and not (CHECKPOINTS_DIR / candidate).exists()
            ):
                return candidate
        return f"{name[:100]}_{uuid.uuid4().hex[:8]}"

    # -------------------------------------------------------------------------
    @staticmethod
    def _installed_digest(path: Path) -> str | None:
        try:
            record = json.loads((path / EXCHANGE_RECORD).read_text("utf-8"))
            digest = record.get("content_sha256")
            return digest if isinstance(digest, str) else None
        except (OSError, ValueError, AttributeError):
            return None

    # -------------------------------------------------------------------------
    def start_import(
        self, archive_path: Path, rename_to: str | None
    ) -> JobStartResponse:
        """Assess the staged archive and start the import job (which owns the file)."""
        try:
            plan = self.assess_archive(archive_path, rename_to)
            job_id = self.jobs.start_job(
                IMPORT_JOB_TYPE,
                self.run_import,
                kwargs={"archive_path": str(archive_path), "plan": plan},
                require_idle=True,
            )
        except Exception:
            archive_path.unlink(missing_ok=True)
            raise
        return JobStartResponse(
            job_id=job_id,
            job_type=IMPORT_JOB_TYPE,
            status="running",
            message=f"Importing checkpoint {plan.name}",
            poll_interval=1.0,
        )

    # -------------------------------------------------------------------------
    def run_import(
        self, archive_path: str, plan: ImportPlan, job_id: str | None = None
    ) -> dict[str, Any]:
        source = Path(archive_path)
        should_stop = self._stop_callback(job_id)
        staging = CHECKPOINTS_DIR / f"{STAGING_PREFIX}{uuid.uuid4().hex}"
        published = False
        installed: list[Path] = []
        final = CHECKPOINTS_DIR / plan.name
        try:
            existing = self.checkpoints.get_checkpoint(plan.name)
            if existing is not None and self._installed_digest(existing.path) == (
                plan.manifest.content_sha256
            ):
                return {"import": ImportOutcome(
                    checkpoint=plan.name,
                    content_sha256=plan.manifest.content_sha256,
                    status="already_imported",
                ).model_dump()}
            CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
            staging.mkdir()
            digest, manifest = self._extract_and_verify(
                source, plan, staging, should_stop, job_id
            )
            self._stop_check(should_stop)
            self._assert_metadata(manifest, staging)
            installed = self._install_shared_resources(manifest, staging)
            self._progress(job_id, 80)
            self._smoke_test(staging)
            self._progress(job_id, 90)
            self._stop_check(should_stop)
            checkpoint_dir = staging / "checkpoint"
            (checkpoint_dir / EXCHANGE_RECORD).write_text(
                json.dumps(
                    {
                        "content_sha256": digest,
                        "imported_utc": datetime.now(timezone.utc).isoformat(),
                        "manifest": manifest.model_dump(),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            try:
                os.rename(checkpoint_dir, final)
            except OSError as exc:
                raise JobExecutionError(
                    f"Could not publish checkpoint {plan.name}: {exc}",
                    code="publish_failed",
                    phase="publish",
                ) from exc
            published = True
            try:
                self.checkpoints.register_completed_checkpoint(plan.name, final)
            except CheckpointRegistryError as exc:
                raise JobExecutionError(
                    str(exc), code="registration_failed", phase="register"
                ) from exc
        except ArchiveCancelled:
            self._rollback(final, published, installed)
            return {}
        except Exception:
            self._rollback(final, published, installed)
            raise
        finally:
            shutil.rmtree(staging, ignore_errors=True)
            source.unlink(missing_ok=True)
        logger.info("Imported checkpoint %s (%s)", plan.name, digest[:12])
        outcome = ImportOutcome(
            checkpoint=plan.name,
            content_sha256=digest,
            status="imported",
            renamed_from=plan.renamed_from,
        )
        return {"import": outcome.model_dump()}

    # -------------------------------------------------------------------------
    @staticmethod
    def _rollback(final: Path, published: bool, installed: list[Path]) -> None:
        """Undo only what this import created; pre-existing resources stay untouched."""
        if published:
            shutil.rmtree(final, ignore_errors=True)
        for path in installed:
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)

    # -------------------------------------------------------------------------
    def _extract_and_verify(
        self,
        source: Path,
        plan: ImportPlan,
        staging: Path,
        should_stop: Callable[[], bool] | None,
        job_id: str | None,
    ) -> tuple[str, ExchangeManifest]:
        try:
            with zipfile.ZipFile(source) as archive:
                entries = inspect_archive(archive)
                payload = [name for name in entries if name != MANIFEST_NAME]
                total = sum(entries[name].size for name in payload) or 1
                done = 0

                def advance(count: int) -> None:
                    nonlocal done
                    done += count
                    self._progress(job_id, 5 + 70 * done / total)

                declared = {item.path: item.size for item in plan.manifest.files}
                hashes: dict[str, str] = {}
                for name in payload:
                    expected = declared.get(name, entries[name].size)
                    if name.startswith(("tokenizer/", "encoder/")):
                        target = staging / name
                    else:
                        target = staging / "checkpoint" / name
                    hashes[name] = extract_member(
                        archive, name, target, expected, should_stop, advance
                    )
        except zipfile.BadZipFile as exc:
            raise JobExecutionError(
                "The archive is corrupt", code="archive_invalid", phase="extract"
            ) from exc
        except ArchiveError as exc:
            raise JobExecutionError(
                str(exc), code="archive_invalid", phase="extract"
            ) from exc

        for item in plan.manifest.files:
            if hashes.get(item.path) != item.sha256:
                raise JobExecutionError(
                    f"Checksum mismatch for {item.path}",
                    code="checksum_mismatch",
                    phase="verify",
                )
        return plan.manifest.content_sha256, plan.manifest

    # -------------------------------------------------------------------------
    def _install_shared_resources(
        self, manifest: ExchangeManifest, staging: Path
    ) -> list[Path]:
        """Install bundled tokenizer/encoder files; identical content is a no-op.

        Returns the paths newly created so a failed import can remove them again.
        """
        created: list[Path] = []
        tokenizer_source = staging / "tokenizer"
        if tokenizer_source.is_dir():
            target = tokenizer_exchange_dir(manifest.tokenizer.name)
            if target.is_dir():
                if _tree_digest(target) != _tree_digest(tokenizer_source):
                    raise JobExecutionError(
                        f"A different tokenizer '{manifest.tokenizer.name}' is already "
                        "installed; refusing to overwrite it",
                        code="tokenizer_conflict",
                        phase="compatibility",
                    )
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(tokenizer_source, target)
                created.append(target)
        encoder_source = staging / "encoder"
        if encoder_source.is_dir() and not encoder_installed(manifest.encoder.revision):
            target = encoder_snapshot_dir(manifest.encoder.revision)
            if target.exists():
                shutil.rmtree(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(encoder_source, target)
            created.append(target)
            refs = target.parent.parent / "refs"
            refs.mkdir(parents=True, exist_ok=True)
            if not (refs / "main").exists():
                (refs / "main").write_text(manifest.encoder.revision, encoding="utf-8")
                created.append(refs / "main")
        if not encoder_installed(manifest.encoder.revision):
            self._rollback(CHECKPOINTS_DIR / ".none", False, created)
            raise JobExecutionError(
                "The pinned BEiT encoder is not installed (offline). Install it locally "
                "or re-export the checkpoint with the encoder included.",
                code="encoder_missing",
                phase="compatibility",
            )
        return created

    # -------------------------------------------------------------------------
    @staticmethod
    def _assert_metadata(manifest: ExchangeManifest, staging: Path) -> None:
        """Check the extracted metadata agrees with the manifest before anything is installed."""
        try:
            metadata = json.loads(
                (staging / "checkpoint" / "configuration" / "metadata.json").read_text("utf-8")
            )
            tokenizer = metadata["tokenizer"]
            max_report_size = metadata["max_report_size"]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise JobExecutionError(
                "The checkpoint metadata must define 'tokenizer' and 'max_report_size'.",
                code="metadata_invalid",
                phase="verify",
            ) from exc
        if tokenizer != manifest.tokenizer.name:
            raise JobExecutionError(
                "The checkpoint metadata names a different tokenizer than the manifest.",
                code="metadata_invalid",
                phase="verify",
            )
        for key in ("pad_token_id", "start_token_id", "end_token_id", "vocabulary_size"):
            if getattr(manifest.tokenizer, key) is None or getattr(manifest.tokenizer, key) != metadata.get(key):
                raise JobExecutionError(
                    f"The checkpoint metadata and the manifest disagree on '{key}'.",
                    code="metadata_invalid",
                    phase="verify",
                )
        if isinstance(max_report_size, bool) or not isinstance(max_report_size, int) or max_report_size < 1:
            raise JobExecutionError(
                "The checkpoint metadata has an invalid 'max_report_size'.",
                code="metadata_invalid",
                phase="verify",
            )

    # -------------------------------------------------------------------------
    @staticmethod
    def _smoke_test(staging: Path) -> None:
        """Load the checkpoint and run one forward pass before it is registered."""
        import numpy as np
        from keras import ops

        from xreport_shared.models.training.processing import TokenizerHandler
        from xreport_shared.repositories.serialization.model import ModelSerializer

        checkpoint_dir = staging / "checkpoint"
        try:
            _, metadata, _ = ModelSerializer().load_training_configuration(
                checkpoint_dir
            )
            handler = TokenizerHandler(metadata)
            if handler.tokenizer is None:
                raise RuntimeError("tokenizer unavailable")
        except Exception as exc:  # noqa: BLE001
            raise JobExecutionError(
                "The tokenizer could not be loaded (it may need network access): "
                f"{str(exc)[:120]}",
                code="tokenizer_missing",
                phase="smoke_test",
            ) from exc
        try:
            model, _, metadata, _, _ = ModelSerializer().load_checkpoint(checkpoint_dir)
            max_report_size = int(metadata.get("max_report_size", 200))
            sequence = np.full(
                (1, max_report_size), metadata["pad_token_id"], dtype=np.int32
            )
            sequence[0, 0] = metadata["start_token_id"]
            image = np.zeros((1, 224, 224, 3), dtype=np.float32)
            # Keras conversion handles CUDA tensors, unlike np.asarray.
            output = np.asarray(
                ops.convert_to_numpy(model([image, sequence], training=False))
            )
            if output.shape[-1] != handler.vocabulary_size or not np.isfinite(output).all():
                raise ValueError("model output does not match the tokenizer vocabulary")
        except Exception as exc:  # noqa: BLE001
            raise JobExecutionError(
                f"The checkpoint failed its load and inference check: {str(exc)[:160]}",
                code="smoke_test_failed",
                phase="smoke_test",
            ) from exc

    # -------------------------------------------------------------------------
    def _stop_callback(self, job_id: str | None) -> Callable[[], bool] | None:
        if job_id is None:
            return None
        return lambda: self.jobs.should_stop(job_id)

    # -------------------------------------------------------------------------
    @staticmethod
    def _stop_check(should_stop: Callable[[], bool] | None) -> None:
        if should_stop is not None and should_stop():
            raise ArchiveCancelled("Import was cancelled")

    # -------------------------------------------------------------------------
    def _progress(self, job_id: str | None, value: float) -> None:
        if job_id is not None:
            self.jobs.update_progress(job_id, value)

###############################################################################
def _tree_digest(directory: Path) -> str:
    files = [
        ManifestFile(
            path=path.relative_to(directory).as_posix(),
            size=path.stat().st_size,
            sha256=sha256_file(path),
        )
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    ]
    return content_digest(files)

###############################################################################
@lru_cache(maxsize=1)
def get_checkpoint_exchange_service() -> CheckpointExchangeService:
    return CheckpointExchangeService()

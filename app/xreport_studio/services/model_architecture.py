from __future__ import annotations

from functools import lru_cache
from typing import Any

from pydantic import ValidationError

from xreport_shared.common.utils.security import validate_checkpoint_name
from xreport_studio.domain.model_architecture import (
    ArchitectureCompatibility,
    ArchitectureIssue,
    ArchitecturePreset,
    ArchitecturePresetRequest,
    ArchitecturePresetsResponse,
    ArchitecturePreviewRequest,
    ArchitecturePreviewResponse,
    CheckpointArchitectureResponse,
    EncodersResponse,
    ModelArchitecture,
)
from xreport_studio.models.training.architecture import (
    BUILTIN_PRESETS,
    DEFAULT_PRESET_KEY,
    REFERENCE_SEQUENCE_LENGTH,
    REFERENCE_VOCABULARY_SIZE,
    SHAPE_FIELDS,
    architecture_from_configuration,
    compare_architectures,
    compare_weights,
    describe,
    encoder_options,
)
from xreport_shared.repositories.checkpoints import CheckpointRepository
from xreport_studio.repositories.model_architecture import (
    ModelArchitectureRepository,
    PresetNameConflictError,
    PresetRecord,
)
from xreport_shared.services.errors import (
    BadRequestError,
    ConflictError,
    InternalServiceError,
    NotFoundError,
)

CUSTOM_PRESET_PREFIX = "custom-"
CHECKPOINT_REFERENCE_PREFIX = "checkpoint:"


###############################################################################
class ModelArchitectureService:

    # -------------------------------------------------------------------------
    def __init__(
        self,
        repository: ModelArchitectureRepository,
        checkpoint_repository: CheckpointRepository,
    ) -> None:
        self.repository = repository
        self.checkpoint_repository = checkpoint_repository

    # -------------------------------------------------------------------------
    @staticmethod
    def _custom_key(preset_id: int) -> str:
        return f"{CUSTOM_PRESET_PREFIX}{preset_id}"

    # -------------------------------------------------------------------------
    @staticmethod
    def _custom_id(key: str) -> int | None:
        suffix = key.removeprefix(CUSTOM_PRESET_PREFIX)
        if key == suffix or not suffix.isdigit():
            return None
        return int(suffix)

    # -------------------------------------------------------------------------
    def _custom_preset(self, record: PresetRecord) -> ArchitecturePreset:
        _, stats = describe(record.architecture)
        return ArchitecturePreset(
            key=self._custom_key(record.preset_id),
            name=record.name,
            description=record.description,
            builtin=False,
            architecture=record.architecture,
            stats=stats,
            created_at=record.created_at,
            updated_at=record.updated_at,
        )

    # -------------------------------------------------------------------------
    def _assert_not_reserved(self, name: str) -> None:
        reserved = {preset.name.casefold() for preset in BUILTIN_PRESETS.values()}
        if name.strip().casefold() in reserved:
            raise ConflictError(detail=f"'{name.strip()}' is a built-in preset name")

    # -------------------------------------------------------------------------
    def get_encoders(self) -> EncodersResponse:
        return EncodersResponse(encoders=encoder_options())

    # -------------------------------------------------------------------------
    def list_presets(self) -> ArchitecturePresetsResponse:
        hidden = (self.repository.hidden_builtin_keys() & BUILTIN_PRESETS.keys()) - {
            DEFAULT_PRESET_KEY
        }
        builtins = [
            ArchitecturePreset(
                key=preset.key,
                name=preset.name,
                description=preset.description,
                builtin=True,
                protected=preset.key == DEFAULT_PRESET_KEY,
                architecture=preset.architecture,
                stats=describe(preset.architecture)[1],
            )
            for preset in BUILTIN_PRESETS.values()
            if preset.key not in hidden
        ]
        custom = [self._custom_preset(record) for record in self.repository.list_presets()]
        return ArchitecturePresetsResponse(
            presets=[*builtins, *custom], hidden_builtins=len(hidden)
        )

    # -------------------------------------------------------------------------
    def resolve_preset(self, key: str) -> tuple[str, ModelArchitecture]:
        """Return the name and architecture of a built-in or custom preset."""
        builtin = BUILTIN_PRESETS.get(key)
        if builtin is not None:
            return builtin.name, builtin.architecture
        preset_id = self._custom_id(key)
        record = self.repository.get_preset(preset_id) if preset_id is not None else None
        if record is None:
            raise NotFoundError(detail=f"Architecture preset not found: {key}")
        return record.name, record.architecture

    # -------------------------------------------------------------------------
    def create_preset(self, request: ArchitecturePresetRequest) -> ArchitecturePreset:
        self._assert_not_reserved(request.name)
        try:
            record = self.repository.create_preset(
                request.name, request.description, request.architecture
            )
        except PresetNameConflictError as exc:
            raise ConflictError(detail=str(exc)) from exc
        return self._custom_preset(record)

    # -------------------------------------------------------------------------
    def update_preset(
        self, key: str, request: ArchitecturePresetRequest
    ) -> ArchitecturePreset:
        if key in BUILTIN_PRESETS:
            raise BadRequestError(detail="Built-in presets cannot be modified")
        preset_id = self._custom_id(key)
        self._assert_not_reserved(request.name)
        try:
            record = (
                self.repository.update_preset(
                    preset_id, request.name, request.description, request.architecture
                )
                if preset_id is not None
                else None
            )
        except PresetNameConflictError as exc:
            raise ConflictError(detail=str(exc)) from exc
        if record is None:
            raise NotFoundError(detail=f"Architecture preset not found: {key}")
        return self._custom_preset(record)

    # -------------------------------------------------------------------------
    def delete_preset(self, key: str) -> None:
        if key == DEFAULT_PRESET_KEY:
            raise BadRequestError(detail="The XREPORT default preset cannot be deleted")
        if key in BUILTIN_PRESETS:
            self.repository.hide_builtin(key)
            return
        preset_id = self._custom_id(key)
        if preset_id is None or not self.repository.delete_preset(preset_id):
            raise NotFoundError(detail=f"Architecture preset not found: {key}")

    # -------------------------------------------------------------------------
    def restore_builtin_presets(self) -> ArchitecturePresetsResponse:
        self.repository.restore_builtins()
        return self.list_presets()

    # -------------------------------------------------------------------------
    def get_checkpoint_architecture(
        self, checkpoint: str
    ) -> CheckpointArchitectureResponse:
        from xreport_shared.repositories.serialization.model import ModelSerializer

        try:
            checkpoint = validate_checkpoint_name(checkpoint)
        except ValueError as exc:
            raise BadRequestError(detail=str(exc)) from exc
        record = self.checkpoint_repository.get_checkpoint(checkpoint)
        if record is None:
            raise NotFoundError(detail=f"Checkpoint is not registered: {checkpoint}")
        if not record.artifact_complete:
            raise InternalServiceError(
                detail=f"Checkpoint artifact is missing or incomplete: {checkpoint}"
            )
        try:
            configuration, metadata, _ = ModelSerializer().load_training_configuration(
                record.path
            )
            architecture, legacy = architecture_from_configuration(configuration)
        except (OSError, ValueError) as exc:
            raise InternalServiceError(
                detail=f"Failed to reconstruct checkpoint architecture: {exc}"
            ) from exc

        preset = configuration.get("architecture_preset")
        summary = configuration.get("architecture_summary")
        return CheckpointArchitectureResponse(
            checkpoint=checkpoint,
            architecture=architecture,
            preset_name=preset.get("name") if isinstance(preset, dict) else None,
            legacy=legacy,
            measured_parameters=(
                summary.get("total_parameters") if isinstance(summary, dict) else None
            ),
            vocabulary_size=self._metadata_int(metadata, "vocabulary_size"),
            sequence_length=self._metadata_int(metadata, "max_report_size"),
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def _metadata_int(metadata: Any, key: str) -> int | None:
        value = metadata.get(key) if isinstance(metadata, dict) else None
        return value if isinstance(value, int) else None

    # -------------------------------------------------------------------------
    def _compare(
        self, architecture: ModelArchitecture, reference: str
    ) -> ArchitectureCompatibility:
        if not reference.startswith(CHECKPOINT_REFERENCE_PREFIX):
            name, other = self.resolve_preset(reference)
            return compare_architectures(architecture, other, f"preset {name}", "preset")

        result = self.get_checkpoint_architecture(
            reference.removeprefix(CHECKPOINT_REFERENCE_PREFIX)
        )
        label = f"checkpoint {result.checkpoint}"
        report = compare_architectures(
            architecture, result.architecture, label, "checkpoint"
        )
        if result.vocabulary_size is not None and result.sequence_length is not None:
            report.weights = compare_weights(
                architecture,
                result.architecture,
                result.vocabulary_size,
                result.sequence_length,
                label,
            )
        return report

    # -------------------------------------------------------------------------
    def preview(self, request: ArchitecturePreviewRequest) -> ArchitecturePreviewResponse:
        try:
            architecture = ModelArchitecture.model_validate(request.architecture)
        except ValidationError as exc:
            return ArchitecturePreviewResponse(
                valid=False,
                issues=[
                    ArchitectureIssue(
                        field=str(error["loc"][0]) if error["loc"] else None,
                        message=str(error["msg"]).removeprefix("Value error, "),
                    )
                    for error in exc.errors()
                ],
            )

        graph, stats = describe(
            architecture, REFERENCE_VOCABULARY_SIZE, REFERENCE_SEQUENCE_LENGTH
        )
        return ArchitecturePreviewResponse(
            valid=True,
            issues=[],
            architecture=architecture,
            graph=graph,
            stats=stats,
            compatibility=(
                self._compare(architecture, request.reference)
                if request.reference
                else None
            ),
            shape_fields=list(SHAPE_FIELDS),
        )


###############################################################################
@lru_cache(maxsize=1)
def get_model_architecture_service() -> ModelArchitectureService:
    return ModelArchitectureService(
        repository=ModelArchitectureRepository(),
        checkpoint_repository=CheckpointRepository(),
    )

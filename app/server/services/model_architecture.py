from __future__ import annotations

from functools import lru_cache

from pydantic import ValidationError

from server.common.utils.security import validate_checkpoint_name
from server.domain.model_architecture import (
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
from server.models.training.architecture import (
    BUILTIN_PRESETS,
    REFERENCE_SEQUENCE_LENGTH,
    REFERENCE_VOCABULARY_SIZE,
    architecture_from_configuration,
    compatibility,
    describe,
    encoder_options,
)
from server.repositories.checkpoints import CheckpointRepository
from server.repositories.model_architecture import (
    ModelArchitectureRepository,
    PresetNameConflictError,
    PresetRecord,
)
from server.repositories.serialization.dataset import DatasetRepository
from server.services.errors import (
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
        builtins = [
            ArchitecturePreset(
                key=preset.key,
                name=preset.name,
                description=preset.description,
                builtin=True,
                architecture=preset.architecture,
                stats=describe(preset.architecture)[1],
            )
            for preset in BUILTIN_PRESETS.values()
        ]
        custom = [self._custom_preset(record) for record in self.repository.list_presets()]
        return ArchitecturePresetsResponse(presets=[*builtins, *custom])

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
        if key in BUILTIN_PRESETS:
            raise BadRequestError(detail="Built-in presets cannot be deleted")
        preset_id = self._custom_id(key)
        if preset_id is None or not self.repository.delete_preset(preset_id):
            raise NotFoundError(detail=f"Architecture preset not found: {key}")

    # -------------------------------------------------------------------------
    def get_checkpoint_architecture(
        self, checkpoint: str
    ) -> CheckpointArchitectureResponse:
        from server.repositories.serialization.model import ModelSerializer

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
            configuration, _, _ = ModelSerializer().load_training_configuration(
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
        )

    # -------------------------------------------------------------------------
    def _reference_architecture(self, reference: str) -> tuple[str, ModelArchitecture]:
        if reference.startswith(CHECKPOINT_REFERENCE_PREFIX):
            name = reference.removeprefix(CHECKPOINT_REFERENCE_PREFIX)
            result = self.get_checkpoint_architecture(name)
            return f"checkpoint {result.checkpoint}", result.architecture
        name, architecture = self.resolve_preset(reference)
        return f"preset {name}", architecture

    # -------------------------------------------------------------------------
    def _dataset_dimensions(self, dataset_name: str | None) -> tuple[int, int]:
        if dataset_name is None:
            return REFERENCE_VOCABULARY_SIZE, REFERENCE_SEQUENCE_LENGTH
        metadata = DatasetRepository().load_training_data(
            only_metadata=True, dataset_name=dataset_name
        )
        vocabulary = metadata.get("vocabulary_size") if metadata else None
        length = metadata.get("max_report_size") if metadata else None
        if not isinstance(vocabulary, int) or not isinstance(length, int):
            raise NotFoundError(
                detail=f"Processed dataset metadata not found: {dataset_name}"
            )
        return vocabulary, length

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

        vocabulary, length = self._dataset_dimensions(request.dataset_name)
        graph, stats = describe(architecture, vocabulary, length)
        report = None
        if request.reference:
            label, reference = self._reference_architecture(request.reference)
            report = compatibility(architecture, reference, label)
        return ArchitecturePreviewResponse(
            valid=True,
            issues=[],
            architecture=architecture,
            graph=graph,
            stats=stats,
            compatibility=report,
        )

###############################################################################
@lru_cache(maxsize=1)
def get_model_architecture_service() -> ModelArchitectureService:
    return ModelArchitectureService(
        repository=ModelArchitectureRepository(),
        checkpoint_repository=CheckpointRepository(),
    )

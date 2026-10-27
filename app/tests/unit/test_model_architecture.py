from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from server.domain.model_architecture import (
    ArchitecturePresetRequest,
    ArchitecturePreviewRequest,
    ModelArchitecture,
)
from server.models.training import architecture as arch
from server.models.training.layers import (
    PositionalEmbedding,
    SoftMaxClassifier,
    TransformerDecoder,
    TransformerEncoder,
)
from server.repositories.database.engine import Database
from server.repositories.model_architecture import ModelArchitectureRepository
from server.repositories.schemas.studio import StudioBase
from server.services.errors import BadRequestError, ConflictError, NotFoundError
from server.services.model_architecture import ModelArchitectureService

WIDTH = 64
TOKENS = 5

###############################################################################
def _memory_database() -> Database:
    database = Database.__new__(Database)
    database.engine = create_engine("sqlite:///:memory:", future=True)
    database.session = sessionmaker(
        bind=database.engine, future=True, expire_on_commit=False
    )
    database.insert_batch_size = 100
    StudioBase.metadata.create_all(database.engine)
    return database

###############################################################################
def _service() -> ModelArchitectureService:
    return ModelArchitectureService(
        repository=ModelArchitectureRepository(database=_memory_database()),
        checkpoint_repository=None,  # type: ignore[arg-type]
    )

###############################################################################
def test_split_heads_require_divisible_width_and_minimum_head_size() -> None:
    with pytest.raises(ValidationError, match="divisible"):
        ModelArchitecture(embedding_dims=100, attention_heads=8, head_dim_mode="split")
    with pytest.raises(ValidationError, match="at least"):
        ModelArchitecture(embedding_dims=64, attention_heads=16, head_dim_mode="split")
    # full-width heads have no divisibility constraint (original XREPORT design)
    assert ModelArchitecture(embedding_dims=100, attention_heads=8).key_dim == 100
    with pytest.raises(ValidationError):
        ModelArchitecture.model_validate({"image_encoder": "resnet50"})

###############################################################################
@pytest.mark.parametrize(
    ("head_dim_mode", "ffn_units"), [("full", WIDTH), ("split", 3 * WIDTH)]
)
def test_analytic_counts_match_built_keras_layers(
    head_dim_mode: str, ffn_units: int
) -> None:
    architecture = ModelArchitecture(
        embedding_dims=WIDTH,
        attention_heads=4,
        head_dim_mode=head_dim_mode,  # type: ignore[arg-type]
        ffn_units=ffn_units,
        classifier_units=96,
    )
    hidden = np.zeros((2, TOKENS, WIDTH), dtype="float32")
    encoder = TransformerEncoder(seed=1, **arch.layer_kwargs(architecture))
    decoder = TransformerDecoder(seed=1, **arch.layer_kwargs(architecture))
    encoder(hidden)
    decoder(hidden, hidden)
    embedding = PositionalEmbedding(50, WIDTH, TOKENS)
    embedding(np.ones((2, TOKENS), dtype="int32"))
    classifier = SoftMaxClassifier(96, 50)
    classifier(hidden)

    assert encoder.count_params() == arch.encoder_block_parameters(architecture)
    assert decoder.count_params() == arch.decoder_block_parameters(architecture)
    assert embedding.count_params() == arch.embedding_parameters(architecture, 50, TOKENS)
    assert classifier.count_params() == arch.classifier_parameters(architecture, 50)

###############################################################################
def test_legacy_layer_config_rebuilds_original_layers() -> None:
    legacy = {"name": "encoder", "embedding_dims": WIDTH, "num_heads": 4, "seed": 1}
    layer = TransformerEncoder.from_config(legacy)
    layer(np.zeros((2, TOKENS, WIDTH), dtype="float32"))
    original = ModelArchitecture(embedding_dims=WIDTH, attention_heads=4, ffn_units=WIDTH)

    assert (layer.key_dim, layer.ffn_units, layer.dropout) == (WIDTH, WIDTH, 0.2)
    assert layer.count_params() == arch.encoder_block_parameters(original)

###############################################################################
def test_default_preset_matches_original_studio_architecture() -> None:
    default = arch.BUILTIN_PRESETS["xreport-default"].architecture
    configuration = {
        "num_encoders": 6,
        "num_decoders": 6,
        "embedding_dims": 768,
        "attention_heads": 8,
        "train_temp": 1.0,
        "freeze_img_encoder": True,
    }
    reconstructed, legacy = arch.architecture_from_configuration(configuration)

    assert legacy is True
    assert reconstructed == default
    assert arch.architecture_from_configuration(
        {"architecture": default.model_dump()}
    ) == (default, False)

###############################################################################
def test_compatibility_separates_training_only_changes_from_shape_changes() -> None:
    base = ModelArchitecture()

    assert arch.compatibility(base, base, "base").level == "identical"
    tuned = base.model_copy(update={"dropout": 0.1, "temperature": 0.8})
    assert arch.compatibility(tuned, base, "base").level == "weights_compatible"
    deeper = base.model_copy(update={"num_decoders": 8, "dropout": 0.1})
    report = arch.compatibility(deeper, base, "base")
    assert report.level == "retrain_required"
    assert {change.field: change.impact for change in report.changes} == {
        "num_decoders": "retrain_required",
        "dropout": "weights_compatible",
    }

###############################################################################
def test_frozen_encoder_is_excluded_from_trainable_parameters() -> None:
    frozen = ModelArchitecture(freeze_image_encoder=True)
    _, stats = arch.describe(frozen)
    _, unfrozen = arch.describe(frozen.model_copy(update={"freeze_image_encoder": False}))

    beit = arch.ENCODER_REGISTRY[frozen.image_encoder].parameters
    assert stats.total_parameters == unfrozen.total_parameters
    assert stats.trainable_parameters == unfrozen.trainable_parameters - beit
    assert stats.frozen_parameters == beit

###############################################################################
def test_beit_parameter_constant_matches_installed_encoder() -> None:
    from server.common.path import ENCODERS_DIR
    from server.models.training.encoder import BeitXRayImageEncoder

    transformers = pytest.importorskip("transformers")
    try:
        model = transformers.AutoModel.from_pretrained(
            "microsoft/beit-base-patch16-224",
            revision=BeitXRayImageEncoder.ENCODER_REVISION,
            cache_dir=ENCODERS_DIR,
            local_files_only=True,
        )
    except OSError:
        pytest.skip("Pinned BEiT encoder is not installed")
    spec = arch.ENCODER_REGISTRY["beit-base-patch16-224"]

    assert sum(parameter.numel() for parameter in model.parameters()) == spec.parameters
    assert model.config.hidden_size == spec.feature_dims

###############################################################################
def test_custom_presets_persist_and_protect_names() -> None:
    service = _service()
    compact = arch.BUILTIN_PRESETS["compact"].architecture
    created = service.create_preset(
        ArchitecturePresetRequest(name="Narrow", description="", architecture=compact)
    )
    assert service.resolve_preset(created.key) == ("Narrow", compact)

    with pytest.raises(ConflictError):
        service.create_preset(
            ArchitecturePresetRequest(name=" narrow ", architecture=compact)
        )
    with pytest.raises(ConflictError):
        service.create_preset(
            ArchitecturePresetRequest(name="XREPORT Default", architecture=compact)
        )
    with pytest.raises(BadRequestError):
        service.delete_preset("compact")

    wider = compact.model_copy(update={"embedding_dims": 512})
    updated = service.update_preset(
        created.key, ArchitecturePresetRequest(name="Wide", architecture=wider)
    )
    assert (updated.name, updated.architecture.embedding_dims) == ("Wide", 512)
    keys = [preset.key for preset in service.list_presets().presets]
    assert keys[: len(arch.BUILTIN_PRESETS)] == list(arch.BUILTIN_PRESETS)
    assert created.key in keys

    service.delete_preset(created.key)
    with pytest.raises(NotFoundError):
        service.resolve_preset(created.key)

###############################################################################
def test_preview_reports_field_issues_instead_of_failing() -> None:
    service = _service()
    invalid = service.preview(
        ArchitecturePreviewRequest(
            architecture={"embedding_dims": 100, "attention_heads": 8, "head_dim_mode": "split"}
        )
    )
    assert invalid.valid is False
    assert invalid.graph is None
    assert "divisible" in invalid.issues[0].message

    valid = service.preview(
        ArchitecturePreviewRequest(architecture={"num_decoders": 2}, reference="xreport-default")
    )
    assert valid.valid is True
    assert valid.stats is not None and valid.stats.depth == 8
    assert valid.compatibility is not None
    assert valid.compatibility.level == "retrain_required"

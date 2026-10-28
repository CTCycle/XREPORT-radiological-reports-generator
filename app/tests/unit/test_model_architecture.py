from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xreport_studio.domain.model_architecture import (
    ArchitecturePresetRequest,
    ArchitecturePreviewRequest,
    ModelArchitecture,
)
from xreport_studio.models.training import architecture as arch
from xreport_shared.models.training.layers import (
    PositionalEmbedding,
    SoftMaxClassifier,
    TransformerDecoder,
    TransformerEncoder,
)
from xreport_shared.repositories.database.engine import Database
from xreport_studio.repositories.model_architecture import ModelArchitectureRepository
from xreport_studio.repositories.schemas.studio import StudioBase
from xreport_shared.services.errors import BadRequestError, ConflictError, NotFoundError
from xreport_studio.services.model_architecture import ModelArchitectureService

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
def test_stack_heads_validate_independently_of_each_other() -> None:
    with pytest.raises(ValidationError, match="divisible"):
        ModelArchitecture(embedding_dims=100, encoder_heads=8, encoder_head_dim_mode="split")
    with pytest.raises(ValidationError, match="at least"):
        ModelArchitecture(embedding_dims=64, decoder_heads=16, decoder_head_dim_mode="split")
    # the decoder keeps its full-width heads (no divisibility constraint)
    mixed = ModelArchitecture(
        embedding_dims=96, encoder_heads=4, encoder_head_dim_mode="split", decoder_heads=7
    )
    assert (mixed.key_dim("encoder"), mixed.key_dim("decoder")) == (24, 96)
    assert ModelArchitecture(decoder_head_dim_mode="custom", decoder_head_dim=32).key_dim("decoder") == 32
    with pytest.raises(ValidationError):
        ModelArchitecture.model_validate({"image_encoder": "resnet50"})


###############################################################################
def test_shared_keys_of_saved_architectures_apply_to_both_stacks() -> None:
    migrated = ModelArchitecture.model_validate(
        {"attention_heads": 4, "head_dim_mode": "split", "ffn_units": 512, "dropout": 0.1,
         "embedding_dims": 256}
    )
    assert (migrated.encoder_heads, migrated.decoder_heads) == (4, 4)
    assert (migrated.encoder_ffn_units, migrated.decoder_ffn_units) == (512, 512)
    assert migrated.decoder_head_dim_mode == "split"
    assert ModelArchitecture.model_validate(migrated.model_dump()) == migrated

###############################################################################
CONFIGURATIONS = [
    {"embedding_dims": WIDTH, "encoder_heads": 4, "encoder_ffn_units": WIDTH,
     "decoder_heads": 4, "decoder_ffn_units": WIDTH},
    {"embedding_dims": WIDTH, "encoder_heads": 4, "encoder_head_dim_mode": "split",
     "encoder_ffn_units": 3 * WIDTH, "encoder_ffn_activation": "gelu",
     "decoder_heads": 2, "decoder_head_dim_mode": "custom", "decoder_head_dim": 24,
     "decoder_ffn_units": 80, "decoder_ffn_activation": "silu",
     "classifier_units": 96, "classifier_activation": "gelu", "norm_epsilon": 1e-5},
]


###############################################################################
@pytest.mark.parametrize("settings", CONFIGURATIONS)
def test_analytic_counts_and_shapes_match_built_keras_layers(settings: dict) -> None:
    architecture = ModelArchitecture(num_encoders=1, num_decoders=1, **settings)
    hidden = np.zeros((2, TOKENS, WIDTH), dtype="float32")
    encoder = TransformerEncoder(seed=1, **arch.stack_kwargs(architecture, "encoder"))
    decoder = TransformerDecoder(seed=1, **arch.stack_kwargs(architecture, "decoder"))
    encoder(hidden)
    decoder(hidden, hidden)
    embedding = PositionalEmbedding(50, WIDTH, TOKENS)
    embedding(np.ones((2, TOKENS), dtype="int32"))
    classifier = SoftMaxClassifier(
        architecture.classifier_units, 50, 1.0, architecture.classifier_activation
    )
    classifier(hidden)

    assert encoder.count_params() == arch.encoder_block_parameters(architecture)
    assert decoder.count_params() == arch.decoder_block_parameters(architecture)
    assert embedding.count_params() == arch.embedding_parameters(architecture, 50, TOKENS)
    assert classifier.count_params() == arch.classifier_parameters(architecture, 50)

    shapes = arch.weight_shapes(architecture, 50, TOKENS)
    built = {tuple(weight.shape) for layer in (encoder, decoder, embedding, classifier)
             for weight in layer.weights}
    stack_shapes = {shape for (group, _, _), shape in shapes.items()
                    if group in {"encoder", "decoder", "embedding", "classifier"}}
    assert stack_shapes == built
    assert sum(int(np.prod(shape)) for key, shape in shapes.items()
               if key[0] != "projection") == (
        encoder.count_params() + decoder.count_params()
        + embedding.count_params() + classifier.count_params()
    )


###############################################################################
def test_layer_options_change_the_computation() -> None:
    from keras import ops

    hidden = np.random.default_rng(0).normal(size=(2, TOKENS, WIDTH)).astype("float32")
    relu = TransformerEncoder(embedding_dims=WIDTH, num_heads=2, seed=1)
    gelu = TransformerEncoder(embedding_dims=WIDTH, num_heads=2, seed=1, activation="gelu")
    relu(hidden), gelu(hidden)
    gelu.set_weights(relu.get_weights())

    assert not np.allclose(
        ops.convert_to_numpy(relu(hidden)), ops.convert_to_numpy(gelu(hidden))
    )
    assert TransformerEncoder(embedding_dims=WIDTH, num_heads=2, seed=1,
                              norm_epsilon=1e-2).addnorm1.epsilon == 1e-2


###############################################################################
def test_legacy_layer_config_rebuilds_original_layers() -> None:
    legacy = {"name": "encoder", "embedding_dims": WIDTH, "num_heads": 4, "seed": 1}
    layer = TransformerEncoder.from_config(legacy)
    layer(np.zeros((2, TOKENS, WIDTH), dtype="float32"))
    original = ModelArchitecture(embedding_dims=WIDTH, encoder_heads=4, encoder_ffn_units=WIDTH)

    assert (layer.key_dim, layer.ffn_units, layer.dropout, layer.activation) == (
        WIDTH, WIDTH, 0.2, "relu")
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
def test_comparison_separates_structure_from_configurable_settings() -> None:
    base = ModelArchitecture()

    def compare(candidate: ModelArchitecture):
        return arch.compare_architectures(candidate, base, "base", "preset")

    assert compare(base).level == "identical"
    tuned = base.model_copy(update={"encoder_dropout": 0.1, "temperature": 0.8,
                                    "decoder_ffn_activation": "gelu"})
    assert compare(tuned).level == "compatible"
    # a custom head size equal to the width, or unused, is no difference at all
    assert compare(base.model_copy(update={"encoder_head_dim_mode": "custom",
                                           "encoder_head_dim": 768})).level == "identical"
    assert compare(base.model_copy(update={"decoder_head_dim": 16})).level == "identical"

    deeper = base.model_copy(update={"num_decoders": 8, "decoder_dropout": 0.1})
    report = compare(deeper)
    assert report.level == "mismatch"
    assert {change.field: change.scope for change in report.changes} == {
        "num_decoders": "shape",
        "decoder_dropout": "behaviour",
    }


###############################################################################
def test_checkpoint_weights_report_the_tensors_that_cannot_load() -> None:
    base = ModelArchitecture()
    same = arch.compare_weights(base, base, 1000, 50, "checkpoint a")
    assert same.status == "loadable" and not same.mismatches

    tuned = base.model_copy(update={"encoder_dropout": 0.0, "temperature": 0.5})
    assert arch.compare_weights(tuned, base, 1000, 50, "checkpoint a").status == "loadable"

    changed = base.model_copy(update={"embedding_dims": 512, "num_decoders": 4})
    report = arch.compare_weights(changed, base, 1000, 50, "checkpoint a")
    assert report.status == "incompatible"
    layers = {(item.layer, item.tensor): item for item in report.mismatches}
    token = layers[("embedding", "token_embeddings")]
    assert (token.expected, token.found) == ([1000, 512], [1000, 768])
    # decoders 5-6 exist only in the checkpoint
    missing = layers[("decoder 5-6", "ffn/dense1/kernel")]
    assert (missing.expected, missing.found) == ([], [768, 768])
    assert ("encoder 1-6", "attention/query/kernel") in layers


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
    from xreport_shared.common.path import ENCODERS_DIR
    from xreport_shared.models.training.encoder import BeitXRayImageEncoder

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
        service.update_preset(
            "compact", ArchitecturePresetRequest(name="Compact 2", architecture=compact)
        )

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
def test_builtin_presets_can_be_hidden_and_restored_except_the_default() -> None:
    service = _service()
    with pytest.raises(BadRequestError):
        service.delete_preset("xreport-default")

    service.delete_preset("large")
    service.delete_preset("compact")
    listing = service.list_presets()
    assert [preset.key for preset in listing.presets] == ["xreport-default", "balanced"]
    assert listing.hidden_builtins == 2
    assert listing.presets[0].protected is True
    # hidden presets still resolve, so trained checkpoints and runs keep working
    assert service.resolve_preset("large")[0] == "Large"

    restored = service.restore_builtin_presets()
    assert [preset.key for preset in restored.presets] == list(arch.BUILTIN_PRESETS)
    assert restored.hidden_builtins == 0


###############################################################################
def test_preview_reports_field_issues_instead_of_failing() -> None:
    service = _service()
    invalid = service.preview(
        ArchitecturePreviewRequest(
            architecture={"embedding_dims": 100, "encoder_heads": 8, "encoder_head_dim_mode": "split"}
        )
    )
    assert invalid.valid is False
    assert invalid.graph is None
    assert invalid.issues[0].field == "encoder_heads"
    assert "divisible" in invalid.issues[0].message

    valid = service.preview(
        ArchitecturePreviewRequest(architecture={"num_decoders": 2}, reference="xreport-default")
    )
    assert valid.valid is True
    assert valid.stats is not None and valid.stats.depth == 8
    assert valid.compatibility is not None
    assert valid.compatibility.level == "mismatch"
    assert valid.compatibility.weights is None
    assert "num_decoders" in valid.shape_fields

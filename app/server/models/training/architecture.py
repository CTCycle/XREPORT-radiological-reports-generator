from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from server.domain.model_architecture import (
    BEIT_ENCODER_ID,
    ArchitectureCompatibility,
    ArchitectureEdge,
    ArchitectureGraph,
    ArchitectureNode,
    ArchitectureStats,
    EncoderOption,
    FieldChange,
    ModelArchitecture,
)

# Dimensions used to size an architecture when no processed dataset is chosen:
# the default Studio tokenizer (distilbert-base-uncased) and report length.
REFERENCE_VOCABULARY_SIZE = 30522
REFERENCE_SEQUENCE_LENGTH = 200
IMAGE_SHAPE = (224, 224, 3)
FLOAT32_BYTES = 4

# Fields that change no weight shape: a checkpoint trained with one value can be
# rebuilt or fine-tuned with another. Every other field changes weight shapes.
WEIGHTS_COMPATIBLE_FIELDS = ("freeze_image_encoder", "dropout", "temperature")

###############################################################################
@dataclass(frozen=True)
class ImageEncoderSpec:
    id: str
    label: str
    repository: str
    output_tokens: int
    feature_dims: int
    parameters: int
    description: str


ENCODER_REGISTRY: dict[str, ImageEncoderSpec] = {
    BEIT_ENCODER_ID: ImageEncoderSpec(
        id=BEIT_ENCODER_ID,
        label="BEiT Base 16/224",
        repository="microsoft/beit-base-patch16-224",
        output_tokens=197,
        feature_dims=768,
        parameters=85_761_984,
        description=(
            "Pinned pretrained vision transformer used by every XREPORT checkpoint. "
            "Its 768-wide patch features are projected to the model width."
        ),
    ),
}

###############################################################################
@dataclass(frozen=True)
class BuiltinPreset:
    key: str
    name: str
    description: str
    architecture: ModelArchitecture


BUILTIN_PRESETS: dict[str, BuiltinPreset] = {
    preset.key: preset
    for preset in (
        BuiltinPreset(
            key="xreport-default",
            name="XREPORT default",
            description=(
                "The standard XREPORT transformer: six full-width encoder and "
                "decoder layers on a frozen BEiT encoder."
            ),
            architecture=ModelArchitecture(),
        ),
        BuiltinPreset(
            key="compact",
            name="Compact",
            description="Two shallow, narrow stacks for quick experiments and small datasets.",
            architecture=ModelArchitecture(
                embedding_dims=256,
                num_encoders=2,
                num_decoders=2,
                attention_heads=4,
                head_dim_mode="split",
                ffn_units=512,
                dropout=0.1,
                classifier_units=512,
            ),
        ),
        BuiltinPreset(
            key="balanced",
            name="Balanced",
            description="Standard split-head attention at medium width and depth.",
            architecture=ModelArchitecture(
                embedding_dims=512,
                num_encoders=4,
                num_decoders=4,
                attention_heads=8,
                head_dim_mode="split",
                ffn_units=1024,
                dropout=0.1,
                classifier_units=1024,
            ),
        ),
        BuiltinPreset(
            key="large",
            name="Large",
            description=(
                "BEiT-width split-head transformer with a 4x feed-forward and a "
                "trainable image encoder. Needs a GPU with ample memory."
            ),
            architecture=ModelArchitecture(
                freeze_image_encoder=False,
                embedding_dims=768,
                num_encoders=6,
                num_decoders=6,
                attention_heads=12,
                head_dim_mode="split",
                ffn_units=3072,
                dropout=0.1,
                classifier_units=1024,
            ),
        ),
    )
}

# -----------------------------------------------------------------------------
def encoder_options() -> list[EncoderOption]:
    from server.models.training.encoder import BeitXRayImageEncoder

    return [
        EncoderOption(
            id=spec.id,
            label=spec.label,
            repository=spec.repository,
            revision=BeitXRayImageEncoder.ENCODER_REVISION,
            input_shape=list(IMAGE_SHAPE),
            output_tokens=spec.output_tokens,
            feature_dims=spec.feature_dims,
            parameters=spec.parameters,
            pretrained=True,
            inference_compatible=True,
            description=spec.description,
        )
        for spec in ENCODER_REGISTRY.values()
    ]

# -----------------------------------------------------------------------------
def layer_kwargs(architecture: ModelArchitecture) -> dict[str, Any]:
    """Keyword arguments shared by every transformer encoder and decoder layer."""
    return {
        "embedding_dims": architecture.embedding_dims,
        "num_heads": architecture.attention_heads,
        "ffn_units": architecture.ffn_units,
        "dropout": architecture.dropout,
        "key_dim": architecture.key_dim,
    }

# [ANALYTIC PARAMETER COUNTS]
# These mirror the Keras layers in layers.py and encoder.py; unit tests compare
# them with the parameter counts of the built layers.

# -----------------------------------------------------------------------------
def attention_parameters(width: int, heads: int, key_dim: int) -> int:
    # query, key and value projections (with bias) plus the output projection
    return 4 * width * heads * key_dim + 3 * heads * key_dim + width


def feed_forward_parameters(width: int, units: int) -> int:
    return width * units + units + units * width + width


def encoder_block_parameters(architecture: ModelArchitecture) -> int:
    width = architecture.embedding_dims
    attention = attention_parameters(
        width, architecture.attention_heads, architecture.key_dim
    )
    return attention + 2 * (2 * width) + feed_forward_parameters(
        width, architecture.ffn_units
    )


def decoder_block_parameters(architecture: ModelArchitecture) -> int:
    width = architecture.embedding_dims
    attention = attention_parameters(
        width, architecture.attention_heads, architecture.key_dim
    )
    return 2 * attention + 3 * (2 * width) + feed_forward_parameters(
        width, architecture.ffn_units
    )


def embedding_parameters(
    architecture: ModelArchitecture, vocabulary_size: int, sequence_length: int
) -> int:
    return (vocabulary_size + sequence_length) * architecture.embedding_dims


def classifier_parameters(architecture: ModelArchitecture, vocabulary_size: int) -> int:
    units = architecture.classifier_units
    return architecture.embedding_dims * units + units + units * vocabulary_size + vocabulary_size


def projection_parameters(architecture: ModelArchitecture) -> int:
    features = ENCODER_REGISTRY[architecture.image_encoder].feature_dims
    return features * architecture.embedding_dims + architecture.embedding_dims

# -----------------------------------------------------------------------------
def describe(
    architecture: ModelArchitecture,
    vocabulary_size: int = REFERENCE_VOCABULARY_SIZE,
    sequence_length: int = REFERENCE_SEQUENCE_LENGTH,
) -> tuple[ArchitectureGraph, ArchitectureStats]:
    encoder = ENCODER_REGISTRY[architecture.image_encoder]
    width = architecture.embedding_dims
    tokens = encoder.output_tokens
    encoder_block = encoder_block_parameters(architecture)
    decoder_block = decoder_block_parameters(architecture)
    block_properties = {
        "attention_heads": architecture.attention_heads,
        "head_dim_mode": architecture.head_dim_mode,
        "key_dim": architecture.key_dim,
        "ffn_units": architecture.ffn_units,
        "dropout": architecture.dropout,
    }

    nodes = [
        ArchitectureNode(
            id="image_input",
            kind="input",
            label="Image input",
            detail="224 x 224 RGB radiograph",
            lane="image",
            parameters=0,
            trainable=False,
            output_shape=[None, *IMAGE_SHAPE],
        ),
        ArchitectureNode(
            id="image_encoder",
            kind="image_encoder",
            label=encoder.label,
            detail=f"Pretrained · {tokens} x {encoder.feature_dims}",
            lane="image",
            parameters=encoder.parameters,
            trainable=not architecture.freeze_image_encoder,
            output_shape=[None, tokens, encoder.feature_dims],
            properties={
                "repository": encoder.repository,
                "frozen": architecture.freeze_image_encoder,
            },
        ),
        ArchitectureNode(
            id="projection",
            kind="projection",
            label="Feature projection",
            detail=f"Dense {encoder.feature_dims} → {width}",
            lane="image",
            parameters=projection_parameters(architecture),
            trainable=True,
            output_shape=[None, tokens, width],
        ),
        ArchitectureNode(
            id="encoder_stack",
            kind="encoder_stack",
            label="Encoder stack",
            detail="Self-attention + feed-forward",
            lane="image",
            parameters=encoder_block * architecture.num_encoders,
            trainable=True,
            repeat=architecture.num_encoders,
            output_shape=[None, tokens, width],
            properties={**block_properties, "block_parameters": encoder_block},
        ),
        ArchitectureNode(
            id="token_input",
            kind="input",
            label="Report tokens",
            detail=f"{sequence_length} token ids",
            lane="text",
            parameters=0,
            trainable=False,
            output_shape=[None, sequence_length],
        ),
        ArchitectureNode(
            id="embedding",
            kind="embedding",
            label="Token embedding",
            detail=f"{vocabulary_size} tokens + {sequence_length} positions",
            lane="text",
            parameters=embedding_parameters(architecture, vocabulary_size, sequence_length),
            trainable=True,
            output_shape=[None, sequence_length, width],
            properties={
                "vocabulary_size": vocabulary_size,
                "sequence_length": sequence_length,
            },
        ),
        ArchitectureNode(
            id="decoder_stack",
            kind="decoder_stack",
            label="Decoder stack",
            detail="Self- and cross-attention, FFN",
            lane="text",
            parameters=decoder_block * architecture.num_decoders,
            trainable=True,
            repeat=architecture.num_decoders,
            output_shape=[None, sequence_length, width],
            properties={**block_properties, "block_parameters": decoder_block},
        ),
        ArchitectureNode(
            id="classifier",
            kind="classifier",
            label="Token classifier",
            detail=(
                f"{width} → {architecture.classifier_units} → {vocabulary_size}, "
                f"T={architecture.temperature:g}"
            ),
            lane="output",
            parameters=classifier_parameters(architecture, vocabulary_size),
            trainable=True,
            output_shape=[None, sequence_length, vocabulary_size],
            properties={
                "classifier_units": architecture.classifier_units,
                "temperature": architecture.temperature,
            },
        ),
    ]
    edges = [
        ArchitectureEdge(source="image_input", target="image_encoder"),
        ArchitectureEdge(source="image_encoder", target="projection"),
        ArchitectureEdge(source="projection", target="encoder_stack"),
        ArchitectureEdge(source="token_input", target="embedding"),
        ArchitectureEdge(source="embedding", target="decoder_stack"),
        ArchitectureEdge(
            source="encoder_stack", target="decoder_stack", label="cross-attention"
        ),
        ArchitectureEdge(source="decoder_stack", target="classifier"),
    ]

    total = sum(node.parameters for node in nodes)
    frozen = encoder.parameters if architecture.freeze_image_encoder else 0
    trainable = total - frozen
    stats = ArchitectureStats(
        total_parameters=total,
        trainable_parameters=trainable,
        frozen_parameters=frozen,
        weights_bytes=total * FLOAT32_BYTES,
        # weights plus gradients and the two AdamW moments of trainable weights
        training_memory_bytes=(total + 3 * trainable) * FLOAT32_BYTES,
        depth=architecture.num_encoders + architecture.num_decoders,
        vocabulary_size=vocabulary_size,
        sequence_length=sequence_length,
    )

    return ArchitectureGraph(nodes=nodes, edges=edges), stats

# -----------------------------------------------------------------------------
def compatibility(
    architecture: ModelArchitecture, reference: ModelArchitecture, label: str
) -> ArchitectureCompatibility:
    current = architecture.model_dump()
    previous = reference.model_dump()
    changes = [
        FieldChange(
            field=field,
            before=previous[field],
            after=current[field],
            impact=(
                "weights_compatible"
                if field in WEIGHTS_COMPATIBLE_FIELDS
                else "retrain_required"
            ),
        )
        for field in current
        if current[field] != previous[field]
    ]
    if not changes:
        return ArchitectureCompatibility(
            reference=label,
            level="identical",
            changes=[],
            message=f"Identical to {label}.",
        )
    if all(change.impact == "weights_compatible" for change in changes):
        return ArchitectureCompatibility(
            reference=label,
            level="weights_compatible",
            changes=changes,
            message=(
                f"Weight shapes match {label}; only training behaviour changes."
            ),
        )
    return ArchitectureCompatibility(
        reference=label,
        level="retrain_required",
        changes=changes,
        message=(
            f"Weight shapes differ from {label}. The transformer must be trained "
            "from scratch; the pretrained BEiT weights are still reused."
        ),
    )

# -----------------------------------------------------------------------------
def architecture_from_configuration(
    configuration: dict[str, Any],
) -> tuple[ModelArchitecture, bool]:
    """Reconstruct the architecture of a checkpoint configuration.

    Returns the architecture and whether it was derived from the flat keys of
    checkpoints trained before architecture snapshots existed.
    """
    snapshot = configuration.get("architecture")
    if isinstance(snapshot, dict):
        return ModelArchitecture.model_validate(snapshot), False

    width = int(configuration.get("embedding_dims", 256))
    return (
        ModelArchitecture(
            freeze_image_encoder=bool(configuration.get("freeze_img_encoder", False)),
            embedding_dims=width,
            num_encoders=int(configuration.get("num_encoders", 4)),
            num_decoders=int(configuration.get("num_decoders", 4)),
            attention_heads=int(configuration.get("attention_heads", 8)),
            head_dim_mode="full",
            ffn_units=width,
            dropout=0.2,
            classifier_units=1024,
            temperature=float(configuration.get("train_temp", 1.0)),
        ),
        True,
    )

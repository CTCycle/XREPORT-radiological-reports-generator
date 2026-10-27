from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

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
    Stack,
    WeightCompatibility,
    WeightMismatch,
)

# Dimensions used to size an architecture when no processed dataset is chosen:
# the default Studio tokenizer (distilbert-base-uncased) and report length.
REFERENCE_VOCABULARY_SIZE = 30522
REFERENCE_SEQUENCE_LENGTH = 200
IMAGE_SHAPE = (224, 224, 3)
FLOAT32_BYTES = 4

DEFAULT_PRESET_KEY = "xreport-default"
BEIT_REVISION = "f02e8f77db4703e3fbd3766e3375a4619c5a4863"

# Effective settings that decide the shape of a weight tensor. Any other setting
# (dropout, activations, epsilon, temperature, freezing) leaves every tensor
# shape unchanged, so weights trained with one value load with another.
SHAPE_FIELDS = (
    "image_encoder",
    "embedding_dims",
    "num_encoders",
    "encoder_heads",
    "encoder_key_dim",
    "encoder_ffn_units",
    "num_decoders",
    "decoder_heads",
    "decoder_key_dim",
    "decoder_ffn_units",
    "classifier_units",
)

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


def shared_stacks(
    heads: int, head_dim_mode: str, ffn_units: int, dropout: float
) -> dict[str, Any]:
    """Identical encoder and decoder settings for a built-in preset."""
    return {
        f"{stack}_{key}": value
        for stack in ("encoder", "decoder")
        for key, value in (
            ("heads", heads),
            ("head_dim_mode", head_dim_mode),
            ("ffn_units", ffn_units),
            ("dropout", dropout),
        )
    }


BUILTIN_PRESETS: dict[str, BuiltinPreset] = {
    preset.key: preset
    for preset in (
        BuiltinPreset(
            key=DEFAULT_PRESET_KEY,
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
                classifier_units=512,
                **shared_stacks(4, "split", 512, 0.1),
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
                classifier_units=1024,
                **shared_stacks(8, "split", 1024, 0.1),
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
                classifier_units=1024,
                **shared_stacks(12, "split", 3072, 0.1),
            ),
        ),
    )
}

# -----------------------------------------------------------------------------
def encoder_options() -> list[EncoderOption]:
    return [
        EncoderOption(
            id=spec.id,
            label=spec.label,
            repository=spec.repository,
            revision=BEIT_REVISION,
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
def stack_kwargs(architecture: ModelArchitecture, stack: Stack) -> dict[str, Any]:
    """Keyword arguments of the transformer encoder or decoder layers of a stack."""
    return {
        "embedding_dims": architecture.embedding_dims,
        "num_heads": getattr(architecture, f"{stack}_heads"),
        "ffn_units": getattr(architecture, f"{stack}_ffn_units"),
        "dropout": getattr(architecture, f"{stack}_dropout"),
        "key_dim": architecture.key_dim(stack),
        "activation": getattr(architecture, f"{stack}_ffn_activation"),
        "norm_epsilon": architecture.norm_epsilon,
    }

# [ANALYTIC PARAMETER COUNTS]
# These mirror the Keras layers in layers.py and encoder.py; unit tests compare
# them with the parameter counts and weight shapes of the built layers.

# -----------------------------------------------------------------------------
def attention_parameters(width: int, heads: int, key_dim: int) -> int:
    # query, key and value projections (with bias) plus the output projection
    return 4 * width * heads * key_dim + 3 * heads * key_dim + width


def feed_forward_parameters(width: int, units: int) -> int:
    return width * units + units + units * width + width


def encoder_block_parameters(architecture: ModelArchitecture) -> int:
    width = architecture.embedding_dims
    attention = attention_parameters(
        width, architecture.encoder_heads, architecture.key_dim("encoder")
    )
    return attention + 2 * (2 * width) + feed_forward_parameters(
        width, architecture.encoder_ffn_units
    )


def decoder_block_parameters(architecture: ModelArchitecture) -> int:
    width = architecture.embedding_dims
    attention = attention_parameters(
        width, architecture.decoder_heads, architecture.key_dim("decoder")
    )
    return 2 * attention + 3 * (2 * width) + feed_forward_parameters(
        width, architecture.decoder_ffn_units
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

    def block_properties(stack: Stack) -> dict[str, Any]:
        return {
            "heads": getattr(architecture, f"{stack}_heads"),
            "head_dim_mode": getattr(architecture, f"{stack}_head_dim_mode"),
            "key_dim": architecture.key_dim(stack),
            "ffn_units": getattr(architecture, f"{stack}_ffn_units"),
            "ffn_activation": getattr(architecture, f"{stack}_ffn_activation"),
            "dropout": getattr(architecture, f"{stack}_dropout"),
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
            fixed=True,
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
            detail=(
                f"{architecture.encoder_heads} heads x {architecture.key_dim('encoder')}"
                f" · FFN {architecture.encoder_ffn_units}"
            ),
            lane="image",
            parameters=encoder_block * architecture.num_encoders,
            trainable=True,
            repeat=architecture.num_encoders,
            output_shape=[None, tokens, width],
            properties={**block_properties("encoder"), "block_parameters": encoder_block},
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
            fixed=True,
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
            detail=(
                f"{architecture.decoder_heads} heads x {architecture.key_dim('decoder')}"
                f" · FFN {architecture.decoder_ffn_units}"
            ),
            lane="text",
            parameters=decoder_block * architecture.num_decoders,
            trainable=True,
            repeat=architecture.num_decoders,
            output_shape=[None, sequence_length, width],
            properties={**block_properties("decoder"), "block_parameters": decoder_block},
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
                "activation": architecture.classifier_activation,
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
def effective_settings(architecture: ModelArchitecture) -> dict[str, Any]:
    """Settings that act on the model, with head size resolved to its real value.

    Raw fields such as an unused custom head size, or a custom size equal to the
    model width, are not differences between two architectures.
    """
    return {
        "image_encoder": architecture.image_encoder,
        "freeze_image_encoder": architecture.freeze_image_encoder,
        "embedding_dims": architecture.embedding_dims,
        "norm_epsilon": architecture.norm_epsilon,
        **{
            f"{prefix}{key}": value
            for stack, prefix, count in (
                ("encoder", "encoder_", "num_encoders"),
                ("decoder", "decoder_", "num_decoders"),
            )
            for key, value in (
                ("heads", getattr(architecture, f"{stack}_heads")),
                ("key_dim", architecture.key_dim(stack)),
                ("ffn_units", getattr(architecture, f"{stack}_ffn_units")),
                ("ffn_activation", getattr(architecture, f"{stack}_ffn_activation")),
                ("dropout", getattr(architecture, f"{stack}_dropout")),
            )
        },
        "num_encoders": architecture.num_encoders,
        "num_decoders": architecture.num_decoders,
        "classifier_units": architecture.classifier_units,
        "classifier_activation": architecture.classifier_activation,
        "temperature": architecture.temperature,
    }

# -----------------------------------------------------------------------------
def compare_architectures(
    architecture: ModelArchitecture,
    reference: ModelArchitecture,
    label: str,
    reference_kind: Literal["preset", "checkpoint"],
) -> ArchitectureCompatibility:
    """Compare two architecture definitions by their effective settings."""
    current = effective_settings(architecture)
    previous = effective_settings(reference)
    changes = [
        FieldChange(
            field=field,
            before=previous[field],
            after=current[field],
            scope="shape" if field in SHAPE_FIELDS else "behaviour",
        )
        for field in current
        if current[field] != previous[field]
    ]
    if not changes:
        level, message = "identical", f"Same architecture as {label}."
    elif all(change.scope == "behaviour" for change in changes):
        level, message = (
            "compatible",
            f"Same structure as {label}; only configurable settings differ, "
            "so every weight tensor has the same shape.",
        )
    else:
        shape = sum(change.scope == "shape" for change in changes)
        level, message = (
            "mismatch",
            f"{shape} structural setting{'s' if shape != 1 else ''} differ from "
            f"{label}, so weight tensors have different shapes.",
        )
    return ArchitectureCompatibility(
        reference=label,
        reference_kind=reference_kind,
        level=level,  # type: ignore[arg-type]
        changes=changes,
        message=message,
    )

# -----------------------------------------------------------------------------
def weight_shapes(
    architecture: ModelArchitecture, vocabulary_size: int, sequence_length: int
) -> dict[tuple[str, str, str], tuple[int, ...]]:
    """Trainable tensors of the model as (group, block, tensor) -> shape.

    The group is a stack or component, the block its 1-based index within the
    stack ("" outside stacks). The shapes mirror the Keras layers (see
    layers.py); unit tests compare them with a built model.
    """
    width = architecture.embedding_dims
    features = ENCODER_REGISTRY[architecture.image_encoder].feature_dims
    shapes: dict[tuple[str, str, str], tuple[int, ...]] = {
        ("projection", "", "dense/kernel"): (features, width),
        ("projection", "", "dense/bias"): (width,),
        ("embedding", "", "token_embeddings"): (vocabulary_size, width),
        ("embedding", "", "position_embeddings"): (sequence_length, width),
    }

    def attention(group: str, block: str, name: str, heads: int, key_dim: int) -> None:
        for part in ("query", "key", "value"):
            shapes[(group, block, f"{name}/{part}/kernel")] = (width, heads, key_dim)
            shapes[(group, block, f"{name}/{part}/bias")] = (heads, key_dim)
        shapes[(group, block, f"{name}/output/kernel")] = (heads, key_dim, width)
        shapes[(group, block, f"{name}/output/bias")] = (width,)

    def feed_forward(group: str, block: str, units: int) -> None:
        shapes[(group, block, "ffn/dense1/kernel")] = (width, units)
        shapes[(group, block, "ffn/dense1/bias")] = (units,)
        shapes[(group, block, "ffn/dense2/kernel")] = (units, width)
        shapes[(group, block, "ffn/dense2/bias")] = (width,)

    def norms(group: str, block: str, count: int) -> None:
        for index in range(1, count + 1):
            shapes[(group, block, f"norm{index}/gamma")] = (width,)
            shapes[(group, block, f"norm{index}/beta")] = (width,)

    for index in range(1, architecture.num_encoders + 1):
        block = str(index)
        attention("encoder", block, "attention", architecture.encoder_heads, architecture.key_dim("encoder"))
        norms("encoder", block, 2)
        feed_forward("encoder", block, architecture.encoder_ffn_units)
    for index in range(1, architecture.num_decoders + 1):
        block = str(index)
        key_dim = architecture.key_dim("decoder")
        attention("decoder", block, "self_attention", architecture.decoder_heads, key_dim)
        attention("decoder", block, "cross_attention", architecture.decoder_heads, key_dim)
        norms("decoder", block, 3)
        feed_forward("decoder", block, architecture.decoder_ffn_units)

    units = architecture.classifier_units
    shapes[("classifier", "", "dense1/kernel")] = (width, units)
    shapes[("classifier", "", "dense1/bias")] = (units,)
    shapes[("classifier", "", "dense2/kernel")] = (units, vocabulary_size)
    shapes[("classifier", "", "dense2/bias")] = (vocabulary_size,)
    return shapes

# -----------------------------------------------------------------------------
def _block_range(blocks: list[int]) -> str:
    runs: list[str] = []
    start = previous = blocks[0]
    for block in [*blocks[1:], None]:  # type: ignore[list-item]
        if block is not None and block == previous + 1:
            previous = block
            continue
        runs.append(str(start) if start == previous else f"{start}-{previous}")
        if block is not None:
            start = previous = block
    return ", ".join(runs)

# -----------------------------------------------------------------------------
def compare_weights(
    architecture: ModelArchitecture,
    checkpoint: ModelArchitecture,
    vocabulary_size: int,
    sequence_length: int,
    label: str,
) -> WeightCompatibility:
    """Check which tensors of a draft cannot take the weights stored in a checkpoint.

    The draft is assumed to use the checkpoint's vocabulary and report length.
    Tensors that differ in the same way across blocks are reported once.
    """
    expected = weight_shapes(architecture, vocabulary_size, sequence_length)
    found = weight_shapes(checkpoint, vocabulary_size, sequence_length)
    grouped: dict[tuple[str, str, tuple[int, ...], tuple[int, ...]], list[int]] = {}
    for key in expected.keys() | found.keys():
        group, block, tensor = key
        if expected.get(key) != found.get(key):
            grouped.setdefault(
                (group, tensor, expected.get(key, ()), found.get(key, ())), []
            ).append(int(block) if block else 0)

    mismatches = [
        WeightMismatch(
            layer=f"{group} {_block_range(sorted(blocks))}" if blocks[0] else group,
            tensor=tensor,
            expected=list(want),
            found=list(have),
        )
        for (group, tensor, want, have), blocks in sorted(grouped.items())
    ]
    if not mismatches:
        return WeightCompatibility(
            status="loadable",
            message=f"Every weight tensor of {label} fits this architecture.",
            mismatches=[],
        )
    return WeightCompatibility(
        status="incompatible",
        message=(
            f"{len(mismatches)} tensor group{'s' if len(mismatches) != 1 else ''} "
            f"cannot load the trained weights of {label}: weights can only be "
            "reused when every tensor shape matches."
        ),
        mismatches=mismatches,
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
        ModelArchitecture.model_validate(
            {
                "freeze_image_encoder": bool(configuration.get("freeze_img_encoder", False)),
                "embedding_dims": width,
                "num_encoders": int(configuration.get("num_encoders", 4)),
                "num_decoders": int(configuration.get("num_decoders", 4)),
                "attention_heads": int(configuration.get("attention_heads", 8)),
                "head_dim_mode": "full",
                "ffn_units": width,
                "dropout": 0.2,
                "classifier_units": 1024,
                "temperature": float(configuration.get("train_temp", 1.0)),
            }
        ),
        True,
    )

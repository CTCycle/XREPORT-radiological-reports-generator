from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

BEIT_ENCODER_ID = "beit-base-patch16-224"
MIN_HEAD_DIM = 8
MAX_HEAD_DIM = 1024

HeadDimMode = Literal["full", "split", "custom"]
Activation = Literal["relu", "gelu", "silu"]
Stack = Literal["encoder", "decoder"]
ArchitectureMatch = Literal["identical", "compatible", "mismatch"]
WeightStatus = Literal["loadable", "incompatible"]
ChangeScope = Literal["shape", "behaviour"]

# Keys of architectures saved before encoder and decoder were configured
# independently: each one sets the same value on both stacks.
SHARED_STACK_KEYS = {
    "attention_heads": "heads",
    "head_dim_mode": "head_dim_mode",
    "ffn_units": "ffn_units",
    "dropout": "dropout",
}


###############################################################################
class ModelArchitecture(BaseModel):
    """Configurable, constrained XREPORT transformer architecture.

    Encoder and decoder stacks are configured independently. Components that are
    part of the fixed XREPORT design (image preprocessing, BEiT weights,
    positional embeddings, vocabulary and report length, softmax output) are not
    fields.
    """

    image_encoder: Literal["beit-base-patch16-224"] = Field(
        default=BEIT_ENCODER_ID, description="Pretrained image encoder from the registry"
    )
    freeze_image_encoder: bool = Field(
        default=True, description="Keep pretrained image encoder weights frozen"
    )
    embedding_dims: int = Field(
        default=768, ge=64, le=1024, description="Model width shared by all transformer blocks"
    )
    norm_epsilon: float = Field(
        default=1e-4, ge=1e-8, le=1e-2, description="Epsilon of every residual layer norm"
    )

    num_encoders: int = Field(default=6, ge=1, le=12, description="Transformer encoder layers")
    encoder_head_dim_mode: HeadDimMode = Field(
        default="full",
        description="full: model width per head; split: width / heads; custom: encoder_head_dim",
    )
    encoder_head_dim: int = Field(
        default=64, ge=MIN_HEAD_DIM, le=MAX_HEAD_DIM, description="Head size in custom mode"
    )
    encoder_heads: int = Field(default=8, ge=1, le=16, description="Encoder attention heads")
    encoder_ffn_units: int = Field(
        default=768, ge=64, le=4096, description="Hidden units of each encoder feed-forward network"
    )
    encoder_ffn_activation: Activation = Field(
        default="relu", description="Activation of the encoder feed-forward network"
    )
    encoder_dropout: float = Field(default=0.2, ge=0.0, le=0.5, description="Encoder dropout rate")

    num_decoders: int = Field(default=6, ge=1, le=12, description="Transformer decoder layers")
    decoder_head_dim_mode: HeadDimMode = Field(
        default="full",
        description="full: model width per head; split: width / heads; custom: decoder_head_dim",
    )
    decoder_head_dim: int = Field(
        default=64, ge=MIN_HEAD_DIM, le=MAX_HEAD_DIM, description="Head size in custom mode"
    )
    decoder_heads: int = Field(default=8, ge=1, le=16, description="Decoder attention heads")
    decoder_ffn_units: int = Field(
        default=768, ge=64, le=4096, description="Hidden units of each decoder feed-forward network"
    )
    decoder_ffn_activation: Activation = Field(
        default="relu", description="Activation of the decoder feed-forward network"
    )
    decoder_dropout: float = Field(default=0.2, ge=0.0, le=0.5, description="Decoder dropout rate")

    classifier_units: int = Field(
        default=1024, ge=64, le=4096, description="Hidden units of the token classifier"
    )
    classifier_activation: Activation = Field(
        default="relu", description="Activation of the classifier hidden layer"
    )
    temperature: float = Field(
        default=1.0, ge=0.1, le=2.0, description="Softmax temperature of the classifier"
    )
    model_config = ConfigDict(extra="forbid")

    # -------------------------------------------------------------------------
    @model_validator(mode="before")
    @classmethod
    def expand_shared_stack_keys(cls, data: Any) -> Any:
        if not isinstance(data, dict) or not SHARED_STACK_KEYS.keys() & data.keys():
            return data
        expanded = {key: value for key, value in data.items() if key not in SHARED_STACK_KEYS}
        for old, suffix in SHARED_STACK_KEYS.items():
            if old in data:
                for stack in ("encoder", "decoder"):
                    expanded.setdefault(f"{stack}_{suffix}", data[old])
        return expanded

    # -------------------------------------------------------------------------
    @field_validator("encoder_heads", "decoder_heads")
    @classmethod
    def validate_split_heads(cls, heads: int, info: ValidationInfo) -> int:
        stack = str(info.field_name).removesuffix("_heads")
        width = info.data.get("embedding_dims")
        if info.data.get(f"{stack}_head_dim_mode") != "split" or width is None:
            return heads
        if width % heads != 0:
            raise ValueError(
                f"Model width {width} must be divisible by {heads} heads in split mode"
            )
        if width // heads < MIN_HEAD_DIM:
            raise ValueError(
                f"Split heads need at least {MIN_HEAD_DIM} dimensions each; "
                "reduce the heads or increase the model width"
            )
        return heads

    # -------------------------------------------------------------------------
    def key_dim(self, stack: Stack) -> int:
        mode = getattr(self, f"{stack}_head_dim_mode")
        if mode == "split":
            return self.embedding_dims // getattr(self, f"{stack}_heads")
        if mode == "custom":
            return getattr(self, f"{stack}_head_dim")
        return self.embedding_dims


###############################################################################
class EncoderOption(BaseModel):
    id: str
    label: str
    repository: str
    revision: str
    input_shape: list[int]
    output_tokens: int
    feature_dims: int
    parameters: int
    pretrained: bool
    inference_compatible: bool
    description: str


###############################################################################
class EncodersResponse(BaseModel):
    encoders: list[EncoderOption]


###############################################################################
class ArchitectureStats(BaseModel):
    total_parameters: int
    trainable_parameters: int
    frozen_parameters: int
    weights_bytes: int
    training_memory_bytes: int
    depth: int
    vocabulary_size: int
    sequence_length: int


###############################################################################
class ArchitectureNode(BaseModel):
    id: str
    kind: str
    label: str
    detail: str
    lane: Literal["image", "text", "output"]
    parameters: int
    trainable: bool
    repeat: int = 1
    output_shape: list[int | None]
    fixed: bool = False
    properties: dict[str, Any] = Field(default_factory=dict)


###############################################################################
class ArchitectureEdge(BaseModel):
    source: str
    target: str
    label: str = ""


###############################################################################
class ArchitectureGraph(BaseModel):
    nodes: list[ArchitectureNode]
    edges: list[ArchitectureEdge]


###############################################################################
class FieldChange(BaseModel):
    field: str
    before: Any
    after: Any
    scope: ChangeScope


###############################################################################
class WeightMismatch(BaseModel):
    layer: str
    tensor: str
    expected: list[int]
    found: list[int]


###############################################################################
class WeightCompatibility(BaseModel):
    status: WeightStatus
    message: str
    mismatches: list[WeightMismatch]


###############################################################################
class ArchitectureCompatibility(BaseModel):
    reference: str
    reference_kind: Literal["preset", "checkpoint"]
    level: ArchitectureMatch
    changes: list[FieldChange]
    message: str
    weights: WeightCompatibility | None = None


###############################################################################
class ArchitectureIssue(BaseModel):
    field: str | None = None
    message: str


###############################################################################
class ArchitecturePreviewRequest(BaseModel):
    architecture: dict[str, Any]
    reference: str | None = Field(
        default=None,
        min_length=1,
        max_length=160,
        description="Preset key or checkpoint:<name> to compare against",
    )
    model_config = ConfigDict(extra="forbid")


###############################################################################
class ArchitecturePreviewResponse(BaseModel):
    valid: bool
    issues: list[ArchitectureIssue]
    architecture: ModelArchitecture | None = None
    graph: ArchitectureGraph | None = None
    stats: ArchitectureStats | None = None
    compatibility: ArchitectureCompatibility | None = None
    shape_fields: list[str] = Field(default_factory=list)


###############################################################################
class ArchitecturePreset(BaseModel):
    key: str
    name: str
    description: str
    builtin: bool
    protected: bool = False
    architecture: ModelArchitecture
    stats: ArchitectureStats
    created_at: datetime | None = None
    updated_at: datetime | None = None


###############################################################################
class ArchitecturePresetsResponse(BaseModel):
    presets: list[ArchitecturePreset]
    hidden_builtins: int = 0


###############################################################################
class ArchitecturePresetRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=80)
    description: str = Field(default="", max_length=500)
    architecture: ModelArchitecture
    model_config = ConfigDict(extra="forbid")


###############################################################################
class CheckpointArchitectureResponse(BaseModel):
    checkpoint: str
    architecture: ModelArchitecture
    preset_name: str | None = None
    legacy: bool
    measured_parameters: int | None = None
    vocabulary_size: int | None = None
    sequence_length: int | None = None

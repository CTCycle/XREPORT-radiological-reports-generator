from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

BEIT_ENCODER_ID = "beit-base-patch16-224"
MIN_HEAD_DIM = 8

HeadDimMode = Literal["full", "split"]
CompatibilityLevel = Literal["identical", "weights_compatible", "retrain_required"]

###############################################################################
class ModelArchitecture(BaseModel):
    """Configurable, constrained XREPORT transformer architecture."""

    image_encoder: Literal["beit-base-patch16-224"] = Field(
        default=BEIT_ENCODER_ID, description="Pretrained image encoder from the registry"
    )
    freeze_image_encoder: bool = Field(
        default=True, description="Keep pretrained image encoder weights frozen"
    )
    embedding_dims: int = Field(
        default=768, ge=64, le=1024, description="Model width shared by all transformer blocks"
    )
    num_encoders: int = Field(default=6, ge=1, le=12, description="Transformer encoder layers")
    num_decoders: int = Field(default=6, ge=1, le=12, description="Transformer decoder layers")
    attention_heads: int = Field(default=8, ge=1, le=16, description="Attention heads per block")
    head_dim_mode: HeadDimMode = Field(
        default="full",
        description="full: every head projects to the model width; split: width / heads",
    )
    ffn_units: int = Field(
        default=768, ge=64, le=4096, description="Hidden units of each feed-forward network"
    )
    dropout: float = Field(default=0.2, ge=0.0, le=0.5, description="Dropout rate")
    classifier_units: int = Field(
        default=1024, ge=64, le=4096, description="Hidden units of the token classifier"
    )
    temperature: float = Field(
        default=1.0, ge=0.1, le=2.0, description="Softmax temperature of the classifier"
    )
    model_config = ConfigDict(extra="forbid")

    # -------------------------------------------------------------------------
    @model_validator(mode="after")
    def validate_head_dimensions(self) -> ModelArchitecture:
        if self.head_dim_mode == "split":
            if self.embedding_dims % self.attention_heads != 0:
                raise ValueError(
                    f"Model width {self.embedding_dims} must be divisible by "
                    f"{self.attention_heads} attention heads in split mode"
                )
            if self.embedding_dims // self.attention_heads < MIN_HEAD_DIM:
                raise ValueError(
                    f"Split heads need at least {MIN_HEAD_DIM} dimensions each; "
                    "reduce the heads or increase the model width"
                )
        return self

    # -------------------------------------------------------------------------
    @property
    def key_dim(self) -> int:
        if self.head_dim_mode == "split":
            return self.embedding_dims // self.attention_heads
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
    impact: Literal["weights_compatible", "retrain_required"]

###############################################################################
class ArchitectureCompatibility(BaseModel):
    reference: str
    level: CompatibilityLevel
    changes: list[FieldChange]
    message: str

###############################################################################
class ArchitectureIssue(BaseModel):
    field: str | None = None
    message: str

###############################################################################
class ArchitecturePreviewRequest(BaseModel):
    architecture: dict[str, Any]
    dataset_name: str | None = Field(default=None, min_length=1, max_length=128)
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

###############################################################################
class ArchitecturePreset(BaseModel):
    key: str
    name: str
    description: str
    builtin: bool
    architecture: ModelArchitecture
    stats: ArchitectureStats
    created_at: datetime | None = None
    updated_at: datetime | None = None

###############################################################################
class ArchitecturePresetsResponse(BaseModel):
    presets: list[ArchitecturePreset]

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

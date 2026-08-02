from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal

from pydantic import BaseModel, Field

###############################################################################
@dataclass(frozen=True)
class InferenceImage:
    filename: str
    content_type: str
    data: bytes
    size_bytes: int

###############################################################################
@dataclass(frozen=True)
class ProviderGenerationResult:
    reports: dict[str, str]
    display_sections: dict[str, dict[str, str]]
    metadata: list[dict[str, object]]
    provenance: dict[str, object]

###############################################################################
GenerationProfile = Literal["deterministic", "concise", "detailed"]
OutputSection = Literal["raw_report", "findings", "impression"]
ValidationStatus = Literal["blocked", "incompatible", "disabled", "pending", "passed"]

_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")

###############################################################################
class ModelCapabilities(BaseModel):
    clinical_context: bool = False
    prior_report: bool = False
    multiple_current_views: bool = False
    findings: bool = False
    impression: bool = False
    grounding: bool = False

    model_config = {"extra": "forbid", "strict": True}

###############################################################################
class ModelResourcePolicy(BaseModel):
    max_snapshot_size_bytes: int | None = Field(default=None, ge=0)
    reason: str | None = None

    model_config = {"extra": "forbid", "strict": True}

###############################################################################
class ModelRuntimeConstraints(BaseModel):
    min_transformers: str | None = None
    max_transformers_exclusive: str | None = None
    required_modules: list[str] = Field(default_factory=list)

    model_config = {"extra": "forbid", "strict": True}

###############################################################################
class InferenceManifestEntry(BaseModel):
    """Strict, reviewable contract for one configured Hugging Face model."""

    model_ref: str
    repository_id: str
    provider: Literal["huggingface"]
    enabled: bool = True
    display_name: str
    description: str
    category: str
    recommended: bool = False
    research_only: bool = True
    gated: bool = False
    license: str
    parameter_size: str | None = None
    local_size_bytes: int | None = Field(default=None, ge=0)
    revision: str
    model_loader: Literal["image_text_to_text", "causal_lm", "blip_conditional_generation"]
    processor_loader: Literal["auto", "image", "blip"]
    adapter: Literal[
        "medgemma",
        "generate_cxr_blip",
        "standard_image_text",
        "maira2",
        "chexagent_impression",
        "cxrmate2",
    ]
    prompt_profile: str
    output_sections: list[OutputSection] = Field(min_length=1)
    input_semantics: Literal["single_image", "independent_images", "single_study"]
    max_current_images: int = Field(ge=1, le=16)
    supports_clinical_context: bool = False
    supports_prior_images: bool = False
    preferred_dtype: Literal["auto", "float32", "float16", "bfloat16"] = "auto"
    quantization: list[str] = Field(min_length=1)
    trust_remote_code: bool = False
    remote_code_approved: bool = False
    validation_status: ValidationStatus
    validation_message: str | None = None
    resource_policy: ModelResourcePolicy = Field(default_factory=ModelResourcePolicy)
    runtime_constraints: ModelRuntimeConstraints = Field(
        default_factory=ModelRuntimeConstraints
    )
    required_files: list[str] = Field(min_length=1)
    weight_file_sets: list[list[str]] = Field(min_length=1)
    processor_repository_id: str | None = None
    processor_revision: str | None = None
    capabilities: ModelCapabilities = Field(default_factory=ModelCapabilities)

    model_config = {"extra": "forbid", "strict": True}

    # -------------------------------------------------------------------------
    @staticmethod
    def _validate_revision(value: str, field_name: str) -> str:
        if not _REVISION_PATTERN.fullmatch(value):
            raise ValueError(f"{field_name} must be an exact 40-character commit SHA")
        return value

    # -------------------------------------------------------------------------
    def model_post_init(self, __context: object) -> None:
        if self.provider != "huggingface":
            raise ValueError("Only the huggingface provider is valid in this manifest")
        if self.model_ref != f"huggingface:{self.repository_id}":
            raise ValueError("model_ref must match repository_id")
        self._validate_revision(self.revision, "revision")
        if self.processor_revision is not None:
            self._validate_revision(self.processor_revision, "processor_revision")
        if (self.processor_repository_id is None) != (self.processor_revision is None):
            raise ValueError(
                "processor_repository_id and processor_revision must be provided together"
            )
        if any(not group for group in self.weight_file_sets):
            raise ValueError("weight_file_sets must contain no empty alternatives")
        if self.remote_code_approved and not self.trust_remote_code:
            raise ValueError("remote_code_approved requires trust_remote_code")
        if self.validation_status == "passed" and not self.enabled:
            raise ValueError("A disabled manifest entry cannot be marked passed")

###############################################################################
class InferenceManifest(BaseModel):
    schema_version: Literal[2]
    models: list[InferenceManifestEntry] = Field(min_length=1, max_length=5)

    model_config = {"extra": "forbid", "strict": True}

###############################################################################
class ModelAvailability(BaseModel):
    model_ref: str
    provider: Literal["huggingface", "xreport"]
    display_name: str
    description: str
    status: Literal["ready", "not_installed", "unvalidated", "gated", "runtime_unavailable", "incompatible", "disabled"]
    status_message: str | None = None
    enabled: bool = True
    validation_status: ValidationStatus = "pending"
    validation_message: str | None = None
    category: str
    recommended: bool = False
    research_only: bool = True
    gated: bool = False
    parameter_size: str | None = None
    local_size_bytes: int | None = None
    input_semantics: Literal["single_image", "independent_images", "single_study"]
    capabilities: ModelCapabilities
    model_revision: str | None = None
    model_loader: str | None = None
    processor_loader: str | None = None
    adapter: str | None = None
    trust_remote_code: bool = False
    remote_code_approved: bool = False
    output_sections: list[OutputSection] = Field(default_factory=list)
    max_current_images: int = 1
    supports_prior_images: bool = False
    supports_clinical_context: bool = False
    preferred_dtype: str = "auto"
    quantization: list[str] = Field(default_factory=list)
    prompt_profile: str | None = None
    license: str | None = None
    resource_policy: ModelResourcePolicy = Field(default_factory=ModelResourcePolicy)
    runtime_constraints: ModelRuntimeConstraints = Field(
        default_factory=ModelRuntimeConstraints
    )
    processor_repository_id: str | None = None
    processor_revision: str | None = None

###############################################################################
class ProviderAvailability(BaseModel):
    status: Literal["ready", "not_installed", "unvalidated", "gated", "runtime_unavailable", "incompatible", "disabled"]
    message: str | None = None

###############################################################################
class InferenceModelsResponse(BaseModel):
    models: list[ModelAvailability]
    providers: dict[str, ProviderAvailability]

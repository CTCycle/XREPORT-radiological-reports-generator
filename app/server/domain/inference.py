from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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
ValidationStatus = Literal[
    "blocked",
    "incompatible",
    "disabled",
    "pending",
    "degraded",
    "passed",
]
InstallationState = Literal[
    "not_installed",
    "staged",
    "active",
    "corrupt",
    "failed",
    "downloading",
]
ModelOrigin = Literal["public", "custom"]
HardwareDemand = Literal["low", "moderate", "high", "very_high"]
AccessPolicy = Literal["open", "gated"]

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
    access_policy: AccessPolicy = "open"
    access_url: str | None = None
    anatomy_coverage: str = "chest_xray"
    coverage_note: str | None = None
    hardware_demand: HardwareDemand = "moderate"
    parameter_label: str | None = None
    license: str
    parameter_size: str | None = None
    download_size_bytes: int | None = Field(default=None, ge=0)
    local_size_bytes: int | None = Field(default=None, ge=0)
    revision: str
    model_loader: Literal[
        "auto_model",
        "image_text_to_text",
        "causal_lm",
    ]
    processor_loader: Literal["auto"]
    adapter: Literal[
        "medgemma",
        "chexone",
        "cxrmate_multi",
        "cxrmate_ed",
        "cxrmate2",
    ]
    prompt_profile: str
    output_sections: list[OutputSection] = Field(min_length=1)
    input_semantics: Literal["single_image", "independent_images", "single_study"]
    max_current_images: int = Field(ge=1, le=16)
    supports_clinical_context: bool = False
    supports_prior_images: bool = False
    preferred_dtype: Literal["auto", "float32", "float16", "bfloat16"]
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
    processor_files: list[str] = Field(default_factory=list)
    processor_target_prefix: str = "processor"
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
        if any(
            not item
            or Path(item).is_absolute()
            or ".." in Path(item).parts
            or Path(item).name != item
            for item in self.processor_files
        ):
            raise ValueError("processor_files must contain only repository filenames")
        prefix = Path(self.processor_target_prefix)
        if (
            not self.processor_target_prefix
            or prefix.is_absolute()
            or ".." in prefix.parts
            or prefix.name != self.processor_target_prefix
        ):
            raise ValueError(
                "processor_target_prefix must be a safe relative directory"
            )
        if any(not group for group in self.weight_file_sets):
            raise ValueError("weight_file_sets must contain no empty alternatives")
        if self.remote_code_approved and not self.trust_remote_code:
            raise ValueError("remote_code_approved requires trust_remote_code")
        if self.validation_status == "passed" and not self.enabled:
            raise ValueError("A disabled manifest entry cannot be marked passed")


###############################################################################
class InferenceManifest(BaseModel):
    schema_version: Literal[3]
    models: list[InferenceManifestEntry] = Field(min_length=5, max_length=5)

    model_config = {"extra": "forbid", "strict": True}

    # -------------------------------------------------------------------------
    def model_post_init(self, __context: object) -> None:
        refs = [entry.model_ref for entry in self.models]
        if len(set(refs)) != len(refs):
            raise ValueError(
                "The public inference catalogue cannot contain duplicate model refs"
            )
        if any(entry.provider != "huggingface" for entry in self.models):
            raise ValueError(
                "The public inference catalogue may contain only Hugging Face models"
            )


###############################################################################
class ModelAvailability(BaseModel):
    model_ref: str
    provider: Literal["huggingface", "xreport"]
    origin: ModelOrigin = "public"
    display_name: str
    description: str
    status: Literal[
        "ready",
        "not_installed",
        "unvalidated",
        "downloading",
        "gated",
        "runtime_unavailable",
        "incompatible",
        "disabled",
    ]
    status_message: str | None = None
    enabled: bool = True
    validation_status: ValidationStatus = "pending"
    validation_message: str | None = None
    validation_receipt_status: Literal["missing", "passed", "invalid"] = "missing"
    validation_receipt_message: str | None = None
    category: str
    recommended: bool = False
    research_only: bool = True
    gated: bool = False
    access_policy: AccessPolicy = "open"
    access_url: str | None = None
    anatomy_coverage: str = "chest_xray"
    coverage_note: str | None = None
    hardware_demand: HardwareDemand = "moderate"
    parameter_label: str | None = None
    parameter_size: str | None = None
    download_size_bytes: int | None = Field(default=None, ge=0)
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
    processor_files: list[str] = Field(default_factory=list)
    processor_target_prefix: str = "processor"
    required_files: list[str] = Field(default_factory=list)
    weight_file_sets: list[list[str]] = Field(default_factory=list)
    installation_state: InstallationState = "not_installed"
    local_path: str | None = None
    active_revision: str | None = None
    candidate_revision: str | None = None
    integrity_status: str = "unknown"
    cloud_assessment: dict[str, object] | None = None
    update_available: bool = False
    available_actions: list[str] = Field(default_factory=list)


###############################################################################
class ProviderAvailability(BaseModel):
    status: Literal[
        "ready",
        "not_installed",
        "unvalidated",
        "downloading",
        "gated",
        "runtime_unavailable",
        "incompatible",
        "disabled",
    ]
    message: str | None = None


###############################################################################
class InferenceModelsResponse(BaseModel):
    models: list[ModelAvailability]
    providers: dict[str, ProviderAvailability]


###############################################################################
class ModelUpdateCheckResponse(BaseModel):
    model_ref: str
    repository_id: str
    installed_revision: str | None = None
    latest_revision: str | None = None
    update_available: bool = False
    source: str
    checked_at: str
    error: str | None = None


###############################################################################
class ModelUpdateCheckRequest(BaseModel):
    model_ref: str


###############################################################################
class InferenceGenerateRequest(BaseModel):
    model_ref: str
    generation_profile: GenerationProfile
    clinical_context: str = ""


###############################################################################
class ModelMaintenanceRequest(BaseModel):
    model_ref: str
    action: Literal[
        "download", "repair", "reinstall", "download_update", "delete_local"
    ]
    revision: str | None = None

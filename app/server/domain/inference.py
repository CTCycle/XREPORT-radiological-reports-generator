from __future__ import annotations

from dataclasses import dataclass
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
GenerationProfile = Literal["deterministic", "concise", "detailed"]

###############################################################################
class ModelCapabilities(BaseModel):
    clinical_context: bool = False
    prior_report: bool = False
    multiple_current_views: bool = False
    findings: bool = False
    impression: bool = False
    grounding: bool = False

###############################################################################
class ModelAvailability(BaseModel):
    model_ref: str
    provider: Literal["huggingface", "xreport"]
    display_name: str
    description: str
    status: Literal["ready", "not_installed", "gated", "runtime_unavailable", "incompatible", "disabled"]
    status_message: str | None = None
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
    output_sections: list[str] = Field(default_factory=list)
    max_current_images: int = 1
    supports_prior_images: bool = False
    supports_clinical_context: bool = False
    preferred_dtype: str = "auto"
    quantization: list[str] = Field(default_factory=list)
    prompt_profile: str | None = None
    license: str | None = None

###############################################################################
class ProviderAvailability(BaseModel):
    status: Literal["ready", "not_installed", "gated", "runtime_unavailable", "incompatible", "disabled"]
    message: str | None = None

###############################################################################
class InferenceModelsResponse(BaseModel):
    models: list[ModelAvailability]
    providers: dict[str, ProviderAvailability]

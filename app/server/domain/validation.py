from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

VALIDATION_METRICS = frozenset(
    {"text_statistics", "image_statistics", "pixels_distribution"}
)
CHECKPOINT_EVALUATION_METRICS = frozenset({"evaluation_report", "bleu_score"})


###############################################################################
def _validate_metric_names(
    metrics: list[str],
    supported: frozenset[str],
    label: str,
) -> list[str]:
    unsupported = sorted(set(metrics) - supported)
    if unsupported:
        raise ValueError(f"Unsupported {label}: {', '.join(unsupported)}")
    return metrics


###############################################################################
class ValidationRequest(BaseModel):
    """Request model for dataset validation."""

    dataset_name: str = Field(..., min_length=1, max_length=128)
    metrics: list[str] = Field(..., min_length=1, max_length=4)
    sample_size: float = Field(..., ge=0.01, le=1.0)
    seed: int | None = None
    model_config = ConfigDict(extra="forbid")

    @field_validator("metrics")
    @classmethod
    def validate_metrics(cls, metrics: list[str]) -> list[str]:
        return _validate_metric_names(metrics, VALIDATION_METRICS, "validation metrics")


###############################################################################
class PixelDistribution(BaseModel):
    """Model for pixel intensity distribution."""

    bins: list[int]
    counts: list[int]


###############################################################################
class ImageStatistics(BaseModel):
    """Model for image statistics."""

    count: int
    mean_height: float
    mean_width: float
    mean_pixel_value: float
    std_pixel_value: float
    mean_noise_std: float
    mean_noise_ratio: float


###############################################################################
class TextStatistics(BaseModel):
    """Model for text statistics."""

    count: int
    total_words: int
    unique_words: int
    avg_words_per_report: float
    min_words_per_report: int
    max_words_per_report: int


###############################################################################
class ValidationResponse(BaseModel):
    """Response model for dataset validation."""

    success: bool
    message: str
    pixel_distribution: PixelDistribution | None = None
    image_statistics: ImageStatistics | None = None
    text_statistics: TextStatistics | None = None


###############################################################################
class ValidationReportResponse(BaseModel):
    """Response model for a persisted validation report."""

    dataset_name: str
    date: str | None = None
    sample_size: float | None = None
    metrics: list[str] = Field(default_factory=list)
    pixel_distribution: PixelDistribution | None = None
    image_statistics: ImageStatistics | None = None
    text_statistics: TextStatistics | None = None
    artifacts: dict[str, dict[str, str]] | None = None


###############################################################################
class CheckpointEvaluationRequest(BaseModel):
    """Request model for checkpoint evaluation."""

    checkpoint: str = Field(..., min_length=1, max_length=128)
    metrics: list[str] = Field(..., min_length=1, max_length=4)
    num_samples: int = Field(..., ge=1, le=1000)
    metric_configs: dict[str, dict[str, float | int]] | None = None
    seed: int | None = None

    model_config = ConfigDict(extra="forbid")

    @field_validator("metrics")
    @classmethod
    def validate_metrics(cls, metrics: list[str]) -> list[str]:
        return _validate_metric_names(
            metrics,
            CHECKPOINT_EVALUATION_METRICS,
            "checkpoint evaluation metrics",
        )


###############################################################################
class CheckpointEvaluationResults(BaseModel):
    """Evaluation metric results."""

    loss: float | None = None
    accuracy: float | None = None
    bleu_score: float | None = None


###############################################################################
class CheckpointEvaluationResponse(BaseModel):
    """Response model for checkpoint evaluation."""

    success: bool
    message: str
    results: CheckpointEvaluationResults | None = None


###############################################################################
class CheckpointEvaluationReportResponse(BaseModel):
    """Response model for a persisted checkpoint evaluation report."""

    checkpoint: str
    date: str | None = None
    metrics: list[str] = Field(default_factory=list)
    metric_configs: dict[str, dict[str, float | int]] = Field(default_factory=dict)
    results: CheckpointEvaluationResults | None = None

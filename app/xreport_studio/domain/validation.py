from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from xreport_studio.configurations.tokenizers import TOKENIZER_ID_PATTERN

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
    tokenizer: str | None = Field(
        default=None, max_length=200, pattern=TOKENIZER_ID_PATTERN
    )
    model_config = ConfigDict(extra="forbid")

    # -------------------------------------------------------------------------
    @field_validator("metrics")
    @classmethod
    def validate_metrics(cls, metrics: list[str]) -> list[str]:
        return _validate_metric_names(metrics, VALIDATION_METRICS, "validation metrics")

    # -------------------------------------------------------------------------
    @model_validator(mode="after")
    def validate_tokenizer(self) -> "ValidationRequest":
        if self.tokenizer is not None and "text_statistics" not in self.metrics:
            raise ValueError("Token statistics require the text_statistics metric")
        return self


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
class ValidationRunMetricModel(BaseModel):
    """A metric computed by a run and how many records it covered."""

    metric: str
    records_computed: int | None = None


###############################################################################
class ValidationRunSummary(BaseModel):
    """Configuration and coverage of one validation run."""

    validation_run_id: int
    date: str | None = None
    sample_size: float
    seed: int | None = None
    records_requested: int | None = None
    records_evaluated: int | None = None
    text_vocabulary_size: int | None = None
    tokenizer: str | None = None
    max_report_size: int | None = None
    metrics: list[ValidationRunMetricModel] = Field(default_factory=list)


###############################################################################
class AggregateStatistic(BaseModel):
    """Distribution summary of one measure; unavailable values are null."""

    domain: Literal["text", "image", "token"]
    measure: str
    count: int
    total: float | None = None
    minimum: float | None = None
    maximum: float | None = None
    mean: float | None = None
    median: float | None = None
    std: float | None = None
    p05: float | None = None
    p25: float | None = None
    p75: float | None = None
    p95: float | None = None


###############################################################################
class HistogramBinModel(BaseModel):
    lower: float
    upper: float
    count: int


###############################################################################
class HistogramModel(BaseModel):
    name: str
    bins: list[HistogramBinModel]


###############################################################################
class ValidationReportResponse(BaseModel):
    """Response model for one persisted validation run."""

    dataset_name: str
    run: ValidationRunSummary
    aggregates: list[AggregateStatistic] = Field(default_factory=list)
    histograms: list[HistogramModel] = Field(default_factory=list)


###############################################################################
class ValidationRunListResponse(BaseModel):
    """Validation history of a dataset, newest first."""

    dataset_name: str
    runs: list[ValidationRunSummary]


###############################################################################
class RecordStatisticsResponse(BaseModel):
    """One page of per-record statistics of a run."""

    total: int
    rows: list[dict[str, str | int | float | bool | None]]


###############################################################################
class CheckpointEvaluationRequest(BaseModel):
    """Request model for checkpoint evaluation."""

    checkpoint: str = Field(..., min_length=1, max_length=128)
    metrics: list[str] = Field(..., min_length=1, max_length=4)
    num_samples: int = Field(..., ge=1, le=1000)
    metric_configs: dict[str, dict[str, float | int]] | None = None
    seed: int | None = None

    model_config = ConfigDict(extra="forbid")

    # -------------------------------------------------------------------------
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

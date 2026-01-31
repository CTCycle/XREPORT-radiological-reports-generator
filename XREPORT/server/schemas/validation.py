from __future__ import annotations

from pydantic import BaseModel, ConfigDict


###############################################################################
class ValidationRequest(BaseModel):
    """Request model for dataset validation."""
    
    dataset_name: str
    metrics: list[str]
    sample_size: float = 1.0
    seed: int | None = None
    
    model_config = ConfigDict(extra="forbid")


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
    metrics: list[str] = []
    pixel_distribution: PixelDistribution | None = None
    image_statistics: ImageStatistics | None = None
    text_statistics: TextStatistics | None = None
    artifacts: dict[str, dict[str, str]] | None = None


###############################################################################
class CheckpointEvaluationRequest(BaseModel):
    """Request model for checkpoint evaluation."""
    
    checkpoint: str
    metrics: list[str]  # ["evaluation_report", "bleu_score"]
    num_samples: int = 10  # Number of samples for BLEU calculation
    
    model_config = ConfigDict(extra="forbid")


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


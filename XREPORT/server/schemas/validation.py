from __future__ import annotations

from pydantic import BaseModel, ConfigDict


###############################################################################
class ValidationRequest(BaseModel):
    """Request model for dataset validation."""
    
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

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from XREPORT.server.schemas.validation import (
    ValidationRequest,
    ValidationResponse,
)
from XREPORT.server.utils.logger import logger
from XREPORT.server.utils.services.validation import DatasetValidator
from XREPORT.server.utils.services.training.serializer import DataSerializer
from XREPORT.server.utils.configurations.server import server_settings

router = APIRouter(prefix="/validation", tags=["validation"])


###############################################################################
@router.post(
    "/run",
    response_model=ValidationResponse,
    status_code=status.HTTP_200_OK,
)
async def run_validation(request: ValidationRequest) -> ValidationResponse:
    """Run validation analytics on the current dataset."""
    try:
        serializer = DataSerializer()
        
        # Log validation configuration
        sample_pct = request.sample_size * 100
        logger.info(f"Starting dataset validation with {sample_pct:.1f}% sample size")
        
        dataset = serializer.load_source_dataset(
            sample_size=request.sample_size,
            seed=request.seed if request.seed is not None else server_settings.global_settings.seed
        )
        
        if dataset.empty:
            return ValidationResponse(
                success=False,
                message="No data found in the database to validate.",
            )
        
        logger.info(f"Loaded {len(dataset)} records for validation")
            
        # Validate that stored image paths exist
        dataset = serializer.validate_img_paths(dataset)
        
        if dataset.empty:
            return ValidationResponse(
                success=False,
                message="No valid image paths found in the dataset.",
            )
        
        logger.info(f"Starting analysis on {len(dataset)} validated records")
        
        validator = DatasetValidator(dataset)
        response = ValidationResponse(success=True, message="Validation completed successfully")
        
        metrics_requested = [m for m in request.metrics]
        logger.info(f"Metrics to compute: {', '.join(metrics_requested)}")
        
        if "text_statistics" in request.metrics:
            logger.info(f"[1/3] Calculating text statistics for {len(dataset)} reports...")
            response.text_statistics = validator.calculate_text_statistics()
            logger.info(f"[1/3] Text statistics complete: {response.text_statistics.total_words} total words, {response.text_statistics.unique_words} unique")
            
        if "image_statistics" in request.metrics:
            logger.info(f"[2/3] Calculating image statistics for {len(dataset)} images (this may take a while)...")
            response.image_statistics = validator.calculate_image_statistics()
            logger.info(f"[2/3] Image statistics complete: analyzed {response.image_statistics.count} images")
            
        if "pixels_distribution" in request.metrics:
            logger.info(f"[3/3] Calculating pixel intensity distribution for {len(dataset)} images...")
            response.pixel_distribution = validator.calculate_pixel_distribution()
            logger.info(f"[3/3] Pixel distribution complete")
        
        logger.info("Dataset validation completed successfully")
            
        return response

    except Exception as e:
        logger.exception("Validation failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Validation failed: {str(e)}",
        ) from e

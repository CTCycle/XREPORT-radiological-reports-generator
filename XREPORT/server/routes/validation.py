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
        
        dataset = serializer.load_source_dataset(
            sample_size=request.sample_size,
            seed=request.seed if request.seed is not None else server_settings.global_settings.seed
        )
        
        if dataset.empty:
            return ValidationResponse(
                success=False,
                message="No data found in the database to validate.",
            )
            
        # Ensure paths are correct
        dataset = serializer.update_img_path(dataset)
        
        validator = DatasetValidator(dataset)
        response = ValidationResponse(success=True, message="Validation completed successfully")
        
        if "text_statistics" in request.metrics:
            logger.info("Calculating text statistics...")
            response.text_statistics = validator.calculate_text_statistics()
            
        if "image_statistics" in request.metrics:
            logger.info("Calculating image statistics...")
            response.image_statistics = validator.calculate_image_statistics()
            
        if "pixels_distribution" in request.metrics:
            logger.info("Calculating pixel distribution...")
            response.pixel_distribution = validator.calculate_pixel_distribution()
            
        return response

    except Exception as e:
        logger.exception("Validation failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Validation failed: {str(e)}",
        ) from e

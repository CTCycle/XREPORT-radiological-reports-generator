from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, status

from XREPORT.server.schemas.validation import (
    ValidationRequest,
    ValidationResponse,
    CheckpointEvaluationRequest,
    CheckpointEvaluationResponse,
    CheckpointEvaluationResults,
)
from XREPORT.server.schemas.jobs import (
    JobStartResponse,
    JobStatusResponse,
    JobCancelResponse,
)
from XREPORT.server.utils.logger import logger
from XREPORT.server.utils.jobs import JobManager, job_manager
from XREPORT.server.utils.services.validation import DatasetValidator
from XREPORT.server.utils.repository.serializer import DataSerializer, ModelSerializer
from XREPORT.server.utils.configurations.server import ServerSettings, server_settings
from XREPORT.server.utils.learning.training.dataloader import XRAYDataLoader
from XREPORT.server.utils.services.evaluation import CheckpointEvaluator


# -----------------------------------------------------------------------------
def run_validation_job(
    request_data: dict[str, Any],
    job_id: str,
) -> dict[str, Any]:
    """Blocking validation function that runs in background thread."""
    # Use global job_manager imported at top level
    jm = job_manager
    
    serializer = DataSerializer()
    
    sample_size = request_data.get("sample_size", 1.0)
    # Access server_settings carefully if inside thread, but here we pass request_data.
    # We should have passed the seed in request_data from the caller.
    seed = request_data.get("seed", 42)
    metrics = request_data.get("metrics", [])
    
    # Log validation configuration
    sample_pct = sample_size * 100
    logger.info(f"Starting dataset validation with {sample_pct:.1f}% sample size")
    
    jm.update_progress(job_id, 5.0)
    if jm.should_stop(job_id):
        return {}
    
    dataset = serializer.load_source_dataset(
        sample_size=sample_size,
        seed=seed
    )
    
    if dataset.empty:
        return {
            "success": False,
            "message": "No data found in the database to validate.",
        }
    
    logger.info(f"Loaded {len(dataset)} records for validation")
    jm.update_progress(job_id, 15.0)
    if jm.should_stop(job_id):
        return {}
    
    # Extract dataset_name from source data
    if "dataset_name" in dataset.columns and not dataset.empty:
        dataset_name = dataset["dataset_name"].iloc[0]
    else:
        dataset_name = "default"
        
    # Validate that stored image paths exist
    dataset = serializer.validate_img_paths(dataset)
    
    if dataset.empty:
        return {
            "success": False,
            "message": "No valid image paths found in the dataset.",
        }
    
    logger.info(f"Starting analysis on {len(dataset)} validated records")
    jm.update_progress(job_id, 20.0)
    if jm.should_stop(job_id):
        return {}
    
    validator = DatasetValidator(dataset, dataset_name=dataset_name)
    result: dict[str, Any] = {
        "success": True,
        "message": "Validation completed successfully",
    }
    
    logger.info(f"Metrics to compute: {', '.join(metrics)}")
    
    progress_per_metric = 25.0
    current_progress = 20.0
    
    if "text_statistics" in metrics:
        if jm.should_stop(job_id):
            return {}
        logger.info(f"[1/3] Calculating text statistics for {len(dataset)} reports...")
        text_stats, text_records_df = validator.calculate_text_statistics()
        result["text_statistics"] = {
            "count": text_stats.count,
            "total_words": text_stats.total_words,
            "unique_words": text_stats.unique_words,
            "avg_words_per_report": text_stats.avg_words_per_report,
            "min_words_per_report": text_stats.min_words_per_report,
            "max_words_per_report": text_stats.max_words_per_report,
        }
        logger.info(f"[1/3] Text statistics complete: {text_stats.total_words} total words, {text_stats.unique_words} unique")
        
        # Persist per-record text statistics
        if not text_records_df.empty:
            serializer.save_text_statistics(text_records_df)
            logger.info(f"Saved {len(text_records_df)} text statistics records to database")
        
        current_progress += progress_per_metric
        jm.update_progress(job_id, current_progress)
        
    if "image_statistics" in metrics:
        if jm.should_stop(job_id):
            return {}
        logger.info(f"[2/3] Calculating image statistics for {len(dataset)} images (this may take a while)...")
        image_stats, image_records_df = validator.calculate_image_statistics()
        result["image_statistics"] = {
            "count": image_stats.count,
            "mean_height": image_stats.mean_height,
            "mean_width": image_stats.mean_width,
            "mean_pixel_value": image_stats.mean_pixel_value,
            "std_pixel_value": image_stats.std_pixel_value,
            "mean_noise_std": image_stats.mean_noise_std,
            "mean_noise_ratio": image_stats.mean_noise_ratio,
        }
        logger.info(f"[2/3] Image statistics complete: analyzed {image_stats.count} images")
        
        # Persist per-record image statistics
        if not image_records_df.empty:
            serializer.save_images_statistics(image_records_df)
            logger.info(f"Saved {len(image_records_df)} image statistics records to database")
        
        current_progress += progress_per_metric
        jm.update_progress(job_id, current_progress)
        
    if "pixels_distribution" in metrics:
        if jm.should_stop(job_id):
            return {}
        logger.info(f"[3/3] Calculating pixel intensity distribution for {len(dataset)} images...")
        pixel_dist = validator.calculate_pixel_distribution()
        result["pixel_distribution"] = {
            "bins": pixel_dist.bins,
            "counts": pixel_dist.counts,
        }
        logger.info("[3/3] Pixel distribution complete")
        
        current_progress += progress_per_metric
        jm.update_progress(job_id, current_progress)
    
    logger.info("Dataset validation completed successfully")
    jm.update_progress(job_id, 100.0)
    
    return result


# -----------------------------------------------------------------------------
def run_checkpoint_evaluation_job(
    request_data: dict[str, Any],
    job_id: str,
) -> dict[str, Any]:
    """Blocking checkpoint evaluation function that runs in background thread."""
    # Use global job_manager imported at top level
    jm = job_manager
    
    checkpoint = request_data.get("checkpoint", "")
    metrics = request_data.get("metrics", [])
    num_samples = request_data.get("num_samples", 10)
    
    logger.info(f"Starting checkpoint evaluation: {checkpoint}")
    logger.info(f"Metrics: {metrics}, Samples: {num_samples}")
    
    jm.update_progress(job_id, 10.0)
    if jm.should_stop(job_id):
        return {}
    
    # Load the model checkpoint
    model_serializer = ModelSerializer()
    try:
        model, train_config, model_metadata, _, _ = model_serializer.load_checkpoint(checkpoint)
    except FileNotFoundError:
        return {
            "success": False,
            "message": f"Checkpoint not found: {checkpoint}",
            "results": None,
        }
    
    model.summary(expand_nested=True)
    jm.update_progress(job_id, 30.0)
    if jm.should_stop(job_id):
        return {}
    
    # Initialize evaluator
    evaluator = CheckpointEvaluator(model, train_config, model_metadata)
    results: dict[str, Any] = {}
    
    # Run evaluation_report metric (requires validation dataset)
    if "evaluation_report" in metrics:
        if jm.should_stop(job_id):
            return {}
        logger.info("Running evaluation report (loss and accuracy)...")
        
        # Load validation data
        data_serializer = DataSerializer()
        _, validation_data, _ = data_serializer.load_training_data()
        
        if validation_data.empty:
            logger.warning("No validation data available for evaluation report")
        else:
            # Validate image paths
            validation_data = data_serializer.validate_img_paths(validation_data)
            
            if not validation_data.empty:
                # Build validation dataset
                loader = XRAYDataLoader(train_config)
                validation_dataset = loader.build_training_dataloader(validation_data)
                
                # Run evaluation
                eval_results = evaluator.evaluate_model(validation_dataset)
                results["loss"] = eval_results.get("loss")
                results["accuracy"] = eval_results.get("accuracy")
        
        jm.update_progress(job_id, 60.0)
    
    # Run BLEU score metric
    if "bleu_score" in metrics:
        if jm.should_stop(job_id):
            return {}
        logger.info(f"Calculating BLEU score with {num_samples} samples...")
        
        # Load validation data with text for BLEU comparison
        data_serializer = DataSerializer()
        _, validation_data, _ = data_serializer.load_training_data()
        
        if validation_data.empty:
            logger.warning("No validation data available for BLEU calculation")
        else:
            # Validate image paths
            validation_data = data_serializer.validate_img_paths(validation_data)
            
            if not validation_data.empty:
                bleu = evaluator.calculate_bleu_score(
                    validation_data, 
                    num_samples=num_samples
                )
                results["bleu_score"] = bleu
        
        jm.update_progress(job_id, 90.0)
    
    jm.update_progress(job_id, 100.0)
    
    return {
        "success": True,
        "message": f"Evaluation completed for {checkpoint}",
        "results": results,
    }


###############################################################################
class ValidationEndpoint:
    """Endpoint for dataset validation and checkpoint evaluation analytics."""

    JOB_TYPE_VALIDATION = "validation"
    JOB_TYPE_EVALUATION = "checkpoint_evaluation"

    def __init__(
        self,
        router: APIRouter,
        job_manager: JobManager,
        server_settings: ServerSettings,
    ) -> None:
        self.router = router
        self.job_manager = job_manager
        self.server_settings = server_settings

    # -----------------------------------------------------------------------------
    async def run_validation(self, request: ValidationRequest) -> JobStartResponse:
        """Run validation analytics on the current dataset."""
        if self.job_manager.is_job_running("validation"):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Validation is already in progress",
            )
        
        # Prepare request data with default seed if not provided
        request_data = request.model_dump()
        if not request_data.get("seed"):
            request_data["seed"] = self.server_settings.global_settings.seed
            
        # Start background job
        job_id = self.job_manager.start_job(
            job_type="validation",
            runner=run_validation_job,
            kwargs={
                "request_data": request_data,
            },
        )

        job_status = self.job_manager.get_job_status(job_id)
        if job_status is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to initialize validation job",
            )
        
        return JobStartResponse(
            job_id=job_id,
            job_type=job_status["job_type"],
            status=job_status["status"],
            message="Validation job started",
        )

    # -----------------------------------------------------------------------------
    async def evaluate_checkpoint(
        self,
        request: CheckpointEvaluationRequest,
    ) -> JobStartResponse:
        """Evaluate a model checkpoint using selected metrics."""
        if self.job_manager.is_job_running("checkpoint_evaluation"):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Checkpoint evaluation is already in progress",
            )
        
        # Start background job
        job_id = self.job_manager.start_job(
            job_type="checkpoint_evaluation",
            runner=run_checkpoint_evaluation_job,
            kwargs={
                "request_data": request.model_dump(),
            },
        )

        job_status = self.job_manager.get_job_status(job_id)
        if job_status is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to initialize checkpoint evaluation job",
            )
        
        return JobStartResponse(
            job_id=job_id,
            job_type=job_status["job_type"],
            status=job_status["status"],
            message=f"Checkpoint evaluation job started for {request.checkpoint}",
        )

    # -----------------------------------------------------------------------------
    async def get_validation_job_status(self, job_id: str) -> JobStatusResponse:
        job_status = self.job_manager.get_job_status(job_id)
        if job_status is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job not found: {job_id}",
            )
        return JobStatusResponse(**job_status)

    # -----------------------------------------------------------------------------
    async def cancel_validation_job(self, job_id: str) -> JobCancelResponse:
        job_status = self.job_manager.get_job_status(job_id)
        if job_status is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job not found: {job_id}",
            )
        
        success = self.job_manager.cancel_job(job_id)
        
        return JobCancelResponse(
            job_id=job_id,
            success=success,
            message="Cancellation requested" if success else "Job cannot be cancelled",
        )

    # -----------------------------------------------------------------------------
    def add_routes(self) -> None:
        """Register all validation-related routes."""
        self.router.add_api_route(
            "/run",
            self.run_validation,
            methods=["POST"],
            response_model=JobStartResponse,
            status_code=status.HTTP_202_ACCEPTED,
        )
        self.router.add_api_route(
            "/checkpoint",
            self.evaluate_checkpoint,
            methods=["POST"],
            response_model=JobStartResponse,
            status_code=status.HTTP_202_ACCEPTED,
        )
        self.router.add_api_route(
            "/jobs/{job_id}",
            self.get_validation_job_status,
            methods=["GET"],
            response_model=JobStatusResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/jobs/{job_id}",
            self.cancel_validation_job,
            methods=["DELETE"],
            response_model=JobCancelResponse,
            status_code=status.HTTP_200_OK,
        )


###############################################################################
router = APIRouter(prefix="/validation", tags=["validation"])
validation_endpoint = ValidationEndpoint(
    router=router,
    job_manager=job_manager,
    server_settings=server_settings,
)
validation_endpoint.add_routes()

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, status

from XREPORT.server.schemas.validation import (
    ValidationRequest,
    ValidationResponse,
    ValidationReportResponse,
    CheckpointEvaluationRequest,
    CheckpointEvaluationResponse,
    CheckpointEvaluationResults,
    CheckpointEvaluationReportResponse,
)
from XREPORT.server.schemas.jobs import (
    JobStartResponse,
    JobStatusResponse,
    JobCancelResponse,
)
from XREPORT.server.utils.logger import logger
from XREPORT.server.services.jobs import JobManager, job_manager
from XREPORT.server.services.validation import DatasetValidator
from XREPORT.server.repositories.serializer import DataSerializer, ModelSerializer
from XREPORT.server.configurations.server import ServerSettings, server_settings
from XREPORT.server.learning.training.dataloader import XRAYDataLoader
from XREPORT.server.services.evaluation import CheckpointEvaluator


# -----------------------------------------------------------------------------
class ProgressRange:
    def __init__(self, job_id: str, start: float, end: float) -> None:
        self.job_id = job_id
        self.start = start
        self.end = end

    # -------------------------------------------------------------------------
    def update(self, fraction: float) -> None:
        clamped = min(1.0, max(0.0, fraction))
        progress = self.start + (self.end - self.start) * clamped
        job_manager.update_progress(self.job_id, progress)


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
    dataset_name = request_data.get("dataset_name")

    # Log validation configuration
    sample_pct = sample_size * 100
    logger.info(f"Starting dataset validation with {sample_pct:.1f}% sample size")

    jm.update_progress(job_id, 5.0)
    if jm.should_stop(job_id):
        return {}

    dataset = serializer.load_source_dataset(
        sample_size=sample_size,
        seed=seed,
        dataset_name=dataset_name,
    )

    if dataset.empty:
        return {
            "success": False,
            "message": (
                f"No data found for dataset: {dataset_name}."
                if dataset_name
                else "No data found in the database to validate."
            ),
        }

    logger.info(f"Loaded {len(dataset)} records for validation")
    jm.update_progress(job_id, 15.0)
    if jm.should_stop(job_id):
        return {}

    # Ensure dataset_name is set for downstream persistence
    if not dataset_name:
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
        "dataset_name": dataset_name,
        "sample_size": sample_size,
        "metrics": metrics,
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
        logger.info(
            f"[1/3] Text statistics complete: {text_stats.total_words} total words, {text_stats.unique_words} unique"
        )

        # Persist per-record text statistics
        if not text_records_df.empty:
            serializer.save_text_statistics(text_records_df)
            logger.info(
                f"Saved {len(text_records_df)} text statistics records to database"
            )

        current_progress += progress_per_metric
        jm.update_progress(job_id, current_progress)

    if "image_statistics" in metrics:
        if jm.should_stop(job_id):
            return {}
        logger.info(
            f"[2/3] Calculating image statistics for {len(dataset)} images (this may take a while)..."
        )
        progress_range = ProgressRange(
            job_id, current_progress, current_progress + progress_per_metric
        )
        image_stats, image_records_df = validator.calculate_image_statistics(
            progress_callback=progress_range.update,
        )
        result["image_statistics"] = {
            "count": image_stats.count,
            "mean_height": image_stats.mean_height,
            "mean_width": image_stats.mean_width,
            "mean_pixel_value": image_stats.mean_pixel_value,
            "std_pixel_value": image_stats.std_pixel_value,
            "mean_noise_std": image_stats.mean_noise_std,
            "mean_noise_ratio": image_stats.mean_noise_ratio,
        }
        logger.info(
            f"[2/3] Image statistics complete: analyzed {image_stats.count} images"
        )

        # Persist per-record image statistics
        if not image_records_df.empty:
            serializer.save_images_statistics(image_records_df)
            logger.info(
                f"Saved {len(image_records_df)} image statistics records to database"
            )

        current_progress += progress_per_metric
        jm.update_progress(job_id, current_progress)

    if "pixels_distribution" in metrics:
        if jm.should_stop(job_id):
            return {}
        logger.info(
            f"[3/3] Calculating pixel intensity distribution for {len(dataset)} images..."
        )
        progress_range = ProgressRange(
            job_id, current_progress, current_progress + progress_per_metric
        )
        pixel_dist = validator.calculate_pixel_distribution(
            progress_callback=progress_range.update,
        )
        result["pixel_distribution"] = {
            "bins": pixel_dist.bins,
            "counts": pixel_dist.counts,
        }
        logger.info("[3/3] Pixel distribution complete")

        current_progress += progress_per_metric
        jm.update_progress(job_id, current_progress)

    logger.info("Dataset validation completed successfully")
    jm.update_progress(job_id, 100.0)

    report_payload = {
        "dataset_name": dataset_name,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "sample_size": sample_size,
        "metrics": metrics,
        "text_statistics": result.get("text_statistics"),
        "image_statistics": result.get("image_statistics"),
        "pixel_distribution": result.get("pixel_distribution"),
        "artifacts": None,
    }
    serializer.save_validation_report(report_payload)

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
    metric_configs = request_data.get("metric_configs") or {}
    if not isinstance(metric_configs, dict):
        metric_configs = {}
    seed = request_data.get("seed", 42)

    logger.info(f"Starting checkpoint evaluation: {checkpoint}")
    logger.info(f"Metrics: {metrics}, Samples: {num_samples}")

    jm.update_progress(job_id, 10.0)
    if jm.should_stop(job_id):
        return {}

    # Load the model checkpoint
    model_serializer = ModelSerializer()
    try:
        model, train_config, model_metadata, _, _ = model_serializer.load_checkpoint(
            checkpoint
        )
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
    resolved_metric_configs: dict[str, dict[str, float | int]] = {}

    def resolve_fraction(
        config: dict[str, Any] | None,
        default_fraction: float = 1.0,
    ) -> float:
        if not isinstance(config, dict):
            return default_fraction
        fraction = config.get("data_fraction", default_fraction)
        if not isinstance(fraction, (int, float)):
            return default_fraction
        return float(min(1.0, max(0.01, fraction)))

    validation_data = None
    if "evaluation_report" in metrics or "bleu_score" in metrics:
        data_serializer = DataSerializer()
        _, validation_data, _ = data_serializer.load_training_data()
        if validation_data.empty:
            logger.warning("No validation data available for checkpoint evaluation")
        else:
            validation_data = data_serializer.validate_img_paths(validation_data)

    # Run evaluation_report metric (requires validation dataset)
    if "evaluation_report" in metrics:
        if jm.should_stop(job_id):
            return {}
        logger.info("Running evaluation report (loss and accuracy)...")

        if validation_data is None or validation_data.empty:
            logger.warning("No validation data available for evaluation report")
        else:
            evaluation_config = metric_configs.get("evaluation_report")
            evaluation_fraction = resolve_fraction(evaluation_config, default_fraction=1.0)
            resolved_metric_configs["evaluation_report"] = {
                "data_fraction": evaluation_fraction,
            }

            eval_data = validation_data
            if evaluation_fraction < 1.0:
                eval_data = validation_data.sample(
                    frac=evaluation_fraction,
                    random_state=seed,
                )

            if not eval_data.empty:
                loader = XRAYDataLoader(train_config)
                validation_dataset = loader.build_training_dataloader(eval_data)

                eval_results = evaluator.evaluate_model(validation_dataset)
                results["loss"] = eval_results.get("loss")
                results["accuracy"] = eval_results.get("accuracy")

        jm.update_progress(job_id, 60.0)

    # Run BLEU score metric
    if "bleu_score" in metrics:
        if jm.should_stop(job_id):
            return {}
        logger.info(f"Calculating BLEU score with {num_samples} samples...")

        if validation_data is None or validation_data.empty:
            logger.warning("No validation data available for BLEU calculation")
        else:
            if not validation_data.empty:
                bleu_config = metric_configs.get("bleu_score")
                bleu_fraction = resolve_fraction(bleu_config, default_fraction=1.0)
                if bleu_config is None:
                    bleu_fraction = 1.0
                bleu_samples = max(1, int(num_samples))
                if bleu_fraction < 1.0:
                    bleu_samples = max(1, int(len(validation_data) * bleu_fraction))

                resolved_metric_configs["bleu_score"] = {
                    "data_fraction": bleu_fraction,
                    "num_samples": bleu_samples,
                }

                bleu = evaluator.calculate_bleu_score(
                    validation_data, num_samples=bleu_samples
                )
                results["bleu_score"] = bleu

        jm.update_progress(job_id, 90.0)

    jm.update_progress(job_id, 100.0)

    report_payload = {
        "checkpoint": checkpoint,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": metrics,
        "metric_configs": resolved_metric_configs,
        "results": results,
    }
    try:
        serializer = DataSerializer()
        serializer.save_checkpoint_evaluation_report(report_payload)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to save checkpoint evaluation report: %s", exc)

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
            poll_interval=self.server_settings.jobs.polling_interval,
        )

    # -------------------------------------------------------------------------
    async def get_validation_report(
        self, dataset_name: str
    ) -> ValidationReportResponse:
        serializer = DataSerializer()
        report = serializer.get_validation_report(dataset_name)
        if report is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No validation report found for dataset: {dataset_name}",
            )
        return ValidationReportResponse(**report)

    # -------------------------------------------------------------------------
    async def get_checkpoint_evaluation_report(
        self, checkpoint: str
    ) -> CheckpointEvaluationReportResponse:
        serializer = DataSerializer()
        report = serializer.get_checkpoint_evaluation_report(checkpoint)
        if report is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No evaluation report found for checkpoint: {checkpoint}",
            )
        return CheckpointEvaluationReportResponse(**report)

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
            poll_interval=self.server_settings.jobs.polling_interval,
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
            "/checkpoint/reports/{checkpoint}",
            self.get_checkpoint_evaluation_report,
            methods=["GET"],
            response_model=CheckpointEvaluationReportResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/reports/{dataset_name}",
            self.get_validation_report,
            methods=["GET"],
            response_model=ValidationReportResponse,
            status_code=status.HTTP_200_OK,
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

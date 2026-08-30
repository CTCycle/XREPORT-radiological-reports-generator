from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from typing import Any

from server.services.errors import (
    BadRequestError,
    ConflictError,
    InternalServiceError,
    NotFoundError,
)
import pandas as pd

from server.domain.validation import (
    ValidationRequest,
    ValidationReportResponse,
    CheckpointEvaluationRequest,
    CheckpointEvaluationReportResponse,
)
from server.domain.jobs import (
    JobStartResponse,
)
from server.common.utils.logger import logger
from server.common.utils.security import validate_checkpoint_name
from server.services.jobs import JobExecutionError, JobManager, get_job_manager
from server.services.validation import DatasetValidator
from server.repositories.serialization.validation import ValidationRepository
from server.repositories.serialization.dataset import (
    DatasetIntegrityError,
    DatasetRepository,
)
from server.repositories.serialization.model import ModelSerializer
from server.repositories.checkpoints import CheckpointRepository
from server.configurations.startup import get_server_settings
from server.models.training.dataloader import XRAYDataLoader
from server.services.evaluation import (
    CheckpointEvaluator,
    CheckpointInputMismatchError,
)
from server.configurations import ServerSettings

###############################################################################
def resolve_metric_fraction(
    config: dict[str, Any] | None,
    default_fraction: float = 1.0,
) -> float:
    if not isinstance(config, dict):
        return default_fraction
    fraction = config.get("data_fraction", default_fraction)
    if not isinstance(fraction, (int, float)):
        return default_fraction
    return float(min(1.0, max(0.01, fraction)))

###############################################################################
class ProgressRange:

    # -------------------------------------------------------------------------
    def __init__(self, job_id: str, start: float, end: float) -> None:
        self.job_id = job_id
        self.start = start
        self.end = end

    # -------------------------------------------------------------------------
    def update(self, fraction: float) -> None:
        clamped = min(1.0, max(0.0, fraction))
        progress = self.start + (self.end - self.start) * clamped
        get_job_manager().update_progress(self.job_id, progress)

###############################################################################
def run_validation_job(
    request_data: dict[str, Any],
    job_id: str,
) -> dict[str, Any]:
    """Blocking validation function that runs in background thread."""
    jm = get_job_manager()
    dataset_repository = DatasetRepository()
    sample_size = request_data["sample_size"]
    seed = request_data.get("seed", 42)
    metrics = request_data["metrics"]
    dataset_name = str(request_data["dataset_name"]).strip()
    sample_pct = sample_size * 100
    logger.info(f"Starting dataset validation with {sample_pct:.1f}% sample size")
    jm.update_progress(job_id, 5.0)
    if jm.should_stop(job_id):
        return {}
    dataset = _load_validation_dataset(
        dataset_repository,
        sample_size,
        seed,
        dataset_name,
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
    try:
        dataset = dataset_repository.validate_img_paths(dataset)
    except DatasetIntegrityError as exc:
        raise JobExecutionError(
            str(exc),
            code="dataset_integrity_failed",
            phase="input_validation",
        ) from exc
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
    metric_results = _run_validation_metrics(validator, dataset, metrics, job_id, jm)
    if metric_results is None:
        return {}
    result.update(metric_results[0])
    image_records = metric_results[1]
    logger.info("Dataset validation completed successfully")
    jm.update_progress(job_id, 100.0)
    _save_validation_report(
        ValidationRepository(),
        dataset_name,
        sample_size,
        metrics,
        result,
        image_records,
    )
    return result

###############################################################################
def _run_validation_metrics(
    validator: DatasetValidator,
    dataset: pd.DataFrame,
    metrics: Any,
    job_id: str,
    jm: JobManager,
) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
    progress_per_metric = 25.0
    current_progress = 20.0
    result: dict[str, Any] = {}
    image_records: list[dict[str, Any]] = []
    if "text_statistics" in metrics:
        if jm.should_stop(job_id):
            return None
        result["text_statistics"] = _run_text_validation_metric(validator, dataset)
        current_progress += progress_per_metric
        jm.update_progress(job_id, current_progress)
    if "image_statistics" in metrics:
        if jm.should_stop(job_id):
            return None
        image_result, image_records = _run_image_validation_metric(
            validator, dataset, job_id, current_progress, progress_per_metric
        )
        result["image_statistics"] = image_result
        current_progress += progress_per_metric
        jm.update_progress(job_id, current_progress)
    if "pixels_distribution" in metrics:
        if jm.should_stop(job_id):
            return None
        result["pixel_distribution"] = _run_pixel_validation_metric(
            validator, dataset, job_id, current_progress, progress_per_metric
        )
        current_progress += progress_per_metric
        jm.update_progress(job_id, current_progress)
    return result, image_records

###############################################################################
def _load_validation_dataset(
    repository: DatasetRepository,
    sample_size: Any,
    seed: Any,
    dataset_name: Any,
) -> pd.DataFrame:
    return repository.load_source_dataset(
        sample_size=sample_size,
        seed=seed,
        dataset_name=dataset_name,
    )

###############################################################################
def _run_text_validation_metric(
    validator: DatasetValidator,
    dataset: pd.DataFrame,
) -> dict[str, Any]:
    logger.info(f"[1/3] Calculating text statistics for {len(dataset)} reports...")
    text_stats, _text_records_df = validator.calculate_text_statistics()
    logger.info(
        f"[1/3] Text statistics complete: {text_stats.total_words} total words, {text_stats.unique_words} unique"
    )
    return {
        "count": text_stats.count,
        "total_words": text_stats.total_words,
        "unique_words": text_stats.unique_words,
        "avg_words_per_report": text_stats.avg_words_per_report,
        "min_words_per_report": text_stats.min_words_per_report,
        "max_words_per_report": text_stats.max_words_per_report,
    }

###############################################################################
def _run_image_validation_metric(
    validator: DatasetValidator,
    dataset: pd.DataFrame,
    job_id: str,
    current_progress: float,
    progress_per_metric: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    logger.info(
        f"[2/3] Calculating image statistics for {len(dataset)} images (this may take a while)..."
    )
    progress_range = ProgressRange(
        job_id, current_progress, current_progress + progress_per_metric
    )
    image_stats, image_records_df = validator.calculate_image_statistics(
        progress_callback=progress_range.update,
    )
    logger.info(f"[2/3] Image statistics complete: analyzed {image_stats.count} images")
    image_records = (
        image_records_df.to_dict(orient="records")
        if not image_records_df.empty
        else []
    )
    return {
        "count": image_stats.count,
        "mean_height": image_stats.mean_height,
        "mean_width": image_stats.mean_width,
        "mean_pixel_value": image_stats.mean_pixel_value,
        "std_pixel_value": image_stats.std_pixel_value,
        "mean_noise_std": image_stats.mean_noise_std,
        "mean_noise_ratio": image_stats.mean_noise_ratio,
    }, image_records

###############################################################################
def _run_pixel_validation_metric(
    validator: DatasetValidator,
    dataset: pd.DataFrame,
    job_id: str,
    current_progress: float,
    progress_per_metric: float,
) -> dict[str, Any]:
    logger.info(
        f"[3/3] Calculating pixel intensity distribution for {len(dataset)} images..."
    )
    progress_range = ProgressRange(
        job_id, current_progress, current_progress + progress_per_metric
    )
    pixel_dist = validator.calculate_pixel_distribution(
        progress_callback=progress_range.update,
    )
    logger.info("[3/3] Pixel distribution complete")
    return {"bins": pixel_dist.bins, "counts": pixel_dist.counts}

###############################################################################
def _save_validation_report(
    repository: ValidationRepository,
    dataset_name: str,
    sample_size: Any,
    metrics: Any,
    result: dict[str, Any],
    image_records: list[dict[str, Any]],
) -> None:
    repository.save_validation_report(
        {
            "dataset_name": dataset_name,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "sample_size": sample_size,
            "metrics": metrics,
            "text_statistics": result.get("text_statistics"),
            "image_statistics": result.get("image_statistics"),
            "pixel_distribution": result.get("pixel_distribution"),
            "artifacts": None,
            "image_records": image_records,
        }
    )

###############################################################################
def run_checkpoint_evaluation_job(
    request_data: dict[str, Any],
    job_id: str,
) -> dict[str, Any]:
    """Blocking checkpoint evaluation function that runs in background thread."""
    jm = get_job_manager()
    raw_checkpoint = request_data.get("checkpoint", "")
    try:
        checkpoint = validate_checkpoint_name(str(raw_checkpoint))
    except ValueError as exc:
        return {
            "success": False,
            "message": str(exc),
            "results": None,
        }
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
    checkpoint_data = _load_checkpoint_for_evaluation(checkpoint)
    if checkpoint_data is None:
        return {
            "success": False,
            "message": f"Checkpoint not found: {checkpoint}",
            "results": None,
        }
    model, train_config, model_metadata = checkpoint_data
    model.summary(expand_nested=True)
    jm.update_progress(job_id, 30.0)
    if jm.should_stop(job_id):
        return {}
    evaluator = CheckpointEvaluator(model, train_config, model_metadata)
    validation_data = _load_checkpoint_validation_data(metrics, train_config)
    metric_results = _run_checkpoint_metrics(
        evaluator,
        train_config,
        validation_data,
        metrics,
        metric_configs,
        num_samples,
        seed,
        job_id,
        jm,
    )
    if metric_results is None:
        return {}
    results, resolved_metric_configs = metric_results
    jm.update_progress(job_id, 100.0)
    _save_checkpoint_evaluation_report(
        checkpoint,
        metrics,
        resolved_metric_configs,
        results,
    )
    return {
        "success": True,
        "message": f"Evaluation completed for {checkpoint}",
        "results": results,
    }

###############################################################################
def _run_checkpoint_metrics(
    evaluator: CheckpointEvaluator,
    train_config: Any,
    validation_data: pd.DataFrame | None,
    metrics: Any,
    metric_configs: dict[str, Any],
    num_samples: Any,
    seed: Any,
    job_id: str,
    jm: JobManager,
) -> tuple[dict[str, Any], dict[str, dict[str, float | int]]] | None:
    results: dict[str, Any] = {}
    resolved_metric_configs: dict[str, dict[str, float | int]] = {}
    if "evaluation_report" in metrics:
        if jm.should_stop(job_id):
            return None
        metric_results, metric_config = _run_evaluation_report_metric(
            evaluator,
            train_config,
            validation_data,
            metric_configs.get("evaluation_report"),
            seed,
        )
        results.update(metric_results)
        if metric_config is not None:
            resolved_metric_configs["evaluation_report"] = metric_config
        jm.update_progress(job_id, 60.0)
    if "bleu_score" in metrics:
        if jm.should_stop(job_id):
            return None
        bleu_result, bleu_config = _run_bleu_metric(
            evaluator,
            validation_data,
            metric_configs.get("bleu_score"),
            num_samples,
        )
        if bleu_result is not None:
            results["bleu_score"] = bleu_result
        if bleu_config is not None:
            resolved_metric_configs["bleu_score"] = bleu_config
        jm.update_progress(job_id, 90.0)
    return results, resolved_metric_configs

###############################################################################
def _load_checkpoint_for_evaluation(
    checkpoint: str,
) -> tuple[Any, Any, Any] | None:
    checkpoint_record = CheckpointRepository().get_checkpoint(checkpoint)
    if checkpoint_record is None or not checkpoint_record.artifact_complete:
        return None
    try:
        model, train_config, model_metadata, _, _ = ModelSerializer().load_checkpoint(
            checkpoint_record.path
        )
    except FileNotFoundError:
        return None
    return model, train_config, model_metadata

###############################################################################
def _load_checkpoint_validation_data(
    metrics: Any,
    train_config: dict[str, Any],
) -> pd.DataFrame | None:
    if "evaluation_report" not in metrics and "bleu_score" not in metrics:
        return None
    dataset_name = str(train_config.get("dataset_name") or "").strip()
    if not dataset_name:
        raise JobExecutionError(
            "Checkpoint configuration does not identify its processed dataset.",
            code="checkpoint_dataset_unavailable",
            phase="input_validation",
        )

    repository = DatasetRepository()
    _, validation_data, _ = repository.load_training_data(dataset_name=dataset_name)
    if not isinstance(validation_data, pd.DataFrame) or validation_data.empty:
        raise JobExecutionError(
            f"No validation data is available for checkpoint dataset '{dataset_name}'.",
            code="checkpoint_dataset_unavailable",
            phase="input_validation",
        )
    try:
        return repository.validate_img_paths(validation_data)
    except DatasetIntegrityError as exc:
        raise JobExecutionError(
            f"Checkpoint dataset '{dataset_name}' failed image-path validation: {exc}",
            code="dataset_integrity_failed",
            phase="input_validation",
        ) from exc

###############################################################################
def _run_evaluation_report_metric(
    evaluator: CheckpointEvaluator,
    train_config: Any,
    validation_data: pd.DataFrame | None,
    metric_config: Any,
    seed: Any,
) -> tuple[dict[str, Any], dict[str, float] | None]:
    logger.info("Running evaluation report (loss and accuracy)...")
    if validation_data is None or validation_data.empty:
        logger.warning("No validation data available for evaluation report")
        return {}, None
    evaluation_fraction = resolve_metric_fraction(metric_config, default_fraction=1.0)
    eval_data = validation_data
    if evaluation_fraction < 1.0:
        eval_data = validation_data.sample(frac=evaluation_fraction, random_state=seed)
    if eval_data.empty:
        return {}, {"data_fraction": evaluation_fraction}
    validation_dataset = XRAYDataLoader(train_config).build_training_dataloader(eval_data)
    try:
        evaluator.preflight_validation_dataset(validation_dataset)
        eval_results = evaluator.evaluate_model(validation_dataset)
    except CheckpointInputMismatchError as exc:
        raise JobExecutionError(
            str(exc),
            code="checkpoint_input_mismatch",
            phase="input_validation",
        ) from exc
    return {
        "loss": eval_results.get("loss"),
        "accuracy": eval_results.get("accuracy"),
    }, {"data_fraction": evaluation_fraction}

###############################################################################
def _run_bleu_metric(
    evaluator: CheckpointEvaluator,
    validation_data: pd.DataFrame | None,
    metric_config: Any,
    num_samples: Any,
) -> tuple[float | None, dict[str, float | int] | None]:
    logger.info(f"Calculating BLEU score with {num_samples} samples...")
    if validation_data is None or validation_data.empty:
        logger.warning("No validation data available for BLEU calculation")
        return None, None
    bleu_fraction = resolve_metric_fraction(metric_config, default_fraction=1.0)
    if metric_config is None:
        bleu_fraction = 1.0
    bleu_samples = max(1, int(num_samples))
    if bleu_fraction < 1.0:
        bleu_samples = max(1, int(len(validation_data) * bleu_fraction))
    config = {"data_fraction": bleu_fraction, "num_samples": bleu_samples}
    return evaluator.calculate_bleu_score(validation_data, num_samples=bleu_samples), config

###############################################################################
def _save_checkpoint_evaluation_report(
    checkpoint: str,
    metrics: Any,
    metric_configs: dict[str, dict[str, float | int]],
    results: dict[str, Any],
) -> None:
    try:
        ValidationRepository().save_checkpoint_evaluation_report(
            {
                "checkpoint": checkpoint,
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "metrics": metrics,
                "metric_configs": metric_configs,
                "results": results,
            }
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to save checkpoint evaluation report: %s", exc)

###############################################################################
class ValidationService:
    """Endpoint for dataset validation and checkpoint evaluation analytics."""

    JOB_TYPE_VALIDATION = "validation"
    JOB_TYPE_EVALUATION = "checkpoint_evaluation"

    # -------------------------------------------------------------------------
    def __init__(
        self,
        job_manager: JobManager,
        server_settings: ServerSettings,
        checkpoint_repository: CheckpointRepository | None = None,
    ) -> None:
        self.job_manager = job_manager
        self.server_settings = server_settings
        self.checkpoint_repository = checkpoint_repository or CheckpointRepository()

    # -------------------------------------------------------------------------
    async def run_validation(self, request: ValidationRequest) -> JobStartResponse:
        """Run validation analytics on the current dataset."""
        if self.job_manager.is_job_running("validation"):
            raise ConflictError(
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
            raise InternalServiceError(
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
        serializer = ValidationRepository()
        report = serializer.get_validation_report(dataset_name)
        if report is None:
            raise NotFoundError(
                detail=f"No validation report found for dataset: {dataset_name}",
            )
        return ValidationReportResponse(**report)

    # -------------------------------------------------------------------------
    async def get_checkpoint_evaluation_report(
        self, checkpoint: str
    ) -> CheckpointEvaluationReportResponse:
        try:
            checkpoint_name = validate_checkpoint_name(checkpoint)
        except ValueError as exc:
            raise BadRequestError(
                detail=str(exc),
            ) from exc
        serializer = ValidationRepository()
        report = serializer.get_checkpoint_evaluation_report(checkpoint_name)
        if report is None:
            raise NotFoundError(
                detail=f"No evaluation report found for checkpoint: {checkpoint_name}",
            )
        return CheckpointEvaluationReportResponse(**report)

    # -------------------------------------------------------------------------
    async def evaluate_checkpoint(
        self,
        request: CheckpointEvaluationRequest,
    ) -> JobStartResponse:
        """Evaluate a model checkpoint using selected metrics."""
        if self.job_manager.is_job_running("checkpoint_evaluation"):
            raise ConflictError(
                detail="Checkpoint evaluation is already in progress",
            )

        try:
            checkpoint_name = validate_checkpoint_name(request.checkpoint)
        except ValueError as exc:
            raise BadRequestError(
                detail=str(exc),
            ) from exc

        checkpoint_record = self.checkpoint_repository.get_checkpoint(checkpoint_name)
        if checkpoint_record is None:
            raise NotFoundError(detail=f"Checkpoint is not registered: {checkpoint_name}")
        if not checkpoint_record.artifact_complete:
            raise ConflictError(
                detail=f"Checkpoint artifact is missing or incomplete: {checkpoint_name}"
            )

        request_data = request.model_dump()
        request_data["checkpoint"] = checkpoint_name

        # Start background job
        job_id = self.job_manager.start_job(
            job_type="checkpoint_evaluation",
            runner=run_checkpoint_evaluation_job,
            kwargs={
                "request_data": request_data,
            },
        )

        job_status = self.job_manager.get_job_status(job_id)
        if job_status is None:
            raise InternalServiceError(
                detail="Failed to initialize checkpoint evaluation job",
            )

        return JobStartResponse(
            job_id=job_id,
            job_type=job_status["job_type"],
            status=job_status["status"],
            message=f"Checkpoint evaluation job started for {checkpoint_name}",
            poll_interval=self.server_settings.jobs.polling_interval,
        )

###############################################################################
@lru_cache(maxsize=1)
def get_validation_service() -> ValidationService:
    return ValidationService(
        job_manager=get_job_manager(),
        server_settings=get_server_settings(),
        checkpoint_repository=CheckpointRepository(),
    )



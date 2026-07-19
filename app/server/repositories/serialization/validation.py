from __future__ import annotations

from typing import Any

import pandas as pd
from sqlalchemy import exists, select

from server.common.constants import CHECKPOINT_EVALUATIONS_TABLE
from server.repositories.schemas import (
    Checkpoint,
    CheckpointEvaluation,
    Dataset,
    ValidationRun,
)
from server.repositories.schemas.normalization import normalize_key
from server.repositories.serialization.support import RepositorySupport

###############################################################################
class ValidationRepository(RepositorySupport):
    """Persistence boundary for validation and checkpoint evaluation reports."""

    # -------------------------------------------------------------------------
    def save_validation_report(self, report: dict[str, Any]) -> None:
        dataset_name = str(report.get("dataset_name") or "").strip()
        if not dataset_name:
            raise ValueError("Validation report requires dataset_name")
        with self.database.transaction() as session:
            dataset = session.execute(
                select(Dataset).where(Dataset.name_key == normalize_key(dataset_name))
            ).scalar_one_or_none()
            if dataset is None:
                dataset = Dataset(
                    name=dataset_name,
                    name_key=normalize_key(dataset_name),
                    created_at=self._now_utc(),
                )
                session.add(dataset)
                session.flush()
            text_stats = report.get("text_statistics") or {}
            image_stats = report.get("image_statistics") or {}
            pixel_distribution = report.get("pixel_distribution") or {}
            session.add(
                ValidationRun(
                    dataset_id=dataset.dataset_id,
                    executed_at=self._coerce_datetime(report.get("date")),
                    sample_size=float(report.get("sample_size") or 1.0),
                    metrics_json=report.get("metrics") or [],
                    artifacts_json=report.get("artifacts") or {},
                    status="succeeded",
                    text_count=int(text_stats.get("count", 0) or 0),
                    text_total_words=int(text_stats.get("total_words", 0) or 0),
                    text_unique_words=int(text_stats.get("unique_words", 0) or 0),
                    text_avg_words=float(
                        text_stats.get("avg_words_per_report", 0.0) or 0.0
                    ),
                    text_min_words=int(
                        text_stats.get("min_words_per_report", 0) or 0
                    ),
                    text_max_words=int(
                        text_stats.get("max_words_per_report", 0) or 0
                    ),
                    image_count=int(image_stats.get("count", 0) or 0),
                    image_mean_height=float(
                        image_stats.get("mean_height", 0.0) or 0.0
                    ),
                    image_mean_width=float(
                        image_stats.get("mean_width", 0.0) or 0.0
                    ),
                    image_mean_value=float(
                        image_stats.get("mean_pixel_value", 0.0) or 0.0
                    ),
                    image_std_value=float(
                        image_stats.get("std_pixel_value", 0.0) or 0.0
                    ),
                    image_mean_noise_std=float(
                        image_stats.get("mean_noise_std", 0.0) or 0.0
                    ),
                    image_mean_noise_ratio=float(
                        image_stats.get("mean_noise_ratio", 0.0) or 0.0
                    ),
                    pixel_bins_json=pixel_distribution.get("bins") or [],
                    pixel_counts_json=pixel_distribution.get("counts") or [],
                )
            )

    # -------------------------------------------------------------------------
    def get_validation_report(self, dataset_name: str) -> dict[str, Any] | None:
        dataset_id = self._get_dataset_id(str(dataset_name or "").strip())
        if dataset_id is None:
            return None
        with self.database.read_session() as session:
            row = session.execute(
                select(
                    ValidationRun.validation_run_id,
                    ValidationRun.executed_at,
                    ValidationRun.sample_size,
                    ValidationRun.metrics_json,
                    ValidationRun.artifacts_json,
                    ValidationRun.text_count,
                    ValidationRun.text_total_words,
                    ValidationRun.text_unique_words,
                    ValidationRun.text_avg_words,
                    ValidationRun.text_min_words,
                    ValidationRun.text_max_words,
                    ValidationRun.image_count,
                    ValidationRun.image_mean_height,
                    ValidationRun.image_mean_width,
                    ValidationRun.image_mean_value,
                    ValidationRun.image_std_value,
                    ValidationRun.image_mean_noise_std,
                    ValidationRun.image_mean_noise_ratio,
                    ValidationRun.pixel_bins_json,
                    ValidationRun.pixel_counts_json,
                )
                .where(ValidationRun.dataset_id == dataset_id)
                .order_by(ValidationRun.validation_run_id.desc())
                .limit(1)
            ).first()
        if row is None:
            return None

        metrics = self._parse_json(row[3], default=[])
        artifacts = self._parse_json(row[4], default={})
        text_statistics = None
        if row[5] is not None:
            text_statistics = {
                "count": int(row[5] or 0),
                "total_words": int(row[6] or 0),
                "unique_words": int(row[7] or 0),
                "avg_words_per_report": float(row[8] or 0.0),
                "min_words_per_report": int(row[9] or 0),
                "max_words_per_report": int(row[10] or 0),
            }

        image_statistics = None
        if int(row[11] or 0) > 0:
            image_statistics = {
                "count": int(row[11] or 0),
                "mean_height": float(row[12] or 0.0),
                "mean_width": float(row[13] or 0.0),
                "mean_pixel_value": float(row[14] or 0.0),
                "std_pixel_value": float(row[15] or 0.0),
                "mean_noise_std": float(row[16] or 0.0),
                "mean_noise_ratio": float(row[17] or 0.0),
            }

        pixel_distribution = None
        pixel_bins = self._parse_json(row[18], default=[])
        pixel_counts = self._parse_json(row[19], default=[])
        if pixel_bins and pixel_counts:
            pixel_distribution = {
                "bins": [int(value) for value in pixel_bins],
                "counts": [int(value) for value in pixel_counts],
            }

        return {
            "dataset_name": dataset_name,
            "date": self._format_datetime(row[1]),
            "sample_size": float(row[2] or 0.0),
            "metrics": metrics if isinstance(metrics, list) else [],
            "text_statistics": text_statistics,
            "image_statistics": image_statistics,
            "pixel_distribution": pixel_distribution,
            "artifacts": artifacts if isinstance(artifacts, dict) else {},
        }

    # -------------------------------------------------------------------------
    def validation_report_exists(self, dataset_name: str) -> bool:
        dataset_id = self._get_dataset_id(str(dataset_name or "").strip())
        if dataset_id is None:
            return False
        with self.database.read_session() as session:
            return bool(
                session.execute(
                    select(exists().where(ValidationRun.dataset_id == dataset_id))
                ).scalar()
            )

    # -------------------------------------------------------------------------
    def save_checkpoint_evaluation_report(self, report: dict[str, Any]) -> None:
        checkpoint = str(report.get("checkpoint") or "").strip()
        if not checkpoint:
            raise ValueError("Checkpoint evaluation report requires a checkpoint name")
        self.upsert_table(
            pd.DataFrame(
                [
                    {
                        "checkpoint_id": self._ensure_checkpoint(checkpoint),
                        "executed_at": self._coerce_datetime(report.get("date")),
                        "metrics_json": report.get("metrics") or [],
                        "metric_configs_json": report.get("metric_configs") or {},
                        "results_json": report.get("results") or {},
                    }
                ]
            ),
            CHECKPOINT_EVALUATIONS_TABLE,
        )

    # -------------------------------------------------------------------------
    def get_checkpoint_evaluation_report(
        self, checkpoint: str
    ) -> dict[str, Any] | None:
        checkpoint_name = str(checkpoint or "").strip()
        if not checkpoint_name:
            return None
        with self.database.read_session() as session:
            row = session.execute(
                select(
                    CheckpointEvaluation.executed_at,
                    CheckpointEvaluation.metrics_json,
                    CheckpointEvaluation.metric_configs_json,
                    CheckpointEvaluation.results_json,
                )
                .join(
                    Checkpoint,
                    Checkpoint.checkpoint_id == CheckpointEvaluation.checkpoint_id,
                )
                .where(Checkpoint.name_key == normalize_key(checkpoint_name))
                .order_by(CheckpointEvaluation.evaluation_id.desc())
                .limit(1)
            ).first()
        if row is None:
            return None
        metrics = self._parse_json(row[1], default=[])
        metric_configs = self._parse_json(row[2], default={})
        results = self._parse_json(row[3], default={})
        return {
            "checkpoint": checkpoint_name,
            "date": self._format_datetime(row[0]),
            "metrics": metrics if isinstance(metrics, list) else [],
            "metric_configs": metric_configs if isinstance(metric_configs, dict) else {},
            "results": results if isinstance(results, dict) else {},
        }

    # -------------------------------------------------------------------------
    def checkpoint_evaluation_report_exists(self, checkpoint: str) -> bool:
        checkpoint_name = str(checkpoint or "").strip()
        if not checkpoint_name:
            return False
        with self.database.read_session() as session:
            return bool(
                session.execute(
                    select(
                        exists().where(
                            CheckpointEvaluation.checkpoint_id
                            == Checkpoint.checkpoint_id,
                            Checkpoint.name_key == normalize_key(checkpoint_name),
                        )
                    )
                ).scalar()
            )

    # -------------------------------------------------------------------------
    def list_checkpoint_evaluations(
        self, checkpoint: str | None = None, *, limit: int = 50, offset: int = 0
    ) -> list[dict[str, Any]]:
        if limit < 1 or limit > 500:
            raise ValueError("limit must be between 1 and 500")
        if offset < 0:
            raise ValueError("offset must be >= 0")
        statement = (
            select(
                CheckpointEvaluation.request_id,
                CheckpointEvaluation.status,
                CheckpointEvaluation.executed_at,
                CheckpointEvaluation.metrics_json,
                CheckpointEvaluation.results_json,
                Checkpoint.name,
            )
            .join(Checkpoint, Checkpoint.checkpoint_id == CheckpointEvaluation.checkpoint_id)
            .order_by(
                CheckpointEvaluation.executed_at.desc(),
                CheckpointEvaluation.evaluation_id.desc(),
            )
            .limit(limit)
            .offset(offset)
        )
        if checkpoint:
            statement = statement.where(
                Checkpoint.name_key == normalize_key(checkpoint)
            )
        with self.database.read_session() as session:
            rows = session.execute(statement).all()
        return [
            {
                "request_id": row[0],
                "status": row[1],
                "date": self._format_datetime(row[2]),
                "metrics": self._parse_json(row[3], default=[]),
                "results": self._parse_json(row[4], default={}),
                "checkpoint": row[5],
            }
            for row in rows
        ]

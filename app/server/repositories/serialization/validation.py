from __future__ import annotations

from typing import Any

import pandas as pd
from sqlalchemy import exists, func, select

from server.common.constants import CHECKPOINT_EVALUATIONS_TABLE
from server.repositories.schemas.studio import (
    Checkpoint,
    CheckpointEvaluation,
    Dataset,
    DatasetRecord,
    DatasetVersion,
    RecordImageStat,
    RecordTextStat,
    ValidationAggregate,
    ValidationHistogramBin,
    ValidationRun,
    ValidationRunMetric,
)
from server.repositories.schemas.normalization import normalize_key
from server.repositories.checkpoints import CheckpointRepository
from server.repositories.serialization.support import RepositorySupport

RECORD_STAT_TABLES = {"text": RecordTextStat, "image": RecordImageStat}
AGGREGATE_VALUE_COLUMNS = (
    "total",
    "minimum",
    "maximum",
    "mean",
    "median",
    "std",
    "p05",
    "p25",
    "p75",
    "p95",
)

###############################################################################
class ValidationRepository(RepositorySupport):
    """Persistence boundary for validation and checkpoint evaluation reports."""

    # -------------------------------------------------------------------------
    def __init__(
        self,
        database=None,
        checkpoint_repository: CheckpointRepository | None = None,
    ) -> None:
        super().__init__(database)
        self.checkpoint_repository = checkpoint_repository or CheckpointRepository(
            self.database
        )

    # -------------------------------------------------------------------------
    def _get_dataset_id(self, dataset_name: str) -> int | None:
        with self.database.read_session() as session:
            dataset_id = session.execute(
                select(Dataset.dataset_id).where(
                    Dataset.name_key == normalize_key(dataset_name)
                )
            ).scalar_one_or_none()
        return int(dataset_id) if dataset_id is not None else None

    # -------------------------------------------------------------------------
    def save_validation_run(self, run: dict[str, Any]) -> int:
        """Persist one validation run and all of its statistics atomically.

        Every call inserts a new run; earlier runs are never modified.
        """
        dataset_name = str(run.get("dataset_name") or "").strip()
        if not dataset_name:
            raise ValueError("Validation run requires dataset_name")
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
            version_id = session.execute(
                select(DatasetVersion.dataset_version_id)
                .where(DatasetVersion.dataset_id == dataset.dataset_id)
                .order_by(DatasetVersion.version_number.desc())
                .limit(1)
            ).scalar_one_or_none()
            record = ValidationRun(
                dataset_id=dataset.dataset_id,
                dataset_version_id=version_id,
                executed_at=self._coerce_datetime(run.get("date")),
                sample_size=float(run.get("sample_size") or 1.0),
                seed=run.get("seed"),
                records_requested=run.get("records_requested"),
                records_evaluated=run.get("records_evaluated"),
                text_vocabulary_size=run.get("text_vocabulary_size"),
                tokenizer=run.get("tokenizer"),
                max_report_size=run.get("max_report_size"),
                config_json=run.get("config"),
            )
            session.add(record)
            session.flush()
            run_id = int(record.validation_run_id)
            session.add_all(
                ValidationRunMetric(
                    validation_run_id=run_id,
                    metric=str(item["metric"]),
                    records_computed=item.get("records_computed"),
                )
                for item in run.get("metrics") or []
            )
            session.add_all(
                RecordTextStat(validation_run_id=run_id, **row)
                for row in run.get("text_records") or []
                if row.get("record_id") is not None
            )
            session.add_all(
                RecordImageStat(validation_run_id=run_id, **row)
                for row in run.get("image_records") or []
                if row.get("record_id") is not None
            )
            session.add_all(
                ValidationAggregate(validation_run_id=run_id, **row)
                for row in run.get("aggregates") or []
            )
            for name, bins in (run.get("histograms") or {}).items():
                session.add_all(
                    ValidationHistogramBin(
                        validation_run_id=run_id,
                        histogram=str(name),
                        bin_index=index,
                        bin_lower=float(lower),
                        bin_upper=float(upper),
                        count=int(count),
                    )
                    for index, (lower, upper, count) in enumerate(bins)
                )
        return run_id

    # -------------------------------------------------------------------------
    @staticmethod
    def _run_row(session: Any, dataset_id: int, run_id: int | None) -> ValidationRun | None:
        statement = select(ValidationRun).where(ValidationRun.dataset_id == dataset_id)
        if run_id is not None:
            statement = statement.where(ValidationRun.validation_run_id == run_id)
        return session.execute(
            statement.order_by(ValidationRun.validation_run_id.desc()).limit(1)
        ).scalar_one_or_none()

    # -------------------------------------------------------------------------
    def _summarize_run(self, session: Any, run: ValidationRun) -> dict[str, Any]:
        metrics = session.execute(
            select(ValidationRunMetric)
            .where(ValidationRunMetric.validation_run_id == run.validation_run_id)
            .order_by(ValidationRunMetric.metric)
        ).scalars()
        return {
            "validation_run_id": int(run.validation_run_id),
            "date": self._format_datetime(run.executed_at),
            "sample_size": float(run.sample_size),
            "seed": run.seed,
            "records_requested": run.records_requested,
            "records_evaluated": run.records_evaluated,
            "text_vocabulary_size": run.text_vocabulary_size,
            "tokenizer": run.tokenizer,
            "max_report_size": run.max_report_size,
            "metrics": [
                {"metric": item.metric, "records_computed": item.records_computed}
                for item in metrics
            ],
        }

    # -------------------------------------------------------------------------
    def get_validation_report(
        self, dataset_name: str, run_id: int | None = None
    ) -> dict[str, Any] | None:
        """Return one run (the latest unless `run_id` is given) with its aggregates."""
        dataset_id = self._get_dataset_id(str(dataset_name or "").strip())
        if dataset_id is None:
            return None
        with self.database.read_session() as session:
            run = self._run_row(session, dataset_id, run_id)
            if run is None:
                return None
            summary = self._summarize_run(session, run)
            aggregates = session.execute(
                select(ValidationAggregate)
                .where(ValidationAggregate.validation_run_id == run.validation_run_id)
                .order_by(ValidationAggregate.domain, ValidationAggregate.measure)
            ).scalars()
            aggregate_rows = [
                {
                    "domain": item.domain,
                    "measure": item.measure,
                    "count": item.count,
                    **{name: getattr(item, name) for name in AGGREGATE_VALUE_COLUMNS},
                }
                for item in aggregates
            ]
            histogram_rows = session.execute(
                select(ValidationHistogramBin)
                .where(
                    ValidationHistogramBin.validation_run_id == run.validation_run_id
                )
                .order_by(
                    ValidationHistogramBin.histogram, ValidationHistogramBin.bin_index
                )
            ).scalars()
            histograms: dict[str, list[dict[str, float]]] = {}
            for item in histogram_rows:
                histograms.setdefault(item.histogram, []).append(
                    {
                        "lower": item.bin_lower,
                        "upper": item.bin_upper,
                        "count": int(item.count),
                    }
                )
        return {
            "dataset_name": dataset_name,
            "run": summary,
            "aggregates": aggregate_rows,
            "histograms": [
                {"name": name, "bins": bins} for name, bins in histograms.items()
            ],
        }

    # -------------------------------------------------------------------------
    def list_validation_runs(
        self, dataset_name: str, *, limit: int = 50, offset: int = 0
    ) -> list[dict[str, Any]] | None:
        if limit < 1 or limit > 500:
            raise ValueError("limit must be between 1 and 500")
        if offset < 0:
            raise ValueError("offset must be >= 0")
        dataset_id = self._get_dataset_id(str(dataset_name or "").strip())
        if dataset_id is None:
            return None
        with self.database.read_session() as session:
            runs = (
                session.execute(
                    select(ValidationRun)
                    .where(ValidationRun.dataset_id == dataset_id)
                    .order_by(ValidationRun.validation_run_id.desc())
                    .limit(limit)
                    .offset(offset)
                )
                .scalars()
                .all()
            )
            return [self._summarize_run(session, run) for run in runs]

    # -------------------------------------------------------------------------
    def list_run_records(
        self,
        dataset_name: str,
        run_id: int,
        kind: str,
        *,
        sort: str | None = None,
        descending: bool = False,
        search: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any] | None:
        """Page per-record statistics of one run (`kind` is `text` or `image`)."""
        table = RECORD_STAT_TABLES.get(kind)
        if table is None:
            raise ValueError(f"Unsupported record statistics kind: {kind}")
        if limit < 1 or limit > 500:
            raise ValueError("limit must be between 1 and 500")
        if offset < 0:
            raise ValueError("offset must be >= 0")
        value_columns = [
            column.name
            for column in table.__table__.columns
            if column.name not in ("validation_run_id", "record_id")
        ]
        sort_name = sort or "image_name"
        if sort_name != "image_name" and sort_name not in value_columns:
            raise ValueError(f"Unsupported sort column: {sort_name}")
        dataset_id = self._get_dataset_id(str(dataset_name or "").strip())
        if dataset_id is None:
            return None
        sort_column = (
            DatasetRecord.image_name
            if sort_name == "image_name"
            else getattr(table, sort_name)
        )
        filters = [
            table.validation_run_id == run_id,
            DatasetRecord.dataset_id == dataset_id,
        ]
        if search:
            filters.append(DatasetRecord.image_name.icontains(search, autoescape=True))
        with self.database.read_session() as session:
            if self._run_row(session, dataset_id, run_id) is None:
                return None
            total = session.execute(
                select(func.count())
                .select_from(table)
                .join(DatasetRecord, DatasetRecord.record_id == table.record_id)
                .where(*filters)
            ).scalar_one()
            ordering = sort_column.desc() if descending else sort_column.asc()
            rows = session.execute(
                select(table, DatasetRecord.image_name)
                .join(DatasetRecord, DatasetRecord.record_id == table.record_id)
                .where(*filters)
                .order_by(ordering, table.record_id)
                .limit(limit)
                .offset(offset)
            ).all()
        return {
            "total": int(total),
            "rows": [
                {
                    "record_id": row[0].record_id,
                    "image_name": row[1],
                    **{name: getattr(row[0], name) for name in value_columns},
                }
                for row in rows
            ],
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
        checkpoint_record = self.checkpoint_repository.get_checkpoint(checkpoint)
        if checkpoint_record is None:
            raise ValueError(f"Checkpoint is not registered: {checkpoint}")
        self.upsert_table(
            pd.DataFrame(
                [
                    {
                        "checkpoint_id": checkpoint_record.checkpoint_id,
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
            "metric_configs": metric_configs
            if isinstance(metric_configs, dict)
            else {},
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
                CheckpointEvaluation.executed_at,
                CheckpointEvaluation.metrics_json,
                CheckpointEvaluation.results_json,
                Checkpoint.name,
            )
            .join(
                Checkpoint,
                Checkpoint.checkpoint_id == CheckpointEvaluation.checkpoint_id,
            )
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
                "date": self._format_datetime(row[0]),
                "metrics": self._parse_json(row[1], default=[]),
                "results": self._parse_json(row[2], default={}),
                "checkpoint": row[3],
            }
            for row in rows
        ]

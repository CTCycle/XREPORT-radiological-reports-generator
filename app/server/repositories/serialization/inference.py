from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.orm import selectinload

from server.repositories.checkpoints import CheckpointRepository
from server.repositories.schemas import InferenceReport, InferenceRun
from server.repositories.schemas.normalization import normalize_key
from server.repositories.serialization.support import RepositorySupport

###############################################################################
class InferenceRepository(RepositorySupport):
    """Persistence boundary for inference and checkpoint history."""

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
    def save_generated_reports(
        self,
        reports: list[dict[str, str]],
        *,
        provider: str,
        model_ref: str,
        model_revision: str | None,
        generation_profile: str,
        generation_config: dict[str, Any],
        clinical_context: str,
        request_id: str | None,
        status: str,
        execution_time_seconds: float | None,
        executed_at: datetime | None = None,
    ) -> None:
        if not reports:
            return
        normalized_request_id = (
            str(request_id or "").strip() or f"gen_{uuid.uuid4().hex[:12]}"
        )
        checkpoint_id: int | None = None
        if provider == "xreport":
            checkpoint = self.checkpoint_repository.get_checkpoint(
                model_ref.removeprefix("xreport:")
            )
            if checkpoint is None:
                raise ValueError(
                    f"Checkpoint is not registered: {model_ref.removeprefix('xreport:')}"
                )
            checkpoint_id = checkpoint.checkpoint_id
        with self.database.transaction() as session:
            run = session.execute(
                select(InferenceRun).where(
                    InferenceRun.request_id == normalized_request_id
                )
            ).scalar_one_or_none()
            values = {
                "checkpoint_id": checkpoint_id,
                "provider": provider,
                "model_ref": model_ref,
                "model_revision": model_revision,
                "generation_profile": generation_profile,
                "generation_config_json": generation_config,
                "clinical_context": clinical_context.strip() or None,
                "status": status,
                "execution_time_seconds": execution_time_seconds,
                "executed_at": executed_at or self._now_utc(),
            }
            if run is None:
                run = InferenceRun(request_id=normalized_request_id, **values)
                session.add(run)
                session.flush()
            else:
                for key, value in values.items():
                    setattr(run, key, value)
                session.execute(
                    delete(InferenceReport).where(
                        InferenceReport.inference_run_id == run.inference_run_id
                    )
                )
            session.add_all(
                InferenceReport(
                    inference_run_id=run.inference_run_id,
                    input_image_name=str(report["image"]),
                    input_image_name_key=normalize_key(str(report["image"])),
                    image_index=index,
                    generated_report=str(report["report"]),
                    record_id=None,
                )
                for index, report in enumerate(reports)
            )

    # -------------------------------------------------------------------------
    @staticmethod
    def _effective_report(report: InferenceReport) -> str:
        return report.edited_report if report.edited_report is not None else report.generated_report

    @classmethod
    def _report_sections(
        cls, run: InferenceRun, report: InferenceReport
    ) -> dict[str, str]:
        configuration = cls._parse_json(run.generation_config_json, default={})
        if not isinstance(configuration, dict):
            return {}
        display_sections = configuration.get("display_sections")
        if not isinstance(display_sections, dict):
            return {}
        values = display_sections.get(report.input_image_name)
        if not isinstance(values, dict):
            values = next(
                (
                    candidate
                    for name, candidate in display_sections.items()
                    if normalize_key(str(name)) == normalize_key(report.input_image_name)
                    and isinstance(candidate, dict)
                ),
                None,
            )
        if not isinstance(values, dict):
            return {}
        return {
            str(section): str(value)
            for section, value in values.items()
            if isinstance(value, str)
        }

    @classmethod
    def _report_summary(cls, run: InferenceRun, report: InferenceReport) -> dict[str, Any]:
        effective = cls._effective_report(report)
        preview = " ".join(effective.split())[:280]
        return {
            "image_index": report.image_index,
            "input_image_name": report.input_image_name,
            "preview": preview,
            "edited": report.edited_report is not None,
            "edited_at": cls._format_datetime(report.edited_at),
        }

    @classmethod
    def _report_detail(cls, run: InferenceRun, report: InferenceReport) -> dict[str, Any]:
        return {
            "image_index": report.image_index,
            "input_image_name": report.input_image_name,
            "generated_report": report.generated_report,
            "edited_report": report.edited_report,
            "effective_report": cls._effective_report(report),
            "edited": report.edited_report is not None,
            "edited_at": cls._format_datetime(report.edited_at),
            "sections": cls._report_sections(run, report),
        }

    @classmethod
    def _run_metadata(cls, run: InferenceRun) -> dict[str, Any]:
        return {
            "request_id": run.request_id,
            "provider": run.provider,
            "model_ref": run.model_ref,
            "model_revision": run.model_revision,
            "generation_profile": run.generation_profile,
            "clinical_context": run.clinical_context,
            "status": run.status,
            "execution_time_seconds": run.execution_time_seconds,
            "date": cls._format_datetime(run.executed_at),
        }

    @classmethod
    def _summary_payload(cls, run: InferenceRun) -> dict[str, Any]:
        reports = sorted(run.reports, key=lambda report: report.image_index)
        payload = cls._run_metadata(run)
        payload.update(
            {
                "reports": [cls._report_summary(run, report) for report in reports],
                "image_names": [report.input_image_name for report in reports],
                "report_count": len(reports),
                "provenance_available": bool(
                    isinstance(cls._parse_json(run.generation_config_json, default={}), dict)
                    and cls._parse_json(run.generation_config_json, default={}).get("provenance")
                ),
            }
        )
        return payload

    @classmethod
    def _detail_payload(cls, run: InferenceRun) -> dict[str, Any]:
        reports = sorted(run.reports, key=lambda report: report.image_index)
        payload = cls._run_metadata(run)
        configuration = cls._parse_json(run.generation_config_json, default={})
        if not isinstance(configuration, dict):
            configuration = {}
        sections: list[str] = []
        for report in reports:
            for section in cls._report_sections(run, report):
                if section not in sections:
                    sections.append(section)
        payload.update(
            {
                "generation_config": configuration,
                "reports": [cls._report_detail(run, report) for report in reports],
                "output_sections": sections,
            }
        )
        return payload

    # -------------------------------------------------------------------------
    def list_inference_history(
        self,
        model_ref: str | None = None,
        status: str | None = None,
        *,
        sort: str = "newest",
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        if limit < 1 or limit > 500:
            raise ValueError("limit must be between 1 and 500")
        if offset < 0:
            raise ValueError("offset must be >= 0")
        if sort not in {"newest", "oldest"}:
            raise ValueError("sort must be newest or oldest")
        filters = []
        if model_ref:
            filters.append(InferenceRun.model_ref == model_ref)
        if status:
            filters.append(InferenceRun.status == status)
        order_direction = "desc" if sort == "newest" else "asc"
        order_date = (
            InferenceRun.executed_at.desc()
            if order_direction == "desc"
            else InferenceRun.executed_at.asc()
        )
        order_id = (
            InferenceRun.inference_run_id.desc()
            if order_direction == "desc"
            else InferenceRun.inference_run_id.asc()
        )
        stmt = (
            select(InferenceRun)
            .options(selectinload(InferenceRun.reports))
            .where(*filters)
            .order_by(order_date, order_id)
            .limit(limit)
            .offset(offset)
        )
        with self.database.read_session() as session:
            total = session.execute(
                select(func.count(InferenceRun.inference_run_id)).where(*filters)
            ).scalar_one()
            runs = session.execute(stmt).scalars().all()
        return {
            "items": [self._summary_payload(run) for run in runs],
            "total": int(total),
            "limit": limit,
            "offset": offset,
        }

    # -------------------------------------------------------------------------
    def get_inference_history(self, request_id: str) -> dict[str, Any] | None:
        normalized_request_id = str(request_id or "").strip()
        if not normalized_request_id:
            return None
        with self.database.read_session() as session:
            run = session.execute(
                select(InferenceRun)
                .options(selectinload(InferenceRun.reports))
                .where(InferenceRun.request_id == normalized_request_id)
            ).scalar_one_or_none()
            return self._detail_payload(run) if run is not None else None

    # -------------------------------------------------------------------------
    def update_inference_reports(
        self, request_id: str, updates: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        normalized_request_id = str(request_id or "").strip()
        image_indexes = [int(update["image_index"]) for update in updates]
        if len(image_indexes) != len(set(image_indexes)):
            raise ValueError("Report image indexes must be unique")
        with self.database.transaction() as session:
            run = session.execute(
                select(InferenceRun)
                .options(selectinload(InferenceRun.reports))
                .where(InferenceRun.request_id == normalized_request_id)
            ).scalar_one_or_none()
            if run is None:
                return None
            reports_by_index = {report.image_index: report for report in run.reports}
            unknown_indexes = sorted(set(image_indexes) - set(reports_by_index))
            if unknown_indexes:
                raise ValueError(
                    "Unknown report image index(es): "
                    + ", ".join(str(index) for index in unknown_indexes)
                )
            edited_at = self._now_utc()
            for update in updates:
                report = reports_by_index[int(update["image_index"])]
                edited_report = str(update["edited_report"])
                if edited_report == report.generated_report:
                    report.edited_report = None
                    report.edited_at = None
                else:
                    report.edited_report = edited_report
                    report.edited_at = edited_at
            session.flush()
            return self._detail_payload(run)

    # -------------------------------------------------------------------------
    def delete_inference_history(self, request_id: str) -> bool:
        normalized_request_id = str(request_id or "").strip()
        with self.database.transaction() as session:
            run = session.execute(
                select(InferenceRun).where(
                    InferenceRun.request_id == normalized_request_id
                )
            ).scalar_one_or_none()
            if run is None:
                return False
            session.delete(run)
            return True

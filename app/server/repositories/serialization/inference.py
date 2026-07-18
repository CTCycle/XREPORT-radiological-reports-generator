from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import delete, select

from server.repositories.schemas import InferenceReport, InferenceRun
from server.repositories.schemas.normalization import normalize_key
from server.repositories.serialization.support import RepositorySupport


###############################################################################
class InferenceRepository(RepositorySupport):
    """Persistence boundary for inference and checkpoint history."""

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
        normalized_request_id = str(request_id or "").strip() or f"gen_{uuid.uuid4().hex[:12]}"
        checkpoint_id: int | None = None
        if provider == "xreport":
            checkpoint_id = self._ensure_checkpoint(model_ref.removeprefix("xreport:"))
        with self.database.transaction() as session:
            run = session.execute(
                select(InferenceRun).where(InferenceRun.request_id == normalized_request_id)
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
    def list_inference_history(
        self, model_ref: str | None = None, *, limit: int = 50, offset: int = 0
    ) -> list[dict[str, Any]]:
        if limit < 1 or limit > 500:
            raise ValueError("limit must be between 1 and 500")
        if offset < 0:
            raise ValueError("offset must be >= 0")
        stmt = (
            select(InferenceRun)
            .order_by(InferenceRun.executed_at.desc(), InferenceRun.inference_run_id.desc())
            .limit(limit)
            .offset(offset)
        )
        if model_ref:
            stmt = stmt.where(InferenceRun.model_ref == model_ref)
        with self.database.read_session() as session:
            runs = session.execute(stmt).scalars().all()
        return [
            {
                "request_id": run.request_id,
                "provider": run.provider,
                "model_ref": run.model_ref,
                "model_revision": run.model_revision,
                "generation_profile": run.generation_profile,
                "generation_config": self._parse_json(run.generation_config_json, default={}),
                "clinical_context": run.clinical_context,
                "status": run.status,
                "execution_time_seconds": run.execution_time_seconds,
                "date": self._format_datetime(run.executed_at),
            }
            for run in runs
        ]

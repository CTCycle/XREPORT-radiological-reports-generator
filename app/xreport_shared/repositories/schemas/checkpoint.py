from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import Integer, String, Text, UniqueConstraint, event
from sqlalchemy.orm import Mapped, mapped_column

from xreport_shared.repositories.schemas.normalization import normalize_key
from xreport_shared.repositories.schemas.types import UTCDateTime

###############################################################################
class CheckpointColumns:
    """Checkpoint registry columns; each application maps its own table."""

    __tablename__ = "checkpoints"
    checkpoint_id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True
    )
    name: Mapped[str] = mapped_column(String, nullable=False)
    name_key: Mapped[str] = mapped_column(String(255), nullable=False)
    path: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime(),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    last_seen_at: Mapped[datetime] = mapped_column(
        UTCDateTime(), nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    __table_args__ = (UniqueConstraint("name_key", name="uq_checkpoints_name_key"),)

###############################################################################
@event.listens_for(CheckpointColumns, "before_insert", propagate=True)
def _populate_checkpoint_name_key(
    _mapper: Any, _connection: Any, target: CheckpointColumns
) -> None:
    target.name_key = normalize_key(target.name)

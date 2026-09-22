"""preserve original inference reports while allowing draft edits"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

import server.repositories.schemas.types


revision: str = "e91a4f6c2d73"
down_revision: Union[str, None] = "f48a7c2e91b6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


###############################################################################
def upgrade() -> None:
    op.add_column(
        "inference_reports",
        sa.Column("edited_report", sa.Text(), nullable=True),
    )
    op.add_column(
        "inference_reports",
        sa.Column(
            "edited_at",
            server.repositories.schemas.types.UTCDateTime(timezone=True),
            nullable=True,
        ),
    )


###############################################################################
def downgrade() -> None:
    op.drop_column("inference_reports", "edited_at")
    op.drop_column("inference_reports", "edited_report")

"""preserve original inference reports while allowing draft edits"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

import server.repositories.schemas.types


revision: str = "e91a4f6c2d73"
down_revision: Union[str, None] = "f48a7c2e91b6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    tables = set(sa.inspect(op.get_bind()).get_table_names())
    if "inference_reports" not in tables:
        return

    columns = {
        column["name"]
        for column in sa.inspect(op.get_bind()).get_columns("inference_reports")
    }
    if "edited_report" not in columns:
        op.add_column(
            "inference_reports",
            sa.Column("edited_report", sa.Text(), nullable=True),
        )
    if "edited_at" not in columns:
        op.add_column(
            "inference_reports",
            sa.Column(
                "edited_at",
                server.repositories.schemas.types.UTCDateTime(timezone=True),
                nullable=True,
            ),
        )


def downgrade() -> None:
    with op.batch_alter_table("inference_reports") as batch_op:
        batch_op.drop_column("edited_at")
        batch_op.drop_column("edited_report")

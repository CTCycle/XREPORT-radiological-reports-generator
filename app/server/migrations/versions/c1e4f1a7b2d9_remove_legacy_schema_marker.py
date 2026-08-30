"""remove the legacy application schema marker"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "c1e4f1a7b2d9"
down_revision: Union[str, None] = "ad03b780d44b"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


###############################################################################
def upgrade() -> None:
    """Remove the marker now that Alembic owns schema versioning."""
    op.drop_table("schema_metadata", if_exists=True)


###############################################################################
def downgrade() -> None:
    """Restore the legacy marker for controlled development downgrades."""
    marker = op.create_table(
        "schema_metadata",
        sa.Column("schema_name", sa.String(length=64), nullable=False),
        sa.Column("schema_version", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("schema_name"),
    )
    op.bulk_insert(
        marker,
        [{"schema_name": "xreport", "schema_version": 1}],
    )

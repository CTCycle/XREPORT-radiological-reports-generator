"""Add hidden built-in architecture presets."""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "b3c1d5a7e9f2"
down_revision: Union[str, None] = "8071b092e5f8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

###############################################################################
def upgrade() -> None:
    op.create_table(
        "hidden_architecture_presets",
        sa.Column("preset_key", sa.String(length=80), nullable=False),
        sa.PrimaryKeyConstraint("preset_key"),
    )

###############################################################################
def downgrade() -> None:
    op.drop_table("hidden_architecture_presets")

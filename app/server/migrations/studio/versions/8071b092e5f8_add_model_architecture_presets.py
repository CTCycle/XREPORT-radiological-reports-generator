"""Add user-defined model architecture presets."""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

import server.repositories.schemas.types

revision: str = "8071b092e5f8"
down_revision: Union[str, None] = "9d4d3bce4ede"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

###############################################################################
def upgrade() -> None:
    op.create_table(
        "model_architecture_presets",
        sa.Column("preset_id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("name", sa.String(length=80), nullable=False),
        sa.Column("name_key", sa.String(length=80), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column(
            "architecture_json",
            server.repositories.schemas.types.JSONSequence(),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            server.repositories.schemas.types.UTCDateTime(timezone=True),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            server.repositories.schemas.types.UTCDateTime(timezone=True),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("preset_id"),
        sa.UniqueConstraint(
            "name_key", name="uq_model_architecture_presets_name_key"
        ),
    )

###############################################################################
def downgrade() -> None:
    op.drop_table("model_architecture_presets")

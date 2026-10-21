"""persist the UI theme preference with application settings"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "a83cdb64e721"
down_revision: Union[str, None] = "e91a4f6c2d73"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


###############################################################################
def upgrade() -> None:
    with op.batch_alter_table("application_settings") as batch_op:
        batch_op.add_column(
            sa.Column(
                "theme_preference",
                sa.String(length=8),
            nullable=True,
            )
        )
        batch_op.create_check_constraint(
            "ck_application_settings_theme_preference",
            "theme_preference IS NULL OR theme_preference IN ('light', 'dark', 'system')",
        )


###############################################################################
def downgrade() -> None:
    with op.batch_alter_table("application_settings") as batch_op:
        batch_op.drop_constraint(
            "ck_application_settings_theme_preference",
            type_="check",
        )
        batch_op.drop_column("theme_preference")

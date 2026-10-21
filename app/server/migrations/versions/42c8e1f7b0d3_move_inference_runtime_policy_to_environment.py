"""move inference process policy out of application settings"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "42c8e1f7b0d3"
down_revision: Union[str, None] = "b1e7a2c4d908"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("application_settings") as batch_op:
        batch_op.drop_constraint("ck_application_settings_device", type_="check")
        batch_op.drop_column("inference_device")
        batch_op.drop_column("inference_hf_local_only")


def downgrade() -> None:
    with op.batch_alter_table("application_settings") as batch_op:
        batch_op.add_column(
            sa.Column(
                "inference_hf_local_only",
                sa.Boolean(),
                nullable=False,
                server_default=sa.true(),
            )
        )
        batch_op.add_column(
            sa.Column(
                "inference_device",
                sa.String(length=16),
                nullable=False,
                server_default="auto",
            )
        )
        batch_op.create_check_constraint(
            "ck_application_settings_device",
            "inference_device IN ('auto', 'cpu', 'cuda')",
        )
        batch_op.alter_column("inference_hf_local_only", server_default=None)
        batch_op.alter_column("inference_device", server_default=None)

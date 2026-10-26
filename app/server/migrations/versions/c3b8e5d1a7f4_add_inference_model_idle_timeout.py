"""add the idle timeout that releases resident inference models"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "c3b8e5d1a7f4"
down_revision: Union[str, None] = "7d3f9a1c5e82"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    columns = {
        column["name"] for column in sa.inspect(op.get_bind()).get_columns("application_settings")
    }
    if "inference_model_idle_timeout" in columns:
        return
    with op.batch_alter_table("application_settings") as batch_op:
        batch_op.add_column(
            sa.Column(
                "inference_model_idle_timeout",
                sa.BigInteger(),
                nullable=False,
                server_default="300",
            )
        )
        batch_op.create_check_constraint(
            "ck_application_settings_model_idle_timeout",
            "inference_model_idle_timeout >= 0",
        )
    with op.batch_alter_table("application_settings") as batch_op:
        batch_op.alter_column("inference_model_idle_timeout", server_default=None)


def downgrade() -> None:
    with op.batch_alter_table("application_settings") as batch_op:
        batch_op.drop_constraint(
            "ck_application_settings_model_idle_timeout", type_="check"
        )
        batch_op.drop_column("inference_model_idle_timeout")

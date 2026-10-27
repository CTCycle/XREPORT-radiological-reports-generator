"""add user-adjustable generation limits for local inference"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "d4a9c6e2b3f1"
down_revision: Union[str, None] = "c3b8e5d1a7f4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_COLUMNS = (
    "inference_max_new_tokens_deterministic",
    "inference_max_new_tokens_concise",
    "inference_max_new_tokens_detailed",
    "inference_detailed_num_beams",
)


def upgrade() -> None:
    columns = {
        column["name"] for column in sa.inspect(op.get_bind()).get_columns("application_settings")
    }
    if all(name in columns for name in _COLUMNS):
        return
    with op.batch_alter_table("application_settings") as batch_op:
        for name in _COLUMNS:
            batch_op.add_column(
                sa.Column(name, sa.BigInteger(), nullable=False, server_default="0")
            )
        batch_op.create_check_constraint(
            "ck_application_settings_max_new_tokens",
            "inference_max_new_tokens_deterministic >= 0 AND inference_max_new_tokens_deterministic <= 4096"
            " AND inference_max_new_tokens_concise >= 0 AND inference_max_new_tokens_concise <= 4096"
            " AND inference_max_new_tokens_detailed >= 0 AND inference_max_new_tokens_detailed <= 4096",
        )
        batch_op.create_check_constraint(
            "ck_application_settings_detailed_num_beams",
            "inference_detailed_num_beams >= 0 AND inference_detailed_num_beams <= 8",
        )
    with op.batch_alter_table("application_settings") as batch_op:
        for name in _COLUMNS:
            batch_op.alter_column(name, server_default=None)


def downgrade() -> None:
    with op.batch_alter_table("application_settings") as batch_op:
        batch_op.drop_constraint("ck_application_settings_detailed_num_beams", type_="check")
        batch_op.drop_constraint("ck_application_settings_max_new_tokens", type_="check")
        for name in _COLUMNS:
            batch_op.drop_column(name)

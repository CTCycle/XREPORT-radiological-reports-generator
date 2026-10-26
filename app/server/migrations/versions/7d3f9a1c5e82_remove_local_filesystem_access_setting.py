"""remove the local filesystem access application setting"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "7d3f9a1c5e82"
down_revision: Union[str, None] = "42c8e1f7b0d3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    columns = {
        column["name"] for column in sa.inspect(op.get_bind()).get_columns("application_settings")
    }
    if "allow_local_filesystem_access" not in columns:
        return
    with op.batch_alter_table("application_settings") as batch_op:
        batch_op.drop_column("allow_local_filesystem_access")


def downgrade() -> None:
    with op.batch_alter_table("application_settings") as batch_op:
        batch_op.add_column(
            sa.Column(
                "allow_local_filesystem_access",
                sa.Boolean(),
                nullable=False,
                server_default=sa.true(),
            )
        )
        batch_op.alter_column("allow_local_filesystem_access", server_default=None)

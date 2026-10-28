"""Record tokenizer revision and special token ids on processing runs."""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "d1f4b7c20a53"
down_revision: Union[str, None] = "c7e2a94f1b08"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

COLUMNS = (
    ("tokenizer_revision", sa.String(length=64)),
    ("pad_token_id", sa.Integer()),
    ("start_token_id", sa.Integer()),
    ("end_token_id", sa.Integer()),
)


###############################################################################
def upgrade() -> None:
    with op.batch_alter_table("processing_runs", schema=None) as batch_op:
        for name, kind in COLUMNS:
            batch_op.add_column(sa.Column(name, kind, nullable=True))


###############################################################################
def downgrade() -> None:
    with op.batch_alter_table("processing_runs", schema=None) as batch_op:
        for name, _kind in reversed(COLUMNS):
            batch_op.drop_column(name)

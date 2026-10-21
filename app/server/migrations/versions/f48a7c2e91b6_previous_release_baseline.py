"""Anchor the published v3.2.0 database revision."""

from typing import Sequence, Union


revision: str = "f48a7c2e91b6"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """The previous release already created the schema represented by b1."""


def downgrade() -> None:
    """The compatibility anchor has no schema changes to reverse."""

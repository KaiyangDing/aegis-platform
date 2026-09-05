"""usage_ledger 计量账本

Revision ID: 0001
Revises:
Create Date: 2026-09-05

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0001"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "usage_ledger",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("request_id", sa.String(length=64), nullable=False),
        sa.Column("tenant_id", sa.String(length=64), nullable=False),
        sa.Column("session_id", sa.String(length=64), nullable=True),
        sa.Column("tier", sa.String(length=16), nullable=False),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("model", sa.String(length=64), nullable=False),
        sa.Column("prompt_tokens", sa.Integer(), nullable=False),
        sa.Column("completion_tokens", sa.Integer(), nullable=False),
        sa.Column("cached", sa.Boolean(), nullable=False),
        sa.Column("usage_missing", sa.Boolean(), nullable=False),
        sa.Column("cost", sa.Numeric(precision=12, scale=6), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_usage_ledger_created_at"), "usage_ledger", ["created_at"], unique=False
    )
    op.create_index(
        op.f("ix_usage_ledger_request_id"), "usage_ledger", ["request_id"], unique=False
    )
    op.create_index(
        "ix_usage_tenant_created",
        "usage_ledger",
        ["tenant_id", "created_at"],
        unique=False,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_usage_tenant_created", table_name="usage_ledger")
    op.drop_index(op.f("ix_usage_ledger_request_id"), table_name="usage_ledger")
    op.drop_index(op.f("ix_usage_ledger_created_at"), table_name="usage_ledger")
    op.drop_table("usage_ledger")

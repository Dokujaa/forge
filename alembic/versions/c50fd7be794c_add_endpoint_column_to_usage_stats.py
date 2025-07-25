"""Add endpoint column to usage_stats

Revision ID: c50fd7be794c
Revises: ca1ac51334ec
Create Date: 2025-04-08 20:46:46.713120

"""

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision = "c50fd7be794c"
down_revision = "ca1ac51334ec"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table("usage_stats", schema=None) as batch_op:
        batch_op.add_column(sa.Column("endpoint", sa.String(), nullable=False))
        batch_op.add_column(sa.Column("input_tokens", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("output_tokens", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("requests_count", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("cost", sa.Float(), nullable=True))
        batch_op.add_column(sa.Column("created_at", sa.DateTime(), nullable=True))

        # Drop the old FK using its specific name
        batch_op.drop_constraint(
            op.f("fk_usage_stats_user_id_users"), type_="foreignkey"
        )
        # Create the new FK with ON DELETE CASCADE
        batch_op.create_foreign_key(
            op.f("fk_usage_stats_user_id_users"),  # Keep the same conventional name
            "users",
            ["user_id"],
            ["id"],
            ondelete="CASCADE",
        )

        batch_op.drop_column("completion_tokens")
        batch_op.drop_column("timestamp")
        batch_op.drop_column("error_count")
        batch_op.drop_column("prompt_tokens")
        batch_op.drop_column("success_count")
        batch_op.drop_column("request_count")

    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table("usage_stats", schema=None) as batch_op:
        batch_op.add_column(sa.Column("request_count", sa.INTEGER(), nullable=True))
        batch_op.add_column(sa.Column("success_count", sa.INTEGER(), nullable=True))
        batch_op.add_column(sa.Column("prompt_tokens", sa.INTEGER(), nullable=True))
        batch_op.add_column(sa.Column("error_count", sa.INTEGER(), nullable=True))
        batch_op.add_column(sa.Column("timestamp", sa.DATETIME(), nullable=True))
        batch_op.add_column(sa.Column("completion_tokens", sa.INTEGER(), nullable=True))

        # Drop the ON DELETE CASCADE FK
        batch_op.drop_constraint(
            op.f("fk_usage_stats_user_id_users"), type_="foreignkey"
        )
        # Recreate the original FK without ON DELETE CASCADE
        batch_op.create_foreign_key(
            op.f("fk_usage_stats_user_id_users"), "users", ["user_id"], ["id"]
        )

        batch_op.drop_column("created_at")
        batch_op.drop_column("cost")
        batch_op.drop_column("requests_count")
        batch_op.drop_column("output_tokens")
        batch_op.drop_column("input_tokens")
        batch_op.drop_column("endpoint")

    # ### end Alembic commands ###

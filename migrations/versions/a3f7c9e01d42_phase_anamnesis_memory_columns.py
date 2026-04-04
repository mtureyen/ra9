"""Phase Anamnesis — activation, source, drift, mood, suppression columns

Revision ID: a3f7c9e01d42
Revises: db76770ffb64
Create Date: 2026-04-04 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'a3f7c9e01d42'
down_revision: Union[str, None] = 'db76770ffb64'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Activation tracking
    op.add_column('memories', sa.Column('last_access_at', sa.DateTime(timezone=True), nullable=True))
    op.add_column('memories', sa.Column('current_activation', sa.Float(), server_default='0.5', nullable=False))

    # Source tracking
    op.add_column('memories', sa.Column('source_strength', sa.Float(), server_default='1.0', nullable=False))

    # Deferred retrieval (E1)
    op.add_column('memories', sa.Column('deferred_activation', sa.Float(), server_default='0.0', nullable=False))

    # Formation period (E6)
    op.add_column('memories', sa.Column('formation_period', sa.Boolean(), server_default='false', nullable=False))

    # Retrieval-induced drift (E9)
    op.add_column('memories', sa.Column('original_embedding', Vector(1024), nullable=True))

    # Mood-dependent gating (E10)
    op.add_column('memories', sa.Column('encoding_mood', postgresql.JSONB(astext_type=sa.Text()), server_default='{}', nullable=False))

    # Intentional suppression (E13)
    op.add_column('memories', sa.Column('suppression_level', sa.Float(), server_default='0.0', nullable=False))
    op.add_column('memories', sa.Column('suppression_decay_start', sa.DateTime(timezone=True), nullable=True))

    # Spacing effect (E27)
    op.add_column('memories', sa.Column('retrieval_timestamps', postgresql.JSONB(astext_type=sa.Text()), server_default='[]', nullable=False))

    # GIN index on tags for efficient array lookups
    op.create_index('idx_memories_tags_gin', 'memories', ['tags'], unique=False, postgresql_using='gin')

    # Data migration: copy embedding → original_embedding for existing rows
    op.execute("UPDATE memories SET original_embedding = embedding WHERE original_embedding IS NULL")


def downgrade() -> None:
    op.drop_index('idx_memories_tags_gin', table_name='memories', postgresql_using='gin')
    op.drop_column('memories', 'retrieval_timestamps')
    op.drop_column('memories', 'suppression_decay_start')
    op.drop_column('memories', 'suppression_level')
    op.drop_column('memories', 'encoding_mood')
    op.drop_column('memories', 'original_embedding')
    op.drop_column('memories', 'formation_period')
    op.drop_column('memories', 'deferred_activation')
    op.drop_column('memories', 'source_strength')
    op.drop_column('memories', 'current_activation')
    op.drop_column('memories', 'last_access_at')

import os
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from dotenv import load_dotenv
from sqlalchemy import create_engine

from emotive.db.models import Base

# Load .env from project root
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

# Read DATABASE_URL from environment (.env loaded above)
url = os.environ.get("DATABASE_URL")
if not url:
    url = config.get_main_option("sqlalchemy.url")
if not url:
    raise RuntimeError("DATABASE_URL is not set. Create a .env file or export it.")


def run_migrations_offline() -> None:
    context.configure(url=url, target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = create_engine(url)
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

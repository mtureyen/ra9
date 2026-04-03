import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

# Load .env from project root
_env_path = Path(__file__).resolve().parents[3] / ".env"
load_dotenv(_env_path)

DATABASE_URL = os.environ.get("DATABASE_URL")
if DATABASE_URL is None:
    raise RuntimeError(
        "DATABASE_URL environment variable is not set. "
        "Create a .env file in the project root or export it in your shell."
    )

engine = create_engine(
    DATABASE_URL,
    echo=False,
    pool_size=10,
    max_overflow=20,
    pool_timeout=60,
    pool_pre_ping=True,
)
SessionFactory = sessionmaker(bind=engine)


def get_session() -> Session:
    """Create a new database session. Caller is responsible for closing it."""
    return SessionFactory()

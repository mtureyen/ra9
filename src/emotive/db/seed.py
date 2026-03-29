"""Seed the database with initial singleton rows."""

from sqlalchemy import select

from .engine import get_session
from .models import Temperament


def seed_temperament() -> None:
    """Insert the singleton temperament row if it doesn't exist."""
    session = get_session()
    try:
        stmt = select(Temperament).where(Temperament.id == 1)
        exists = session.execute(stmt).scalar_one_or_none()
        if exists is None:
            session.add(Temperament(id=1))
            session.commit()
    finally:
        session.close()


def seed_all() -> None:
    """Run all seed operations."""
    seed_temperament()


if __name__ == "__main__":
    seed_all()
    print("Seeding complete.")

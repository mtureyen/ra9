# ra9

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env  # fill in your values
```

## Database

```bash
brew install postgresql@17 pgvector
brew services start postgresql@17
createdb emotive_ai
psql emotive_ai -c "CREATE EXTENSION IF NOT EXISTS vector;"
alembic upgrade head
python -m emotive.db.seed
```

## Run

```bash
python -m emotive.server          # Start MCP server
python -m emotive.cli.close_sessions    # Close orphaned sessions
python -m emotive.cli.export_obsidian   # Export memories to Obsidian vault
```

## Tests

```bash
pytest                # Run all tests
pytest -v             # Verbose
pytest tests/test_config/  # Run specific module
```

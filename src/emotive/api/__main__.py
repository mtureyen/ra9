"""Entry point: python -m emotive.api

Starts the FastAPI server on 127.0.0.1:8000.
The brain boots automatically on startup.
"""

import uvicorn


def main() -> None:
    uvicorn.run(
        "emotive.api.server:app",
        host="127.0.0.1",
        port=8000,
        log_level="info",
    )


if __name__ == "__main__":
    main()

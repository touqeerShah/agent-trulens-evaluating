"""SQLite session and optional dashboard helpers for local TruLens runs."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path


os.environ.setdefault("TRULENS_OTEL_TRACING", "1")


def _otel_database_url(database_url: str) -> str:
    """Keep legacy SQLite files intact and select a fresh OTEL schema file."""
    prefix = "sqlite:///"
    if not database_url.startswith(prefix):
        return database_url

    path = Path(database_url[len(prefix) :])
    if not path.exists():
        return database_url

    with sqlite3.connect(path) as connection:
        has_events_table = connection.execute(
            "select 1 from sqlite_master where type='table' and name='trulens_events'"
        ).fetchone()
    if has_events_table:
        return database_url

    for version in range(2, 100):
        candidate = path.with_name(f"{path.stem}_v{version}{path.suffix}")
        if not candidate.exists():
            print(
                f"Existing TruLens database {path} has no trulens_events table; "
                f"using new OTEL database {candidate}."
            )
            return f"{prefix}{candidate}"
    raise RuntimeError("Could not find an unused OTEL SQLite database filename.")


def create_session(database_url: str = "sqlite:///default_otel.sqlite", reset: bool = False):
    from trulens.core.database.connector.default import DefaultDBConnector
    from trulens.core.experimental import Feature
    from trulens.core.session import TruSession

    resolved_database_url = _otel_database_url(database_url)
    session = TruSession(
        connector=DefaultDBConnector(database_url=resolved_database_url),
        experimental_feature_flags=[Feature.OTEL_TRACING],
    )
    if reset:
        session.reset_database()
    return session


def launch_dashboard(port: int = 8001):
    from trulens.dashboard import run_dashboard

    return run_dashboard(port=port)

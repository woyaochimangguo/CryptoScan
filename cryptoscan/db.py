from __future__ import annotations

from contextlib import contextmanager
from threading import Lock
from typing import Iterator

from sqlmodel import Session, SQLModel, create_engine

from .config import settings
from . import models  # noqa: F401  (register tables)

_engine = create_engine(
    f"sqlite:///{settings.db_path}",
    echo=False,
    connect_args={"check_same_thread": False, "timeout": 30},
)
_init_lock = Lock()
_initialized = False


def _ensure_sqlite_columns() -> None:
    """Small migration shim for local SQLite during early development."""
    wanted = {
        "strategy_id": "VARCHAR DEFAULT 'legacy'",
        "strategy_name": "VARCHAR DEFAULT ''",
        "strategy_version": "VARCHAR DEFAULT ''",
        "policy_id": "VARCHAR DEFAULT ''",
        "model_profile": "VARCHAR DEFAULT ''",
        "risk_profile": "VARCHAR DEFAULT 'paper_default'",
    }
    with _engine.begin() as conn:
        existing = {
            row[1]
            for row in conn.exec_driver_sql("PRAGMA table_info(episode)").fetchall()
        }
        for name, ddl in wanted.items():
            if name not in existing:
                conn.exec_driver_sql(f"ALTER TABLE episode ADD COLUMN {name} {ddl}")
        conn.exec_driver_sql(
            """
            UPDATE episode
            SET strategy_id = 'oi_funding_flip',
                strategy_name = 'OI + funding flip',
                strategy_version = '1.0.0'
            WHERE (strategy_id IS NULL OR strategy_id = '' OR strategy_id = 'legacy')
              AND trigger = 'oi_funding_flip'
            """
        )


def init_db() -> None:
    global _initialized
    if _initialized:
        return
    with _init_lock:
        if _initialized:
            return
        SQLModel.metadata.create_all(_engine)
        _ensure_sqlite_columns()
        with _engine.begin() as conn:
            conn.exec_driver_sql("PRAGMA journal_mode=WAL")
            conn.exec_driver_sql("PRAGMA busy_timeout=30000")
        _initialized = True


@contextmanager
def session_scope() -> Iterator[Session]:
    session = Session(_engine, expire_on_commit=False)
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_engine():
    return _engine

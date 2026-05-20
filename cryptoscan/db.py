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


def init_db() -> None:
    global _initialized
    if _initialized:
        return
    with _init_lock:
        if _initialized:
            return
        SQLModel.metadata.create_all(_engine)
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

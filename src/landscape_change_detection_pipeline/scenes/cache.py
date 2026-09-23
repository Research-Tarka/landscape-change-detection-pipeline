"""Resume cache for scene discovery/download: one SQLite (WAL) database per split/worker.

Purpose
-------
Scene discovery (this module's caller, :mod:`.download`) and the later fetch
stage both need to know "has this scene already been handled?"
without re-querying Earth Engine or re-downloading pixels on every resumed
run. This uses a small SQLite database
per split/worker, opened in WAL mode so one process's writes don't block
concurrent readers in the same split, with a per-path lock guarding the
write path itself (SQLite's own locking handles cross-process safety; the
in-process lock avoids two threads in the same worker racing the same
connection).

Why SQLite-WAL per split/worker, not one shared database
----------------------------------------------------------
Splits (LPT batches) run as separate processes/GEE projects,
potentially on different machines. A single shared database
would need network-safe locking; one file per split avoids that entirely --
each split owns its own cache file, and results are merged later by simply
reading every split's database, not by concurrent writes to one file.

Schema
------
One row per ``(tile_id, sensor, scene_id)`` triple, recording whether the
scene was kept (queued for download) or rejected (and why), so a resumed run
can skip straight past scenes already resolved either way.
"""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

CACHE_FILENAME_TEMPLATE = "scene_cache_split{split}.sqlite3"

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS scenes (
    tile_id     TEXT NOT NULL,
    sensor      TEXT NOT NULL,
    scene_id    TEXT NOT NULL,
    year        INTEGER NOT NULL,
    month       INTEGER,
    status      TEXT NOT NULL,   -- 'kept' or 'rejected'
    reason      TEXT,            -- populated when status='rejected'
    cloud_pct   REAL,
    cached_at   TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (tile_id, sensor, scene_id)
);
"""

#: Indexes reference ``month``, so they must be created only after the
#: migration below has guaranteed the column exists -- otherwise
#: ``CREATE INDEX ... (..., month)`` fails with "no such column: month" on a
#: database created before that column existed.
_CREATE_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_scenes_tile_sensor_year
    ON scenes (tile_id, sensor, year);
CREATE INDEX IF NOT EXISTS idx_scenes_tile_sensor_year_month
    ON scenes (tile_id, sensor, year, month);
"""

#: Added when this database may already exist from before ``month`` existed;
#: ``CREATE TABLE IF NOT EXISTS`` alone would silently skip adding it to an
#: already-created table.
_MIGRATIONS = (
    "ALTER TABLE scenes ADD COLUMN month INTEGER",
)

#: Guards connection creation/writes within one process; SQLite's own file
#: locking (relaxed by WAL mode) handles safety across processes/workers.
_lock = threading.Lock()


def cache_path_for_split(cache_dir: str | Path, split: int) -> Path:
    """The cache database path for one split/worker."""
    return Path(cache_dir) / CACHE_FILENAME_TEMPLATE.format(split=split)


@contextmanager
def open_cache(cache_dir: str | Path, split: int) -> Iterator[sqlite3.Connection]:
    """Open (creating if needed) the resume-cache database for one split.

    WAL mode lets readers (e.g. a status report) run concurrently with the
    worker's own writes without blocking on each other.
    """
    path = cache_path_for_split(cache_dir, split)
    path.parent.mkdir(parents=True, exist_ok=True)

    with _lock:
        conn = sqlite3.connect(str(path), timeout=30.0)
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.executescript(_CREATE_TABLE)
            existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(scenes)").fetchall()}
            if "month" not in existing_cols:
                for migration in _MIGRATIONS:
                    conn.execute(migration)
            conn.executescript(_CREATE_INDEXES)
            conn.commit()
        except Exception:
            conn.close()
            raise

    try:
        yield conn
    finally:
        conn.close()


def is_cached(conn: sqlite3.Connection, tile_id: str, sensor: str, scene_id: str) -> bool:
    """Whether ``scene_id`` has already been resolved (kept or rejected) for this tile/sensor."""
    row = conn.execute(
        "SELECT 1 FROM scenes WHERE tile_id = ? AND sensor = ? AND scene_id = ?",
        (tile_id, sensor, scene_id),
    ).fetchone()
    return row is not None


def cached_scene_ids(conn: sqlite3.Connection, tile_id: str, sensor: str, year: int) -> set[str]:
    """All scene ids already resolved for ``(tile_id, sensor, year)``, kept or rejected."""
    rows = conn.execute(
        "SELECT scene_id FROM scenes WHERE tile_id = ? AND sensor = ? AND year = ?",
        (tile_id, sensor, year),
    ).fetchall()
    return {r[0] for r in rows}


def record_scene(
    conn: sqlite3.Connection,
    tile_id: str,
    sensor: str,
    scene_id: str,
    year: int,
    status: str,
    reason: Optional[str] = None,
    cloud_pct: Optional[float] = None,
    month: Optional[int] = None,
) -> None:
    """Insert or update one scene's resolution ('kept' or 'rejected')."""
    if status not in ("kept", "rejected"):
        raise ValueError(f"status must be 'kept' or 'rejected', got {status!r}")
    with _lock:
        conn.execute(
            """
            INSERT INTO scenes (tile_id, sensor, scene_id, year, month, status, reason, cloud_pct)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (tile_id, sensor, scene_id)
            DO UPDATE SET status = excluded.status, reason = excluded.reason,
                          cloud_pct = excluded.cloud_pct, year = excluded.year,
                          month = excluded.month
            """,
            (tile_id, sensor, scene_id, year, month, status, reason, cloud_pct),
        )
        conn.commit()


def kept_scene_ids(conn: sqlite3.Connection, tile_id: str, sensor: str, year: int) -> list[str]:
    """Scene ids already marked 'kept' for ``(tile_id, sensor, year)`` -- the resume set to fetch."""
    rows = conn.execute(
        "SELECT scene_id FROM scenes WHERE tile_id = ? AND sensor = ? AND year = ? AND status = 'kept'",
        (tile_id, sensor, year),
    ).fetchall()
    return [r[0] for r in rows]

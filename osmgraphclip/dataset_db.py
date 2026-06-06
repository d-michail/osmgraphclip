"""
Unified SQLite abstraction replacing downloads.csv and index.csv.

A single ``dataset.db`` file inside the dataset output directory holds two
tables: ``downloads`` (written by create_dataset.py) and ``graphs`` (written
by create_graphs.py and create_city_dataset.py).  OsmDataset reads from
``graphs`` via ``graphs_as_dataframe()``.

Thread-safety
-------------
One ``DatasetDB`` instance wraps a single sqlite3 connection opened with
``check_same_thread=False`` and a ``threading.RLock``.  Every public method
acquires the lock before executing SQL.

Process-pool callers (create_dataset.py ProcessPoolExecutor) never share a
DatasetDB instance across process boundaries — workers return result rows to
the main process which does all writes.
"""

from __future__ import annotations

import datetime
import sqlite3
import threading
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd

DB_FILENAME = "dataset.db"

_DDL = """
PRAGMA journal_mode = WAL;
PRAGMA synchronous  = NORMAL;

CREATE TABLE IF NOT EXISTS downloads (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    lat            REAL    NOT NULL,
    lon            REAL    NOT NULL,
    bbox_size      REAL    NOT NULL,
    geojson_prefix TEXT    NOT NULL UNIQUE,
    level          INTEGER,
    timestamp      TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS graphs (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    lat            REAL    NOT NULL,
    lon            REAL    NOT NULL,
    bbox_size      REAL    NOT NULL,
    graph_pickle   TEXT    NOT NULL UNIQUE,
    level          INTEGER,
    method         TEXT    NOT NULL DEFAULT 'geolink',
    richness_score REAL,
    timestamp      TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_downloads_prefix ON downloads (geojson_prefix);
CREATE INDEX IF NOT EXISTS idx_graphs_pickle    ON graphs    (graph_pickle);
CREATE INDEX IF NOT EXISTS idx_graphs_latlon    ON graphs    (lat, lon);
"""


class DatasetDB:
    """Single-file SQLite wrapper for the OSMGraphCLIP dataset pipeline."""

    def __init__(self, path: str | Path) -> None:
        self._path = str(path)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            self._path,
            check_same_thread=False,
            isolation_level=None,  # autocommit; each execute() is its own transaction
        )
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(_DDL)
            # Migration: add method column to existing databases that predate it.
            try:
                self._conn.execute(
                    "ALTER TABLE graphs ADD COLUMN method TEXT NOT NULL DEFAULT 'geolink'"
                )
            except sqlite3.OperationalError:
                pass  # column already exists

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except Exception:
                pass
            self._conn.close()

    def __enter__(self) -> "DatasetDB":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    # ── downloads table ──────────────────────────────────────────────────────

    @property
    def downloaded_prefixes(self) -> Set[str]:
        """Return the set of all geojson_prefix values already in the DB."""
        with self._lock:
            rows = self._conn.execute("SELECT geojson_prefix FROM downloads").fetchall()
        return {row["geojson_prefix"] for row in rows}

    def insert_download(
        self,
        lat: float,
        lon: float,
        bbox_size: float,
        geojson_prefix: str,
        level: Optional[int] = None,
    ) -> None:
        """Insert one download row. Silently ignores duplicate geojson_prefix."""
        ts = _now_iso()
        with self._lock:
            self._conn.execute(
                """
                INSERT OR IGNORE INTO downloads
                    (lat, lon, bbox_size, geojson_prefix, level, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (lat, lon, bbox_size, geojson_prefix, level, ts),
            )

    def downloads_as_list(self) -> List[Dict]:
        """Return all download rows as a list of dicts.

        Keys: lat, lon, timestamp, bbox_size, geojson_prefix, and level
        (only present when non-NULL — matches the old read_downloads_csv contract).
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT lat, lon, timestamp, bbox_size, geojson_prefix, level "
                "FROM downloads ORDER BY id"
            ).fetchall()
        result: List[Dict] = []
        for row in rows:
            d: Dict = {
                "lat":            row["lat"],
                "lon":            row["lon"],
                "timestamp":      row["timestamp"],
                "bbox_size":      row["bbox_size"],
                "geojson_prefix": row["geojson_prefix"],
            }
            if row["level"] is not None:
                d["level"] = int(row["level"])
            result.append(d)
        return result

    def clear_downloads(self) -> None:
        """Delete all rows from downloads (implements --no-resume for downloads)."""
        with self._lock:
            self._conn.execute("DELETE FROM downloads")

    # ── graphs table ─────────────────────────────────────────────────────────

    @property
    def graph_pickles(self) -> Set[str]:
        """Return the set of all graph_pickle basenames already in the DB."""
        with self._lock:
            rows = self._conn.execute("SELECT graph_pickle FROM graphs").fetchall()
        return {row["graph_pickle"] for row in rows}

    def insert_graph(
        self,
        lat: float,
        lon: float,
        bbox_size: float,
        graph_pickle: str,
        level: Optional[int] = None,
        richness_score: Optional[float] = None,
        method: str = 'geolink',
    ) -> None:
        """Insert one graph row. Silently ignores duplicate graph_pickle."""
        ts = _now_iso()
        with self._lock:
            self._conn.execute(
                """
                INSERT OR IGNORE INTO graphs
                    (lat, lon, bbox_size, graph_pickle, level, method, richness_score, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (lat, lon, bbox_size, graph_pickle, level, method, richness_score, ts),
            )

    def graphs_as_dataframe(self) -> pd.DataFrame:
        """Return the graphs table as a DataFrame compatible with OsmDataset.

        Columns: lat, lon, graph_pickle, bbox_size, level, richness_score, timestamp.
        level and richness_score may be all-NaN for single-resolution datasets.
        """
        with self._lock:
            df = pd.read_sql_query(
                "SELECT lat, lon, graph_pickle, bbox_size, level, method, richness_score, timestamp "
                "FROM graphs ORDER BY id",
                self._conn,
            )
        return df

    def clear_graphs(self) -> None:
        """Delete all rows from graphs (implements --no-resume for graphs)."""
        with self._lock:
            self._conn.execute("DELETE FROM graphs")

    def is_multiresolution(self) -> bool:
        """Return True if any download row has a non-NULL level."""
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM downloads WHERE level IS NOT NULL LIMIT 1"
            ).fetchone()
        return row is not None

    def __repr__(self) -> str:
        return f"DatasetDB({self._path!r})"


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

"""MultiscaleDatasetDB: extends DatasetDB with a band_features table.

The band_features table stores per-location multi-scale band feature files
(.npz) produced by create_multiscale_dataset.py.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import List, Set, Tuple

from .dataset_db import DatasetDB

_BAND_DDL = """
CREATE TABLE IF NOT EXISTS band_features (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    lat        REAL    NOT NULL,
    lon        REAL    NOT NULL,
    bands_path TEXT    NOT NULL UNIQUE,
    band_radii TEXT    NOT NULL,
    timestamp  TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_band_features_latlon ON band_features (lat, lon);
"""


class MultiscaleDatasetDB(DatasetDB):
    """DatasetDB subclass that also manages the band_features table."""

    def __init__(self, path: str | Path) -> None:
        super().__init__(path)
        with self._lock:
            self._conn.executescript(_BAND_DDL)

    # ── band_features table ───────────────────────────────────────────────────

    def insert_band_features(
        self,
        lat: float,
        lon: float,
        bands_path: str,
        band_radii: List[float],
    ) -> None:
        """Insert one band_features row. Silently ignores duplicate bands_path."""
        ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with self._lock:
            self._conn.execute(
                """
                INSERT OR IGNORE INTO band_features
                    (lat, lon, bands_path, band_radii, timestamp)
                VALUES (?, ?, ?, ?, ?)
                """,
                (lat, lon, bands_path, json.dumps(band_radii), ts),
            )

    def completed_band_locations(self) -> Set[Tuple[float, float]]:
        """Return the set of (lat, lon) pairs that have a band_features row."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT lat, lon FROM band_features"
            ).fetchall()
        return {(row["lat"], row["lon"]) for row in rows}

    def clear_band_features(self) -> None:
        """Delete all rows from band_features (implements --no-resume)."""
        with self._lock:
            self._conn.execute("DELETE FROM band_features")

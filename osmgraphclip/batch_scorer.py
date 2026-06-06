"""Batch SQL scoring for global OSM location candidate screening.

Scores (lat, lon) candidates by counting OSM features per fixed bbox using a
persistent PostGIS connection.  No GeoDataFrame construction, no per-location
round-trips, no adaptive bbox expansion.

Key speed improvements over the per-location adaptive_probe() approach
-----------------------------------------------------------------------
- VALUES + LATERAL batch queries: amortise connection overhead over 100+ locations
- Fixed bbox size: no expansion loop (up to 7× fewer SQL calls per location)
- No GeoDataFrame construction: avoids GeoJSON parsing + Shapely geometry ops
- Proxy richness score computed in Python from SQL count columns (no richness.py)
- Thread-safe: each worker thread keeps its own persistent connection
"""

from __future__ import annotations

import logging
import math
import os
import queue
import threading
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ─── EPSG:3857 conversion ────────────────────────────────────────────────────

_EARTH_RADIUS_M = 6378137.0  # WGS84 / EPSG:3857 sphere radius


def _latlon_to_3857(lat: float, lon: float) -> Tuple[float, float]:
    """Convert (lat, lon) in WGS84 to (x, y) in EPSG:3857 (Web Mercator).

    Uses the spherical Mercator formula — identical to PostGIS ST_Transform.
    Accurate to < 1 m for |lat| < 85°; sufficient for our bbox queries.
    """
    x = lon * (math.pi / 180.0) * _EARTH_RADIUS_M
    y = math.log(math.tan(math.pi / 4 + (lat * math.pi / 180.0) / 2)) * _EARTH_RADIUS_M
    return x, y


def _bbox_3857(lat: float, lon: float, dist_m: float) -> Tuple[float, float, float, float]:
    """Return (xmin, ymin, xmax, ymax) in EPSG:3857 for a square bbox of half-width dist_m."""
    cx, cy = _latlon_to_3857(lat, lon)
    return cx - dist_m, cy - dist_m, cx + dist_m, cy + dist_m


def _latlons_to_3857_bboxes(
    lats: np.ndarray,
    lons: np.ndarray,
    dist_m: float,
) -> np.ndarray:
    """Vectorised conversion of (lat, lon) arrays to (xmin, ymin, xmax, ymax) in EPSG:3857.

    Returns an array of shape (N, 4).
    """
    xs = lons * (math.pi / 180.0 * _EARTH_RADIUS_M)
    lat_rad = lats * (math.pi / 180.0)
    ys = np.log(np.tan(np.pi / 4 + lat_rad / 2)) * _EARTH_RADIUS_M
    return np.column_stack([xs - dist_m, ys - dist_m, xs + dist_m, ys + dist_m])


# ─── Batch SQL template ───────────────────────────────────────────────────────
# Each location contributes one row to the VALUES clause.
# LATERAL subqueries COUNT(*) and COUNT(column) — the latter counts non-NULLs,
# giving per-category presence without a DISTINCT (faster with a spatial index).

_BATCH_SQL_TEMPLATE = """\
WITH locs(id, xmin, ymin, xmax, ymax) AS (
    VALUES {values}
)
SELECT
    l.id,
    COALESCE(p.n_pts, 0)       AS n_points,
    COALESCE(p.n_amenity, 0)   AS n_amenity,
    COALESCE(p.n_shop, 0)      AS n_shop,
    COALESCE(p.n_leisure, 0)   AS n_leisure,
    COALESCE(p.n_tourism, 0)   AS n_tourism,
    COALESCE(p.n_historic, 0)  AS n_historic,
    COALESCE(p.n_natural, 0)   AS n_natural,
    COALESCE(ln.n_lines, 0)    AS n_lines,
    COALESCE(ln.n_highway, 0)  AS n_highway,
    COALESCE(ln.n_waterway, 0) AS n_waterway,
    COALESCE(ln.n_railway, 0)  AS n_railway,
    COALESCE(pg.n_polys, 0)    AS n_polys,
    COALESCE(pg.n_building, 0) AS n_building,
    COALESCE(pg.n_landuse, 0)  AS n_landuse
FROM locs l
LEFT JOIN LATERAL (
    SELECT COUNT(*)           AS n_pts,
           COUNT(amenity)     AS n_amenity,
           COUNT(shop)        AS n_shop,
           COUNT(leisure)     AS n_leisure,
           COUNT(tourism)     AS n_tourism,
           COUNT(historic)    AS n_historic,
           COUNT("natural")   AS n_natural
    FROM planet_osm_point
    WHERE way && ST_MakeEnvelope(l.xmin, l.ymin, l.xmax, l.ymax, 3857)
) p ON true
LEFT JOIN LATERAL (
    SELECT COUNT(*)        AS n_lines,
           COUNT(highway)  AS n_highway,
           COUNT(waterway) AS n_waterway,
           COUNT(railway)  AS n_railway
    FROM planet_osm_line
    WHERE way && ST_MakeEnvelope(l.xmin, l.ymin, l.xmax, l.ymax, 3857)
) ln ON true
LEFT JOIN LATERAL (
    SELECT COUNT(*)        AS n_polys,
           COUNT(building) AS n_building,
           COUNT(landuse)  AS n_landuse
    FROM planet_osm_polygon
    WHERE way && ST_MakeEnvelope(l.xmin, l.ymin, l.xmax, l.ymax, 3857)
) pg ON true"""

_RESULT_COLS = (
    "id",
    "n_points", "n_amenity", "n_shop", "n_leisure", "n_tourism", "n_historic", "n_natural",
    "n_lines", "n_highway", "n_waterway", "n_railway",
    "n_polys", "n_building", "n_landuse",
)
_ZERO_ROW: Dict[str, int] = {col: 0 for col in _RESULT_COLS if col != "id"}


# ─── Proxy richness score ─────────────────────────────────────────────────────

def compute_proxy_richness(row: dict) -> float:
    """Compute a richness proxy score in [0, 1] from SQL count columns.

    Approximates the composite richness score from osmgraphclip/richness.py
    using only feature counts and category presence flags derived from SQL.
    No GeoDataFrame, no Shapely geometry, no full tag iteration required.

    Component weights (chosen to roughly match the full richness.py score):
      0.30  tag diversity proxy   — how many distinct category columns are non-zero
      0.25  category coverage     — fraction of the 9 detectable categories present
      0.20  feature count score   — normalised to 50 features at full score
      0.15  geometry entropy      — balance across point / line / polygon types
      0.10  density proxy         — normalised to 20 features at full score
    """
    n_points = int(row.get("n_points") or 0)
    n_lines  = int(row.get("n_lines")  or 0)
    n_polys  = int(row.get("n_polys")  or 0)
    total = n_points + n_lines + n_polys
    if total == 0:
        return 0.0

    # --- Category presence (9 of the 10 richness.py categories detectable from SQL) ---
    cat_flags = [
        int(row.get("n_amenity")  or 0) > 0,                              # food/health/education
        int(row.get("n_shop")     or 0) > 0,                              # retail
        (int(row.get("n_highway") or 0) > 0 or
         int(row.get("n_railway") or 0) > 0),                             # transport
        int(row.get("n_leisure")  or 0) > 0,                              # leisure
        int(row.get("n_natural")  or 0) > 0,                              # nature_green
        int(row.get("n_waterway") or 0) > 0,                              # water
        (int(row.get("n_tourism") or 0) > 0 or
         int(row.get("n_historic") or 0) > 0),                            # tourism
        int(row.get("n_building") or 0) > 0,                              # built_environment
        int(row.get("n_landuse")  or 0) > 0,                              # nature_green (landuse)
    ]
    n_cats = sum(cat_flags)

    # --- Geometry entropy ---
    p = [c / total for c in [n_points, n_lines, n_polys] if c > 0]
    geom_entropy = (
        -sum(pi * math.log(pi) for pi in p) / math.log(3)
        if len(p) > 1 else 0.0
    )

    score = (
        0.30 * min(n_cats / 9.0,  1.0)   # tag diversity proxy
        + 0.25 * (n_cats / 10.0)         # category coverage (10-cat denominator)
        + 0.20 * min(total / 50.0, 1.0)  # feature count
        + 0.15 * geom_entropy            # geometry balance
        + 0.10 * min(total / 20.0, 1.0)  # density proxy
    )
    return float(min(score, 1.0))


# ─── Phase 1: Global density pre-scan ────────────────────────────────────────

def run_density_scan(
    postgis_url: str,
    cache_path: str,
    sample_pct: float = 10.0,
    connect_timeout: int = 15,
) -> Set[Tuple[int, int]]:
    """Return the set of non-empty 1-degree (lat_bin, lon_bin) cells.

    Queries planet_osm_point with TABLESAMPLE SYSTEM for speed on a full-planet
    database.  Results are cached to *cache_path* (Parquet) and reloaded on
    subsequent runs without hitting the database.

    Parameters
    ----------
    postgis_url:
        PostgreSQL DSN for the PostGIS database.
    cache_path:
        Parquet file to write / read the density raster.
    sample_pct:
        Percentage of planet_osm_point pages to sample (default 10 %).
        10 % reliably detects cells with ≥ 10 total features.
    connect_timeout:
        psycopg2 connection timeout in seconds.

    Returns
    -------
    Set of (lat_bin, lon_bin) tuples where lat_bin = floor(lat) and
    lon_bin = floor(lon).
    """
    if os.path.exists(cache_path):
        logger.info("Loading density cache from %s", cache_path)
        df = pd.read_parquet(cache_path)
        cells: Set[Tuple[int, int]] = set(
            zip(df["lat_bin"].astype(int), df["lon_bin"].astype(int))
        )
        logger.info("Loaded %d non-empty 1-degree cells from cache", len(cells))
        return cells

    import psycopg2

    logger.info(
        "Running global density scan (planet_osm_point TABLESAMPLE SYSTEM(%.0f))…"
        " This may take a few minutes.",
        sample_pct,
    )
    # TABLESAMPLE SYSTEM reads random data pages — fast even on multi-hundred-GB tables.
    sql = f"""
        SELECT
            floor(ST_Y(ST_Transform(way, 4326)))::int AS lat_bin,
            floor(ST_X(ST_Transform(way, 4326)))::int AS lon_bin,
            COUNT(*) AS cnt
        FROM planet_osm_point TABLESAMPLE SYSTEM({sample_pct})
        GROUP BY lat_bin, lon_bin
        HAVING COUNT(*) >= 1
    """
    conn = psycopg2.connect(
        postgis_url,
        connect_timeout=connect_timeout,
        options="-c statement_timeout=0",  # no timeout for density scan
    )
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()
    finally:
        conn.close()

    df = pd.DataFrame(rows, columns=["lat_bin", "lon_bin", "cnt"])
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    df.to_parquet(cache_path, index=False)
    cells = set(zip(df["lat_bin"].astype(int), df["lon_bin"].astype(int)))
    logger.info("Density scan complete: %d non-empty 1-degree cells found", len(cells))
    return cells


# ─── Phase 2: H3 candidate generation ────────────────────────────────────────

def generate_h3_candidates(
    density_cells: Set[Tuple[int, int]],
    h3_resolution: int = 5,
    rng_seed: int = 42,
    jitter_scale: float = 0.0,
) -> List[Tuple[float, float]]:
    """Generate candidate (lat, lon) points from H3 cells within OSM-data areas.

    Enumerates all H3 cells at *h3_resolution* globally and retains only those
    whose centre falls within a 1-degree density cell from *density_cells*.
    An optional small random offset (jitter) can be added to break up the
    regular hexagonal grid pattern.

    Parameters
    ----------
    density_cells:
        Set of (lat_bin, lon_bin) from run_density_scan() — 1-degree cells
        known to contain OSM data.
    h3_resolution:
        H3 resolution of candidate cells (5 = ~22 km hexagons, ~2 M globally).
    rng_seed:
        Random seed for jitter offset.
    jitter_scale:
        Maximum absolute offset in degrees added to each cell centre.
        A value of 0.05 adds up to ±5 km of uniform random jitter near the equator.
        Set to 0.0 to use exact cell centres (reproducible, slightly less varied).

    Returns
    -------
    List of (lat, lon) tuples, one per retained H3 cell.
    """
    import h3 as h3lib

    rng = np.random.default_rng(rng_seed)
    seen: Dict[str, Tuple[float, float]] = {}

    logger.info(
        "Generating H3 res-%d candidates within %d density cells…",
        h3_resolution, len(density_cells),
    )

    for res0_cell in h3lib.get_res0_cells():
        for cell in h3lib.cell_to_children(res0_cell, h3_resolution):
            lat, lon = h3lib.cell_to_latlng(cell)
            lat_bin = int(math.floor(lat))
            lon_bin = int(math.floor(lon))
            if (lat_bin, lon_bin) in density_cells and cell not in seen:
                if jitter_scale > 0.0:
                    lat = float(np.clip(
                        lat + rng.uniform(-jitter_scale, jitter_scale), -89.9, 89.9
                    ))
                    lon = lon + rng.uniform(-jitter_scale, jitter_scale)
                    # Wrap longitude to [-180, 180]
                    lon = ((lon + 180.0) % 360.0) - 180.0
                seen[cell] = (lat, lon)

    candidates = list(seen.values())
    logger.info(
        "H3 candidate generation: %d candidates at res-%d within density raster",
        len(candidates), h3_resolution,
    )
    return candidates


# ─── BatchScorer ─────────────────────────────────────────────────────────────

class BatchScorer:
    """Count OSM features for a batch of candidate locations via PostGIS.

    Uses a persistent psycopg2 connection and a VALUES + LATERAL SQL query
    that checks up to *batch_size* bboxes in a single round-trip.

    Thread-safety: one BatchScorer per thread — do not share across threads.
    """

    def __init__(
        self,
        postgis_url: str,
        batch_size: int = 100,
        bbox_size_m: float = 500.0,
        query_timeout_s: int = 60,
        connect_timeout: int = 15,
    ) -> None:
        self.postgis_url = postgis_url
        self.batch_size = batch_size
        self.bbox_size_m = bbox_size_m
        self.query_timeout_ms = query_timeout_s * 1000
        self.connect_timeout = connect_timeout
        self._conn = None
        # Cache the SQL templates by batch size to avoid repeated string ops
        self._sql_cache: Dict[int, str] = {}

    # --- Connection management ---

    def connect(self) -> None:
        import psycopg2
        self._conn = psycopg2.connect(
            self.postgis_url,
            connect_timeout=self.connect_timeout,
            options=f"-c statement_timeout={self.query_timeout_ms}",
        )
        self._conn.autocommit = True
        logger.debug("BatchScorer: connected to PostGIS")

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    def __enter__(self) -> "BatchScorer":
        self.connect()
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def _ensure_connected(self) -> None:
        """Reconnect if the connection is closed or broken."""
        import psycopg2
        if self._conn is None or self._conn.closed:
            self.connect()
            return
        try:
            with self._conn.cursor() as cur:
                cur.execute("SELECT 1")
        except Exception:
            logger.debug("BatchScorer: stale connection — reconnecting")
            try:
                self._conn.close()
            except Exception:
                pass
            self.connect()

    def _get_sql(self, n: int) -> str:
        if n not in self._sql_cache:
            values = ", ".join(["(%s,%s,%s,%s,%s)"] * n)
            self._sql_cache[n] = _BATCH_SQL_TEMPLATE.format(values=values)
        return self._sql_cache[n]

    # --- Scoring ---

    def score_batch(self, candidates: List[Tuple[float, float]]) -> List[dict]:
        """Score one batch of (lat, lon) candidates.

        Returns a list of dicts (one per candidate) with keys ``lat``, ``lon``,
        and all columns from _RESULT_COLS.  Results for locations where the
        spatial index returns no rows are returned as all-zero counts.
        """
        if not candidates:
            return []

        # Pre-compute bboxes (vectorised, no pyproj)
        lats = np.array([c[0] for c in candidates])
        lons = np.array([c[1] for c in candidates])
        bboxes = _latlons_to_3857_bboxes(lats, lons, self.bbox_size_m)

        sql = self._get_sql(len(candidates))
        # Flat parameter list: (id, xmin, ymin, xmax, ymax) × n
        flat_params = []
        for idx, (xmin, ymin, xmax, ymax) in enumerate(bboxes):
            flat_params.extend([idx, float(xmin), float(ymin), float(xmax), float(ymax)])

        self._ensure_connected()
        with self._conn.cursor() as cur:
            cur.execute(sql, flat_params)
            rows = cur.fetchall()

        result_by_id: Dict[int, dict] = {
            row[0]: dict(zip(_RESULT_COLS, row)) for row in rows
        }

        out: List[dict] = []
        for idx, (lat, lon) in enumerate(candidates):
            r = dict(result_by_id.get(idx, {"id": idx, **_ZERO_ROW}))
            r["lat"] = lat
            r["lon"] = lon
            out.append(r)
        return out


# ─── Phase 3: Score all candidates (with resume + threading) ─────────────────

def score_candidates(
    candidates: List[Tuple[float, float]],
    postgis_url: str,
    cache_path: str,
    bbox_size_m: float = 500.0,
    batch_size: int = 100,
    n_workers: int = 4,
    query_timeout_s: int = 60,
    flush_interval: int = 10_000,
) -> "List[ScoredLocation]":
    """Score all candidates with resume support and parallel worker threads.

    Each worker thread maintains a persistent PostGIS connection and drains a
    shared work queue.  Results are collected in the main thread, which flushes
    the parquet cache every *flush_interval* newly-completed locations.

    Parameters
    ----------
    candidates:
        Full list of (lat, lon) candidate points to score.
    postgis_url:
        PostGIS connection DSN.
    cache_path:
        Parquet file for resume support (read on start, written every flush).
    bbox_size_m:
        Fixed bbox half-width in metres for feature counting.
    batch_size:
        Locations per SQL query (100 is a good default; reduce if queries time out).
    n_workers:
        Number of parallel PostGIS connections / threads.
    query_timeout_s:
        Per-query statement_timeout in seconds.
    flush_interval:
        Flush the parquet cache after this many new results are collected.

    Returns
    -------
    List of ScoredLocation objects — includes all scored candidates (not filtered
    by feature count), so the caller can apply diversity selection.
    """
    from osmgraphclip.global_sampler import (
        ScoredLocation,
        annotate_h3_and_bucket,
        load_scored_cache,
        save_scored_cache,
    )

    # --- Resume: load existing cache ---
    cache = load_scored_cache(cache_path)

    scored_list: List[ScoredLocation] = []
    for (lat, lon), (richness, bbox_size, total_nodes) in cache.items():
        scored_list.append(
            annotate_h3_and_bucket(lat, lon, richness, bbox_size, total_nodes)
        )

    already_scored: Set[Tuple[float, float]] = set(cache.keys())
    todo: List[Tuple[float, float]] = [
        (lat, lon) for lat, lon in candidates
        if (round(lat, 6), round(lon, 6)) not in already_scored
    ]

    logger.info(
        "score_candidates: %d total, %d from cache, %d remaining | "
        "workers=%d batch_size=%d bbox=%.0fm",
        len(candidates), len(scored_list), len(todo),
        n_workers, batch_size, bbox_size_m,
    )

    if not todo:
        return scored_list

    # --- Thread-safe shared state ---
    work_queue: queue.Queue = queue.Queue()
    result_queue: queue.Queue = queue.Queue()

    # Fill work queue with batches
    n_batches = 0
    for i in range(0, len(todo), batch_size):
        work_queue.put(todo[i: i + batch_size])
        n_batches += 1

    def _worker() -> None:
        """Drain the work queue, scoring batches with a persistent connection."""
        scorer = BatchScorer(postgis_url, batch_size, bbox_size_m, query_timeout_s)
        try:
            scorer.connect()
        except Exception as exc:
            logger.error("Worker failed to connect: %s", exc)
            result_queue.put(None)  # sentinel
            return
        try:
            while True:
                try:
                    batch = work_queue.get_nowait()
                except queue.Empty:
                    break
                try:
                    results = scorer.score_batch(batch)
                    result_queue.put(results)
                except Exception as exc:
                    logger.warning("Batch scoring failed: %s — inserting zeros", exc)
                    # Return zero rows so resume cache skips these locations next run
                    zero_results = [
                        {"lat": lat, "lon": lon, **_ZERO_ROW}
                        for lat, lon in batch
                    ]
                    result_queue.put(zero_results)
                finally:
                    work_queue.task_done()
        finally:
            scorer.close()
            result_queue.put(None)  # sentinel: this worker is done

    # --- Launch workers ---
    workers = [threading.Thread(target=_worker, daemon=True) for _ in range(n_workers)]
    for w in workers:
        w.start()

    # --- Collect results in main thread ---
    n_workers_done = 0
    n_new = 0

    while n_workers_done < n_workers:
        try:
            item = result_queue.get(timeout=180)
        except queue.Empty:
            logger.warning("No results for 180 s — check worker threads")
            continue

        if item is None:
            n_workers_done += 1
            continue

        for r in item:
            total = (
                int(r.get("n_points") or 0)
                + int(r.get("n_lines")  or 0)
                + int(r.get("n_polys")  or 0)
            )
            richness = compute_proxy_richness(r)
            loc = annotate_h3_and_bucket(r["lat"], r["lon"], richness, bbox_size_m, total)
            scored_list.append(loc)
            n_new += 1

        if n_new > 0 and n_new % flush_interval < batch_size:
            save_scored_cache(cache_path, scored_list)
            pct = 100.0 * n_new / max(len(todo), 1)
            logger.info(
                "Progress: %d/%d new scored (%.1f%%) | %d total in cache",
                n_new, len(todo), pct, len(scored_list),
            )

    for w in workers:
        w.join()

    # Final flush
    save_scored_cache(cache_path, scored_list)
    logger.info(
        "Scoring complete: %d new locations scored; %d total in cache",
        n_new, len(scored_list),
    )
    return scored_list

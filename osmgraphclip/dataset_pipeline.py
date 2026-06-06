"""
Shared pipeline utilities used by create_dataset.py and create_graphs.py.
"""

import datetime
import json
import logging
import os
import re
from typing import List, Optional, Set, Tuple

import pandas as pd
import geopandas as gpd

from .osm_tags import load_geojson_to_gdf

logger = logging.getLogger(__name__)


# ── Header normalisation ──────────────────────────────────────────────────────

def _normalize_header(header: str) -> str:
    return header.strip().lower()


# ── GeoJSON loading helpers ───────────────────────────────────────────────────

def _load_geojson_if_exists(path: str) -> Optional[gpd.GeoDataFrame]:
    # Prefer compressed variant; fall back to plain .geojson for existing datasets.
    gz_path = path if path.endswith(".gz") else path + ".gz"
    plain_path = path[:-3] if path.endswith(".gz") else path
    for candidate in (gz_path, plain_path):
        if not os.path.exists(candidate):
            continue
        if os.path.getsize(candidate) == 0:
            logger.warning("Skipping empty (0-byte) file: %s", candidate)
            return None
        try:
            return load_geojson_to_gdf(candidate)
        except Exception as exc:
            logger.warning("Failed to read %s: %s", candidate, exc)
            return None
    return None


def _load_geojson_candidates(paths: List[str]) -> Optional[gpd.GeoDataFrame]:
    gdfs = []
    for path in paths:
        gdf = _load_geojson_if_exists(path)
        if gdf is not None and len(gdf) > 0:
            gdfs.append(gdf)
    if not gdfs:
        return None
    combined = pd.concat([g for g in gdfs if not g.empty], ignore_index=True)
    return gdfs[0].__class__(combined, crs=gdfs[0].crs)


def load_gdfs_for_prefix(output_file_prefix: str) -> Tuple:
    """Load polygon, line, and point GeoDataFrames for a given file prefix."""
    polygon_gdf = _load_geojson_candidates([
        f"{output_file_prefix}_polygon.geojson.gz",
        f"{output_file_prefix}_multipolygon.geojson.gz",
    ])
    line_gdf = _load_geojson_candidates([
        f"{output_file_prefix}_linestring.geojson.gz",
        f"{output_file_prefix}_multilinestring.geojson.gz",
    ])
    point_gdf = _load_geojson_candidates([
        f"{output_file_prefix}_point.geojson.gz",
        f"{output_file_prefix}_multipoint.geojson.gz",
    ])
    return polygon_gdf, line_gdf, point_gdf


# ── Nodata sentinel ───────────────────────────────────────────────────────────

def nodata_sentinel_path(graphs_output_dir: str, i: int) -> str:
    return os.path.join(graphs_output_dir, f"osm_{i}.nodata")


def write_nodata_sentinel(graphs_output_dir: str, i: int, reason: str) -> None:
    """Write a small JSON sentinel file so future resume passes skip this location."""
    path = nodata_sentinel_path(graphs_output_dir, i)
    try:
        with open(path, "w") as f:
            json.dump({
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "reason": reason,
            }, f)
    except Exception as exc:
        logger.warning("Could not write nodata sentinel for location %d: %s", i, exc)


def nodata_is_expired(path: str, ttl_hours: float) -> bool:
    """Return True if the .nodata sentinel at *path* is older than *ttl_hours*."""
    try:
        with open(path) as f:
            data = json.load(f)
        ts = datetime.datetime.fromisoformat(data["timestamp"])
        age = datetime.datetime.now(datetime.timezone.utc) - ts
        return age.total_seconds() > ttl_hours * 3600
    except Exception:
        return False


# ── Directory scanning ────────────────────────────────────────────────────────

_GEOJSON_PREFIX_RE = re.compile(r'^(osm_\d+(?:_L\d+)?)_\w+\.geojson(?:\.gz)?$')


def scan_geojson_prefixes(graphs_output_dir: str) -> Set[str]:
    """Return all geojson_prefix stems present in graphs_output_dir.

    A stem is the part before the geometry-type suffix, e.g. ``osm_42`` or
    ``osm_42_L1``.
    """
    prefixes: Set[str] = set()
    if not os.path.isdir(graphs_output_dir):
        return prefixes
    for fn in os.listdir(graphs_output_dir):
        m = _GEOJSON_PREFIX_RE.match(fn)
        if m:
            prefixes.add(m.group(1))
    return prefixes

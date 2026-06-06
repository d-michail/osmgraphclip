#!/usr/bin/env python3
"""Download OSM GeoJSON files for a set of locations.

Populates the ``downloads`` table in ``dataset.db`` and writes ``metadata.json``
inside ``--output-dir``.  Run ``create_graphs.py`` afterwards to build the
graph pickles that are used for training.
"""

import math
import os
import argparse
import csv
import datetime
import faulthandler
import gzip
import json
import logging
import multiprocessing
import re
import signal
import threading
import psutil
from concurrent.futures import ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED
from concurrent.futures.process import BrokenProcessPool
from typing import Iterator, List, Optional, Tuple, Dict, Set
import pandas as pd
import numpy as np
from osmgraphclip.osm_downloader import OSMDownloader
from osmgraphclip.downloader_config import DownloaderConfig, load_downloader_config, create_downloader
from osmgraphclip.richness import compute_richness_metrics, is_rich_enough, load_tag_frequencies
from osmgraphclip.dataset_pipeline import (
    _normalize_header,
    load_gdfs_for_prefix,
    nodata_is_expired,
    nodata_sentinel_path,
    write_nodata_sentinel,
    scan_geojson_prefixes,
)
from osmgraphclip.dataset_db import DatasetDB

logger = logging.getLogger(__name__)

_shutdown_event = threading.Event()


def _install_signal_handlers():
    def _handler(signum, frame):
        if not _shutdown_event.is_set():
            logger.warning(
                "Signal %d received — finishing in-flight downloads then stopping cleanly...",
                signum,
            )
            _shutdown_event.set()

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


_GEOJSON_SUFFIXES = [
    "_polygon.geojson.gz", "_multipolygon.geojson.gz",
    "_linestring.geojson.gz", "_multilinestring.geojson.gz",
    "_point.geojson.gz", "_multipoint.geojson.gz",
    "_polygon.geojson", "_multipolygon.geojson",
    "_linestring.geojson", "_multilinestring.geojson",
    "_point.geojson", "_multipoint.geojson",
]


def _norm_range(values: List[float]) -> List[float]:
    """Normalise a list to [0, 1] using per-location min/max.

    Returns [1.0, …, 1.0] when all values are equal (no spread to normalise).
    """
    lo, hi = min(values), max(values)
    if hi == lo:
        return [1.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def _compute_selection_scores(
    results: List[Tuple[int, float, str]],
    level_metrics_list: List[Optional[dict]],
    *,
    w_richness: float = 1.0,
    w_entropy: float = 0.0,
    w_categories: float = 0.0,
    w_spatial: float = 0.0,
    w_idf: float = 0.0,
    w_depth: float = 0.0,
    w_size: float = 0.0,
    w_nodes: float = 0.0,
    min_nodes: int = 0,
) -> List[float]:
    """Compute a per-level selection score combining quality metrics and size penalties.

    Quality metrics (richness, entropy, categories, spatial, idf) contribute positively.
    Size and node counts are penalty terms (subtracted).

    ``richness_score``, ``category_coverage``, and ``spatial_coverage`` are already
    globally bounded to [0, 1] and are used as-is.  ``tag_entropy`` and ``idf_score``
    are per-location normalised so they are comparable to the penalty terms.
    ``bbox`` and ``total_nodes`` are log-scaled then per-location normalised to [0, 1],
    which prevents the large absolute bbox range (e.g. 200 m → 20 000 m) from swamping
    the richness signal.

    Levels with fewer than ``min_nodes`` total nodes receive a large negative penalty
    (-2.0) so they are only chosen when no qualifying level exists.
    """
    def _get(m: Optional[dict], key: str, default: float = 0.0) -> float:
        return m.get(key, default) if m else default

    richness  = [_get(m, "richness_score")             for m in level_metrics_list]
    entropy   = [_get(m, "tag_entropy")                for m in level_metrics_list]
    cat_cov   = [_get(m, "category_coverage")          for m in level_metrics_list]
    spatial   = [_get(m, "spatial_coverage")           for m in level_metrics_list]
    idf       = [_get(m, "idf_score")                  for m in level_metrics_list]
    depth     = [_get(m, "semantic_depth")             for m in level_metrics_list]
    raw_nodes = [_get(m, "total_nodes", 0.0)           for m in level_metrics_list]
    nodes     = [max(n, 1.0) for n in raw_nodes]
    bboxes    = [r[1] for r in results]

    entropy_norms = _norm_range(entropy)
    idf_norms     = _norm_range(idf)
    depth_norms   = _norm_range(depth)
    bbox_norms    = _norm_range([math.log(b) for b in bboxes])
    nodes_norms   = _norm_range([math.log(n) for n in nodes])

    scores = [
        w_richness    * richness[k]
        + w_entropy   * entropy_norms[k]
        + w_categories * cat_cov[k]
        + w_spatial   * spatial[k]
        + w_idf       * idf_norms[k]
        + w_depth     * depth_norms[k]
        - w_size      * bbox_norms[k]
        - w_nodes     * nodes_norms[k]
        for k in range(len(results))
    ]

    if min_nodes > 0:
        for k in range(len(results)):
            if raw_nodes[k] < min_nodes:
                scores[k] -= 2.0

    return scores


def _delete_geojson_for_prefix(graphs_output_dir: str, prefix_name: str) -> None:
    """Delete all GeoJSON.gz files associated with a download prefix."""
    full_prefix = os.path.join(graphs_output_dir, prefix_name)
    for suffix in _GEOJSON_SUFFIXES:
        path = full_prefix + suffix
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        except OSError as exc:
            logger.warning("Could not delete %s: %s", path, exc)
    # Fallback: scan directory for any remaining files with this prefix but an
    # unknown/legacy suffix (e.g. old _features.geojson.gz files).
    _fallback_re = re.compile(
        r'^' + re.escape(prefix_name) + r'_.+\.geojson(?:\.gz)?$'
    )
    try:
        for fn in os.listdir(graphs_output_dir):
            if _fallback_re.match(fn):
                path = os.path.join(graphs_output_dir, fn)
                try:
                    os.remove(path)
                    logger.debug("Deleted legacy file: %s", path)
                except OSError as exc:
                    logger.warning("Could not delete %s: %s", path, exc)
    except OSError:
        pass


def _prefix_has_data(prefix: str) -> bool:
    """Return True if any GeoJSON file for *prefix* contains at least one feature.

    Uses stdlib json/gzip rather than pyogrio so it is safe to call from
    forked worker processes (GDAL/pyogrio is not fork-safe on Linux).
    """
    for suffix in _GEOJSON_SUFFIXES:
        path = prefix + suffix
        if not os.path.exists(path):
            continue
        try:
            open_fn = gzip.open if path.endswith(".gz") else open
            with open_fn(path, "rt", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("features"):
                return True
        except Exception:
            pass
    return False


# Predefined bbox profiles: (initial_bbox_size_m, max_bbox_size_m)
# Used with --bbox-profile; individual --bbox-size / --max-bbox-size override these.
# Profiles are also used as resolution levels for --multiresolution (in order, fine → coarse).
BBOX_PROFILES: Dict[str, Tuple[float, float]] = {
    "dense_city": (100.0,    200.0),
    "suburb":     (250.0,    500.0),
    "rural":      (500.0,   2000.0),
    "wilderness": (1000.0,  5000.0),
    "regional":   (5000.0, 20000.0),
}

# Default level bbox sizes (max of each profile) used by --multiresolution when --levels is not given.
DEFAULT_MULTIRESOLUTION_LEVELS: List[float] = [
    BBOX_PROFILES[k][1] for k in BBOX_PROFILES
]

# Predefined city bounding boxes (lat_min, lon_min, lat_max, lon_max)
CITY_BBOXES: Dict[str, Tuple[float, float, float, float]] = {
    "berlin": (52.3382, 13.0883, 52.6755, 13.7611),
    "paris": (48.8155, 2.2241, 48.9022, 2.4699),
    "london": (51.2868, -0.5103, 51.6918, 0.3340),
    "newyork": (40.4774, -74.2591, 40.9176, -73.7004),
    "tokyo": (35.5168, 139.5602, 35.8174, 139.9196),
    "madrid": (40.3120, -3.8882, 40.6437, -3.5179),
    "rome": (41.7951, 12.3730, 42.0502, 12.6410),
    "amsterdam": (52.2784, 4.7283, 52.4310, 5.0790),
    "barcelona": (41.3200, 2.0525, 41.4695, 2.2280),
    "munich": (48.0617, 11.3608, 48.2482, 11.7229),
    "vienna": (48.1185, 16.1827, 48.3233, 16.5774),
    "prague": (50.0004, 14.2244, 50.1774, 14.7068),
    "moscow": (55.4916, 37.3193, 55.9577, 37.9671),
    "stockholm": (59.2093, 17.8089, 59.4524, 18.2710),
    "copenhagen": (55.6145, 12.4532, 55.7271, 12.6509),
    "athens": (37.9008, 23.6255, 38.0479, 23.8171),
}


def generate_grid_locations(
    lat_min: float, lon_min: float, lat_max: float, lon_max: float, sample_spacing_m: float = 500
) -> List[Tuple[float, float]]:
    """Generate a grid of locations covering a bounding box.

    Args:
        lat_min: Minimum latitude of bounding box
        lon_min: Minimum longitude of bounding box
        lat_max: Maximum latitude of bounding box
        lon_max: Maximum longitude of bounding box
        sample_spacing_m: Approximate spacing between samples in meters

    Returns:
        List of (lat, lon) tuples covering the bounding box
    """
    avg_lat = (lat_min + lat_max) / 2
    lat_degree_m = 111320.0
    lon_degree_m = 111320.0 * np.cos(np.radians(avg_lat))

    lat_spacing = sample_spacing_m / lat_degree_m
    lon_spacing = sample_spacing_m / lon_degree_m

    lats = np.arange(lat_min, lat_max, lat_spacing)
    lons = np.arange(lon_min, lon_max, lon_spacing)

    locations = []
    for lat in lats:
        for lon in lons:
            locations.append((float(lat), float(lon)))

    logger.info(f"Generated {len(locations)} grid locations covering the bounding box")
    logger.info(f"Grid dimensions: {len(lats)} x {len(lons)} = {len(locations)} points")

    return locations


def load_locations_from_csv(locations_file: str) -> List[Tuple[float, float]]:
    if not os.path.exists(locations_file):
        raise FileNotFoundError(f"Locations file not found: {locations_file}")

    try:
        df = pd.read_csv(locations_file, encoding="utf-8-sig")
    except pd.errors.EmptyDataError as exc:
        raise ValueError(
            "Locations CSV must include a header row with 'lat' and 'lon' columns."
        ) from exc

    if len(df.columns) == 0:
        raise ValueError(
            "Locations CSV must include a header row with 'lat' and 'lon' columns."
        )

    normalized_columns = {_normalize_header(str(name)): name for name in df.columns}
    lat_column = normalized_columns.get("lat") or normalized_columns.get("latitude")
    lon_column = normalized_columns.get("lon") or normalized_columns.get("longitude")

    if not lat_column or not lon_column:
        raise ValueError(
            "Locations CSV must include 'lat'/'lon' or 'latitude'/'longitude' columns."
        )

    coords_df = df[[lat_column, lon_column]].rename(columns={lat_column: "lat", lon_column: "lon"})
    coords_df["lat"] = pd.to_numeric(coords_df["lat"], errors="coerce")
    coords_df["lon"] = pd.to_numeric(coords_df["lon"], errors="coerce")

    invalid_rows = coords_df[coords_df["lat"].isna() | coords_df["lon"].isna()]
    for row_index in invalid_rows.index.tolist():
        line_index = int(row_index) + 2
        logger.warning(
            "Skipping row %d in %s because lat/lon is missing or invalid.",
            line_index,
            locations_file,
        )

    valid_rows = coords_df.dropna(subset=["lat", "lon"])
    locations = [(float(row.lat), float(row.lon)) for row in valid_rows.itertuples(index=False)]

    if not locations:
        raise ValueError(f"No valid locations found in {locations_file}")

    return locations


def stream_locations_from_csv(locations_file: str) -> Iterator[Tuple[int, float, float]]:
    """Yield (index, lat, lon) one row at a time without loading the full file into memory."""
    if not os.path.exists(locations_file):
        raise FileNotFoundError(f"Locations file not found: {locations_file}")

    with open(locations_file, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        try:
            raw_headers = next(reader)
        except StopIteration:
            return

        normalized = {_normalize_header(h): idx for idx, h in enumerate(raw_headers)}
        lat_idx = normalized.get("lat") if normalized.get("lat") is not None else normalized.get("latitude")
        lon_idx = normalized.get("lon") if normalized.get("lon") is not None else normalized.get("longitude")

        if lat_idx is None or lon_idx is None:
            raise ValueError(
                "Locations CSV must include 'lat'/'lon' or 'latitude'/'longitude' columns."
            )

        for i, row in enumerate(reader):
            try:
                lat = float(row[lat_idx])
                lon = float(row[lon_idx])
                yield (i, lat, lon)
            except (ValueError, IndexError, TypeError):
                logger.warning(
                    "Skipping row %d in %s: lat/lon missing or invalid.", i + 2, locations_file
                )


def _adaptive_download(
    i: int,
    lat: float,
    lon: float,
    output_file_prefix: str,
    initial_bbox_size: float,
    save_road_graph: bool,
    max_bbox_size: float,
    expansion_strategy: str,
    expansion_step: float,
    expansion_factor: float,
    min_total_nodes: int,
    min_unique_labels: int,
    min_richness_score: Optional[float] = None,
    tag_frequencies: Optional[dict] = None,
    bbox_profile: Optional[str] = None,
    downloader_config: Optional[DownloaderConfig] = None,
) -> Tuple[bool, Optional[Tuple], float]:
    """Download OSM data with adaptive bounding box expansion and optional profile escalation.

    Expands the bbox until richness thresholds are met or the current ceiling is reached.
    When a profile is given and a ceiling is reached with zero data, automatically escalates
    to the next larger profile rather than giving up.  Escalation only happens on zero data;
    if any features were found the function stops and returns what it has.

    Returns:
        (success, gdfs, actual_bbox_size)
        where gdfs = (polygon_gdf, line_gdf, point_gdf) or None on failure.
    """
    profile_keys = list(BBOX_PROFILES.keys())
    if bbox_profile and bbox_profile in BBOX_PROFILES:
        start_idx = profile_keys.index(bbox_profile)
        ceilings = [
            (profile_keys[idx], BBOX_PROFILES[profile_keys[idx]][1])
            for idx in range(start_idx, len(profile_keys))
        ]
    else:
        ceilings = [(None, max_bbox_size)]

    ceiling_idx = 0
    current_label, current_max = ceilings[ceiling_idx]
    bbox_size = initial_bbox_size
    polygon_gdf = line_gdf = point_gdf = None

    while True:
        downloader = create_downloader(
            downloader_config or DownloaderConfig(),
            lat=lat, lon=lon, dist=bbox_size,
            output_file=output_file_prefix,
            save_road_graph=save_road_graph,
        )
        downloaded = downloader()

        profile_tag = f"[{current_label}] " if current_label else ""

        if downloaded:
            polygon_gdf, line_gdf, point_gdf = load_gdfs_for_prefix(output_file_prefix)
        else:
            polygon_gdf, line_gdf, point_gdf = None, None, None
            logger.info(
                "Location %d %sbbox=%dm: no OSM features returned.",
                i, profile_tag, int(bbox_size),
            )

        metrics = compute_richness_metrics(
            polygon_gdf, line_gdf, point_gdf,
            tag_frequencies=tag_frequencies,
            bbox_size_m=bbox_size,
        )

        if downloaded:
            logger.info(
                "Location %d %sbbox=%dm: nodes=%d, unique_labels=%d, "
                "richness=%.3f (entropy=%.2f, cat_cov=%.2f, spatial=%.2f, idf=%.2f)",
                i, profile_tag, int(bbox_size),
                metrics["total_nodes"], metrics["unique_labels"],
                metrics["richness_score"], metrics["tag_entropy"],
                metrics["category_coverage"], metrics["spatial_coverage"],
                metrics["idf_score"],
            )

        if is_rich_enough(metrics, min_total_nodes, min_unique_labels, min_richness_score):
            return True, (polygon_gdf, line_gdf, point_gdf), bbox_size

        if expansion_strategy == "linear":
            next_size = bbox_size + expansion_step
        else:
            next_size = bbox_size * expansion_factor

        if next_size > current_max:
            has_data = polygon_gdf is not None or line_gdf is not None or point_gdf is not None
            if not has_data:
                next_ceiling_idx = ceiling_idx + 1
                if next_ceiling_idx < len(ceilings):
                    ceiling_idx = next_ceiling_idx
                    current_label, current_max = ceilings[ceiling_idx]
                    logger.info(
                        "Location %d: no data up to %dm, escalating to profile '%s' (max %dm).",
                        i, int(bbox_size), current_label, int(current_max),
                    )
                    bbox_size = next_size
                    continue
                else:
                    logger.warning(
                        "Location %d: all profiles exhausted up to %dm with no OSM data. Skipping.",
                        i, int(bbox_size),
                    )
                    return False, None, bbox_size
            else:
                logger.warning(
                    "Location %d: max bbox %dm reached (profile '%s', nodes=%d, unique_labels=%d). "
                    "Proceeding with available data.",
                    i, int(current_max), current_label or "custom",
                    metrics["total_nodes"], metrics["unique_labels"],
                )
                return True, (polygon_gdf, line_gdf, point_gdf), bbox_size

        logger.info(
            "Location %d: %snot rich enough, expanding bbox %dm → %dm",
            i, profile_tag, int(bbox_size), int(next_size),
        )
        bbox_size = next_size


def _multiresolution_download(
    i: int,
    lat: float,
    lon: float,
    graphs_output_dir: str,
    level_sizes: List[float],
    save_road_graph: bool,
    tag_frequencies: Optional[dict],
    adaptive_bbox: bool = False,
    min_total_nodes: int = 5,
    min_unique_labels: int = 3,
    min_richness_score: Optional[float] = None,
    expansion_strategy: str = "multiplicative",
    expansion_factor: float = 1.5,
    expansion_step: float = 100.0,
    adaptive_max_factor: float = 4.0,
    best_level_only: bool = False,
    best_level_weights: Optional[dict] = None,
    downloader_config: Optional[DownloaderConfig] = None,
) -> List[Tuple[int, float, str]]:
    """Download GeoJSON files for all resolution levels for one location.

    Scans levels fine → coarse to find the minimum level that returns OSM data,
    then downloads GeoJSON for all levels from that minimum up to the coarsest.

    When *adaptive_bbox* is True, a Phase 3 is appended: if the coarsest level
    is not rich enough, the bbox is expanded beyond the last predefined level up
    to ``last_level_size * adaptive_max_factor``, and the result is appended as
    an extra level (index ``len(level_sizes)``).

    Returns:
        List of (level_idx, bbox_size, geojson_prefix_basename) for each
        successfully downloaded level.  Empty list if no data was found at any
        level.
    """
    # ── Phase 1: find the minimum (finest) level that returns OSM data ────────
    min_level_idx: Optional[int] = None

    for level_idx, bbox_size in enumerate(level_sizes):
        prefix = os.path.join(graphs_output_dir, f"osm_{i}_L{level_idx}")
        downloader = create_downloader(
            downloader_config or DownloaderConfig(),
            lat=lat, lon=lon, dist=bbox_size,
            output_file=prefix, save_road_graph=save_road_graph,
        )
        downloaded = downloader()

        if not downloaded:
            logger.info(
                "Location %d: no data at L%d (bbox=%dm), trying next level",
                i, level_idx, int(bbox_size),
            )
            continue

        if _prefix_has_data(prefix):
            min_level_idx = level_idx
            logger.info(
                "Location %d (%.5f, %.5f): minimum level with data is L%d (bbox=%dm)",
                i, lat, lon, level_idx, int(bbox_size),
            )
            break

        logger.info(
            "Location %d: download returned no features at L%d (bbox=%dm), trying next level",
            i, level_idx, int(bbox_size),
        )

    if min_level_idx is None:
        logger.warning(
            "Location %d (%.5f, %.5f): no OSM data found at any level. Skipping.",
            i, lat, lon,
        )
        return []

    # ── Phase 2: download GeoJSON for all levels ≥ min_level_idx ─────────────
    results: List[Tuple[int, float, str]] = []
    level_richness: List[float] = []
    level_metrics_list: List[Optional[dict]] = []
    last_gdfs: Optional[Tuple] = None
    last_bbox_size: float = level_sizes[-1]
    last_metrics: Optional[dict] = None
    stopped_for_row_limit: bool = False

    for level_idx in range(min_level_idx, len(level_sizes)):
        bbox_size = level_sizes[level_idx]
        prefix = os.path.join(graphs_output_dir, f"osm_{i}_L{level_idx}")

        if level_idx == min_level_idx:
            # Already downloaded in Phase 1; load GDFs only if richness check needed.
            if adaptive_bbox or best_level_only:
                polygon_gdf, line_gdf, point_gdf = load_gdfs_for_prefix(prefix)
                last_gdfs = (polygon_gdf, line_gdf, point_gdf)
        else:
            downloader = create_downloader(
                downloader_config or DownloaderConfig(),
                lat=lat, lon=lon, dist=bbox_size,
                output_file=prefix, save_road_graph=save_road_graph,
            )
            if not downloader():
                if getattr(downloader, "hit_row_limit", False):
                    logger.warning(
                        "Location %d: L%d (bbox=%dm) exceeded row limit — "
                        "stopping further levels.",
                        i, level_idx, int(bbox_size),
                    )
                    stopped_for_row_limit = True
                    break
                logger.warning(
                    "Location %d: download failed at L%d (bbox=%dm). Skipping level.",
                    i, level_idx, int(bbox_size),
                )
                continue
            if not _prefix_has_data(prefix):
                logger.warning(
                    "Location %d: no features at L%d (bbox=%dm). Skipping level.",
                    i, level_idx, int(bbox_size),
                )
                continue
            if adaptive_bbox or best_level_only:
                polygon_gdf, line_gdf, point_gdf = load_gdfs_for_prefix(prefix)
                last_gdfs = (polygon_gdf, line_gdf, point_gdf)

        last_bbox_size = bbox_size
        if (adaptive_bbox or best_level_only) and last_gdfs is not None:
            last_metrics = compute_richness_metrics(
                last_gdfs[0], last_gdfs[1], last_gdfs[2],
                tag_frequencies=tag_frequencies,
                bbox_size_m=bbox_size,
            )

        prefix_name = f"osm_{i}_L{level_idx}"
        results.append((level_idx, bbox_size, prefix_name))
        level_richness.append(last_metrics.get("richness_score", 0.0) if last_metrics is not None else 0.0)
        level_metrics_list.append(last_metrics)
        logger.info(
            "Location %d: downloaded L%d (bbox=%dm) → %s",
            i, level_idx, int(bbox_size), prefix_name,
        )

    # ── Phase 3: adaptive expansion beyond the coarsest predefined level ──────
    if adaptive_bbox and results and last_metrics is not None and not stopped_for_row_limit:
        if not is_rich_enough(last_metrics, min_total_nodes, min_unique_labels, min_richness_score):
            if expansion_strategy == "linear":
                next_start = last_bbox_size + expansion_step
            else:
                next_start = last_bbox_size * expansion_factor

            adaptive_max = last_bbox_size * adaptive_max_factor

            if next_start <= adaptive_max:
                adaptive_level_idx = len(level_sizes)
                adaptive_prefix = os.path.join(
                    graphs_output_dir, f"osm_{i}_L{adaptive_level_idx}"
                )
                logger.info(
                    "Location %d: coarsest level not rich enough (richness=%.3f), "
                    "applying adaptive expansion %dm → max %dm",
                    i, last_metrics.get("richness_score", 0.0),
                    int(next_start), int(adaptive_max),
                )
                success, adaptive_gdfs, actual_size = _adaptive_download(
                    i=i, lat=lat, lon=lon,
                    output_file_prefix=adaptive_prefix,
                    initial_bbox_size=next_start,
                    save_road_graph=save_road_graph,
                    max_bbox_size=adaptive_max,
                    expansion_strategy=expansion_strategy,
                    expansion_step=expansion_step,
                    expansion_factor=expansion_factor,
                    min_total_nodes=min_total_nodes,
                    min_unique_labels=min_unique_labels,
                    min_richness_score=min_richness_score,
                    tag_frequencies=tag_frequencies,
                    bbox_profile=None,
                    downloader_config=downloader_config,
                )
                if success:
                    prefix_name = f"osm_{i}_L{adaptive_level_idx}"
                    results.append((adaptive_level_idx, actual_size, prefix_name))
                    if adaptive_gdfs is not None:
                        ph3_metrics = compute_richness_metrics(
                            adaptive_gdfs[0], adaptive_gdfs[1], adaptive_gdfs[2],
                            tag_frequencies=tag_frequencies,
                            bbox_size_m=actual_size,
                        )
                        level_richness.append(ph3_metrics.get("richness_score", 0.0))
                        level_metrics_list.append(ph3_metrics)
                    else:
                        level_richness.append(0.0)
                        level_metrics_list.append(None)
                    logger.info(
                        "Location %d: adaptive downloaded L%d (bbox=%dm) → %s",
                        i, adaptive_level_idx, int(actual_size), prefix_name,
                    )
            else:
                logger.info(
                    "Location %d: adaptive expansion skipped "
                    "(next_start=%dm already exceeds adaptive_max=%dm)",
                    i, int(next_start), int(adaptive_max),
                )

    if best_level_only and results:
        weights = best_level_weights or {}
        selection_scores = _compute_selection_scores(
            results, level_metrics_list,
            w_richness=weights.get("richness", 1.0),
            w_entropy=weights.get("entropy", 0.0),
            w_categories=weights.get("categories", 0.0),
            w_spatial=weights.get("spatial", 0.0),
            w_idf=weights.get("idf", 0.0),
            w_depth=weights.get("depth", 0.0),
            w_size=weights.get("size", 0.0),
            w_nodes=weights.get("nodes", 0.0),
            min_nodes=int(weights.get("min_nodes", 0)),
        )
        best_idx = max(range(len(results)), key=lambda k: selection_scores[k])
        best_level_idx, best_bbox, best_prefix = results[best_idx]
        # Log per-level statistics so the selection rationale is visible in logs.
        for k, (lvl_idx, lvl_bbox, lvl_prefix) in enumerate(results):
            m = level_metrics_list[k]
            marker = " ★ SELECTED" if k == best_idx else ""
            if m is not None:
                logger.info(
                    "Location %d  L%d bbox=%dm%s: "
                    "nodes=%d  labels=%d  richness=%.3f  depth=%.2f  "
                    "entropy=%.2f  cat_cov=%.2f  spatial=%.2f  idf=%.2f  "
                    "sel_score=%.3f",
                    i, lvl_idx, int(lvl_bbox), marker,
                    m.get("total_nodes", 0), m.get("unique_labels", 0),
                    m.get("richness_score", 0.0), m.get("semantic_depth", 0.0),
                    m.get("tag_entropy", 0.0),
                    m.get("category_coverage", 0.0), m.get("spatial_coverage", 0.0),
                    m.get("idf_score", 0.0), selection_scores[k],
                )
            else:
                logger.info(
                    "Location %d  L%d bbox=%dm%s: richness=%.3f  sel_score=%.3f (no metrics)",
                    i, lvl_idx, int(lvl_bbox), marker,
                    level_richness[k], selection_scores[k],
                )
        # Collect every osm_{i}_L* prefix on disk — previous runs may have left
        # files for levels that were not downloaded in this run (e.g. because the
        # row limit caused an early break), so we can't rely on `results` alone.
        disk_level_re = re.compile(rf'^osm_{i}_L\d+$')
        all_level_prefixes_on_disk = {
            p for p in scan_geojson_prefixes(graphs_output_dir)
            if disk_level_re.match(p)
        }
        # Build the set of prefixes in the current results for quick lookup.
        results_prefixes = {prefix_name for _, _, prefix_name in results}
        orphaned = all_level_prefixes_on_disk - results_prefixes - {best_prefix}
        if orphaned:
            logger.info(
                "Location %d: deleting %d orphaned level prefix(es) from previous runs: %s",
                i, len(orphaned), sorted(orphaned),
            )
        logger.info(
            "Location %d: keeping L%d (bbox=%dm, richness=%.3f, sel_score=%.3f), "
            "discarding %d other level(s)",
            i, best_level_idx, int(best_bbox), level_richness[best_idx],
            selection_scores[best_idx], len(results) - 1,
        )
        for k, (_, _, prefix_name) in enumerate(results):
            if k != best_idx:
                _delete_geojson_for_prefix(graphs_output_dir, prefix_name)
        for prefix_name in orphaned:
            _delete_geojson_for_prefix(graphs_output_dir, prefix_name)
        results = [results[best_idx]]

    return results


def _process_location(
    i: int,
    lat: float,
    lon: float,
    *,
    graphs_output_dir: str,
    args,
    tag_frequencies: Optional[dict],
    multiresolution_levels: List[float],
    adaptive_max_factor: float,
    downloader_config: Optional[DownloaderConfig] = None,
) -> Tuple[str, List[list]]:
    """Download OSM GeoJSON file(s) for one location.

    Returns:
        (outcome, csv_rows) where outcome is ``"processed"`` or ``"skipped_empty"``,
        and csv_rows is a (possibly empty) list of rows to insert into the downloads table.
    """
    output_file_prefix = os.path.join(graphs_output_dir, f"osm_{i}")
    _proc = psutil.Process()
    _mem_mb = _proc.memory_info().rss / 1024 / 1024
    logger.debug("Location %d: worker RSS before = %.0f MB", i, _mem_mb)

    # ── Multiresolution ───────────────────────────────────────────────────────
    if args.multiresolution:
        logger.info(
            "Processing location %d (%.5f, %.5f) [multiresolution, %d levels%s]",
            i, lat, lon, len(multiresolution_levels),
            "+adaptive" if args.adaptive_bbox else "",
        )
        level_results = _multiresolution_download(
            i=i, lat=lat, lon=lon,
            graphs_output_dir=graphs_output_dir,
            level_sizes=multiresolution_levels,
            save_road_graph=args.save_road_graph,
            tag_frequencies=tag_frequencies,
            adaptive_bbox=args.adaptive_bbox,
            min_total_nodes=args.min_total_nodes,
            min_unique_labels=args.min_unique_labels,
            min_richness_score=args.min_richness_score,
            expansion_strategy=args.bbox_expansion_strategy,
            expansion_factor=args.bbox_expansion_factor,
            expansion_step=args.bbox_expansion_step,
            adaptive_max_factor=adaptive_max_factor,
            best_level_only=getattr(args, "best_level_only", False),
            best_level_weights=getattr(args, "best_level_weights", None),
            downloader_config=downloader_config,
        )
        if not level_results:
            write_nodata_sentinel(graphs_output_dir, i, "no OSM data at any resolution level")
            return "skipped_empty", []
        rows = []
        for lvl_idx, lvl_size, prefix_name in level_results:
            timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
            rows.append([lat, lon, timestamp, lvl_size, prefix_name, lvl_idx])
        logger.debug("Location %d: worker RSS after = %.0f MB (Δ%.0f MB)",
                     i, _proc.memory_info().rss / 1024 / 1024,
                     _proc.memory_info().rss / 1024 / 1024 - _mem_mb)
        return "processed", rows

    # ── Single-level ──────────────────────────────────────────────────────────
    actual_bbox_size = args.bbox_size

    if args.adaptive_bbox:
        logger.info("Downloading data for location %d: (%s, %s) [adaptive bbox]", i, lat, lon)
        success, _, actual_bbox_size = _adaptive_download(
            i=i, lat=lat, lon=lon,
            output_file_prefix=output_file_prefix,
            initial_bbox_size=args.bbox_size,
            save_road_graph=args.save_road_graph,
            max_bbox_size=args.max_bbox_size,
            expansion_strategy=args.bbox_expansion_strategy,
            expansion_step=args.bbox_expansion_step,
            expansion_factor=args.bbox_expansion_factor,
            min_total_nodes=args.min_total_nodes,
            min_unique_labels=args.min_unique_labels,
            min_richness_score=args.min_richness_score,
            tag_frequencies=tag_frequencies,
            bbox_profile=args.bbox_profile,
            downloader_config=downloader_config,
        )
        if not success:
            write_nodata_sentinel(graphs_output_dir, i, "no OSM data within adaptive bbox limit")
            return "skipped_empty", []
    else:
        logger.info("Downloading data for location %d: (%s, %s)", i, lat, lon)
        downloader = create_downloader(
            downloader_config or DownloaderConfig(),
            lat=lat, lon=lon,
            dist=args.bbox_size,
            output_file=output_file_prefix,
            save_road_graph=args.save_road_graph,
        )
        if not downloader():
            logger.warning(
                "Skipping location %d (%s, %s): no OSM data returned.", i, lat, lon,
            )
            write_nodata_sentinel(graphs_output_dir, i, "no OSM data returned")
            return "skipped_empty", []

    prefix_name = f"osm_{i}"
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    return "processed", [[lat, lon, timestamp, actual_bbox_size, prefix_name]]


_LOG_FORMAT = "%(asctime)s %(levelname)s [PID%(process)d] %(name)s: %(message)s"
_NOISY_LOGGERS = ("httpx", "httpcore", "huggingface_hub", "transformers")


def _worker_init(log_level: int = logging.INFO):
    """Logging setup for ProcessPoolExecutor worker processes."""
    faulthandler.enable()  # dump C stack trace to stderr on SIGSEGV
    # Workers must not handle SIGINT themselves — the main process owns the signal
    # and communicates shutdown via _shutdown_event.  Without this, Ctrl+C raises
    # KeyboardInterrupt inside a worker, which propagates back through future.result()
    # and crashes the main process before the graceful-shutdown path can run.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    logging.basicConfig(level=log_level, format=_LOG_FORMAT)
    for noisy_logger in _NOISY_LOGGERS:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)


def main():
    logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT)
    for noisy_logger in _NOISY_LOGGERS:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)
    _install_signal_handlers()

    parser = argparse.ArgumentParser(
        description="Download OSM GeoJSON files for a set of locations.",
    )
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to store the dataset.")
    parser.add_argument(
        "--bbox-size", type=float, default=None,
        help="Initial bounding box half-width in meters. Overrides --bbox-profile. Default: 250m.",
    )
    parser.add_argument(
        "--tagw-path", type=str, default="data/all_tags30_frequency1.json",
        help=(
            "Path to GeoLink tag weights JSON file. "
            "Only needed when --adaptive-bbox is active (used for IDF-based richness scoring)."
        ),
    )
    parser.add_argument(
        "--locations-file", type=str,
        help=(
            "Path to CSV file with location coordinates. Reads lat/lon by header name "
            "(order independent), supports lat/lon or latitude/longitude, and ignores extra columns."
        ),
    )
    parser.add_argument(
        "--city", type=str,
        help=f"Name of city to generate dataset for. Available cities: {', '.join(CITY_BBOXES.keys())}",
    )
    parser.add_argument(
        "--sample-spacing", type=float, default=500,
        help="Spacing between grid samples in meters (only used with --city). Default: 500m",
    )
    parser.add_argument(
        "--save-road-graph", action="store_true",
        help="Also download and save OSM road graph as GraphML (*.graphml).",
    )

    # Bounding box profile and adaptive expansion arguments
    parser.add_argument(
        "--bbox-profile", type=str, default=None, choices=list(BBOX_PROFILES.keys()),
        help=(
            "Preset bbox profile that sets --bbox-size and --max-bbox-size together. "
            "Choices: dense_city (100m/200m), suburb (250m/500m), "
            "rural (500m/2000m), wilderness (1000m/5000m), regional (5000m/20000m). "
            "Explicit --bbox-size / --max-bbox-size always override the profile."
        ),
    )
    parser.add_argument(
        "--adaptive-bbox", action="store_true",
        help=(
            "Enable adaptive bounding box expansion. If the downloaded area is semantically "
            "poor (too few nodes or unique labels), the bbox is expanded and retried until "
            "the richness thresholds are met or --max-bbox-size is reached."
        ),
    )
    parser.add_argument(
        "--max-bbox-size", type=float, default=None,
        help=(
            "Maximum bounding box half-width in meters when using --adaptive-bbox "
            "(absolute value, overrides --adaptive-max-factor and --bbox-profile)."
        ),
    )
    parser.add_argument(
        "--adaptive-max-factor", type=float, default=None,
        help=(
            "Maximum bbox expressed as a multiplier of the starting size. "
            "Default: 4.0 when --adaptive-bbox is active and --max-bbox-size is not given."
        ),
    )
    parser.add_argument(
        "--bbox-expansion-strategy", type=str, default="multiplicative",
        choices=["multiplicative", "linear"],
        help=(
            "Strategy for expanding the bbox on each retry. "
            "'multiplicative' multiplies by --bbox-expansion-factor (default); "
            "'linear' adds --bbox-expansion-step."
        ),
    )
    parser.add_argument(
        "--bbox-expansion-factor", type=float, default=1.5,
        help="Multiplicative growth factor for bbox expansion. Default: 1.5.",
    )
    parser.add_argument(
        "--bbox-expansion-step", type=float, default=100.0,
        help="Step size in meters for bbox expansion (linear strategy). Default: 100m.",
    )
    parser.add_argument(
        "--min-total-nodes", type=int, default=5,
        help="Minimum total OSM feature count required before accepting a sample. Default: 5.",
    )
    parser.add_argument(
        "--min-unique-labels", type=int, default=3,
        help="Minimum number of distinct OSM tag label strings required. Default: 3.",
    )
    parser.add_argument(
        "--min-richness-score", type=float, default=0.2,
        help="Minimum composite richness score [0, 1] (--adaptive-bbox only). Default: 0.2.",
    )

    # Multi-resolution arguments
    parser.add_argument(
        "--multiresolution", action="store_true",
        help=(
            "Download GeoJSON at multiple resolution levels for each location. "
            "Scans levels fine → coarse to find the minimum level that returns OSM data, "
            "then downloads GeoJSON for all levels from that minimum up to the coarsest. "
            "Can be combined with --adaptive-bbox."
        ),
    )
    parser.add_argument(
        "--levels", type=float, nargs="+", default=None, metavar="BBOX_M",
        help=(
            "Bbox half-widths in metres for each resolution level (fine → coarse), "
            "used with --multiresolution. "
            f"Default: {' '.join(str(int(s)) for s in DEFAULT_MULTIRESOLUTION_LEVELS)}."
        ),
    )
    parser.add_argument(
        "--best-level-only", action="store_true", default=False,
        dest="best_level_only",
        help=(
            "When used with --multiresolution, keep only the level with the highest "
            "selection score per location and delete the GeoJSON files for all other levels. "
            "Richness is computed for every downloaded level regardless of --adaptive-bbox. "
            "Use --best-level-*-weight flags to tune the selection scoring. "
            "Defaults (node-weight=0.5, size-weight=0.1) target graphs of ~20-30 nodes."
        ),
    )
    parser.add_argument(
        "--best-level-richness-weight", type=float, default=1.0, dest="blw_richness",
        metavar="W",
        help="Weight for composite richness score in best-level selection (default 1.0).",
    )
    parser.add_argument(
        "--best-level-entropy-weight", type=float, default=0.0, dest="blw_entropy",
        metavar="W",
        help="Weight for tag entropy (per-location normalised) in best-level selection (default 0.0).",
    )
    parser.add_argument(
        "--best-level-categories-weight", type=float, default=0.1, dest="blw_categories",
        metavar="W",
        help="Weight for category coverage in best-level selection (default 0.1).",
    )
    parser.add_argument(
        "--best-level-spatial-weight", type=float, default=0.0, dest="blw_spatial",
        metavar="W",
        help="Weight for spatial coverage in best-level selection (default 0.0).",
    )
    parser.add_argument(
        "--best-level-idf-weight", type=float, default=0.0, dest="blw_idf",
        metavar="W",
        help="Weight for IDF score (per-location normalised) in best-level selection (default 0.0).",
    )
    parser.add_argument(
        "--best-level-depth-weight", type=float, default=0.2, dest="blw_depth",
        metavar="W",
        help=(
            "Weight for semantic depth (avg non-null tags per feature, per-location normalised) "
            "in best-level selection. Positive values prefer levels with richly-tagged POIs "
            "over levels dominated by sparse infrastructure. Default 0.2."
        ),
    )
    parser.add_argument(
        "--best-level-size-weight", type=float, default=0.1, dest="blw_size",
        metavar="W",
        help=(
            "Penalty weight for bbox size (log-normalised to [0,1] per location) in "
            "best-level selection. Positive values prefer finer (smaller) bboxes. Default 0.1."
        ),
    )
    parser.add_argument(
        "--best-level-node-weight", type=float, default=0.5, dest="blw_nodes",
        metavar="W",
        help=(
            "Penalty weight for total node count (log-normalised per location) in "
            "best-level selection. Positive values prefer levels with fewer nodes. Default 0.5."
        ),
    )
    parser.add_argument(
        "--best-level-min-nodes", type=int, default=10, dest="blw_min_nodes",
        metavar="N",
        help=(
            "Minimum node count floor for best-level selection. Levels with fewer than N "
            "total OSM features receive a large penalty (-2.0) so they are only chosen when "
            "no level meets the floor. Default 10."
        ),
    )

    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--resume", dest="resume", action="store_true",
        help="Resume: skip already downloaded locations (default).",
    )
    resume_group.add_argument(
        "--no-resume", dest="resume", action="store_false",
        help="Do not resume: re-download all locations and clear the downloads table.",
    )
    parser.set_defaults(resume=True)
    parser.add_argument(
        "--nodata-retry-after", type=float, default=None, metavar="HOURS",
        help=(
            "Retry locations whose .nodata sentinel is older than this many hours. "
            "By default .nodata files are never retried automatically."
        ),
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help=(
            "Number of parallel worker threads. Keep low (1–4) for the public Overpass API. "
            "Default: 4."
        ),
    )
    parser.add_argument(
        "--debug", action="store_true", default=False,
        help="Enable DEBUG-level logging (includes per-row counts, memory stats, etc.).",
    )
    parser.add_argument(
        "--backend", type=str, default="overpass",
        choices=["overpass", "postgis", "auto"],
        help=(
            "Data source backend. 'overpass' uses the Overpass API (default). "
            "'postgis' queries a local PostGIS database loaded with osm2pgsql. "
            "'auto' tries PostGIS first and falls back to Overpass if no data is found."
        ),
    )
    parser.add_argument(
        "--postgis-url", type=str, default=None,
        dest="postgis_url",
        help=(
            "PostgreSQL DSN for the PostGIS backend, "
            "e.g. postgresql://osm:osm@localhost:5432/gis. "
            "Overrides the POSTGIS_URL environment variable / .env file."
        ),
    )
    parser.add_argument(
        "--postgis-max-rows", type=int, default=50_000,
        dest="postgis_max_rows_per_table",
        help=(
            "Maximum rows per geometry table (point/line/polygon) returned by a single "
            "PostGIS query. When a table exceeds this limit the level is skipped and no "
            "coarser levels are attempted. Default: 50000. Set to 0 for unlimited."
        ),
    )
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # ── Build downloader config (loads .env, validates backend/url) ───────────
    try:
        downloader_config = load_downloader_config(args)
    except ValueError as exc:
        parser.error(str(exc))

    # ── Detect explicit user inputs ───────────────────────────────────────────
    _user_gave_max_bbox_size = args.max_bbox_size is not None
    _user_gave_adaptive_max_factor = args.adaptive_max_factor is not None

    if _user_gave_max_bbox_size and _user_gave_adaptive_max_factor:
        parser.error("--max-bbox-size and --adaptive-max-factor are mutually exclusive.")

    if args.best_level_only and not args.multiresolution:
        parser.error("--best-level-only requires --multiresolution.")

    # ── Assemble best-level selection weights ─────────────────────────────────
    args.best_level_weights = {
        "richness":   args.blw_richness,
        "entropy":    args.blw_entropy,
        "categories": args.blw_categories,
        "spatial":    args.blw_spatial,
        "idf":        args.blw_idf,
        "depth":      args.blw_depth,
        "size":       args.blw_size,
        "nodes":      args.blw_nodes,
        "min_nodes":  args.blw_min_nodes,
    }

    # ── Apply bbox profile defaults ───────────────────────────────────────────
    if args.bbox_profile:
        profile_initial, profile_max = BBOX_PROFILES[args.bbox_profile]
        if args.bbox_size is None:
            args.bbox_size = profile_initial
        if args.max_bbox_size is None:
            args.max_bbox_size = profile_max

    if args.bbox_size is None:
        args.bbox_size = 250.0

    _adaptive_max_factor: float = args.adaptive_max_factor if _user_gave_adaptive_max_factor else 4.0

    if not _user_gave_max_bbox_size and args.max_bbox_size is None:
        args.max_bbox_size = args.bbox_size * _adaptive_max_factor

    # ── Resolve multiresolution level sizes ───────────────────────────────────
    multiresolution_levels: List[float] = []
    if args.multiresolution:
        if args.levels:
            multiresolution_levels = sorted(args.levels)
        else:
            multiresolution_levels = list(DEFAULT_MULTIRESOLUTION_LEVELS)
        logger.info(
            "Multiresolution mode: %d levels — %s%s",
            len(multiresolution_levels),
            ", ".join(f"L{idx}={int(s)}m" for idx, s in enumerate(multiresolution_levels)),
            f" + adaptive fallback (×{_adaptive_max_factor:.1f})" if args.adaptive_bbox else "",
        )

    # ── Load tag frequencies for richness scoring ─────────────────────────────
    tag_frequencies: Optional[dict] = None
    if (args.adaptive_bbox or args.best_level_only) and os.path.exists(args.tagw_path):
        try:
            tag_frequencies = load_tag_frequencies(args.tagw_path)
        except Exception as exc:
            logger.warning("Could not load tag frequencies from %s: %s", args.tagw_path, exc)

    # ── Location source ───────────────────────────────────────────────────────
    if args.city:
        city_key = args.city.lower()
        if city_key not in CITY_BBOXES:
            logger.error("Unknown city: %s. Available: %s", args.city, ", ".join(CITY_BBOXES.keys()))
            return
        lat_min, lon_min, lat_max, lon_max = CITY_BBOXES[city_key]
        logger.info("Generating dataset for %s", args.city)
        locations = generate_grid_locations(
            lat_min, lon_min, lat_max, lon_max, sample_spacing_m=args.sample_spacing
        )
        location_source = f"city:{args.city}"
    elif args.locations_file:
        locations = None  # will stream row-by-row
        location_source = args.locations_file
    else:
        logger.error("Either --city or --locations-file must be specified.")
        parser.print_help()
        return

    graphs_output_dir = os.path.join(args.output_dir, "graphs")
    os.makedirs(graphs_output_dir, exist_ok=True)

    db_path = os.path.join(args.output_dir, "dataset.db")
    metadata_path = os.path.join(args.output_dir, "metadata.json")

    # ── Write metadata (download-side fields) ─────────────────────────────────
    metadata: dict = {
        "bbox_half_width_m": args.bbox_size,
        "bbox_profile": args.bbox_profile,
        "location_source": location_source,
    }
    if args.city:
        metadata["city"] = args.city
        metadata["sample_spacing_m"] = args.sample_spacing
        metadata["city_bbox"] = CITY_BBOXES[args.city.lower()]
    if args.adaptive_bbox:
        metadata["adaptive_bbox"] = True
        metadata["adaptive_max_factor"] = _adaptive_max_factor
        metadata["max_bbox_size_m"] = args.max_bbox_size
        metadata["bbox_expansion_strategy"] = args.bbox_expansion_strategy
        metadata["bbox_expansion_factor"] = args.bbox_expansion_factor
        metadata["bbox_expansion_step"] = args.bbox_expansion_step
        metadata["min_total_nodes"] = args.min_total_nodes
        metadata["min_unique_labels"] = args.min_unique_labels
        metadata["min_richness_score"] = args.min_richness_score
    if args.multiresolution:
        metadata["multiresolution"] = True
        metadata["level_sizes_m"] = multiresolution_levels
        metadata["level_names"] = [f"L{idx}" for idx in range(len(multiresolution_levels))]
        # OsmDataset reads bbox_half_width_m; use the coarsest level for compatibility.
        metadata["bbox_half_width_m"] = max(multiresolution_levels)

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    processed_count = 0
    skipped_count = 0
    skipped_existing_count = 0
    recovered_csv_count = 0

    with DatasetDB(db_path) as db:
        if not args.resume:
            db.clear_downloads()
            logger.info("Resume disabled: cleared downloads table in %s.", db_path)

        existing_geojson_prefixes = db.downloaded_prefixes

        # Pre-compute location indices that have any multiresolution level recorded
        # (covers predefined L0..N-1 *and* any adaptive extra levels).
        # Also track level count per location so that --best-level-only can detect
        # locations that still have multiple levels and need to be regenerated.
        if args.multiresolution:
            _mr_re = re.compile(r'^osm_(\d+)_L\d+$')
            # Union DB and disk prefixes so neither source is missed.
            _all_mr_prefixes = existing_geojson_prefixes | scan_geojson_prefixes(graphs_output_dir)
            _combined_mr_counts: Dict[int, int] = {}
            for p in _all_mr_prefixes:
                m = _mr_re.match(p)
                if m:
                    idx = int(m.group(1))
                    _combined_mr_counts[idx] = _combined_mr_counts.get(idx, 0) + 1
            existing_multiresolution_indices: Set[int] = set(_combined_mr_counts.keys())

        # ── Recovery pre-pass (city / list mode only) ─────────────────────────
        # Write rows for GeoJSON files that exist on disk but are absent from
        # downloads.csv — e.g. after a previous interrupted run.
        if args.resume and locations is not None:
            geojson_on_disk = scan_geojson_prefixes(graphs_output_dir)
            for i, (lat, lon) in enumerate(locations):
                if args.multiresolution:
                    # Recover predefined levels using their known bbox sizes.
                    for lvl_idx, lvl_size in enumerate(multiresolution_levels):
                        prefix_name = f"osm_{i}_L{lvl_idx}"
                        if prefix_name in geojson_on_disk and prefix_name not in existing_geojson_prefixes:
                            db.insert_download(lat=lat, lon=lon, bbox_size=lvl_size,
                                               geojson_prefix=prefix_name, level=lvl_idx)
                            existing_geojson_prefixes.add(prefix_name)
                            recovered_csv_count += 1
                            logger.info(
                                "Recovered missing downloads DB row for existing GeoJSON: %s",
                                prefix_name,
                            )
                    # Also recover any adaptive extra levels (index >= len(multiresolution_levels)).
                    # Their bbox_size is unknown from the filename alone, so we use 0.0 as a
                    # placeholder — create_graphs.py uses bbox_size only for the index.csv record.
                    for disk_prefix in geojson_on_disk:
                        import re as _re
                        _m = _re.match(rf'^osm_{i}_L(\d+)$', disk_prefix)
                        if _m and int(_m.group(1)) >= len(multiresolution_levels):
                            if disk_prefix not in existing_geojson_prefixes:
                                adaptive_lvl_idx = int(_m.group(1))
                                db.insert_download(lat=lat, lon=lon, bbox_size=0.0,
                                                   geojson_prefix=disk_prefix, level=adaptive_lvl_idx)
                                existing_geojson_prefixes.add(disk_prefix)
                                recovered_csv_count += 1
                                logger.info(
                                    "Recovered missing downloads DB row for adaptive GeoJSON: %s",
                                    disk_prefix,
                                )
                else:
                    prefix_name = f"osm_{i}"
                    if prefix_name in geojson_on_disk and prefix_name not in existing_geojson_prefixes:
                        db.insert_download(lat=lat, lon=lon, bbox_size=args.bbox_size,
                                           geojson_prefix=prefix_name)
                        existing_geojson_prefixes.add(prefix_name)
                        recovered_csv_count += 1
                        logger.info(
                            "Recovered missing downloads DB row for existing GeoJSON: %s",
                            prefix_name,
                        )

        # ── Parallel processing ───────────────────────────────────────────────
        worker_kwargs = dict(
            graphs_output_dir=graphs_output_dir,
            args=args,
            tag_frequencies=tag_frequencies,
            multiresolution_levels=multiresolution_levels,
            adaptive_max_factor=_adaptive_max_factor,
            downloader_config=downloader_config,
        )

        def _run_pending(pending_list: List[Tuple[int, float, float]]) -> None:
            """Run pending locations in a process pool, restarting after worker crashes."""
            nonlocal processed_count, skipped_count
            remaining = list(pending_list)
            crash_counts: Dict[int, int] = {}
            MAX_CRASHES = 5

            def _handle_future_result(future, i, completed_ids, crashed_ids):
                """Process one completed future; update completed_ids / crashed_ids in place.

                Returns (delta_processed, delta_skipped) so the caller can update counters.
                """
                try:
                    outcome, rows = future.result()
                except BrokenProcessPool:
                    crashed_ids.add(i)
                    return 0, 0
                except KeyboardInterrupt:
                    # Worker was interrupted before SIG_IGN took effect (e.g. very
                    # early in startup).  Treat as a shutdown signal and stop cleanly.
                    _shutdown_event.set()
                    crashed_ids.add(i)
                    return 0, 0
                except Exception as exc:
                    logger.error("Location %d: unexpected error: %s", i, exc, exc_info=True)
                    completed_ids.add(i)
                    return 0, 1
                completed_ids.add(i)
                if rows:
                    for row in rows:
                        db.insert_download(
                            lat=row[0], lon=row[1], bbox_size=row[3],
                            geojson_prefix=row[4],
                            level=row[5] if len(row) > 5 else None,
                        )
                if outcome == "skipped_empty":
                    return 0, 1
                return 1, 0

            while remaining and not _shutdown_event.is_set():
                location_map = {i: (lat, lon) for i, lat, lon in remaining}
                completed_ids: Set[int] = set()
                crashed_ids: Set[int] = set()

                # If any location has previously crashed, run the batch one at a time so
                # that a segfault can only poison its own future (not the whole pool).
                is_retry_round = any(crash_counts.get(t[0], 0) > 0 for t in remaining)

                if is_retry_round:
                    logger.warning(
                        "Retry round: processing %d location(s) serially to isolate crashers.",
                        len(remaining),
                    )
                    for i, lat, lon in remaining:
                        if _shutdown_event.is_set():
                            break
                        with ProcessPoolExecutor(
                            max_workers=1, initializer=_worker_init,
                            initargs=(logging.getLogger().level,),
                            mp_context=multiprocessing.get_context("spawn"),
                        ) as executor:
                            future = executor.submit(_process_location, i, lat, lon, **worker_kwargs)
                            dp, ds = _handle_future_result(future, i, completed_ids, crashed_ids)
                            processed_count += dp
                            skipped_count += ds
                else:
                    executor = ProcessPoolExecutor(
                        max_workers=args.workers, initializer=_worker_init,
                        initargs=(logging.getLogger().level,),
                        mp_context=multiprocessing.get_context("spawn"),
                    )
                    try:
                        # Submit lazily: keep at most max_workers futures in flight
                        # so that cancellation on shutdown is immediate — no queued
                        # futures for idle workers to race and pick up.
                        pending_locs = list(remaining)
                        submit_idx = 0
                        in_flight: dict = {}

                        def _submit_next():
                            nonlocal submit_idx
                            if submit_idx < len(pending_locs):
                                _i, _lat, _lon = pending_locs[submit_idx]
                                _f = executor.submit(
                                    _process_location, _i, _lat, _lon, **worker_kwargs
                                )
                                in_flight[_f] = _i
                                submit_idx += 1

                        for _ in range(min(args.workers, len(pending_locs))):
                            _submit_next()

                        try:
                            while in_flight:
                                done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                                for future in done:
                                    i = in_flight.pop(future)
                                    dp, ds = _handle_future_result(
                                        future, i, completed_ids, crashed_ids
                                    )
                                    processed_count += dp
                                    skipped_count += ds
                                    if _shutdown_event.is_set():
                                        logger.warning(
                                            "Cancelling %d pending download(s).",
                                            len(in_flight),
                                        )
                                        for f in in_flight:
                                            f.cancel()
                                        in_flight.clear()
                                        break
                                    _submit_next()
                                if _shutdown_event.is_set():
                                    break
                        except BrokenProcessPool:
                            for i in in_flight.values():
                                if i not in completed_ids:
                                    crashed_ids.add(i)
                    finally:
                        # On graceful shutdown do not block waiting for in-flight
                        # worker processes — they will finish on their own.
                        executor.shutdown(
                            wait=not _shutdown_event.is_set(),
                            cancel_futures=_shutdown_event.is_set(),
                        )

                # Increment crash counts; give up on locations that crash too often.
                for i in list(crashed_ids):
                    crash_counts[i] = crash_counts.get(i, 0) + 1
                    lat, lon = location_map[i]
                    if crash_counts[i] >= MAX_CRASHES:
                        logger.error(
                            "Location %d (%.5f, %.5f): skipped after %d consecutive worker "
                            "crashes (possible pyogrio/GDAL segfault on a corrupt GeoJSON).",
                            i, lat, lon, crash_counts[i],
                        )
                        write_nodata_sentinel(
                            graphs_output_dir, i,
                            f"worker crashed {crash_counts[i]} times (likely pyogrio segfault)",
                        )
                        skipped_count += 1
                        completed_ids.add(i)

                retrying = crashed_ids - completed_ids
                if retrying:
                    logger.warning(
                        "Worker process crashed; %d location(s) will be retried (ids: %s%s).",
                        len(retrying),
                        ", ".join(str(x) for x in sorted(retrying)[:10]),
                        ", ..." if len(retrying) > 10 else "",
                    )
                remaining = [t for t in remaining if t[0] not in completed_ids]

        if locations is not None:
            # ── List-based (city grid) ────────────────────────────────────────
            pending: List[Tuple[int, float, float]] = []
            for i, (lat, lon) in enumerate(locations):
                if args.resume:
                    _nodata_path = nodata_sentinel_path(graphs_output_dir, i)
                    if os.path.exists(_nodata_path):
                        if (
                            args.nodata_retry_after is not None
                            and nodata_is_expired(_nodata_path, args.nodata_retry_after)
                        ):
                            logger.info(
                                "Location %d (%s, %s): .nodata expired (TTL=%.1fh), will retry.",
                                i, lat, lon, args.nodata_retry_after,
                            )
                            os.remove(_nodata_path)
                        else:
                            logger.info(
                                "Skipping location %d (%s, %s): nodata sentinel exists "
                                "(delete osm_%d.nodata to retry, or use --nodata-retry-after)",
                                i, lat, lon, i,
                            )
                            skipped_existing_count += 1
                            continue
                    if args.multiresolution:
                        already_done = (
                            i in existing_multiresolution_indices
                            and not (args.best_level_only and _combined_mr_counts.get(i, 0) > 1)
                        )
                    else:
                        already_done = f"osm_{i}" in existing_geojson_prefixes
                    if already_done:
                        logger.info("Skipping location %d (%s, %s): already downloaded", i, lat, lon)
                        skipped_existing_count += 1
                        continue
                pending.append((i, lat, lon))

            logger.info(
                "Downloading %d pending locations with %d worker(s) "
                "(%d already done, %d recovered).",
                len(pending), args.workers, skipped_existing_count, recovered_csv_count,
            )
            _run_pending(pending)

        else:
            # ── Streaming (CSV locations file) ────────────────────────────────
            nodata_set: Set[int] = set()
            _nodata_re = re.compile(r'^osm_(\d+)\.nodata$')
            for fn in os.listdir(graphs_output_dir):
                m = _nodata_re.match(fn)
                if m:
                    idx = int(m.group(1))
                    _nodata_path = os.path.join(graphs_output_dir, fn)
                    if (
                        args.nodata_retry_after is not None
                        and nodata_is_expired(_nodata_path, args.nodata_retry_after)
                    ):
                        logger.info(
                            "Location %d: .nodata expired (TTL=%.1fh), will retry.",
                            idx, args.nodata_retry_after,
                        )
                        os.remove(_nodata_path)
                    else:
                        nodata_set.add(idx)

            pending = []
            for i, lat, lon in stream_locations_from_csv(args.locations_file):
                if args.resume:
                    if i in nodata_set:
                        skipped_existing_count += 1
                        continue
                    if args.multiresolution:
                        already_done = (
                            i in existing_multiresolution_indices
                            and not (args.best_level_only and _combined_mr_counts.get(i, 0) > 1)
                        )
                    else:
                        already_done = f"osm_{i}" in existing_geojson_prefixes
                    if already_done:
                        skipped_existing_count += 1
                        continue
                pending.append((i, lat, lon))

            logger.info(
                "Downloading %d pending locations with %d worker(s) "
                "(%d already done, %d nodata sentinel(s) on disk).",
                len(pending), args.workers, skipped_existing_count, len(nodata_set),
            )
            _run_pending(pending)

    _total = f" (total: {len(locations)})" if locations is not None else ""
    if _shutdown_event.is_set():
        logger.warning(
            "⚠️  Interrupted in %s: %d new, %d already done, "
            "%d empty/skipped, %d csv recovered%s",
            args.output_dir, processed_count, skipped_existing_count,
            skipped_count, recovered_csv_count, _total,
        )
    else:
        logger.info(
            "✅ Downloads complete in %s: %d new, %d already done, "
            "%d empty/skipped, %d csv recovered%s",
            args.output_dir, processed_count, skipped_existing_count,
            skipped_count, recovered_csv_count, _total,
        )


if __name__ == "__main__":
    main()

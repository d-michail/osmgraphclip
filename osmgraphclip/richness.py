"""Semantic richness assessment for OSM GeoDataFrames.

Metrics
-------
total_nodes          : total feature count (points + lines + polygons)
n_points / n_lines / n_polygons : per-type counts
unique_labels        : distinct non-"unknown:unknown" label strings (legacy)
unique_keys          : distinct OSM tag keys present
unique_kv_pairs      : distinct (key, value) pairs
tag_entropy          : Shannon entropy over (key, value) pair frequencies
idf_score            : mean IDF weight of present keys (requires tag_frequencies)
categories_present   : count of semantic category groups detected
category_coverage    : fraction of semantic categories present [0, 1]
geometry_entropy     : Shannon entropy over point/line/polygon counts
spatial_coverage     : fraction of spatial grid cells containing a feature [0, 1]
feature_density      : features per km² (requires bbox_size_m)
richness_score       : composite weighted score in [0, 1]
"""

import json
import math
from collections import Counter
from typing import Dict, Optional, Tuple

import numpy as np
import geopandas as gpd

from .osm_tags import _ensure_label_column, OSM_TAG_COLUMNS, _NON_TAG_COLS


# ---------------------------------------------------------------------------
# Semantic category definitions
# Each entry maps a category name to a list of (key, allowed_values | None)
# conditions.  A category is *present* if ANY condition is satisfied for at
# least one feature.  allowed_values=None means "any non-null value".
# ---------------------------------------------------------------------------
SEMANTIC_CATEGORIES: Dict[str, list] = {
    "food_drink": [
        ("amenity", {"restaurant", "cafe", "bar", "fast_food", "pub",
                     "food_court", "ice_cream", "biergarten", "bbq"}),
    ],
    "retail": [
        ("shop", None),
    ],
    "transport": [
        ("highway", None),
        ("railway", None),
        ("public_transport", None),
    ],
    "education": [
        ("amenity", {"school", "university", "college", "kindergarten",
                     "library", "research_institute"}),
    ],
    "nature_green": [
        ("natural", None),
        ("landuse", {"forest", "meadow", "farmland", "grass", "heath",
                     "orchard", "vineyard", "scrub"}),
    ],
    "leisure": [
        ("leisure", None),
    ],
    "water": [
        ("waterway", None),
        ("water", None),
    ],
    "tourism": [
        ("tourism", None),
    ],
    "built_environment": [
        ("building", None),
    ],
    "health": [
        ("amenity", {"hospital", "clinic", "doctors", "pharmacy",
                     "dentist", "nursing_home", "veterinary"}),
    ],
}

_N_CATEGORIES = len(SEMANTIC_CATEGORIES)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    s = str(value).strip()
    return s == "" or s.lower() == "nan"


def _iter_kv_pairs(gdfs):
    """Yield (key, value_str) for every non-missing OSM tag across all GDFs."""
    for gdf in gdfs:
        if gdf is None or len(gdf) == 0:
            continue
        for col in gdf.columns:
            if col in _NON_TAG_COLS:
                continue
            for raw in gdf[col]:
                if _is_missing(raw):
                    continue
                val = str(raw).strip()
                if val.lower() in ("true", "yes", "1"):
                    yield col, "yes"
                else:
                    yield col, val


def _build_kv_data(gdfs):
    """
    Build two data structures from all GDFs in one pass:
      kv_counter  : Counter of {(key, value): occurrence_count}
      key_values  : dict of {key: set_of_values_present}
    """
    kv_counter: Counter = Counter()
    key_values: Dict[str, set] = {}
    for k, v in _iter_kv_pairs(gdfs):
        kv_counter[(k, v)] += 1
        key_values.setdefault(k, set()).add(v)
    return kv_counter, key_values


# ---------------------------------------------------------------------------
# Individual metric computations
# ---------------------------------------------------------------------------

def _tag_metrics(kv_counter: Counter, tag_frequencies: Optional[dict]) -> dict:
    """Tag diversity, entropy, and IDF-weighted score."""
    n_kv = len(kv_counter)
    key_set = {k for k, _ in kv_counter}
    n_keys = len(key_set)

    # Shannon entropy over (key, value) pair counts
    if n_kv > 1:
        total = sum(kv_counter.values())
        probs = np.array(list(kv_counter.values()), dtype=float) / total
        tag_entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
    else:
        tag_entropy = 0.0

    # IDF-weighted score: mean of log(max_freq / key_freq) over present keys.
    # Tags that are rarer in the global dataset contribute a higher weight.
    idf_score = 0.0
    if tag_frequencies and key_set:
        max_freq = max(tag_frequencies.values())
        total_idf = 0.0
        for key in key_set:
            freq = tag_frequencies.get(key)
            if freq is None:
                # fall back to the base word for compound keys like "building use"
                base = key.split()[0]
                freq = tag_frequencies.get(base)
            if freq and freq > 0:
                total_idf += math.log(max_freq / freq + 1)
        idf_score = total_idf / max(n_keys, 1)

    return {
        "unique_keys": n_keys,
        "unique_kv_pairs": n_kv,
        "total_kv_instances": sum(kv_counter.values()),
        "tag_entropy": tag_entropy,
        "idf_score": idf_score,
    }


def _category_coverage(key_values: Dict[str, set]) -> dict:
    """Fraction of semantic categories present."""
    present = 0
    for conditions in SEMANTIC_CATEGORIES.values():
        for key, allowed in conditions:
            if key not in key_values:
                continue
            if allowed is None or (key_values[key] & allowed):
                present += 1
                break  # this category is satisfied; move to the next one
    return {
        "categories_present": present,
        "category_coverage": present / _N_CATEGORIES if _N_CATEGORIES > 0 else 0.0,
    }


def _geometry_entropy(n_points: int, n_lines: int, n_polygons: int) -> float:
    """Shannon entropy over the three geometry type counts."""
    counts = np.array([n_points, n_lines, n_polygons], dtype=float)
    total = counts.sum()
    if total == 0:
        return 0.0
    present = counts[counts > 0]
    if len(present) == 1:
        return 0.0
    probs = present / total
    return float(-np.sum(probs * np.log(probs)))


def _spatial_coverage(gdfs, grid_n: int = 4) -> float:
    """
    Fraction of a grid_n × grid_n grid that contains at least one feature.

    The reference bounding box is the union of all feature extents.  This
    measures how spatially spread-out the features are relative to each other.
    Returns 0.0 when fewer than two distinct locations are present.
    """
    xs, ys = [], []
    for gdf in gdfs:
        if gdf is None or len(gdf) == 0:
            continue
        geoms = gdf.geometry.dropna()
        if len(geoms) == 0:
            continue
        bounds = geoms.total_bounds  # (minx, miny, maxx, maxy)
        xs.extend([bounds[0], bounds[2]])
        ys.extend([bounds[1], bounds[3]])

    if len(xs) < 2:
        return 0.0
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    if (maxx - minx) < 1e-9 or (maxy - miny) < 1e-9:
        return 0.0

    x_edges = np.linspace(minx, maxx, grid_n + 1)
    y_edges = np.linspace(miny, maxy, grid_n + 1)

    occupied: set = set()
    for gdf in gdfs:
        if gdf is None or len(gdf) == 0:
            continue
        for geom in gdf.geometry.dropna():
            cx = geom.centroid.x
            cy = geom.centroid.y
            xi = int(np.clip(np.searchsorted(x_edges[1:], cx), 0, grid_n - 1))
            yi = int(np.clip(np.searchsorted(y_edges[1:], cy), 0, grid_n - 1))
            occupied.add((xi, yi))

    return len(occupied) / (grid_n * grid_n)


def _feature_density(total_nodes: int, bbox_size_m: Optional[float]) -> Optional[float]:
    """Features per km², based on a square bbox of side 2 × bbox_size_m."""
    if bbox_size_m is None or bbox_size_m <= 0:
        return None
    area_km2 = (2 * bbox_size_m / 1000.0) ** 2
    return total_nodes / area_km2


def _composite_score(metrics: dict) -> float:
    """
    Weighted composite richness score in [0, 1].

    Component weights:
      tag_entropy        0.30  – semantic variety across tag values
      category_coverage  0.25  – breadth across semantic groups
      unique_kv_pairs    0.15  – total tag vocabulary size
      geometry_entropy   0.10  – balance across geometry types
      spatial_coverage   0.10  – spatial spread of features
      semantic_depth     0.05  – average non-null tags per feature (POI richness)
      idf_score          0.05  – rare / informative tag bonus
    """
    # Normalize each component to [0, 1] with a soft cap
    # tag_entropy max ≈ log(50) ≈ 3.91 for 50 balanced kv pairs
    tag_entropy_norm = min(metrics["tag_entropy"] / math.log(50), 1.0)

    category_cov = metrics["category_coverage"]  # already [0, 1]

    # kv diversity: 30 distinct pairs → full score
    kv_norm = min(metrics["unique_kv_pairs"] / 30.0, 1.0)

    # geometry entropy max = log(3) ≈ 1.099 (all three types equally present)
    geom_entropy_norm = min(
        metrics["geometry_entropy"] / math.log(3), 1.0
    ) if metrics["geometry_entropy"] > 0 else 0.0

    spatial_cov = metrics["spatial_coverage"]  # already [0, 1]

    # semantic_depth: avg non-null tags per feature; cap at 8 (richly-tagged named POI)
    depth_norm = min(metrics.get("semantic_depth", 0.0) / 8.0, 1.0)

    # idf_score: mean IDF, typical range ≈ [0, 5].  Cap at 4.0 for normalisation.
    idf_norm = min(metrics.get("idf_score", 0.0) / 4.0, 1.0)

    score = (
        0.30 * tag_entropy_norm
        + 0.25 * category_cov
        + 0.15 * kv_norm
        + 0.10 * geom_entropy_norm
        + 0.10 * spatial_cov
        + 0.05 * depth_norm
        + 0.05 * idf_norm
    )
    return float(np.clip(score, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_tag_frequencies(path: str) -> dict:
    """Load the tag frequency JSON file (e.g. data/all_tags30_frequency1.json)."""
    with open(path, "r") as f:
        return json.load(f)


def compute_richness_metrics(
    polygon_gdf: Optional[gpd.GeoDataFrame],
    line_gdf: Optional[gpd.GeoDataFrame],
    point_gdf: Optional[gpd.GeoDataFrame],
    tag_frequencies: Optional[dict] = None,
    bbox_size_m: Optional[float] = None,
    spatial_grid_n: int = 4,
) -> dict:
    """
    Compute semantic richness metrics for a set of OSM GeoDataFrames.

    Args:
        polygon_gdf:      GeoDataFrame with polygon features (or None).
        line_gdf:         GeoDataFrame with line features (or None).
        point_gdf:        GeoDataFrame with point features (or None).
        tag_frequencies:  Optional {tag_key: global_count} dict loaded from
                          all_tags30_frequency1.json.  Enables IDF scoring.
        bbox_size_m:      Optional half-width of the download bbox in metres.
                          Enables feature_density computation.
        spatial_grid_n:   Grid side-length for spatial coverage (default 4).

    Returns:
        dict — see module docstring for the full list of keys.
        All keys are always present; values that require optional inputs are
        None when those inputs are absent (feature_density only).
    """
    n_polygons = len(polygon_gdf) if polygon_gdf is not None and len(polygon_gdf) > 0 else 0
    n_lines = len(line_gdf) if line_gdf is not None and len(line_gdf) > 0 else 0
    n_points = len(point_gdf) if point_gdf is not None and len(point_gdf) > 0 else 0
    total_nodes = n_polygons + n_lines + n_points

    # Unique label strings — legacy metric kept for backward compatibility
    unique_labels: set = set()
    for gdf in (polygon_gdf, line_gdf, point_gdf):
        if gdf is None or len(gdf) == 0:
            continue
        labeled = _ensure_label_column(gdf)
        unique_labels.update(
            lbl for lbl in labeled["label"].unique() if lbl != "unknown:unknown"
        )

    gdfs = (polygon_gdf, line_gdf, point_gdf)
    kv_counter, key_values = _build_kv_data(gdfs)

    metrics: dict = {
        "total_nodes": total_nodes,
        "n_points": n_points,
        "n_lines": n_lines,
        "n_polygons": n_polygons,
        "unique_labels": len(unique_labels),
        **_tag_metrics(kv_counter, tag_frequencies),
        **_category_coverage(key_values),
        "geometry_entropy": _geometry_entropy(n_points, n_lines, n_polygons),
        "spatial_coverage": _spatial_coverage(gdfs, grid_n=spatial_grid_n),
        "feature_density": _feature_density(total_nodes, bbox_size_m),
    }
    metrics["semantic_depth"] = (
        metrics["total_kv_instances"] / total_nodes if total_nodes > 0 else 0.0
    )
    metrics["richness_score"] = _composite_score(metrics)
    return metrics


def is_rich_enough(
    metrics: dict,
    min_total_nodes: int = 5,
    min_unique_labels: int = 3,
    min_richness_score: Optional[float] = None,
) -> bool:
    """
    Return True if the area meets the minimum richness thresholds.

    Args:
        metrics:            Output of compute_richness_metrics().
        min_total_nodes:    Minimum total OSM feature count.
        min_unique_labels:  Minimum distinct label strings.
        min_richness_score: Optional minimum composite richness_score [0, 1].
                            Evaluated in addition to the node/label thresholds.
    """
    if metrics["total_nodes"] < min_total_nodes:
        return False
    if metrics["unique_labels"] < min_unique_labels:
        return False
    if min_richness_score is not None:
        return metrics.get("richness_score", 0.0) >= min_richness_score
    return True

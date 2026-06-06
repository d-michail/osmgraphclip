"""Global geospatial sampling pipeline.

Three phases:
  1. H3 candidate generation  — globally-distributed (lat, lon) points via H3 hexagonal grid
  2. OSM-aware scoring         — adaptive bbox probe to measure OSM richness per point
  3. Diversity-aware selection — stratified sampling across density / spatial / scale axes
"""

from __future__ import annotations

import dataclasses
import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

import h3
import numpy as np
from shapely.geometry import Point, Polygon

from osmgraphclip.downloader_config import DownloaderConfig, create_downloader
from osmgraphclip.richness import compute_richness_metrics, is_rich_enough
from osmgraphclip.dataset_pipeline import load_gdfs_for_prefix

logger = logging.getLogger(__name__)

# GeoJSON suffixes written by every downloader backend
_GEOJSON_SUFFIXES = [
    "_polygon.geojson.gz",        "_multipolygon.geojson.gz",
    "_linestring.geojson.gz",     "_multilinestring.geojson.gz",
    "_point.geojson.gz",          "_multipoint.geojson.gz",
    "_polygon.geojson",           "_multipolygon.geojson",
    "_linestring.geojson",        "_multilinestring.geojson",
    "_point.geojson",             "_multipoint.geojson",
]

# Ordered density bucket definitions (lower bound inclusive, upper exclusive).
# very_low covers ocean/boundary-only locations (timezone, maritime, admin polygons)
# whose richness score is typically ~0.15 even though they contain no useful OSM features.
DENSITY_BUCKETS: Dict[str, Tuple[float, float]] = {
    "very_low": (0.0,   0.20),
    "low":      (0.20,  0.40),
    "medium":   (0.40,  0.65),
    "high":     (0.65,  1.01),  # upper bound > 1.0 to include richness == 1.0
}

DEFAULT_DENSITY_FRACTIONS: Dict[str, float] = {
    "very_low": 0.02,
    "low":      0.12,
    "medium":   0.77,
    "high":     0.09,
}


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ProbeConfig:
    """Picklable probe parameters, passed through to spawned worker processes."""

    initial_bbox_size: float = 250.0    # initial bbox half-width in metres
    max_bbox_size: float = 5000.0       # maximum bbox half-width in metres
    expansion_strategy: str = "multiplicative"  # "multiplicative" | "linear"
    expansion_factor: float = 1.5       # growth factor per retry (multiplicative)
    expansion_step: float = 100.0       # step size in metres per retry (linear)
    min_total_nodes: int = 5            # minimum OSM feature count to be "rich enough"
    min_unique_labels: int = 3          # minimum distinct OSM labels
    min_richness_score: float = 0.1     # minimum composite richness score


@dataclass
class ScoredLocation:
    """A candidate location that has been probed and classified."""

    lat: float
    lon: float
    richness_score: float
    bbox_size: float        # final bbox half-width used (metres)
    total_nodes: int        # OSM feature count at the accepted bbox
    h3_res3: str            # H3 cell index at resolution 3
    h3_res5: str            # H3 cell index at resolution 5
    h3_res7: str            # H3 cell index at resolution 7
    density_bucket: str     # "very_low" | "low" | "medium" | "high"


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — H3-based candidate generation
# ─────────────────────────────────────────────────────────────────────────────

def _h3_cell_to_polygon(cell: str) -> Polygon:
    """Convert an H3 cell to a Shapely polygon using its boundary vertices.

    h3.cell_to_boundary returns (lat, lng) pairs; Shapely expects (lng, lat).
    """
    boundary = h3.cell_to_boundary(cell)
    return Polygon([(lng, lat) for lat, lng in boundary])


def _random_points_in_h3_cell(
    cell: str,
    k: int,
    rng: np.random.Generator,
) -> List[Tuple[float, float]]:
    """Draw k uniform random (lat, lon) points inside an H3 hexagon.

    Uses rejection sampling inside the cell's bounding box.  Regular hexagons
    have an acceptance rate of ~83 % relative to their bbox, so the overhead
    is minimal and no infinite-loop risk exists in practice.
    """
    polygon = _h3_cell_to_polygon(cell)
    min_lng, min_lat, max_lng, max_lat = polygon.bounds

    points: List[Tuple[float, float]] = []
    batch = max(k * 2, 16)  # draw extra candidates to amortise the contains() check
    while len(points) < k:
        lngs = rng.uniform(min_lng, max_lng, batch)
        lats = rng.uniform(min_lat, max_lat, batch)
        for lng, lat in zip(lngs, lats):
            if polygon.contains(Point(lng, lat)):
                points.append((lat, lng))
                if len(points) == k:
                    break
    return points


def generate_h3_candidates(
    resolutions: List[int],
    cell_sample_fraction: float = 0.05,
    points_per_cell: int = 3,
    rng_seed: int = 42,
) -> Iterator[Tuple[float, float]]:
    """Yield globally-distributed (lat, lon) candidate points via H3.

    Strategy
    --------
    1. Start from h3.get_res0_cells() — 122 base cells covering the globe.
    2. For each resolution in *resolutions* (sorted ascending), expand the
       previous level's selected cells to children at the target resolution.
       At resolution < 5, all cells are kept (global coverage).
       At resolution >= 5, subsample *cell_sample_fraction* of the expanded
       cells to keep the total candidate count practical (resolution 7 has
       ~98 M cells globally; 5 % × 3 pts ≈ several million candidates).
    3. The selected cells at each resolution become the seed for the next
       finer level, so fine-grained candidates are concentrated within the
       globally-distributed coarser regions.
    4. For each selected cell, yield *points_per_cell* uniform random
       (lat, lon) points drawn inside the hexagon boundary polygon.

    Parameters
    ----------
    resolutions:
        H3 resolutions to generate candidates at, e.g. [3, 5, 7].  Values
        must be ≥ 0 and are sorted internally.
    cell_sample_fraction:
        Fraction of cells to retain at resolutions >= 5  (0 < f <= 1.0).
        Set to 1.0 to keep all cells (may yield tens of millions of candidates).
    points_per_cell:
        Number of random points to draw inside each selected cell.
    rng_seed:
        Seed for the random number generator.  The same seed always produces
        the same candidate set, which is required for correct resume behaviour.
    """
    rng = np.random.default_rng(rng_seed)
    resolutions = sorted(resolutions)

    # Start from the 122 resolution-0 cells that tile the globe
    seed_cells: List[str] = list(h3.get_res0_cells())

    for res in resolutions:
        # Expand each seed cell to all its descendants at target resolution.
        # h3.cell_to_children(cell, res) handles multi-level jumps directly.
        expanded: List[str] = []
        for cell in seed_cells:
            cell_res = h3.get_resolution(cell)
            if cell_res < res:
                expanded.extend(h3.cell_to_children(cell, res))
            else:
                expanded.append(cell)  # cell_res == res (no expansion needed)

        # Subsample at fine resolutions to keep the pipeline tractable
        if res >= 5 and 0.0 < cell_sample_fraction < 1.0:
            n_keep = max(1, int(len(expanded) * cell_sample_fraction))
            chosen = rng.choice(len(expanded), size=n_keep, replace=False)
            selected = [expanded[int(i)] for i in sorted(chosen)]
        else:
            selected = expanded

        logger.info(
            "H3 resolution %d: %d cells selected (%.1f %% of %d expanded)",
            res,
            len(selected),
            100.0 * len(selected) / max(len(expanded), 1),
            len(expanded),
        )

        for cell in selected:
            yield from _random_points_in_h3_cell(cell, points_per_cell, rng)

        # Selected cells seed the next (finer) resolution so fine-grained
        # candidates are globally distributed alongside the coarser ones.
        seed_cells = selected


def generate_hierarchical_h3_candidates(
    fine_resolutions: List[int],
    coarse_cell_richness: Dict[str, float],
    richness_threshold: float = 0.2,
    sparse_keep_fraction: float = 0.01,
    cell_sample_fraction: float = 0.05,
    points_per_cell: int = 3,
    rng_seed: int = 42,
) -> Iterator[Tuple[float, float]]:
    """Yield fine-resolution candidates concentrated in OSM-rich coarse cells.

    Strategy
    --------
    1. Classify each coarse H3 cell as *rich* (richness ≥ threshold) or *sparse*.
    2. Keep all rich cells; randomly retain *sparse_keep_fraction* of sparse cells
       so that oceans / deserts still contribute a small uniform baseline.
    3. Expand retained cells to each resolution in *fine_resolutions* (ascending),
       optionally sub-sampling at fine levels (same logic as generate_h3_candidates).
    4. Yield *points_per_cell* random points inside each selected fine cell.

    Parameters
    ----------
    fine_resolutions:
        Target H3 resolutions (must be finer than the coarse keys in
        *coarse_cell_richness*).  Sorted internally.
    coarse_cell_richness:
        Mapping from H3 cell index (any resolution) → richness_score [0, 1].
        Typically built from Phase-2a scoring of coarse candidates.
    richness_threshold:
        Minimum richness for a coarse cell to be expanded to fine resolutions.
    sparse_keep_fraction:
        Fraction of below-threshold coarse cells to keep for diversity.
        Set to 0 to include *only* rich cells.
    cell_sample_fraction:
        Sub-sampling fraction for fine resolutions ≥ 5.
    points_per_cell:
        Random points drawn inside each selected fine cell.
    rng_seed:
        RNG seed — must be deterministic for correct resume behaviour.
    """
    if not coarse_cell_richness:
        return

    rng = np.random.default_rng(rng_seed)
    fine_resolutions = sorted(fine_resolutions)

    # Infer coarse resolution from cell keys
    sample_cell = next(iter(coarse_cell_richness))
    coarse_res = h3.get_resolution(sample_cell)

    # Partition into rich and sparse cells
    all_coarse = list(coarse_cell_richness.keys())
    rich_cells = [c for c in all_coarse if coarse_cell_richness[c] >= richness_threshold]
    sparse_cells = [c for c in all_coarse if coarse_cell_richness[c] < richness_threshold]

    # Keep a small random fraction of sparse cells for geographic diversity
    n_sparse = max(1, round(len(sparse_cells) * sparse_keep_fraction)) if sparse_cells else 0
    if n_sparse and n_sparse < len(sparse_cells):
        sparse_idx = rng.choice(len(sparse_cells), size=n_sparse, replace=False)
        sparse_kept = [sparse_cells[int(i)] for i in sparse_idx]
    else:
        sparse_kept = sparse_cells

    seed_cells = rich_cells + sparse_kept

    logger.info(
        "Hierarchical: %d rich + %d sparse (of %d) coarse cells selected "
        "(threshold=%.2f, sparse_keep=%.3f)",
        len(rich_cells), len(sparse_kept), len(all_coarse),
        richness_threshold, sparse_keep_fraction,
    )

    for res in fine_resolutions:
        if res <= coarse_res:
            # At or coarser than the probe resolution — emit directly
            for cell in seed_cells:
                yield from _random_points_in_h3_cell(cell, points_per_cell, rng)
            continue

        # Expand each seed cell to children at the target resolution
        expanded: List[str] = []
        for cell in seed_cells:
            cell_res = h3.get_resolution(cell)
            if cell_res < res:
                expanded.extend(h3.cell_to_children(cell, res))
            else:
                expanded.append(cell)

        if res >= 5 and 0.0 < cell_sample_fraction < 1.0:
            n_keep = max(1, int(len(expanded) * cell_sample_fraction))
            chosen = rng.choice(len(expanded), size=n_keep, replace=False)
            selected = [expanded[int(i)] for i in sorted(chosen)]
        else:
            selected = expanded

        logger.info(
            "H3 resolution %d: %d cells selected (%.1f%% of %d, within rich coarse cells)",
            res, len(selected), 100.0 * len(selected) / max(len(expanded), 1), len(expanded),
        )

        for cell in selected:
            yield from _random_points_in_h3_cell(cell, points_per_cell, rng)

        seed_cells = selected


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — OSM-aware scoring
# ─────────────────────────────────────────────────────────────────────────────

def _cleanup_geojson(output_file_prefix: str) -> None:
    """Delete all GeoJSON files associated with the given file prefix."""
    for suffix in _GEOJSON_SUFFIXES:
        path = output_file_prefix + suffix
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            pass


def adaptive_probe(
    lat: float,
    lon: float,
    output_file_prefix: str,
    probe_config: ProbeConfig,
    downloader_config: DownloaderConfig,
    tag_frequencies: Optional[dict] = None,
) -> Tuple[bool, float, float]:
    """Download OSM data and expand the bbox until richness thresholds are met.

    Modelled after _adaptive_download() in create_dataset.py — same expansion
    loop, simplified for scoring only (no graph building, no profile escalation).
    Temp GeoJSON files are always cleaned up before returning.

    Parameters
    ----------
    lat, lon:
        Location to probe.
    output_file_prefix:
        Path prefix for temp GeoJSON files (the downloader appends geometry-type
        suffixes, e.g. _polygon.geojson.gz).  Should be unique per worker.
    probe_config:
        Expansion parameters and richness thresholds.
    downloader_config:
        Backend configuration (PostGIS / Overpass / auto).
    tag_frequencies:
        Pre-loaded tag frequency dict for IDF-weighted richness scoring.
        Pass None to skip IDF (slightly less accurate richness score).

    Returns
    -------
    (success, richness_score, actual_bbox_size, total_nodes)
    On total failure returns (False, 0.0, probe_config.initial_bbox_size, 0).

    Notes
    -----
    Even when the thresholds are never met (max bbox reached), the function
    returns (True, best_richness, best_bbox_size, best_total_nodes) as long as
    *any* OSM data was downloaded.  This preserves "very_low" richness samples
    (oceans, deserts) that are valuable for the diversity-aware selection stage.
    """
    cfg = probe_config
    bbox_size = cfg.initial_bbox_size
    best_richness = 0.0
    best_bbox_size = bbox_size
    best_total_nodes = 0
    any_data = False
    iteration = 0

    try:
        while True:
            # Use a unique sub-prefix per iteration.  GDAL's /vsigzip/ reader
            # keeps internal state tied to the file path; overwriting the same
            # .geojson.gz in subsequent bbox expansions causes CRC mismatches
            # on the second read.  Fresh paths avoid the stale-cache problem.
            iter_prefix = f"{output_file_prefix}_{iteration}"

            try:
                downloader = create_downloader(
                    downloader_config,
                    lat=lat, lon=lon, dist=bbox_size,
                    output_file=iter_prefix,
                )
                downloaded = downloader()

                if downloaded:
                    polygon_gdf, line_gdf, point_gdf = load_gdfs_for_prefix(iter_prefix)
                    any_data = True
                else:
                    polygon_gdf = line_gdf = point_gdf = None

                metrics = compute_richness_metrics(
                    polygon_gdf, line_gdf, point_gdf,
                    tag_frequencies=tag_frequencies,
                    bbox_size_m=bbox_size,
                )
                richness = float(metrics.get("richness_score", 0.0))
                total_nodes = int(metrics.get("total_nodes", 0))

                # Track the best result seen so far; used as fallback at max bbox
                if richness > best_richness:
                    best_richness = richness
                    best_bbox_size = bbox_size
                    best_total_nodes = total_nodes

                if is_rich_enough(
                    metrics,
                    cfg.min_total_nodes,
                    cfg.min_unique_labels,
                    cfg.min_richness_score,
                ):
                    return True, richness, bbox_size, total_nodes

                # Compute the next candidate bbox size
                if cfg.expansion_strategy == "linear":
                    next_size = bbox_size + cfg.expansion_step
                else:
                    next_size = bbox_size * cfg.expansion_factor

                if next_size > cfg.max_bbox_size:
                    # Return whatever we managed to find — even below-threshold
                    # data is useful for "very_low" density samples.
                    return any_data, best_richness, best_bbox_size, best_total_nodes

                bbox_size = next_size
                iteration += 1

            finally:
                # Delete this iteration's GeoJSON files immediately so the work
                # directory doesn't accumulate files across bbox expansions.
                _cleanup_geojson(iter_prefix)

    except Exception as exc:
        logger.warning("probe (%.5f, %.5f): error — %s", lat, lon, exc)
        return False, 0.0, cfg.initial_bbox_size, 0


def _score_location_worker(
    task: Tuple[int, float, float],
    *,
    work_dir: str,
    probe_config: ProbeConfig,
    downloader_config: DownloaderConfig,
    tag_frequencies: Optional[dict],
) -> Tuple[int, bool, float, float, int]:
    """Score one candidate location in a spawned worker process.

    Top-level function (not a method) so it is picklable with the 'spawn'
    multiprocessing start method required by ProcessPoolExecutor on Linux.

    Parameters
    ----------
    task:
        (candidate_idx, lat, lon)

    Returns
    -------
    (candidate_idx, success, richness_score, actual_bbox_size, total_nodes)
    """
    candidate_idx, lat, lon = task
    # Unique per-process subdirectory avoids filename collisions between workers
    worker_subdir = os.path.join(work_dir, f"w{os.getpid()}")
    os.makedirs(worker_subdir, exist_ok=True)
    prefix = os.path.join(worker_subdir, f"probe_{candidate_idx}")

    success, richness, bbox_size, total_nodes = adaptive_probe(
        lat=lat,
        lon=lon,
        output_file_prefix=prefix,
        probe_config=probe_config,
        downloader_config=downloader_config,
        tag_frequencies=tag_frequencies,
    )
    return candidate_idx, success, richness, bbox_size, total_nodes


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — Diversity-aware selection
# ─────────────────────────────────────────────────────────────────────────────

def classify_density_bucket(richness_score: float) -> str:
    """Map richness_score ∈ [0, 1] to a density bucket name."""
    for name, (lo, hi) in DENSITY_BUCKETS.items():
        if lo <= richness_score < hi:
            return name
    return "high"  # safety fallback for richness == 1.0


def annotate_h3_and_bucket(
    lat: float,
    lon: float,
    richness_score: float,
    bbox_size: float,
    total_nodes: int = 0,
) -> ScoredLocation:
    """Build a ScoredLocation with H3 cell indices and density bucket."""
    return ScoredLocation(
        lat=lat,
        lon=lon,
        richness_score=richness_score,
        bbox_size=bbox_size,
        total_nodes=total_nodes,
        h3_res3=h3.latlng_to_cell(lat, lon, 3),
        h3_res5=h3.latlng_to_cell(lat, lon, 5),
        h3_res7=h3.latlng_to_cell(lat, lon, 7),
        density_bucket=classify_density_bucket(richness_score),
    )


def _largest_remainder_quotas(
    fractions: Dict[str, float],
    total: int,
) -> Dict[str, int]:
    """Allocate integer quotas summing to *total* via the largest-remainder method."""
    keys = list(fractions.keys())
    s = sum(fractions.values())
    exact = {k: fractions[k] / s * total for k in keys}
    floors = {k: int(v) for k, v in exact.items()}
    remainders = {k: exact[k] - floors[k] for k in keys}
    deficit = total - sum(floors.values())
    for k in sorted(keys, key=lambda k: remainders[k], reverse=True)[:deficit]:
        floors[k] += 1
    return floors


def select_diverse_locations(
    scored: List[ScoredLocation],
    num_output: int,
    density_fractions: Optional[Dict[str, float]] = None,
    max_per_coarse_cell: int = 10,
    coarse_resolution: int = 3,
    scale_balance_weight: float = 0.5,
    rng_seed: int = 17,
    min_nodes_threshold: int = 0,
    min_nodes_exempt_fraction: float = 0.01,
) -> List[ScoredLocation]:
    """Stratified diversity-aware selection of *num_output* points.

    Algorithm
    ---------
    1. If *min_nodes_threshold* > 0: split candidates into node-rich (total_nodes
       ≥ threshold) and sparse (total_nodes < threshold, including unknowns with
       total_nodes == 0).  Reserve *min_nodes_exempt_fraction* × num_output slots
       for the sparse pool (uniform geographic baseline); route remaining quota
       through node-rich candidates only.
    2. Compute per-bucket integer quotas from *density_fractions* × *num_output*
       using the largest-remainder method so they sum exactly.
    3. Spatial diversity: within each bucket, group candidates by coarse H3 cell
       (at *coarse_resolution*) and cap each group to *max_per_coarse_cell*
       candidates (after random shuffle within the group).  If a bucket's capped
       pool is smaller than its quota, the deficit is redistributed pro-rata to
       buckets with a surplus.
    4. Scale diversity: within each bucket, weight candidates by
           scale_score = w * bbox_fraction + (1-w) * richness_score
       where bbox_fraction normalises bbox_size to [0, 1] within the bucket.
       Sample without replacement using these weights as probabilities, which
       draws a mix of dense (small bbox) and sparse (large bbox) points.
    5. Concatenate results from all buckets and sort by (bucket, richness_score).

    Parameters
    ----------
    density_fractions:
        Target fraction per bucket.  Defaults to DEFAULT_DENSITY_FRACTIONS.
        Values are normalised internally so they need not sum to 1.
    max_per_coarse_cell:
        Maximum candidates from each H3 cell at *coarse_resolution*.
    coarse_resolution:
        H3 resolution used as the spatial diversity cap level (default 3;
        ~41 k cells globally, each ~12 000 km²).
    scale_balance_weight:
        Weight for bbox_fraction in the scale score [0, 1].  Higher values
        prioritise spread across the bbox-size axis; lower values favour richness.
    min_nodes_threshold:
        If > 0, candidates with total_nodes < threshold are treated as "sparse"
        and only fill a small baseline quota (*min_nodes_exempt_fraction*).
        Candidates with total_nodes == 0 (unknown, from old cache) are treated
        as potentially node-rich and are not filtered out.
    min_nodes_exempt_fraction:
        Fraction of num_output to fill from sparse candidates when
        *min_nodes_threshold* > 0.  Default: 0.01 (1%).
    """
    if density_fractions is None:
        density_fractions = DEFAULT_DENSITY_FRACTIONS

    rng = np.random.default_rng(rng_seed)
    actual_output = min(num_output, len(scored))

    # Optional node-count split: reserve a small baseline for sparse locations,
    # route the rest through node-rich candidates only.
    sparse_baseline: List[ScoredLocation] = []
    if min_nodes_threshold > 0:
        # total_nodes == 0 means "unknown" (old cache entry) — treated as rich
        node_rich = [loc for loc in scored if loc.total_nodes == 0 or loc.total_nodes >= min_nodes_threshold]
        node_sparse = [loc for loc in scored if 0 < loc.total_nodes < min_nodes_threshold]
        n_sparse_slots = max(0, round(actual_output * min_nodes_exempt_fraction))
        n_rich_slots = actual_output - n_sparse_slots
        logger.info(
            "Node filter: %d rich (nodes≥%d or unknown), %d sparse (nodes<%d); "
            "slots: %d rich + %d sparse",
            len(node_rich), min_nodes_threshold,
            len(node_sparse), min_nodes_threshold,
            n_rich_slots, n_sparse_slots,
        )
        # Select sparse baseline via random spatial-capped draw
        if n_sparse_slots > 0 and node_sparse:
            coarse_attr = f"h3_res{coarse_resolution}"
            sparse_cell_groups: Dict[str, List[ScoredLocation]] = defaultdict(list)
            for loc in node_sparse:
                sparse_cell_groups[getattr(loc, coarse_attr, "unknown")].append(loc)
            sparse_pool: List[ScoredLocation] = []
            for group in sparse_cell_groups.values():
                order = rng.permutation(len(group))
                sparse_pool.extend(group[int(i)] for i in order[:max_per_coarse_cell])
            n_take = min(n_sparse_slots, len(sparse_pool))
            sparse_idx = rng.choice(len(sparse_pool), size=n_take, replace=False)
            sparse_baseline = [sparse_pool[int(i)] for i in sparse_idx]
        scored = node_rich
        actual_output = min(n_rich_slots, len(scored))

    # Step 1 — integer quotas
    quotas = _largest_remainder_quotas(density_fractions, actual_output)

    # Step 2 — bucket assignment and spatial cap
    buckets: Dict[str, List[ScoredLocation]] = {b: [] for b in DENSITY_BUCKETS}
    for loc in scored:
        buckets[loc.density_bucket].append(loc)

    coarse_attr = f"h3_res{coarse_resolution}"

    capped: Dict[str, List[ScoredLocation]] = {}
    for bname, locs in buckets.items():
        if not locs:
            capped[bname] = []
            continue
        cell_groups: Dict[str, List[ScoredLocation]] = defaultdict(list)
        for loc in locs:
            cell_groups[getattr(loc, coarse_attr, "unknown")].append(loc)

        pool: List[ScoredLocation] = []
        for group in cell_groups.values():
            # Shuffle within each cell so we don't always take the first-scored
            order = rng.permutation(len(group))
            pool.extend(group[int(i)] for i in order[:max_per_coarse_cell])
        capped[bname] = pool

    # Redistribute deficits from under-subscribed buckets to buckets with surplus
    surplus: Dict[str, int] = {}
    for bname, q in quotas.items():
        pool_size = len(capped[bname])
        if pool_size < q:
            quotas[bname] = pool_size  # take all available; deficit remains
        else:
            surplus[bname] = pool_size - q

    total_deficit = actual_output - sum(quotas.values())
    if total_deficit > 0 and surplus:
        total_surplus = sum(surplus.values())
        for bname, s in surplus.items():
            extra = min(s, round(total_deficit * s / total_surplus))
            quotas[bname] += extra

    # Step 3 — scale-diverse sampling within each bucket
    result: List[ScoredLocation] = []
    for bname, pool in capped.items():
        q = quotas[bname]
        if not pool or q == 0:
            continue
        if len(pool) <= q:
            result.extend(pool)
            continue

        sizes = np.array([loc.bbox_size for loc in pool], dtype=float)
        richnesses = np.array([loc.richness_score for loc in pool], dtype=float)

        sz_min, sz_max = sizes.min(), sizes.max()
        if sz_max > sz_min:
            bbox_frac = (sizes - sz_min) / (sz_max - sz_min)
        else:
            bbox_frac = np.ones_like(sizes)

        scale_scores = (
            scale_balance_weight * bbox_frac
            + (1.0 - scale_balance_weight) * richnesses
            + 1e-8  # avoid zero weights
        )
        weights = scale_scores / scale_scores.sum()

        chosen = rng.choice(len(pool), size=q, replace=False, p=weights)
        result.extend(pool[int(i)] for i in chosen)

    # Merge in sparse baseline (if min_nodes_threshold was used)
    result.extend(sparse_baseline)

    # Step 4 — sort for readability
    bucket_order = list(DENSITY_BUCKETS.keys())
    result.sort(key=lambda loc: (bucket_order.index(loc.density_bucket), loc.richness_score))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Resume helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_scored_cache(
    cache_path: str,
) -> Dict[Tuple[float, float], Tuple[float, float, int]]:
    """Load the parquet scoring cache.

    Returns a dict mapping (round(lat, 6), round(lon, 6)) → (richness_score, bbox_size, total_nodes).
    total_nodes defaults to 0 for entries written by older versions of the cache.
    An empty dict is returned if the cache does not exist or cannot be read.
    """
    if not os.path.exists(cache_path):
        return {}
    try:
        import pandas as pd
        df = pd.read_parquet(cache_path)
        result: Dict[Tuple[float, float], Tuple[float, float, int]] = {}
        has_nodes = "total_nodes" in df.columns
        for row in df.itertuples(index=False):
            key = (round(float(row.lat), 6), round(float(row.lon), 6))
            total_nodes = int(row.total_nodes) if has_nodes else 0
            result[key] = (float(row.richness_score), float(row.bbox_size), total_nodes)
        return result
    except Exception as exc:
        logger.warning("Could not load scored cache from %s: %s", cache_path, exc)
        return {}


def save_scored_cache(
    cache_path: str,
    scored: List[ScoredLocation],
) -> None:
    """Overwrite the parquet cache with the given scored locations."""
    import pandas as pd
    rows = [dataclasses.asdict(loc) for loc in scored]
    pd.DataFrame(rows).to_parquet(cache_path, index=False)

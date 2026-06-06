#!/usr/bin/env python3
"""Generate a globally-diverse set of lat/lon training locations from OSM data.

Replaces create_h3_locations.py with a batch SQL approach that is 10-50× faster:

  1. **Density scan** — one TABLESAMPLE SQL query on planet_osm_point identifies
     non-empty 1-degree cells (~5–10 min, cached).
  2. **Candidate generation** — enumerate all H3 res-5 cells (~22 km hexagons)
     whose centres fall within data-rich 1-degree cells (~seconds).
  3. **Batch SQL scoring** — count OSM features per candidate using VALUES + LATERAL
     queries (100 locations per round-trip, no GeoDataFrame construction, ~10–20 min).
  4. **Diversity-aware selection** — stratified sampling across density/spatial axes
     using the existing select_diverse_locations() algorithm.

Output CSV is directly compatible with create_dataset.py --locations-file.

Usage
-----
# Full run (PostGIS recommended)
python create_h3_locations_v2.py \\
    --output-dir h3_locations \\
    --postgis-url postgresql://localhost/gis \\
    --num-output-points 100000

# Quick smoke test
python create_h3_locations_v2.py \\
    --output-dir /tmp/h3_test \\
    --postgis-url postgresql://localhost/gis \\
    --num-output-points 500 \\
    --debug

# Force re-run (ignore caches)
python create_h3_locations_v2.py \\
    --output-dir h3_locations \\
    --postgis-url postgresql://localhost/gis \\
    --no-resume
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import logging
import os
import random
import signal
import sys
import threading
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from osmgraphclip.batch_scorer import (
    generate_h3_candidates,
    run_density_scan,
    score_candidates,
)
from osmgraphclip.global_sampler import (
    DEFAULT_DENSITY_FRACTIONS,
    ScoredLocation,
    select_diverse_locations,
)

logger = logging.getLogger(__name__)

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_NOISY_LOGGERS = ("httpx", "httpcore", "huggingface_hub", "transformers", "pyogrio")

_shutdown_event = threading.Event()


# ─── Signal handling ──────────────────────────────────────────────────────────

def _install_signal_handlers() -> None:
    def _handler(signum, frame):
        if not _shutdown_event.is_set():
            logger.warning(
                "Signal %d received — cache will be flushed and the run will stop "
                "gracefully after the current batch.", signum,
            )
            _shutdown_event.set()
            sys.exit(0)
    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


# ─── Argument parser ──────────────────────────────────────────────────────────

def _parse_density_fractions(s: str) -> Dict[str, float]:
    """Parse 'very_low=0.02,low=0.12,medium=0.77,high=0.09' → dict."""
    result: Dict[str, float] = {}
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        key, _, val = part.partition("=")
        result[key.strip()] = float(val.strip())
    if not result:
        raise argparse.ArgumentTypeError(f"Cannot parse density fractions: {s!r}")
    return result


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── I/O ───────────────────────────────────────────────────────────────────
    io = p.add_argument_group("I/O")
    io.add_argument(
        "--output-dir", required=True, metavar="DIR",
        help="Directory to write h3_locations.csv, metadata.json, and caches.",
    )
    io.add_argument(
        "--postgis-url", metavar="DSN",
        help="PostgreSQL connection string for the PostGIS database "
             "(or set POSTGIS_URL environment variable / .env file).",
    )
    io.add_argument(
        "--num-output-points", type=int, default=100_000, metavar="N",
        help="Number of locations to write to h3_locations.csv (default: 100000).",
    )

    # ── Phase 1: density scan ─────────────────────────────────────────────────
    ds = p.add_argument_group("Phase 1 — density scan")
    ds.add_argument(
        "--density-sample-pct", type=float, default=10.0, metavar="PCT",
        help="Percentage of planet_osm_point pages to sample for the density scan "
             "(TABLESAMPLE SYSTEM). 10%% is fast and reliable (default: 10.0).",
    )
    ds.add_argument(
        "--density-cache", metavar="PATH",
        help="Override path for the density raster cache "
             "(default: <output-dir>/density_cache.parquet).",
    )

    # ── Phase 2: candidate generation ─────────────────────────────────────────
    cg = p.add_argument_group("Phase 2 — H3 candidate generation")
    cg.add_argument(
        "--h3-resolution", type=int, default=5, metavar="RES",
        help="H3 resolution for candidate cells. "
             "5 = ~22 km hexagons, ~2 M cells globally (default). "
             "4 = ~58 km hexagons, ~288 k cells (faster, coarser). "
             "6 = ~9 km hexagons, ~14 M cells (denser, slower scoring).",
    )
    cg.add_argument(
        "--jitter", type=float, default=0.05, metavar="DEG",
        help="Random jitter in degrees added to H3 cell centres "
             "(breaks up the regular hexagonal grid; default: 0.05 ≈ ±5 km).",
    )

    # ── Phase 3: batch scoring ────────────────────────────────────────────────
    sc = p.add_argument_group("Phase 3 — batch SQL scoring")
    sc.add_argument(
        "--bbox-size", type=float, default=500.0, metavar="M",
        help="Fixed bbox half-width in metres for feature counting (default: 500).",
    )
    sc.add_argument(
        "--min-total-features", type=int, default=5, metavar="N",
        help="Minimum total OSM features required to include a candidate in diversity "
             "selection as a node-rich location (default: 5). "
             "Below-threshold locations can still be selected as the very_low baseline.",
    )
    sc.add_argument(
        "--batch-size", type=int, default=100, metavar="N",
        help="Number of candidate locations per SQL query (default: 100). "
             "Reduce if queries time out; increase for higher throughput.",
    )
    sc.add_argument(
        "--workers", type=int, default=4, metavar="N",
        help="Number of parallel PostGIS connections / worker threads (default: 4).",
    )
    sc.add_argument(
        "--query-timeout", type=int, default=120, metavar="S",
        help="Per-query statement_timeout in seconds (default: 120).",
    )
    sc.add_argument(
        "--scored-cache", metavar="PATH",
        help="Override path for the scoring cache "
             "(default: <output-dir>/scored_cache.parquet).",
    )
    sc.add_argument(
        "--flush-interval", type=int, default=10_000, metavar="N",
        help="Flush the scoring cache every N new results (default: 10000).",
    )

    # ── Phase 4: diversity selection ──────────────────────────────────────────
    dv = p.add_argument_group("Phase 4 — diversity-aware selection")
    dv.add_argument(
        "--density-fractions", type=_parse_density_fractions,
        default=DEFAULT_DENSITY_FRACTIONS,
        metavar="K=V,...",
        help=(
            "Target fraction per density bucket, comma-separated. "
            "Example: 'very_low=0.02,low=0.12,medium=0.77,high=0.09'. "
            "Values are normalised internally."
        ),
    )
    dv.add_argument(
        "--max-per-coarse-cell", type=int, default=10, metavar="N",
        help="Maximum locations selected per H3 res-3 cell (~130 km hexagon). "
             "Prevents geographic clustering (default: 10).",
    )
    dv.add_argument(
        "--min-nodes-selection", type=int, default=5, metavar="N",
        help="Selection-stage minimum total_nodes threshold. Candidates below this "
             "are treated as 'sparse' and fill at most min-nodes-exempt-fraction of "
             "the output (default: 5; 0 = disabled).",
    )
    dv.add_argument(
        "--min-nodes-exempt-fraction", type=float, default=0.01, metavar="F",
        help="Fraction of output reserved for below-threshold (sparse) locations "
             "(default: 0.01 = 1%%).",
    )

    # ── Misc ──────────────────────────────────────────────────────────────────
    misc = p.add_argument_group("Miscellaneous")
    misc.add_argument(
        "--seed", type=int, default=42, metavar="N",
        help="Random seed for candidate generation and diversity selection (default: 42).",
    )
    misc.add_argument(
        "--resume", dest="resume", action="store_true", default=True,
        help="Resume from existing density / scoring caches (default: on).",
    )
    misc.add_argument(
        "--no-resume", dest="resume", action="store_false",
        help="Ignore existing caches and re-run all phases from scratch.",
    )
    misc.add_argument(
        "--debug", action="store_true",
        help="Enable DEBUG-level logging.",
    )

    return p


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _resolve_postgis_url(args: argparse.Namespace) -> str:
    """Resolve PostGIS DSN from CLI arg, env var, or .env file."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    url = args.postgis_url or os.environ.get("POSTGIS_URL")
    if not url:
        raise SystemExit(
            "Error: PostGIS URL required. "
            "Pass --postgis-url or set POSTGIS_URL in the environment / .env file."
        )
    return url


def _write_csv(locations: List[ScoredLocation], path: str) -> None:
    rows = [dataclasses.asdict(loc) for loc in locations]
    df = pd.DataFrame(rows)
    # Reorder columns to match existing h3_locations.csv format
    ordered = [
        "lat", "lon", "richness_score", "bbox_size", "total_nodes",
        "h3_res3", "h3_res5", "h3_res7", "density_bucket",
    ]
    df = df[[c for c in ordered if c in df.columns]]
    df.to_csv(path, index=False)
    logger.info("Wrote %d locations to %s", len(df), path)


def _write_metadata(args: argparse.Namespace, selected: List[ScoredLocation], path: str) -> None:
    bucket_counts: Dict[str, int] = {}
    for loc in selected:
        bucket_counts[loc.density_bucket] = bucket_counts.get(loc.density_bucket, 0) + 1

    meta = {
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "approach": "batch_sql_v2",
        "num_output_points": args.num_output_points,
        "h3_resolution": args.h3_resolution,
        "jitter_deg": args.jitter,
        "bbox_size_m": args.bbox_size,
        "min_total_features": args.min_total_features,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "density_sample_pct": args.density_sample_pct,
        "seed": args.seed,
        "diversity": {
            "density_fractions": args.density_fractions,
            "max_per_coarse_cell": args.max_per_coarse_cell,
            "min_nodes_selection": args.min_nodes_selection,
            "min_nodes_exempt_fraction": args.min_nodes_exempt_fraction,
        },
        "total_selected": len(selected),
        "bucket_counts": bucket_counts,
        "backend": "postgis",
    }
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Wrote metadata to %s", path)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format=_LOG_FORMAT,
    )
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    _install_signal_handlers()

    # Resolve paths
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    density_cache = args.density_cache or str(output_dir / "density_cache.parquet")
    scored_cache  = args.scored_cache  or str(output_dir / "scored_cache.parquet")
    csv_path      = str(output_dir / "h3_locations.csv")
    meta_path     = str(output_dir / "metadata.json")

    postgis_url = _resolve_postgis_url(args)

    # ── Phase 1: global density pre-scan ─────────────────────────────────────
    if not args.resume and os.path.exists(density_cache):
        logger.info("--no-resume: removing density cache %s", density_cache)
        os.remove(density_cache)
    if not args.resume and os.path.exists(scored_cache):
        logger.info("--no-resume: removing scoring cache %s", scored_cache)
        os.remove(scored_cache)

    logger.info("=== Phase 1: Global density scan ===")
    density_cells = run_density_scan(
        postgis_url=postgis_url,
        cache_path=density_cache,
        sample_pct=args.density_sample_pct,
    )
    logger.info("Density scan: %d non-empty 1-degree cells", len(density_cells))

    if not density_cells:
        raise SystemExit("Error: density scan returned 0 cells. Check PostGIS connection.")

    # ── Phase 2: H3 candidate generation ─────────────────────────────────────
    logger.info("=== Phase 2: H3 candidate generation (res-%d) ===", args.h3_resolution)
    candidates = generate_h3_candidates(
        density_cells=density_cells,
        h3_resolution=args.h3_resolution,
        rng_seed=args.seed,
        jitter_scale=args.jitter,
    )
    logger.info("Generated %d candidate locations", len(candidates))

    if not candidates:
        raise SystemExit("Error: no candidates generated — check density cells.")

    # Shuffle candidates so workers process geographically-diverse batches
    # (avoids all workers hitting the same dense-city region simultaneously)
    rng = np.random.default_rng(args.seed + 1)
    indices = rng.permutation(len(candidates))
    candidates = [candidates[int(i)] for i in indices]

    # ── Phase 3: Batch SQL scoring ────────────────────────────────────────────
    logger.info(
        "=== Phase 3: Batch SQL scoring (%d candidates, %d workers) ===",
        len(candidates), args.workers,
    )
    scored = score_candidates(
        candidates=candidates,
        postgis_url=postgis_url,
        cache_path=scored_cache,
        bbox_size_m=args.bbox_size,
        batch_size=args.batch_size,
        n_workers=args.workers,
        query_timeout_s=args.query_timeout,
        flush_interval=args.flush_interval,
    )
    logger.info("Scoring complete: %d total scored locations", len(scored))

    if not scored:
        raise SystemExit("Error: no locations were scored. Check PostGIS connection.")

    # ── Phase 4: Diversity-aware selection ────────────────────────────────────
    logger.info(
        "=== Phase 4: Diversity-aware selection (%d → %d) ===",
        len(scored), args.num_output_points,
    )
    selected = select_diverse_locations(
        scored=scored,
        num_output=args.num_output_points,
        density_fractions=args.density_fractions,
        max_per_coarse_cell=args.max_per_coarse_cell,
        coarse_resolution=3,
        scale_balance_weight=0.5,
        rng_seed=args.seed + 2,
        min_nodes_threshold=args.min_nodes_selection,
        min_nodes_exempt_fraction=args.min_nodes_exempt_fraction,
    )
    logger.info("Selected %d locations", len(selected))

    # ── Output ────────────────────────────────────────────────────────────────
    _write_csv(selected, csv_path)
    _write_metadata(args, selected, meta_path)

    # Summary
    from osmgraphclip.global_sampler import DENSITY_BUCKETS
    for bucket in DENSITY_BUCKETS:
        n = sum(1 for loc in selected if loc.density_bucket == bucket)
        logger.info("  %-10s: %6d  (%.1f%%)", bucket, n, 100.0 * n / max(len(selected), 1))

    logger.info("Done. Output: %s", csv_path)


if __name__ == "__main__":
    main()

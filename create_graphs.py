#!/usr/bin/env python3
"""Build OSM graph pickles from GeoJSON files produced by create_dataset.py.

Reads the ``downloads`` table from ``dataset.db``, scans ``graphs/`` for any
untracked GeoJSON files (logs a warning), and populates the ``graphs`` table
in ``dataset.db`` which is consumed by the training pipeline.

Run create_dataset.py first to populate the GeoJSON files and dataset.db.
"""

import argparse
import datetime
import json
import logging
import os
import pickle
import signal
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple

from osmgraphclip.osm_graph_builder import build_osm_graph, build_zero_graph
from osmgraphclip.dataset_pipeline import (
    load_gdfs_for_prefix,
    scan_geojson_prefixes,
)
from osmgraphclip.dataset_db import DatasetDB

logger = logging.getLogger(__name__)

_shutdown_event = threading.Event()


def _install_signal_handlers():
    def _handler(signum, frame):
        if not _shutdown_event.is_set():
            logger.warning(
                "Signal %d received — finishing in-flight graphs then stopping cleanly...",
                signum,
            )
            _shutdown_event.set()

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


def _build_graph_from_prefix(
    prefix_path: str,
    tagw_path: str,
    embedding_backend: str,
    embedding_model: Optional[str],
    embedding_cache=None,
    method: str = 'geolink',
) -> str:
    """Load GeoJSON files for prefix_path, build the graph, and save a pickle.

    Returns the path to the saved pickle.
    Raises FileNotFoundError if no GeoJSON features are found.
    """
    polygon_gdf, line_gdf, point_gdf = load_gdfs_for_prefix(prefix_path)
    if polygon_gdf is None and line_gdf is None and point_gdf is None:
        raise FileNotFoundError(
            f"No GeoJSON features found for prefix '{prefix_path}'."
        )
    graph = build_osm_graph(
        polygon_gdf=polygon_gdf,
        line_gdf=line_gdf,
        point_gdf=point_gdf,
        tagw_path=tagw_path,
        embedding_backend=embedding_backend,
        embedding_model=embedding_model,
        embedding_cache=embedding_cache,
        method=method,
    )
    pickle_path = f"{prefix_path}_graph.pkl"
    with open(pickle_path, "wb") as f:
        pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)
    return pickle_path


def _process_entry(
    row: dict,
    *,
    graphs_output_dir: str,
    tagw_path: str,
    embedding_backend: str,
    embedding_model: Optional[str],
    embedding_cache=None,
    method: str = 'geolink',
    node_embedding_dim: int = 512,
) -> Tuple[str, List[list]]:
    """Build graph pickle(s) for one downloads entry.

    Returns (outcome, rows) where outcome is ``"processed"`` or
    ``"skipped_empty"``, and rows is a list of row tuples to insert into graphs.
    """
    logger.debug("Processing entry: %s", row["geojson_prefix"])
    prefix_path = os.path.join(graphs_output_dir, row["geojson_prefix"])
    lat = row["lat"]
    lon = row["lon"]
    bbox_size = row["bbox_size"]
    level = row.get("level")

    t0 = datetime.datetime.now()
    try:
        pickle_path = _build_graph_from_prefix(
            prefix_path, tagw_path, embedding_backend, embedding_model,
            embedding_cache=embedding_cache, method=method,
        )
    except FileNotFoundError:
        nodata_path = prefix_path + ".nodata"
        if os.path.exists(nodata_path):
            graph = build_zero_graph(node_embedding_dim)
            pickle_path = f"{prefix_path}_graph.pkl"
            with open(pickle_path, "wb") as f:
                pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)
            pickle_name = os.path.basename(pickle_path)
            logger.info(
                "Location %s: no OSM data — wrote zero graph → %s", row["geojson_prefix"], pickle_name,
            )
            elapsed = (datetime.datetime.now() - t0).total_seconds()
            csv_row = {
                'lat': lat, 'lon': lon, 'bbox_size': bbox_size,
                'graph_pickle': pickle_name, 'level': level, 'method': 'zero',
            }
            return "processed", [csv_row], row["geojson_prefix"], elapsed
        logger.warning("Skipping %s: no GeoJSON features found", row["geojson_prefix"])
        return "skipped_empty", [], row["geojson_prefix"], None
    except Exception as exc:
        logger.error(
            "Failed to build graph for %s: %s", row["geojson_prefix"], exc, exc_info=True,
        )
        return "skipped_empty", [], row["geojson_prefix"], None

    elapsed = (datetime.datetime.now() - t0).total_seconds()
    pickle_name = os.path.basename(pickle_path)
    csv_row = {
        'lat': lat, 'lon': lon, 'bbox_size': bbox_size,
        'graph_pickle': pickle_name, 'level': level, 'method': method,
    }
    return "processed", [csv_row], row["geojson_prefix"], elapsed


def main():
    logging.basicConfig(level=logging.INFO)
    _install_signal_handlers()

    for noisy in ("httpx", "httpcore", "huggingface_hub", "transformers", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    import transformers
    transformers.logging.set_verbosity_error()

    parser = argparse.ArgumentParser(
        description="Build OSM graph pickles from GeoJSON files produced by create_dataset.py.",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Dataset directory (same --output-dir used with create_dataset.py).",
    )
    parser.add_argument(
        "--tagw-path", type=str, default="data/all_tags30_frequency1.json",
        help="Path to GeoLink tag weights JSON file (required for graph construction).",
    )
    parser.add_argument(
        "--embedding-backend", type=str, default="clip", choices=["clip", "sbert"],
        help="Semantic embedding backend for OSM tag features. Default: clip.",
    )
    parser.add_argument(
        "--embedding-model", type=str, default=None,
        help=(
            "Model name or path for the embedding backend. "
            "Defaults to 'openai/clip-vit-base-patch16' for clip and 'all-MiniLM-L6-v2' for sbert."
        ),
    )
    parser.add_argument(
        "--embedding-cache-path", type=str, default=None,
        help=(
            "Path to a SQLite file for persistent word-embedding cache. "
            "Embeddings are stored across runs and reused, avoiding redundant model inference. "
            "The cache is keyed by (model_name, word) and can be shared across datasets."
        ),
    )
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--resume", dest="resume", action="store_true",
        help="Skip already built graph pickles (default).",
    )
    resume_group.add_argument(
        "--no-resume", dest="resume", action="store_false",
        help="Rebuild all graph pickles and clear the graphs table.",
    )
    parser.set_defaults(resume=True)
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Number of parallel worker threads. Default: 4.",
    )
    parser.add_argument(
        "--method", type=str, default="geolink", choices=["geolink"],
        help="Graph construction method. 'geolink' (default) uses topological spatial "
             "relations.",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable DEBUG logging (includes embedding cache hits).",
    )
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.embedding_cache_path:
        from osmgraphclip.embedding_cache import EmbeddingCache
        from osmgraphclip.osm_to_graph import CLIP_DEFAULT_MODEL, SBERT_DEFAULT_MODEL
        _default_model = CLIP_DEFAULT_MODEL if args.embedding_backend == "clip" else SBERT_DEFAULT_MODEL
        embedding_cache = EmbeddingCache(args.embedding_cache_path, args.embedding_model or _default_model)
        logger.info("Embedding cache: %s (model=%s)", args.embedding_cache_path,
                    args.embedding_model or _default_model)
    else:
        embedding_cache = None

    graphs_output_dir = os.path.join(args.output_dir, "graphs")
    db_path = os.path.join(args.output_dir, "dataset.db")
    metadata_path = os.path.join(args.output_dir, "metadata.json")

    if not os.path.isdir(graphs_output_dir):
        logger.error(
            "graphs/ directory not found in %s. Run create_dataset.py first.", args.output_dir,
        )
        return

    # ── Open DB and read downloads (primary input) ────────────────────────────
    db = DatasetDB(db_path)
    download_rows = db.downloads_as_list()
    if not download_rows:
        logger.error(
            "dataset.db not found or downloads table empty in %s. Run create_dataset.py first.",
            args.output_dir,
        )
        db.close()
        return
    logger.info("Loaded %d entries from downloads table.", len(download_rows))

    # ── Warn about GeoJSON files not tracked in downloads.csv ─────────────────
    known_prefixes = {row["geojson_prefix"] for row in download_rows}
    disk_prefixes = scan_geojson_prefixes(graphs_output_dir)
    untracked = disk_prefixes - known_prefixes
    if untracked:
        logger.warning(
            "%d GeoJSON prefix(es) in graphs/ are not recorded in the downloads table "
            "(lat/lon unavailable — skipping): %s",
            len(untracked), sorted(untracked),
        )

    # ── Resume setup ──────────────────────────────────────────────────────────
    if args.resume:
        existing_graph_pickles = db.graph_pickles
    else:
        db.clear_graphs()
        existing_graph_pickles = set()
        logger.info("Resume disabled: cleared graphs table in %s.", db_path)

    # ── Merge graph-side fields into metadata.json ────────────────────────────
    metadata: dict = {}
    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            try:
                metadata = json.load(f)
            except json.JSONDecodeError:
                pass
    metadata.update({
        "tagw_path": args.tagw_path,
        "embedding_backend": args.embedding_backend,
        "embedding_model": args.embedding_model,
        "graph_method": args.method,
    })
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    processed_count = 0
    skipped_count = 0
    skipped_existing_count = 0
    recovered_csv_count = 0

    # ── Recovery pre-pass ─────────────────────────────────────────────────────
    # Register pickles that already exist on disk but are absent from the DB
    # (e.g. after a previous interrupted run).
    if args.resume:
        logger.info("Checking for existing graph pickles on disk to recover missing DB entries...")
        for row in download_rows:
            pickle_name = f"{row['geojson_prefix']}_graph.pkl"
            pickle_path = os.path.join(graphs_output_dir, pickle_name)
            if os.path.exists(pickle_path) and pickle_name not in existing_graph_pickles:
                db.insert_graph(
                    lat=row["lat"], lon=row["lon"], bbox_size=row["bbox_size"],
                    graph_pickle=pickle_name, level=row.get("level"),
                    method=args.method,
                )
                existing_graph_pickles.add(pickle_name)
                recovered_csv_count += 1
                logger.info(
                    "Recovered missing DB row for existing pickle: %s", pickle_name,
                )

    # ── Build pending list ────────────────────────────────────────────────────
    logger.info("Building list of pending graph entries...")
    pending: List[dict] = []
    for row in download_rows:
        pickle_name = f"{row['geojson_prefix']}_graph.pkl"
        if args.resume and pickle_name in existing_graph_pickles:
            pickle_path = os.path.join(graphs_output_dir, pickle_name)
            if os.path.exists(pickle_path):
                skipped_existing_count += 1
                continue
            logger.warning(
                "Graph pickle %s is in the DB but missing from disk; rebuilding.", pickle_name,
            )
        pending.append(row)

    logger.info(
        "Building graphs for %d pending entries with %d worker(s) "
        "(%d already done, %d recovered).",
        len(pending), args.workers, skipped_existing_count, recovered_csv_count,
    )

    def _fmt_eta(seconds: float) -> str:
        seconds = int(seconds)
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        if h:
            return f"{h}h{m:02d}m{s:02d}s"
        if m:
            return f"{m}m{s:02d}s"
        return f"{s}s"

    t_start = datetime.datetime.now()

    def _collect_future(future, futures_map):
        nonlocal processed_count, skipped_count
        row = futures_map[future]
        try:
            outcome, rows, prefix, build_elapsed = future.result()
        except Exception as exc:
            logger.error(
                "Entry %s: unexpected error: %s", row["geojson_prefix"], exc, exc_info=True,
            )
            skipped_count += 1
            return
        if outcome == "skipped_empty":
            skipped_count += 1
        else:
            processed_count += 1
        if rows:
            for r in rows:
                db.insert_graph(
                    lat=r['lat'], lon=r['lon'], bbox_size=r['bbox_size'],
                    graph_pickle=r['graph_pickle'],
                    level=r.get('level'),
                    method=r.get('method', 'geolink'),
                )

    from osmgraphclip.osm_graph_builder import _get_builder
    _builder = _get_builder(
        args.method, args.tagw_path, "cpu",
        args.embedding_backend, args.embedding_model, embedding_cache,
    )
    node_embedding_dim = _builder.embedding_dim

    worker_kwargs = dict(
        graphs_output_dir=graphs_output_dir,
        tagw_path=args.tagw_path,
        embedding_backend=args.embedding_backend,
        embedding_model=args.embedding_model,
        embedding_cache=embedding_cache,
        method=args.method,
        node_embedding_dim=node_embedding_dim,
    )

    total = len(pending)
    completed = 0
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(_process_entry, row, **worker_kwargs): row
                for row in pending
            }
            for future in as_completed(futures):
                _collect_future(future, futures)
                completed += 1
                wall = (datetime.datetime.now() - t_start).total_seconds()
                eta_str = (
                    f" | ~{_fmt_eta(wall / completed * (total - completed))} remaining"
                    if wall > 0 and completed < total else ""
                )
                prefix = futures[future]["geojson_prefix"]
                try:
                    _, _, _, build_elapsed = future.result()
                except Exception:
                    build_elapsed = None
                if build_elapsed is not None:
                    logger.info("[%d/%d] %s — %.1fs%s", completed, total, prefix, build_elapsed, eta_str)
                else:
                    logger.info("[%d/%d] %s SKIPPED%s", completed, total, prefix, eta_str)
                if _shutdown_event.is_set():
                    logger.warning("Cancelling %d pending graph(s).", total - completed)
                    for f in futures:
                        f.cancel()
                    break
    finally:
        db.close()
        if embedding_cache is not None:
            embedding_cache.close()

    if _shutdown_event.is_set():
        logger.warning(
            "⚠️  Interrupted in %s: %d new, %d already done, "
            "%d failed/skipped, %d recovered, %d cancelled.",
            args.output_dir, processed_count, skipped_existing_count,
            skipped_count, recovered_csv_count, total - completed,
        )
    else:
        logger.info(
            "✅ Graph construction complete in %s: %d new, %d already done, "
            "%d failed/skipped, %d recovered.",
            args.output_dir, processed_count, skipped_existing_count,
            skipped_count, recovered_csv_count,
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Download OSM data and extract multi-scale band features for a set of locations.

For each location this script:
  1. Downloads fine-grain OSM data (--bbox-size, default 1000 m) as GeoJSON files
     compatible with create_graphs.py — stored in <output-dir>/graphs/.
  2. For each band radius (--band-radii, default 2000 10000 20000 m) downloads
     OSM data within a concentric disk at that radius, extracts spatial statistics
     and an SBERT semantic embedding, and discards the intermediate GeoJSON.
  3. Saves per-location band features as <output-dir>/bands/osm_{i}_bands.npz.

Run create_graphs.py afterwards to build graph pickles from the fine-grain GeoJSON.

.npz file layout
----------------
  band_radii           : float32[n_bands]
  spatial_features     : float32[n_bands, 47]       — global spatial features
  spatial_feature_names: object[47]
  subbin_spatial       : float32[n_bands, 2, 16]    — inner / outer ring
  subbin_feature_names : object[16]
  sector_spatial       : float32[n_bands, 4, 11]    — N / E / S / W sectors
  sector_feature_names : object[11]
  global_embeddings    : float32[n_bands, D]         — distance-weighted SBERT
  subbin_embeddings    : float32[n_bands, 2, D]
  sector_embeddings    : float32[n_bands, 4, D]
"""

import argparse
import json
import logging
import os
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Set, Tuple

from osmgraphclip.downloader_config import DownloaderConfig, load_downloader_config, create_downloader
from osmgraphclip.dataset_pipeline import (
    load_gdfs_for_prefix,
    nodata_sentinel_path,
    write_nodata_sentinel,
)
from osmgraphclip.embedding_cache import EmbeddingCache
from osmgraphclip.band_extractor import BandExtractor
from osmgraphclip.multiscale_db import MultiscaleDatasetDB

from create_dataset import (
    CITY_BBOXES,
    generate_grid_locations,
    stream_locations_from_csv,
)

logger = logging.getLogger(__name__)
_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_NOISY_LOGGERS = (
    "httpx", "httpcore", "huggingface_hub", "transformers", "sentence_transformers",
)

_shutdown_event = threading.Event()

_GEOJSON_SUFFIXES = (
    "_polygon.geojson.gz", "_multipolygon.geojson.gz",
    "_linestring.geojson.gz", "_multilinestring.geojson.gz",
    "_point.geojson.gz", "_multipoint.geojson.gz",
)


def _install_signal_handlers() -> None:
    def _handler(signum, frame):
        if not _shutdown_event.is_set():
            logger.warning("Signal %d received — shutting down cleanly...", signum)
            _shutdown_event.set()
    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


def _delete_geojson_for_prefix(directory: str, prefix_name: str) -> None:
    full = os.path.join(directory, prefix_name)
    for suffix in _GEOJSON_SUFFIXES:
        path = full + suffix
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        except OSError as exc:
            logger.debug("Could not remove %s: %s", path, exc)


def _process_location(
    i: int,
    lat: float,
    lon: float,
    *,
    graphs_output_dir: str,
    bands_output_dir: str,
    tmp_dir: str,
    bbox_size: float,
    band_radii: List[float],
    band_max_rows: int,
    downloader_cfg: DownloaderConfig,
    band_extractor: BandExtractor,
    downloaded_prefixes: Set[str],
    completed_band_locations: Set[Tuple[float, float]],
    resume: bool,
) -> Tuple[str, Optional[list], Optional[tuple]]:  # noqa: E501
    """Download fine-grain GeoJSON + extract band features for one location.

    Returns (outcome, fine_row, band_row):
      fine_row  = [lat, lon, bbox_size, prefix_name] or None
      band_row  = (lat, lon, bands_npz_name, band_radii) or None
    """
    if _shutdown_event.is_set():
        return "shutdown", None, None

    fine_prefix_name = f"osm_{i}"
    fine_prefix_full = os.path.join(graphs_output_dir, fine_prefix_name)
    bands_npz_name = f"osm_{i}_bands.npz"
    bands_npz_path = os.path.join(bands_output_dir, bands_npz_name)
    _nodata_path = nodata_sentinel_path(graphs_output_dir, i)

    fine_done = resume and (
        fine_prefix_name in downloaded_prefixes
        or os.path.exists(_nodata_path)
    )
    bands_done = resume and (lat, lon) in completed_band_locations

    if fine_done and bands_done:
        logger.debug("Location %d: already complete, skipping.", i)
        return "skipped", None, None

    _t_start = time.perf_counter()

    # ── Fine-grain download ───────────────────────────────────────────────────
    fine_row = None
    _t_fine = 0.0
    if not fine_done:
        logger.info(
            "Location %d (%.5f, %.5f): downloading fine-grain bbox=%dm",
            i, lat, lon, int(bbox_size),
        )
        _t_fine_start = time.perf_counter()
        try:
            downloader = create_downloader(
                downloader_cfg,
                lat=lat, lon=lon,
                dist=bbox_size,
                output_file=fine_prefix_full,
            )
            downloaded = downloader()
        except Exception as exc:
            logger.warning("Location %d: fine-grain download error: %s", i, exc)
            downloaded = False
        _t_fine = time.perf_counter() - _t_fine_start

        if downloaded:
            fine_row = [lat, lon, bbox_size, fine_prefix_name]
            logger.info("Location %d: fine-grain download complete (%.2fs).", i, _t_fine)
        else:
            logger.warning(
                "Location %d (%.5f, %.5f): no OSM data for fine-grain bbox.",
                i, lat, lon,
            )
            write_nodata_sentinel(graphs_output_dir, i, "no OSM data for fine-grain bbox")
            # Insert a downloads row so create_graphs.py can build a zero graph.
            fine_row = [lat, lon, 0.0, fine_prefix_name]

    # ── Band downloads and feature extraction ────────────────────────────────
    band_row = None
    _t_bands = 0.0
    if not bands_done:
        logger.info(
            "Location %d: extracting bands: %s",
            i, " ".join(f"{int(r)}m" for r in band_radii),
        )
        _t_bands_start = time.perf_counter()
        bands_data: List[Tuple] = []

        # Optimised path: query PostGIS once at the largest radius and filter
        # smaller bands spatially in Python, avoiding N-1 extra round-trips.
        max_radius = max(band_radii)
        _tried_multi = False
        try:
            downloader = create_downloader(
                downloader_cfg,
                lat=lat, lon=lon,
                dist=max_radius,
                output_file=os.path.join(tmp_dir, f"osm_{i}_bmax"),
            )
            if hasattr(downloader, "fetch_gdfs_multi_radii"):
                bands_data = downloader.fetch_gdfs_multi_radii(band_radii, band_max_rows=band_max_rows)
                _tried_multi = True
                logger.debug(
                    "Location %d: fetched all %d band radii in one PostGIS query.",
                    i, len(band_radii),
                )
        except Exception as exc:
            logger.warning(
                "Location %d: multi-radius fetch failed (%s); falling back to per-band queries.",
                i, exc,
            )

        if not _tried_multi:
            for radius in band_radii:
                try:
                    downloader = create_downloader(
                        downloader_cfg,
                        lat=lat, lon=lon,
                        dist=radius,
                        output_file=os.path.join(tmp_dir, f"osm_{i}_b{int(radius)}"),
                    )
                    if hasattr(downloader, "fetch_gdfs"):
                        p_gdf, l_gdf, pt_gdf = downloader.fetch_gdfs(band_max_rows=band_max_rows)
                    else:
                        tmp_prefix_full = os.path.join(tmp_dir, f"osm_{i}_b{int(radius)}")
                        downloaded = downloader()
                        if downloaded:
                            try:
                                p_gdf, l_gdf, pt_gdf = load_gdfs_for_prefix(tmp_prefix_full)
                            except Exception as exc:
                                logger.warning(
                                    "Location %d: failed to load GDFs for radius=%dm: %s",
                                    i, int(radius), exc,
                                )
                                p_gdf = l_gdf = pt_gdf = None
                        else:
                            p_gdf = l_gdf = pt_gdf = None
                        _delete_geojson_for_prefix(tmp_dir, f"osm_{i}_b{int(radius)}")
                except Exception as exc:
                    logger.warning(
                        "Location %d: band radius=%dm download error: %s", i, int(radius), exc,
                    )
                    p_gdf = l_gdf = pt_gdf = None

                bands_data.append((radius, p_gdf, l_gdf, pt_gdf))

        try:
            bands_result = band_extractor.extract_all_bands(lat, lon, bands_data)
            band_extractor.to_npz(bands_result, bands_npz_path)
            band_row = (lat, lon, bands_npz_name, band_radii)
            _t_bands = time.perf_counter() - _t_bands_start
            logger.info(
                "Location %d: band features saved → %s (%.2fs)", i, bands_npz_name, _t_bands,
            )
        except Exception as exc:
            _t_bands = time.perf_counter() - _t_bands_start
            logger.error("Location %d: band feature extraction failed: %s", i, exc)

    _t_total = time.perf_counter() - _t_start
    logger.info(
        "Location %d (%.5f, %.5f): total=%.2fs  fine=%.2fs  bands=%.2fs",
        i, lat, lon, _t_total, _t_fine, _t_bands,
    )

    return "processed", fine_row, band_row


def main() -> None:
    logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT)
    for noisy in _NOISY_LOGGERS:
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _install_signal_handlers()

    parser = argparse.ArgumentParser(
        description=(
            "Download OSM data and extract multi-scale band features for a set of locations. "
            "Run create_graphs.py afterwards to build graph pickles."
        ),
    )
    parser.add_argument("--output-dir", required=True,
                        help="Dataset output directory.")
    parser.add_argument("--locations-file",
                        help="CSV with lat/lon columns (mutually exclusive with --city).")
    parser.add_argument(
        "--city",
        help=f"Built-in city grid. Available: {', '.join(CITY_BBOXES.keys())}",
    )
    parser.add_argument("--sample-spacing", type=float, default=500,
                        help="Grid spacing in metres (city mode). Default: 500m")
    parser.add_argument("--bbox-size", type=float, default=1000,
                        help="Fine-grain bbox half-width in metres. Default: 1000m")
    parser.add_argument(
        "--band-radii", type=float, nargs="+", default=[2000.0, 10000.0, 20000.0],
        metavar="RADIUS_M",
        help="Band radii in metres (concentric disks). Default: 2000 10000 20000",
    )
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2",
                        help="SBERT model name. Default: all-MiniLM-L6-v2")
    parser.add_argument("--embedding-cache-path",
                        help="SQLite file for persistent SBERT embedding cache.")
    parser.add_argument("--backend", default="overpass", choices=["overpass", "postgis", "auto"],
                        help="OSM download backend. Default: overpass")
    parser.add_argument("--postgis-url",
                        help="PostgreSQL DSN for PostGIS backend (or POSTGIS_URL env var).")
    parser.add_argument("--postgis-max-rows", type=int, default=50000,
                        dest="postgis_max_rows_per_table",
                        help="Max rows per geometry table from PostGIS (fine-grain). Default: 50000")
    parser.add_argument("--band-max-rows", type=int, default=100000,
                        dest="band_max_rows",
                        help="SQL LIMIT per table for band queries (0=unlimited). "
                             "Prevents fetchall() hanging on huge bounding boxes. Default: 100000")
    parser.add_argument("--resume", action="store_true", default=True,
                        help="Skip already-processed locations (default).")
    parser.add_argument("--no-resume", action="store_false", dest="resume",
                        help="Re-download and re-extract all locations.")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel worker threads. Default: 4")
    parser.add_argument("--device", default="cpu",
                        help="Torch device for SBERT model. Default: cpu")
    parser.add_argument("--debug", action="store_true",
                        help="Enable DEBUG-level logging.")

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    if not args.city and not args.locations_file:
        parser.error("Either --city or --locations-file must be specified.")
    if args.city and args.locations_file:
        parser.error("--city and --locations-file are mutually exclusive.")

    # ── Setup directories ─────────────────────────────────────────────────────
    graphs_output_dir = os.path.join(args.output_dir, "graphs")
    bands_output_dir = os.path.join(args.output_dir, "bands")
    tmp_dir = os.path.join(args.output_dir, "tmp_bands")
    for d in (graphs_output_dir, bands_output_dir, tmp_dir):
        os.makedirs(d, exist_ok=True)

    db_path = os.path.join(args.output_dir, "dataset.db")
    metadata_path = os.path.join(args.output_dir, "metadata.json")

    # ── Load locations ────────────────────────────────────────────────────────
    if args.city:
        city_key = args.city.lower()
        if city_key not in CITY_BBOXES:
            logger.error(
                "Unknown city: %s. Available: %s", args.city, ", ".join(CITY_BBOXES.keys())
            )
            return
        lat_min, lon_min, lat_max, lon_max = CITY_BBOXES[city_key]
        locations: List[Tuple[int, float, float]] = [
            (i, lat, lon)
            for i, (lat, lon) in enumerate(
                generate_grid_locations(lat_min, lon_min, lat_max, lon_max, args.sample_spacing)
            )
        ]
        location_source = f"city:{args.city}"
        logger.info("City grid: %d locations for %s", len(locations), args.city)
    else:
        locations = list(stream_locations_from_csv(args.locations_file))
        location_source = args.locations_file
        logger.info("CSV: %d locations from %s", len(locations), args.locations_file)

    # ── Write metadata ────────────────────────────────────────────────────────
    band_radii = sorted(args.band_radii)
    metadata: dict = {
        "bbox_size_m": args.bbox_size,
        "band_radii_m": band_radii,
        "location_source": location_source,
        "embedding_model": args.embedding_model,
    }
    if args.city:
        metadata["city"] = args.city
        metadata["sample_spacing_m"] = args.sample_spacing
        metadata["city_bbox"] = CITY_BBOXES[args.city.lower()]
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # ── Initialize components ─────────────────────────────────────────────────
    downloader_cfg = load_downloader_config(args)

    embedding_cache: Optional[EmbeddingCache] = None
    if args.embedding_cache_path:
        embedding_cache = EmbeddingCache(args.embedding_cache_path, args.embedding_model)
        logger.info("SBERT embedding cache: %s", args.embedding_cache_path)

    band_extractor = BandExtractor(
        sbert_model_name=args.embedding_model,
        device=args.device,
        cache=embedding_cache,
    )

    processed = skipped = errors = 0

    with MultiscaleDatasetDB(db_path) as db:
        if not args.resume:
            db.clear_downloads()
            db.clear_band_features()
            logger.info("--no-resume: cleared downloads and band_features tables.")

        downloaded_prefixes = db.downloaded_prefixes
        completed_band_locs = db.completed_band_locations()

        logger.info(
            "Resume state: %d fine-grain downloads done, %d band extractions done.",
            len(downloaded_prefixes), len(completed_band_locs),
        )

        futures = {}
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for i, lat, lon in locations:
                if _shutdown_event.is_set():
                    break
                fut = executor.submit(
                    _process_location,
                    i, lat, lon,
                    graphs_output_dir=graphs_output_dir,
                    bands_output_dir=bands_output_dir,
                    tmp_dir=tmp_dir,
                    bbox_size=args.bbox_size,
                    band_radii=band_radii,
                    band_max_rows=args.band_max_rows,
                    downloader_cfg=downloader_cfg,
                    band_extractor=band_extractor,
                    downloaded_prefixes=downloaded_prefixes,
                    completed_band_locations=completed_band_locs,
                    resume=args.resume,
                )
                futures[fut] = (i, lat, lon)

            for fut in as_completed(futures):
                i, lat, lon = futures[fut]
                try:
                    outcome, fine_row, band_row = fut.result()
                except Exception as exc:
                    logger.error("Location %d (%.5f, %.5f): unexpected error: %s", i, lat, lon, exc)
                    errors += 1
                    continue

                if outcome == "skipped":
                    skipped += 1
                    continue
                if outcome == "shutdown":
                    continue

                if fine_row is not None:
                    lat_, lon_, bbox_, prefix_ = fine_row
                    db.insert_download(lat_, lon_, bbox_, prefix_)

                if band_row is not None:
                    lat_, lon_, bands_path_, radii_ = band_row
                    db.insert_band_features(lat_, lon_, bands_path_, radii_)

                processed += 1
                if processed % 100 == 0:
                    logger.info(
                        "Progress: %d processed, %d skipped, %d errors",
                        processed, skipped, errors,
                    )

    logger.info(
        "Done — processed=%d  skipped=%d  errors=%d", processed, skipped, errors,
    )

    # Clean up empty tmp directory
    try:
        os.rmdir(tmp_dir)
    except OSError:
        pass

    if embedding_cache is not None:
        embedding_cache.close()


if __name__ == "__main__":
    main()

"""Downloader backend configuration and factory.

Loads backend settings from .env / environment variables and CLI args,
then constructs the appropriate downloader instance.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import argparse

logger = logging.getLogger(__name__)


@dataclass
class DownloaderConfig:
    """Picklable config for the download backend.

    Passed through worker_kwargs so spawned worker processes receive it
    without needing to re-parse arguments.
    """
    backend: str = "overpass"          # "overpass" | "postgis" | "auto"
    postgis_url: Optional[str] = None  # DSN e.g. postgresql://osm:osm@localhost:5432/gis
    postgis_connect_timeout: int = 10
    postgis_query_timeout: int = 30    # per-query statement_timeout in seconds
    postgis_max_rows_per_table: int = 50_000


def load_downloader_config(args: "argparse.Namespace") -> DownloaderConfig:
    """Build a DownloaderConfig from argparse namespace + environment/.env.

    Priority: CLI --postgis-url > POSTGIS_URL env var > .env file.
    """
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    postgis_url = getattr(args, "postgis_url", None) or os.environ.get("POSTGIS_URL")
    backend = getattr(args, "backend", "overpass")

    if backend in ("postgis", "auto") and not postgis_url:
        raise ValueError(
            "--backend postgis/auto requires a PostGIS connection string. "
            "Pass --postgis-url or set POSTGIS_URL in the environment / .env file."
        )

    return DownloaderConfig(
        backend=backend,
        postgis_url=postgis_url,
        postgis_max_rows_per_table=getattr(args, "postgis_max_rows_per_table", 0),
    )


def create_downloader(
    config: DownloaderConfig,
    lat: float,
    lon: float,
    dist: float,
    output_file: str,
    save_road_graph: bool = False,
):
    """Instantiate the correct downloader for the given config.

    Returns one of: PostGISDownloader, AutoDownloader, OSMDownloader.
    Imports are lazy so workers using PostGIS never load osmnx.
    """
    if config.backend == "postgis":
        from osmgraphclip.postgis_downloader import PostGISDownloader
        return PostGISDownloader(
            lat=lat, lon=lon, dist=dist,
            output_file=output_file,
            config=config,
            save_road_graph=save_road_graph,
            max_rows_per_table=config.postgis_max_rows_per_table,
        )
    if config.backend == "auto":
        from osmgraphclip.auto_downloader import AutoDownloader
        return AutoDownloader(
            lat=lat, lon=lon, dist=dist,
            output_file=output_file,
            config=config,
            save_road_graph=save_road_graph,
        )
    # default: overpass
    from osmgraphclip.osm_downloader import OSMDownloader
    return OSMDownloader(
        lat=lat, lon=lon, dist=dist,
        output_file=output_file,
        save_road_graph=save_road_graph,
    )

"""AutoDownloader: tries PostGIS first, falls back to Overpass API.

osmnx (and GDAL) are only imported in __call__ when the Overpass fallback is
actually needed, so workers where PostGIS succeeds never load osmnx.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from osmgraphclip.downloader_config import DownloaderConfig

logger = logging.getLogger(__name__)


class AutoDownloader:
    """Try PostGIS; fall back to Overpass if PostGIS returns no data or errors."""

    def __init__(
        self,
        lat: float,
        lon: float,
        dist: float,
        output_file: str,
        config: "DownloaderConfig",
        save_road_graph: bool = False,
    ) -> None:
        from osmgraphclip.postgis_downloader import PostGISDownloader

        self._postgis = PostGISDownloader(
            lat=lat, lon=lon, dist=dist,
            output_file=output_file,
            config=config,
            save_road_graph=False,  # PostGIS never writes road graphs
        )
        # Store args for lazy Overpass construction — osmnx not loaded unless needed.
        self._lat = lat
        self._lon = lon
        self._dist = dist
        self._output_file = output_file
        self._save_road_graph = save_road_graph

    def __call__(self) -> bool:
        try:
            result = self._postgis()
            if result:
                logger.debug(
                    "AutoDownloader: used PostGIS for (%.5f, %.5f)", self._lat, self._lon
                )
                return True
            logger.info(
                "AutoDownloader: PostGIS returned no data for (%.5f, %.5f), "
                "falling back to Overpass", self._lat, self._lon,
            )
        except Exception as exc:
            logger.warning(
                "AutoDownloader: PostGIS failed for (%.5f, %.5f): %s — "
                "falling back to Overpass", self._lat, self._lon, exc,
            )
        # Lazy import: osmnx only loaded here, when the fallback is actually needed.
        from osmgraphclip.osm_downloader import OSMDownloader
        overpass = OSMDownloader(
            lat=self._lat, lon=self._lon, dist=self._dist,
            output_file=self._output_file,
            save_road_graph=self._save_road_graph,
        )
        return overpass()

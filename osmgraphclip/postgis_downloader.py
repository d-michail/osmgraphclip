"""PostGIS-backed downloader.

Queries a local PostGIS database (loaded with osm2pgsql --hstore) instead of
the Overpass API.  Writes the same per-geometry-type .geojson.gz files as
OSMDownloader so the rest of the pipeline is unchanged.

Import requirements:
    osm2pgsql must be invoked with --hstore so that the `tags` hstore column
    is present (needed for the `telecom` tag which is not a native column).

    Example import command:
        osm2pgsql --create --slim --hstore -d <dsn> planet-latest.osm.pbf
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import warnings
from typing import Optional, List, Tuple, TYPE_CHECKING

import geopandas as gpd
from shapely.geometry import shape

# ST_ClipByBox2D can produce empty geometries (bbox overlaps but geometry does not).
# We already filter these with `~gdf.geometry.is_empty`, so suppress the noisy
# GeoPandas deprecation warning that fires during the notna() call.
warnings.filterwarnings("ignore", "GeoSeries.notna", UserWarning)

if TYPE_CHECKING:
    from osmgraphclip.downloader_config import DownloaderConfig

logger = logging.getLogger(__name__)

# ── Bbox conversion ───────────────────────────────────────────────────────────

def _bbox_3857(lat: float, lon: float, dist_m: float) -> Tuple[float, float, float, float]:
    """Return (minx, miny, maxx, maxy) in EPSG:3857 for a square bbox centred
    at (lat, lon) with half-width dist_m metres."""
    from pyproj import Transformer
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    cx, cy = transformer.transform(float(lon), float(lat))
    return cx - dist_m, cy - dist_m, cx + dist_m, cy + dist_m


# ── SQL templates ─────────────────────────────────────────────────────────────
# Native osm2pgsql columns + hstore extractions + geometry.
# `tags` hstore column requires --hstore at import time.

_SELECT_COLS = (
    "osm_id",
    # Original native columns
    "amenity", "building", "highway", "landuse", "railway",
    "public_transport", "waterway", "water",
    "tags->'telecom' AS telecom",
    "leisure", '"natural"', "shop", "tourism",
    # Additional native columns
    "historic", "man_made", "power", "barrier", "aeroway",
    "military", "office", "place", "sport", "religion",
    "boundary", "surface", "service", "denomination",
    # hstore extractions
    "tags->'craft' AS craft",
    "tags->'emergency' AS emergency",
    "tags->'healthcare' AS healthcare",
    # All remaining hstore tags as a JSON blob (captures name, opening_hours, cuisine, etc.)
    "hstore_to_json(tags)::text AS _extra_tags",
    # Geometry column is built dynamically in __call__ to clip to the query bbox.
)

# Geometry column with bbox clipping.  Four %s placeholders for (minx, miny, maxx, maxy) in
# EPSG:3857.  ST_ClipByBox2D clips the stored geometry to the query envelope so that large
# polygons (e.g. national parks) are not returned at their full resolution when only a small
# corner intersects the bbox.
_GEOM_COL = (
    "ST_AsGeoJSON(ST_Transform("
    "ST_ClipByBox2D(way, ST_MakeEnvelope(%s, %s, %s, %s, 3857)), 4326)) AS _geometry"
)

_SQL = "SELECT {cols} FROM {table} WHERE way && ST_MakeEnvelope(%s, %s, %s, %s, 3857)"

_TABLES = {
    "point":   "planet_osm_point",
    "line":    "planet_osm_line",
    "polygon": "planet_osm_polygon",
}

# Tag column names matching _SELECT_COLS order (osm_id excluded — not a tag).
_TAG_COLS = (
    "amenity", "building", "highway", "landuse", "railway",
    "public_transport", "waterway", "water", "telecom",
    "leisure", "natural", "shop", "tourism",
    "historic", "man_made", "power", "barrier", "aeroway",
    "military", "office", "place", "sport", "religion",
    "boundary", "surface", "service", "denomination",
    "craft", "emergency", "healthcare",
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _rows_to_gdf(rows: list) -> Optional[gpd.GeoDataFrame]:
    """Convert psycopg2 rows (osm_id, tag values..., JSON geometry) to a GeoDataFrame.

    Row layout: osm_id, <30 tag cols>, _extra_tags, _geometry  — see _SELECT_COLS.
    osm_id (index 0), _extra_tags (index -2), and _geometry (index -1) are
    handled explicitly; the 30 native tag cols occupy row[1:-2].
    """
    if not rows:
        return None
    features = []
    for row in rows:
        geom_json = row[-1]
        if geom_json is None:
            continue
        try:
            geom = shape(json.loads(geom_json))
        except Exception:
            continue
        # Merge all hstore tags first, then let native columns take precedence.
        extra_json = row[-2]
        extra_tags = json.loads(extra_json) if extra_json else {}
        native_props = dict(zip(_TAG_COLS, row[1:-2]))
        props = {**extra_tags, **native_props}
        features.append({"geometry": geom, **props})
    if not features:
        return None
    return gpd.GeoDataFrame(features, geometry="geometry", crs="EPSG:4326")


def _write_gdf(gdf: gpd.GeoDataFrame, path: str) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write(gdf.to_json(na="drop"))


# ── Downloader ────────────────────────────────────────────────────────────────

class PostGISDownloader:
    """Fetch OSM features from a PostGIS database for a bbox and write .geojson.gz files."""

    def __init__(
        self,
        lat: float,
        lon: float,
        dist: float,
        output_file: str,
        config: "DownloaderConfig",
        save_road_graph: bool = False,
        max_rows_per_table: int = 0,
        skip_on_row_limit: bool = True,
    ) -> None:
        self.lat = lat
        self.lon = lon
        self.dist = dist
        self.output_file = output_file
        self.postgis_url = config.postgis_url
        self.connect_timeout = config.postgis_connect_timeout
        self.query_timeout_ms = config.postgis_query_timeout * 1000
        self.max_rows_per_table = max_rows_per_table  # 0 = unlimited
        self.skip_on_row_limit = skip_on_row_limit
        self.hit_row_limit = False  # set to True when a table exceeds max_rows_per_table
        if save_road_graph:
            logger.debug(
                "PostGISDownloader: save_road_graph is not supported and will be ignored."
            )

    def _query_rows(
        self,
        enforce_row_limit: bool = True,
        sql_row_limit: Optional[int] = None,
    ) -> Optional[List[tuple]]:
        """Connect to PostGIS and return all matching rows, or None on failure.

        sql_row_limit adds a SQL LIMIT clause so PostgreSQL stops after that many
        rows per table, avoiding the full data transfer that made fetchall() hang
        for huge bounding boxes.  When omitted it falls back to max_rows_per_table
        (or no limit when max_rows_per_table is 0).

        When enforce_row_limit is True and a table exceeds max_rows_per_table,
        sets hit_row_limit and returns None.  When False, logs a warning but
        continues accumulating rows from all tables.
        """
        import psycopg2

        minx, miny, maxx, maxy = _bbox_3857(self.lat, self.lon, self.dist)
        bbox_params = (minx, miny, maxx, maxy)
        select_cols = ", ".join(list(_SELECT_COLS) + [_GEOM_COL])
        self.hit_row_limit = False

        # Effective SQL-level cap: stops data transfer early for huge bboxes.
        # Fetch limit+1 so we can detect when the cap was reached.
        _sql_cap = sql_row_limit if sql_row_limit is not None else self.max_rows_per_table
        _sql_limit_clause = f" LIMIT {_sql_cap + 1}" if _sql_cap else ""

        conn = None
        try:
            conn = psycopg2.connect(
                self.postgis_url,
                connect_timeout=self.connect_timeout,
                options=f"-c statement_timeout={self.query_timeout_ms}",
            )
        except psycopg2.OperationalError as exc:
            logger.warning("PostGISDownloader: connection failed: %s", exc)
            return None

        all_rows: List[tuple] = []
        try:
            with conn.cursor() as cur:
                for _label, table in _TABLES.items():
                    sql = _SQL.format(cols=select_cols, table=table) + _sql_limit_clause
                    try:
                        cur.execute(sql, bbox_params + bbox_params)
                        rows = cur.fetchall()
                        logger.debug(
                            "PostGISDownloader (%.5f, %.5f) bbox=%dm: %s returned %d rows",
                            self.lat, self.lon, int(self.dist), table, len(rows),
                        )
                        # Detect row-cap hit: we fetched limit+1 rows, trim to limit.
                        _check_limit = self.max_rows_per_table or _sql_cap
                        if _check_limit and len(rows) > _check_limit:
                            rows = rows[:_check_limit]
                            self.hit_row_limit = True
                            if enforce_row_limit:
                                logger.warning(
                                    "PostGISDownloader (%.5f, %.5f) bbox=%dm: %s hit row limit "
                                    "(%d). Skipping this level.",
                                    self.lat, self.lon, int(self.dist), table, _check_limit,
                                )
                                return None
                            logger.warning(
                                "PostGISDownloader (%.5f, %.5f) bbox=%dm: %s hit row limit "
                                "(%d). Proceeding with capped rows.",
                                self.lat, self.lon, int(self.dist), table, _check_limit,
                            )
                        all_rows.extend(rows)
                    except psycopg2.ProgrammingError as exc:
                        logger.warning(
                            "PostGISDownloader: query error on %s: %s. "
                            "Ensure osm2pgsql was run with --hstore.",
                            table, exc,
                        )
                        conn.rollback()
                        return None
        except Exception as exc:
            logger.exception("PostGISDownloader: unexpected error: %s", exc)
            return None
        finally:
            conn.close()

        logger.debug(
            "PostGISDownloader (%.5f, %.5f) bbox=%dm: %d total rows across all tables",
            self.lat, self.lon, int(self.dist), len(all_rows),
        )
        return all_rows if all_rows else None

    def fetch_gdfs(
        self,
        band_max_rows: int = 0,
    ) -> Tuple[Optional[gpd.GeoDataFrame], Optional[gpd.GeoDataFrame], Optional[gpd.GeoDataFrame]]:
        """Return (polygon_gdf, line_gdf, point_gdf) directly without writing files.

        band_max_rows caps the SQL LIMIT per table (0 = use max_rows_per_table).
        Returns (None, None, None) on failure or empty result.
        """
        sql_limit = band_max_rows if band_max_rows else self.max_rows_per_table
        all_rows = self._query_rows(enforce_row_limit=False, sql_row_limit=sql_limit or None)
        if not all_rows:
            return None, None, None

        gdf = _rows_to_gdf(all_rows)
        if gdf is None or gdf.empty:
            return None, None, None

        by_type: dict = {}
        features_with_geom = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
        for geom_type, subset in features_with_geom.groupby(features_with_geom.geometry.geom_type):
            if not subset.empty:
                by_type[geom_type.lower()] = subset

        polygon_gdf = by_type.get("polygon") if "polygon" in by_type else by_type.get("multipolygon")
        line_gdf    = by_type.get("linestring") if "linestring" in by_type else by_type.get("multilinestring")
        point_gdf   = by_type.get("point")
        return polygon_gdf, line_gdf, point_gdf

    def fetch_gdfs_multi_radii(
        self, radii: List[float], band_max_rows: int = 0,
    ) -> List[Tuple[float, Optional[gpd.GeoDataFrame], Optional[gpd.GeoDataFrame], Optional[gpd.GeoDataFrame]]]:
        """Query once at max(radii) and return spatially-filtered GDFs per radius.

        Opens a single PostGIS connection and executes the 3 table queries once
        at the largest bbox, then partitions the result in Python by centroid
        distance for each smaller radius.  This avoids N-1 redundant round-trips
        when multiple concentric band radii share the same centre.

        band_max_rows caps the SQL LIMIT per table (0 = use max_rows_per_table).
        A SQL-level cap is always applied to prevent fetchall() hanging on huge
        bounding boxes (e.g. regions with large national-park polygons).

        Returns list of (radius, polygon_gdf, line_gdf, point_gdf) in ascending
        radius order.  Falls back to (radius, None, None, None) on failure.
        """
        sorted_radii = sorted(radii)
        max_radius = sorted_radii[-1]
        sql_limit = band_max_rows if band_max_rows else self.max_rows_per_table

        orig_dist = self.dist
        self.dist = max_radius
        try:
            all_rows = self._query_rows(enforce_row_limit=False, sql_row_limit=sql_limit or None)
        finally:
            self.dist = orig_dist

        if not all_rows:
            return [(r, None, None, None) for r in sorted_radii]

        gdf = _rows_to_gdf(all_rows)
        if gdf is None or gdf.empty:
            return [(r, None, None, None) for r in sorted_radii]

        gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
        if gdf.empty:
            return [(r, None, None, None) for r in sorted_radii]

        # Compute centroid distance from the query centre in metres (UTM).
        try:
            center_gdf = gpd.GeoDataFrame(
                geometry=[shape({"type": "Point", "coordinates": [self.lon, self.lat]})],
                crs="EPSG:4326",
            )
            utm_crs = center_gdf.estimate_utm_crs()
        except Exception:
            from pyproj import CRS
            utm_crs = CRS.from_proj4(
                f"+proj=aeqd +lat_0={self.lat} +lon_0={self.lon} +datum=WGS84 +units=m"
            )

        try:
            gdf_utm = gdf.to_crs(utm_crs)
            center_utm = gpd.GeoDataFrame(
                geometry=[shape({"type": "Point", "coordinates": [self.lon, self.lat]})],
                crs="EPSG:4326",
            ).to_crs(utm_crs).geometry[0]
            gdf["_dist_m"] = gdf_utm.geometry.centroid.distance(center_utm).values
        except Exception as exc:
            logger.warning(
                "fetch_gdfs_multi_radii: UTM distance computation failed (%s); "
                "returning all rows for every radius.", exc
            )
            gdf["_dist_m"] = 0.0

        results = []
        for radius in sorted_radii:
            gdf_r = gdf[gdf["_dist_m"] <= radius].drop(columns=["_dist_m"])
            if gdf_r.empty:
                results.append((radius, None, None, None))
                continue

            by_type: dict = {}
            for geom_type, subset in gdf_r.groupby(gdf_r.geometry.geom_type):
                if not subset.empty:
                    by_type[geom_type.lower()] = subset

            polygon_gdf = by_type.get("polygon") if "polygon" in by_type else by_type.get("multipolygon")
            line_gdf    = by_type.get("linestring") if "linestring" in by_type else by_type.get("multilinestring")
            point_gdf   = by_type.get("point")
            results.append((radius, polygon_gdf, line_gdf, point_gdf))

        return results

    def __call__(self) -> bool:
        """Query PostGIS and write per-geometry-type .geojson.gz files.

        Returns True if at least one feature was written, False otherwise.
        Connection is opened and closed within this call (process-pool safe).
        """
        # Remove stale 0-byte files left by a previous crashed run so the resume logic
        # does not mistake them for valid data.
        for _geom_suffix in ("point", "linestring", "polygon", "multipolygon", "multilinestring"):
            _stale = f"{self.output_file}_{_geom_suffix}.geojson.gz"
            if os.path.exists(_stale) and os.path.getsize(_stale) == 0:
                os.remove(_stale)
                logger.warning("PostGISDownloader: removed stale 0-byte file %s", _stale)

        all_rows = self._query_rows(enforce_row_limit=True)
        if not all_rows:
            return False

        any_written = False
        try:
            gdf = _rows_to_gdf(all_rows)
            if gdf is None or gdf.empty:
                return False

            features_with_geom = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
            for geom_type, subset in features_with_geom.groupby(
                features_with_geom.geometry.geom_type
            ):
                if subset.empty:
                    continue
                filename = f"{self.output_file}_{geom_type.lower()}.geojson.gz"
                _write_gdf(subset, filename)
                size_kb = os.path.getsize(filename) / 1024
                logger.debug(
                    "PostGISDownloader: wrote %d %s features → %s (%.1f KB)",
                    len(subset), geom_type, filename, size_kb,
                )
                any_written = True
        except Exception as exc:
            logger.exception("PostGISDownloader: unexpected error: %s", exc)

        return any_written

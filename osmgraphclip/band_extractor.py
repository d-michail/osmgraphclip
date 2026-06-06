"""Multi-scale band feature extractor for OSM GeoDataFrames.

For each radius band (concentric disk) the extractor produces:

Global spatial features (~47 floats)
  - Feature densities (total, by geometry type)
  - Per-category density + fraction (10 semantic categories × 2)
  - Land-use area fractions  (residential / commercial / industrial / natural / recreation)
  - Road-length density by hierarchy (motorway-trunk / primary / secondary / residential / path)
  - Major-road accessibility ratio
  - Distance-to-nearest (min + mean) for 6 key categories

Sub-bin spatial  [2, 16]  — inner ring vs outer ring
  - Category densities × 10, land-use fractions × 5, total feature density

Sector spatial   [4, 11]  — N / E / S / W quadrants
  - Category densities × 10, total feature density

SBERT embeddings (distance-weighted over natural-language feature descriptions)
  - global_embeddings   [sbert_dim]
  - subbin_embeddings   [2, sbert_dim]
  - sector_embeddings   [4, sbert_dim]

.npz file written by to_npz() contains:
  band_radii           : float32[n_bands]
  spatial_features     : float32[n_bands, 47]
  spatial_feature_names: object[47]
  subbin_spatial       : float32[n_bands, 2, 16]
  subbin_feature_names : object[16]
  sector_spatial       : float32[n_bands, 4, 11]
  sector_feature_names : object[11]
  global_embeddings    : float32[n_bands, D]
  subbin_embeddings    : float32[n_bands, 2, D]
  sector_embeddings    : float32[n_bands, 4, D]
"""

from __future__ import annotations

import logging
import math
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import geopandas as gpd
from shapely.geometry import Point

# Centroid on geographic CRS is approximate but acceptable for bearing calculation.
warnings.filterwarnings("ignore", "Geometry is in a geographic CRS", UserWarning)

logger = logging.getLogger(__name__)

# ── Semantic categories (aligned with richness.py) ────────────────────────────

_SEMANTIC_CATEGORIES: Dict[str, list] = {
    "food_drink":        [("amenity", {"restaurant", "cafe", "bar", "fast_food", "pub",
                                       "food_court", "ice_cream", "biergarten", "bbq"})],
    "retail":            [("shop", None)],
    "transport":         [("highway", None), ("railway", None), ("public_transport", None)],
    "education":         [("amenity", {"school", "university", "college", "kindergarten",
                                       "library", "research_institute"})],
    "nature_green":      [("natural", None),
                          ("landuse", {"forest", "meadow", "farmland", "grass", "heath",
                                       "orchard", "vineyard", "scrub"})],
    "leisure":           [("leisure", None)],
    "water":             [("waterway", None), ("water", None)],
    "tourism":           [("tourism", None)],
    "built_environment": [("building", None)],
    "health":            [("amenity", {"hospital", "clinic", "doctors", "pharmacy",
                                       "dentist", "nursing_home", "veterinary"})],
}
_CATEGORY_NAMES = list(_SEMANTIC_CATEGORIES.keys())
_N_CATEGORIES = len(_CATEGORY_NAMES)

# ── Road-class hierarchy ──────────────────────────────────────────────────────

_ROAD_CLASS_MAP: Dict[str, str] = {
    "motorway": "motorway_trunk", "motorway_link": "motorway_trunk",
    "trunk": "motorway_trunk", "trunk_link": "motorway_trunk",
    "primary": "primary", "primary_link": "primary",
    "secondary": "secondary", "secondary_link": "secondary",
    "tertiary": "secondary", "tertiary_link": "secondary",
    "residential": "residential", "living_street": "residential",
    "service": "residential", "unclassified": "residential",
    "footway": "path", "path": "path", "cycleway": "path",
    "pedestrian": "path", "steps": "path", "track": "path",
}
_ROAD_CLASSES = ["motorway_trunk", "primary", "secondary", "residential", "path"]
_MAJOR_ROAD_CLASSES = {"motorway_trunk", "primary"}

# ── Land-use classes ──────────────────────────────────────────────────────────

_LANDUSE_RESIDENTIAL = {"residential", "housing", "apartments"}
_LANDUSE_COMMERCIAL   = {"commercial", "retail", "office"}
_LANDUSE_INDUSTRIAL   = {"industrial", "port", "quarry"}
_LANDUSE_NATURAL      = {"forest", "farmland", "meadow", "grass", "heath",
                          "orchard", "vineyard", "scrub", "village_green"}
_LANDUSE_RECREATION   = {"recreation_ground", "cemetery", "greenfield"}
_BUILDING_RESIDENTIAL = {"residential", "house", "apartments", "detached",
                          "semidetached_house", "terrace"}
_BUILDING_COMMERCIAL  = {"commercial", "retail", "office", "shop"}
_BUILDING_INDUSTRIAL  = {"industrial", "warehouse", "factory"}
_LEISURE_RECREATION   = {"park", "garden", "playground", "nature_reserve", "golf_course"}

_LU_CLASSES = ["residential", "commercial", "industrial", "natural", "recreation"]

# ── Natural-language description templates ────────────────────────────────────

_DESCRIPTION_TEMPLATES = {
    "amenity":          "a {value}",
    "shop":             "a {value} shop",
    "building":         "a {value} building",
    "highway":          "a {value}",
    "landuse":          "a {value} area",
    "natural":          "natural {value}",
    "leisure":          "a {value}",
    "tourism":          "a {value}",
    "waterway":         "a {value}",
    "water":            "a {value} water body",
    "railway":          "a {value}",
    "public_transport": "a {value} stop",
    "man_made":         "a {value}",
    "historic":         "a historic {value}",
    "office":           "a {value} office",
    "sport":            "a {value} sports facility",
    "healthcare":       "a {value}",
    "emergency":        "a {value}",
    "craft":            "a {value} workshop",
    "military":         "a military {value}",
    "power":            "a power {value}",
    "aeroway":          "an aviation {value}",
    "barrier":          "a {value} barrier",
    "place":            "a {value}",
    "telecom":          "a telecommunications facility",
}
# Tag columns to check for descriptions, in priority order
_DESC_PRIORITY = [
    "amenity", "shop", "tourism", "leisure", "historic", "office", "sport",
    "healthcare", "emergency", "craft", "public_transport", "railway",
    "waterway", "water", "natural", "landuse", "highway", "building",
    "man_made", "power", "aeroway", "barrier", "military", "place", "telecom",
]

# Categories for which distance-to-nearest is computed
_DIST_CATEGORIES = ["food_drink", "retail", "transport", "education", "leisure", "health"]

# ── Feature name lists ────────────────────────────────────────────────────────

GLOBAL_FEATURE_NAMES: List[str] = (
    ["total_density", "point_density", "line_density", "polygon_density"]
    + [f"cat_{c}_density" for c in _CATEGORY_NAMES]
    + [f"cat_{c}_frac"    for c in _CATEGORY_NAMES]
    + [f"lu_{lu}_frac"    for lu in _LU_CLASSES]
    + [f"road_{rc}_density" for rc in _ROAD_CLASSES]
    + ["road_major_ratio"]
    + [f"dist_min_{c}_m" for c in _DIST_CATEGORIES]
    + [f"dist_mean_{c}_m" for c in _DIST_CATEGORIES]
)

SUBBIN_FEATURE_NAMES: List[str] = (
    [f"cat_{c}_density" for c in _CATEGORY_NAMES]
    + [f"lu_{lu}_frac"  for lu in _LU_CLASSES]
    + ["feature_density"]
)

SECTOR_FEATURE_NAMES: List[str] = (
    [f"cat_{c}_density" for c in _CATEGORY_NAMES]
    + ["feature_density"]
)

SECTOR_NAMES = ["N", "E", "S", "W"]
SUBBIN_NAMES = ["inner", "outer"]

_N_GLOBAL  = len(GLOBAL_FEATURE_NAMES)
_N_SUBBIN  = len(SUBBIN_FEATURE_NAMES)
_N_SECTOR  = len(SECTOR_FEATURE_NAMES)


# ── Low-level helpers ─────────────────────────────────────────────────────────

def _is_missing(val) -> bool:
    if val is None:
        return True
    if isinstance(val, float) and math.isnan(val):
        return True
    s = str(val).strip()
    return s == "" or s.lower() in ("nan", "none")


def _clean(val: str) -> str:
    return val.strip().lower().replace("_", " ")


def _describe_row(row) -> str:
    """Produce a natural-language description for one OSM feature row."""
    for tag in _DESC_PRIORITY:
        if tag not in row.index:
            continue
        val = row[tag]
        if _is_missing(val):
            continue
        v = _clean(str(val))
        template = _DESCRIPTION_TEMPLATES.get(tag, f"a {{value}} {tag}")
        return template.format(value=v)
    return "an unclassified feature"


def _assign_category(row) -> Optional[str]:
    """Return the first matching semantic category for a feature row, or None."""
    for cat, conditions in _SEMANTIC_CATEGORIES.items():
        for key, allowed in conditions:
            if key not in row.index:
                continue
            val = row[key]
            if _is_missing(val):
                continue
            if allowed is None:
                return cat
            if str(val).strip() in allowed:
                return cat
    return None


def _classify_road(highway_val: str) -> Optional[str]:
    return _ROAD_CLASS_MAP.get(str(highway_val).strip().lower())


def _classify_landuse(row) -> Optional[str]:
    """Assign a land-use class to a polygon row."""
    lu = row.get("landuse", None) if hasattr(row, "get") else None
    if lu is None and hasattr(row, "__getitem__"):
        try: lu = row["landuse"]
        except (KeyError, TypeError): pass

    nat = None
    leisure = None
    building = None
    for attr in ("natural", "leisure", "building"):
        try:
            v = row[attr] if hasattr(row, "__getitem__") else getattr(row, attr, None)
        except (KeyError, TypeError):
            v = None
        if attr == "natural":   nat = v
        elif attr == "leisure": leisure = v
        else:                   building = v

    if not _is_missing(lu):
        s = str(lu).strip().lower()
        if s in _LANDUSE_RESIDENTIAL: return "residential"
        if s in _LANDUSE_COMMERCIAL:  return "commercial"
        if s in _LANDUSE_INDUSTRIAL:  return "industrial"
        if s in _LANDUSE_NATURAL:     return "natural"
        if s in _LANDUSE_RECREATION:  return "recreation"

    if not _is_missing(nat):
        return "natural"

    if not _is_missing(leisure):
        if str(leisure).strip().lower() in _LEISURE_RECREATION:
            return "recreation"

    if not _is_missing(building):
        s = str(building).strip().lower()
        if s in _BUILDING_RESIDENTIAL: return "residential"
        if s in _BUILDING_COMMERCIAL:  return "commercial"
        if s in _BUILDING_INDUSTRIAL:  return "industrial"

    return None


def _haversine_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Bearing in degrees (0 = N, 90 = E, 180 = S, 270 = W)."""
    dlon = math.radians(lon2 - lon1)
    r1 = math.radians(lat1)
    r2 = math.radians(lat2)
    x = math.sin(dlon) * math.cos(r2)
    y = math.cos(r1) * math.sin(r2) - math.sin(r1) * math.cos(r2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def _haversine_bearing_vec(lat1: float, lon1: float,
                           lats2: np.ndarray, lons2: np.ndarray) -> np.ndarray:
    """Vectorized bearing from a single centre to an array of points."""
    dlon = np.radians(lons2 - lon1)
    r1 = math.radians(lat1)
    r2 = np.radians(lats2)
    x = np.sin(dlon) * np.cos(r2)
    y = math.cos(r1) * np.sin(r2) - math.sin(r1) * np.cos(r2) * np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


def _sector_index(bearing: float) -> int:
    """Map bearing to sector index: 0=N, 1=E, 2=S, 3=W."""
    return int((bearing + 45) % 360 // 90)


def _dist_weight(d: np.ndarray, radius_m: float) -> np.ndarray:
    """Distance-decay weights: exp(-3·d/radius), so boundary ≈ 0.05."""
    return np.exp(-3.0 * d / max(radius_m, 1.0))


def _area_m2(radius_m: float) -> float:
    return math.pi * radius_m ** 2


def _weighted_mean_embedding(
    embeddings: np.ndarray, weights: np.ndarray, sbert_dim: int
) -> np.ndarray:
    """Weighted average of embedding rows. Returns zero vector if no rows."""
    if len(embeddings) == 0 or weights.sum() < 1e-12:
        return np.zeros(sbert_dim, dtype=np.float32)
    w = weights / weights.sum()
    return (embeddings * w[:, None]).sum(axis=0).astype(np.float32)


# ── Feature extraction helpers ────────────────────────────────────────────────

def _utm_crs(lat: float, lon: float):
    center = gpd.GeoDataFrame(geometry=[Point(lon, lat)], crs="EPSG:4326")
    try:
        return center.estimate_utm_crs()
    except Exception:
        # Fallback for locations outside UTM coverage (polar regions, edge cases):
        # azimuthal equidistant centred on the point preserves distances globally.
        from pyproj import CRS
        return CRS.from_proj4(
            f"+proj=aeqd +lat_0={lat} +lon_0={lon} +datum=WGS84 +units=m"
        )


def _project_to_utm(gdf: gpd.GeoDataFrame, utm_crs) -> Optional[gpd.GeoDataFrame]:
    if gdf is None or len(gdf) == 0:
        return None
    try:
        return gdf.to_crs(utm_crs)
    except Exception as exc:
        logger.debug("UTM projection failed: %s", exc)
        return None


def _extract_all_features(
    polygon_gdf, line_gdf, point_gdf, utm_crs, center_utm: Point, radius_m: float
) -> Tuple[
    np.ndarray,   # distances  [N]
    np.ndarray,   # bearings   [N]
    List[str],    # categories [N] (may be None stored as "")
    List[str],    # road_classes [N] (may be "")
    List[str],    # landuse_classes [N] (polygon only, else "")
    np.ndarray,   # line_lengths [N]  (non-zero for lines)
    np.ndarray,   # poly_areas [N]    (non-zero for polygons)
    List[str],    # descriptions [N]
    np.ndarray,   # geom_types [N]  0=polygon,1=line,2=point
]:
    """Flatten all features into parallel arrays with per-feature attributes.

    Geometry operations (centroid distances, lengths, areas, bearings) are
    fully vectorised via GeoDataFrame bulk methods.  Tag-based operations
    (_describe_row, _assign_category, _classify_landuse) use DataFrame.apply
    which is still faster than iterrows because it avoids per-row Series
    construction overhead and the costly repeated gdf_utm.loc[idx] lookups.
    """
    all_distances: List[np.ndarray] = []
    all_bearings:  List[np.ndarray] = []
    all_categories:   List[str] = []
    all_road_classes: List[str] = []
    all_lu_classes:   List[str] = []
    all_line_lengths: List[np.ndarray] = []
    all_poly_areas:   List[np.ndarray] = []
    all_descriptions: List[str] = []
    all_geom_types:   List[np.ndarray] = []

    def _process_gdf(gdf, geom_type_idx: int) -> None:
        if gdf is None or len(gdf) == 0:
            return

        # Drop rows with null / empty geometries up-front.
        valid = gdf.geometry.notna() & ~gdf.geometry.is_empty
        if not valid.all():
            gdf = gdf[valid]
        if len(gdf) == 0:
            return

        n = len(gdf)

        # ── Vectorised geometry operations ────────────────────────────────────
        gdf_utm = _project_to_utm(gdf, utm_crs)

        if gdf_utm is not None:
            centroids_utm = gdf_utm.geometry.centroid
            dist_arr = centroids_utm.distance(center_utm).to_numpy(dtype=np.float32)
        else:
            dist_arr = np.zeros(n, dtype=np.float32)

        # Bearing: preserve original coordinate-system convention (center_utm.y / .x)
        # so that sector assignments are identical to the pre-vectorisation code.
        centroids_wgs = gdf.geometry.centroid
        bearing_arr = _haversine_bearing_vec(
            center_utm.y, center_utm.x,
            centroids_wgs.y.to_numpy(), centroids_wgs.x.to_numpy(),
        ).astype(np.float32)

        if gdf_utm is not None and geom_type_idx == 1:
            length_arr = gdf_utm.geometry.length.fillna(0.0).to_numpy(dtype=np.float32)
        else:
            length_arr = np.zeros(n, dtype=np.float32)

        if gdf_utm is not None and geom_type_idx == 0:
            area_arr = gdf_utm.geometry.area.fillna(0.0).to_numpy(dtype=np.float32)
        else:
            area_arr = np.zeros(n, dtype=np.float32)

        # ── Tag-based operations (apply avoids per-row pandas overhead) ───────
        desc_list = gdf.apply(_describe_row, axis=1).tolist()
        cat_list  = gdf.apply(lambda r: _assign_category(r) or "", axis=1).tolist()

        if "highway" in gdf.columns:
            hw_series = gdf["highway"]
            road_list = [
                (_classify_road(str(hw)) if hw and not _is_missing(hw) else "")
                for hw in hw_series
            ]
        else:
            road_list = [""] * n

        if geom_type_idx == 0:
            lu_list = gdf.apply(_classify_landuse, axis=1).fillna("").tolist()
        else:
            lu_list = [""] * n

        all_distances.append(dist_arr)
        all_bearings.append(bearing_arr)
        all_descriptions.extend(desc_list)
        all_categories.extend(cat_list)
        all_road_classes.extend(road_list)
        all_lu_classes.extend(lu_list)
        all_line_lengths.append(length_arr)
        all_poly_areas.append(area_arr)
        all_geom_types.append(np.full(n, geom_type_idx, dtype=np.int8))

    _process_gdf(polygon_gdf, 0)
    _process_gdf(line_gdf, 1)
    _process_gdf(point_gdf, 2)

    def _concat(arrays: List[np.ndarray], dtype) -> np.ndarray:
        return np.concatenate(arrays, dtype=dtype) if arrays else np.array([], dtype=dtype)

    return (
        _concat(all_distances,  np.float32),
        _concat(all_bearings,   np.float32),
        all_categories,
        all_road_classes,
        all_lu_classes,
        _concat(all_line_lengths, np.float32),
        _concat(all_poly_areas,   np.float32),
        all_descriptions,
        _concat(all_geom_types,   np.int8),
    )


def _compute_category_densities(
    categories: List[str], mask: np.ndarray, area_m2: float
) -> np.ndarray:
    """Category densities [n_categories] per km² for features selected by mask."""
    area_km2 = max(area_m2 / 1e6, 1e-9)
    cats = [categories[i] for i in range(len(categories)) if mask[i]]
    counts = np.array(
        [cats.count(c) for c in _CATEGORY_NAMES], dtype=np.float32
    )
    return counts / area_km2


def _compute_category_fractions(
    categories: List[str], mask: np.ndarray
) -> np.ndarray:
    """Category fractions [n_categories] as proportion of total features in mask."""
    cats = [categories[i] for i in range(len(categories)) if mask[i]]
    total = max(len(cats), 1)
    return np.array([cats.count(c) / total for c in _CATEGORY_NAMES], dtype=np.float32)


def _compute_lu_fractions(
    lu_classes: List[str],
    poly_areas: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Land-use area fractions [5] within features selected by mask."""
    total_area = poly_areas[mask].sum()
    if total_area < 1.0:
        return np.zeros(len(_LU_CLASSES), dtype=np.float32)
    fracs = []
    for lu in _LU_CLASSES:
        lu_mask = mask & np.array([c == lu for c in lu_classes], dtype=bool)
        fracs.append(float(poly_areas[lu_mask].sum()) / float(total_area))
    return np.array(fracs, dtype=np.float32)


def _compute_road_features(
    road_classes: List[str],
    line_lengths: np.ndarray,
    mask: np.ndarray,
    area_m2: float,
) -> Tuple[np.ndarray, float]:
    """Road-length densities [5] (km/km²) and major-road ratio."""
    area_km2 = max(area_m2 / 1e6, 1e-9)
    densities = []
    total_len = 0.0
    major_len = 0.0
    for rc in _ROAD_CLASSES:
        rc_mask = mask & np.array([c == rc for c in road_classes], dtype=bool)
        length_m = float(line_lengths[rc_mask].sum())
        densities.append(length_m / 1000.0 / area_km2)
        total_len += length_m
        if rc in _MAJOR_ROAD_CLASSES:
            major_len += length_m
    major_ratio = major_len / max(total_len, 1e-3)
    return np.array(densities, dtype=np.float32), float(major_ratio)


def _compute_distance_to_nearest(
    categories: List[str], distances: np.ndarray, mask: np.ndarray, radius_m: float
) -> np.ndarray:
    """Min and mean distance-to-nearest for each of _DIST_CATEGORIES.

    Uses radius_m as sentinel when a category has no features in the band.
    """
    cats_arr = np.array(categories)
    feats = distances[mask]
    cats_masked = cats_arr[mask]

    result = []
    for cat in _DIST_CATEGORIES:
        cat_dists = feats[cats_masked == cat]
        if len(cat_dists) == 0:
            result.append(float(radius_m))   # min
            result.append(float(radius_m))   # mean
        else:
            result.append(float(cat_dists.min()))
            result.append(float(cat_dists.mean()))
    return np.array(result, dtype=np.float32)


# ── Main extractor class ──────────────────────────────────────────────────────

class BandExtractor:
    """Extracts rich spatial + semantic features for multi-scale radius bands.

    A single instance is safe to share across threads: the SBERT model's
    encode() is stateless; EmbeddingCache is protected by its own lock.
    """

    def __init__(
        self,
        sbert_model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
        cache=None,
    ) -> None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading SBERT model '%s' on device '%s'", sbert_model_name, device)
        self._model = SentenceTransformer(sbert_model_name, device=device)
        self._model_name = sbert_model_name
        self._cache = cache
        self._sbert_dim: int = self._model.get_sentence_embedding_dimension()

    def _batch_encode(self, texts: List[str]) -> Dict[str, np.ndarray]:
        """Encode a list of unique texts, using cache where possible."""
        result: Dict[str, np.ndarray] = {}
        to_encode: List[str] = []
        for t in texts:
            if self._cache is not None:
                cached = self._cache.get(t)
                if cached is not None:
                    result[t] = cached
                    continue
            if t not in result:
                to_encode.append(t)

        if to_encode:
            embeddings = self._model.encode(
                to_encode, convert_to_numpy=True, show_progress_bar=False
            ).astype(np.float32)
            for text, emb in zip(to_encode, embeddings):
                result[text] = emb
                if self._cache is not None:
                    self._cache.put(text, emb)

        return result

    def extract_band(
        self,
        lat: float,
        lon: float,
        polygon_gdf: Optional[gpd.GeoDataFrame],
        line_gdf: Optional[gpd.GeoDataFrame],
        point_gdf: Optional[gpd.GeoDataFrame],
        radius_m: float,
        precomputed_embs: Optional[Dict[str, np.ndarray]] = None,
    ) -> dict:
        """Extract all features for one concentric-disk band.

        Returns a dict with keys:
          spatial          : float32[_N_GLOBAL]        — global spatial features
          subbin_spatial   : float32[2, _N_SUBBIN]     — inner / outer ring
          sector_spatial   : float32[4, _N_SECTOR]     — N / E / S / W
          global_embedding : float32[sbert_dim]
          subbin_embeddings: float32[2, sbert_dim]
          sector_embeddings: float32[4, sbert_dim]
        """
        sbert_dim = self._sbert_dim

        # Prepare UTM coordinate system centred on the band centre
        utm_crs = _utm_crs(lat, lon)
        center_utm_gdf = gpd.GeoDataFrame(
            geometry=[Point(lon, lat)], crs="EPSG:4326"
        ).to_crs(utm_crs)
        center_utm = center_utm_gdf.geometry[0]

        # --- Flatten all features into parallel arrays -----------------------
        (
            distances, bearings,
            categories, road_classes, lu_classes,
            line_lengths, poly_areas,
            descriptions, geom_types,
        ) = _extract_all_features(
            polygon_gdf, line_gdf, point_gdf,
            utm_crs, center_utm, radius_m,
        )

        n_feats = len(distances)
        disk_area = _area_m2(radius_m)
        area_km2  = disk_area / 1e6
        all_mask  = np.ones(n_feats, dtype=bool)

        # --- SBERT: batch-encode all unique descriptions ---------------------
        unique_descs = list(dict.fromkeys(descriptions))  # dedup, preserve order
        if precomputed_embs is not None:
            # Use caller-supplied map; fall back to on-the-fly for any missing key.
            missing = [d for d in unique_descs if d not in precomputed_embs]
            desc_emb_map = {**precomputed_embs, **self._batch_encode(missing)} if missing else precomputed_embs
        else:
            desc_emb_map = self._batch_encode(unique_descs) if unique_descs else {}
        if n_feats > 0:
            feat_embeddings = np.stack(
                [desc_emb_map.get(d, np.zeros(sbert_dim, np.float32)) for d in descriptions]
            )  # [n_feats, sbert_dim]
            weights_all = _dist_weight(distances, radius_m)
        else:
            feat_embeddings = np.zeros((0, sbert_dim), np.float32)
            weights_all = np.zeros(0, np.float32)

        # ── Global spatial features ──────────────────────────────────────────
        total_density = n_feats / max(area_km2, 1e-9)
        n_pts  = int((geom_types == 2).sum())
        n_lns  = int((geom_types == 1).sum())
        n_poly = int((geom_types == 0).sum())

        cat_densities  = _compute_category_densities(categories, all_mask, disk_area)
        cat_fractions  = _compute_category_fractions(categories, all_mask)
        lu_fractions   = _compute_lu_fractions(lu_classes, poly_areas, all_mask)
        road_densities, major_ratio = _compute_road_features(
            road_classes, line_lengths, all_mask, disk_area
        )
        dist_nearest = _compute_distance_to_nearest(
            categories, distances, all_mask, radius_m
        )

        global_spatial = np.concatenate([
            [total_density,
             n_pts  / max(area_km2, 1e-9),
             n_lns  / max(area_km2, 1e-9),
             n_poly / max(area_km2, 1e-9)],
            cat_densities,
            cat_fractions,
            lu_fractions,
            road_densities,
            [major_ratio],
            dist_nearest,
        ]).astype(np.float32)

        global_embedding = _weighted_mean_embedding(feat_embeddings, weights_all, sbert_dim)

        # ── Sub-bin features (inner / outer ring) ────────────────────────────
        half_r = radius_m / 2.0
        inner_mask = distances < half_r
        outer_mask = ~inner_mask
        inner_area = _area_m2(half_r)
        outer_area  = disk_area - inner_area

        subbin_spatial   = np.zeros((2, _N_SUBBIN), dtype=np.float32)
        subbin_embeddings = np.zeros((2, sbert_dim), dtype=np.float32)

        for bin_idx, (bmask, barea) in enumerate([
            (inner_mask, inner_area),
            (outer_mask, outer_area),
        ]):
            if bmask.sum() == 0:
                continue
            sb_cat_d = _compute_category_densities(categories, bmask, barea)
            sb_lu_f  = _compute_lu_fractions(lu_classes, poly_areas, bmask)
            sb_density = bmask.sum() / max(barea / 1e6, 1e-9)
            subbin_spatial[bin_idx] = np.concatenate([sb_cat_d, sb_lu_f, [sb_density]])

            bweights = _dist_weight(distances[bmask], radius_m)
            subbin_embeddings[bin_idx] = _weighted_mean_embedding(
                feat_embeddings[bmask], bweights, sbert_dim
            )

        # ── Sector features (N / E / S / W) ──────────────────────────────────
        sector_indices = np.array([_sector_index(b) for b in bearings], dtype=np.int8)
        sector_spatial   = np.zeros((4, _N_SECTOR), dtype=np.float32)
        sector_embeddings = np.zeros((4, sbert_dim), dtype=np.float32)
        sector_area = disk_area / 4.0

        for s_idx in range(4):
            smask = sector_indices == s_idx
            if smask.sum() == 0:
                continue
            s_cat_d   = _compute_category_densities(categories, smask, sector_area)
            s_density = smask.sum() / max(sector_area / 1e6, 1e-9)
            sector_spatial[s_idx] = np.concatenate([s_cat_d, [s_density]])

            sweights = _dist_weight(distances[smask], radius_m)
            sector_embeddings[s_idx] = _weighted_mean_embedding(
                feat_embeddings[smask], sweights, sbert_dim
            )

        return {
            "spatial":           global_spatial,
            "subbin_spatial":    subbin_spatial,
            "sector_spatial":    sector_spatial,
            "global_embedding":  global_embedding,
            "subbin_embeddings": subbin_embeddings,
            "sector_embeddings": sector_embeddings,
        }

    def extract_all_bands(
        self,
        lat: float,
        lon: float,
        bands_data: List[Tuple[float, Optional[gpd.GeoDataFrame], Optional[gpd.GeoDataFrame], Optional[gpd.GeoDataFrame]]],
    ) -> Dict[float, dict]:
        """Extract features for all bands.

        Args:
            lat, lon    : Center coordinate.
            bands_data  : List of (radius_m, polygon_gdf, line_gdf, point_gdf).

        Returns:
            Dict mapping radius_m → band feature dict.

        SBERT descriptions are deduplicated across all bands and encoded in a
        single batch call, which is cheaper than encoding each band separately.
        """
        # ── Collect all unique descriptions across every band ─────────────────
        all_unique_descs: List[str] = []
        seen: set = set()
        for _, p_gdf, l_gdf, pt_gdf in bands_data:
            for gdf in (p_gdf, l_gdf, pt_gdf):
                if gdf is None or len(gdf) == 0:
                    continue
                valid = gdf.geometry.notna() & ~gdf.geometry.is_empty
                gdf_v = gdf[valid] if not valid.all() else gdf
                for _, row in gdf_v.iterrows():
                    d = _describe_row(row)
                    if d not in seen:
                        seen.add(d)
                        all_unique_descs.append(d)

        precomputed_embs = self._batch_encode(all_unique_descs) if all_unique_descs else {}

        results: Dict[float, dict] = {}
        for radius, p_gdf, l_gdf, pt_gdf in bands_data:
            try:
                results[radius] = self.extract_band(
                    lat, lon, p_gdf, l_gdf, pt_gdf, radius,
                    precomputed_embs=precomputed_embs,
                )
            except Exception as exc:
                logger.warning("Band extraction failed for radius=%.0fm: %s", radius, exc)
                results[radius] = self._empty_band()
        return results

    def _empty_band(self) -> dict:
        d = self._sbert_dim
        return {
            "spatial":           np.zeros(_N_GLOBAL,  dtype=np.float32),
            "subbin_spatial":    np.zeros((2, _N_SUBBIN), dtype=np.float32),
            "sector_spatial":    np.zeros((4, _N_SECTOR), dtype=np.float32),
            "global_embedding":  np.zeros(d, dtype=np.float32),
            "subbin_embeddings": np.zeros((2, d), dtype=np.float32),
            "sector_embeddings": np.zeros((4, d), dtype=np.float32),
        }

    def to_npz(self, bands_result: Dict[float, dict], output_path: str) -> None:
        """Save band features to a compressed .npz file.

        Arrays:
          band_radii           : float32[n_bands]
          spatial_features     : float32[n_bands, 47]
          spatial_feature_names: object[47]
          subbin_spatial       : float32[n_bands, 2, 16]
          subbin_feature_names : object[16]
          sector_spatial       : float32[n_bands, 4, 11]
          sector_feature_names : object[11]
          global_embeddings    : float32[n_bands, D]
          subbin_embeddings    : float32[n_bands, 2, D]
          sector_embeddings    : float32[n_bands, 4, D]
        """
        radii = sorted(bands_result.keys())
        B = [bands_result[r] for r in radii]
        np.savez_compressed(
            output_path,
            band_radii            = np.array(radii, dtype=np.float32),
            spatial_features      = np.stack([b["spatial"]           for b in B]).astype(np.float32),
            spatial_feature_names = np.array(GLOBAL_FEATURE_NAMES, dtype=object),
            subbin_spatial        = np.stack([b["subbin_spatial"]    for b in B]).astype(np.float32),
            subbin_feature_names  = np.array(SUBBIN_FEATURE_NAMES, dtype=object),
            sector_spatial        = np.stack([b["sector_spatial"]    for b in B]).astype(np.float32),
            sector_feature_names  = np.array(SECTOR_FEATURE_NAMES, dtype=object),
            global_embeddings     = np.stack([b["global_embedding"]  for b in B]).astype(np.float32),
            subbin_embeddings     = np.stack([b["subbin_embeddings"] for b in B]).astype(np.float32),
            sector_embeddings     = np.stack([b["sector_embeddings"] for b in B]).astype(np.float32),
        )

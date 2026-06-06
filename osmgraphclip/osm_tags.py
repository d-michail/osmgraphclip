"""
Tag-related constants and pure-geopandas helpers shared between the dataset
creation path (no torch) and the graph building path.
"""

import gzip
import io
import math
import os
import re

import geopandas as gpd

_LANG_VARIANT_RE = re.compile(r'^(.+):[a-z]{2,3}$')

OSM_TAG_COLUMNS = (
    # Original 13
    "amenity",
    "building",
    "highway",
    "landuse",
    "railway",
    "public_transport",
    "waterway",
    "water",
    "telecom",
    "leisure",
    "natural",
    "shop",
    "tourism",
    # Native osm2pgsql columns
    "historic",
    "man_made",
    "power",
    "barrier",
    "aeroway",
    "military",
    "office",
    "place",
    "sport",
    "religion",
    "boundary",
    "surface",
    "service",
    "denomination",
    # hstore extractions (require --hstore at osm2pgsql import)
    "craft",
    "emergency",
    "healthcare",
)


_NON_TAG_COLS = frozenset({"geometry", "label", "_geometry", "osm_id", "id"})


def _is_missing_value(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    text = str(value).strip()
    return text == "" or text.lower() == "nan"


def _lang_variant_base(tag: str) -> str | None:
    """Return the base tag if this tag is a language variant, else None.
    E.g. 'name:en' -> 'name', 'official_name:fr' -> 'official_name', 'amenity' -> None.
    """
    m = _LANG_VARIANT_RE.match(tag)
    return m.group(1) if m else None


def _build_preferred_lang_tags(row) -> dict:
    """
    For every group of language-variant tags in this row, pick one preferred tag.
    Priority: {base}:en > {base} (plain) > first non-missing variant.
    Returns a dict mapping base_tag -> preferred_tag_key.
    """
    groups: dict[str, list[str]] = {}
    for tag in row.index:
        base = _lang_variant_base(tag)
        if base is not None:
            groups.setdefault(base, []).append(tag)

    preferred = {}
    for base, variants in groups.items():
        en_tag = f'{base}:en'
        if en_tag in variants and not _is_missing_value(row[en_tag]):
            preferred[base] = en_tag
            continue
        if base in row.index and not _is_missing_value(row[base]):
            preferred[base] = base
            continue
        for variant in variants:
            if not _is_missing_value(row[variant]):
                preferred[base] = variant
                break

    return preferred


def _label_from_row(row) -> str:
    preferred = _build_preferred_lang_tags(row)

    parts = []
    for tag in row.index:
        if tag in _NON_TAG_COLS:
            continue

        base = _lang_variant_base(tag)
        if base is not None:
            # Skip language variants that aren't the preferred one for their group
            if preferred.get(base) != tag:
                continue
        else:
            # Skip plain base tag if a language variant was preferred over it
            if tag in preferred and preferred[tag] != tag:
                continue

        raw_value = row[tag]
        if _is_missing_value(raw_value):
            continue

        if isinstance(raw_value, bool):
            if raw_value:
                parts.append(tag)
            continue

        value_text = str(raw_value).strip()
        if value_text.lower() in ("true", "yes", "1"):
            parts.append(tag)
        else:
            parts.append(f"{tag}:{value_text}")

    if not parts:
        return "unknown:unknown"

    return ";".join(parts)


def _ensure_label_column(gdf: gpd.GeoDataFrame = None) -> gpd.GeoDataFrame:
    if gdf is None or len(gdf) == 0:
        return gdf

    gdf = gdf.copy()
    generated_labels = gdf.apply(_label_from_row, axis=1)

    if "label" not in gdf.columns:
        gdf["label"] = generated_labels
        return gdf

    missing_mask = gdf["label"].apply(_is_missing_value)
    gdf.loc[missing_mask, "label"] = generated_labels[missing_mask]
    return gdf


def load_geojson_to_gdf(geojson_path: str, geom_type: str = None) -> gpd.GeoDataFrame:
    """
    Load GeoJSON (or gzip-compressed GeoJSON) file into a GeoDataFrame.

    Args:
        geojson_path: Path to GeoJSON or .geojson.gz file
        geom_type: Filter by geometry type ('Point', 'LineString', 'Polygon')

    Returns:
        GeoDataFrame
    """
    os.environ.setdefault("OGR_GEOJSON_MAX_OBJ_SIZE", "0")
    if geojson_path.endswith(".gz"):
        # Bypass GDAL's /vsigzip/ virtual filesystem, which caches file metadata
        # and produces decompression errors when the same path is overwritten
        # (e.g. during adaptive bbox expansion in create_dataset.py).
        with gzip.open(geojson_path, "rb") as f:
            buf = io.BytesIO(f.read())
        gdf = gpd.read_file(buf, engine="pyogrio")
    else:
        gdf = gpd.read_file(geojson_path, engine="pyogrio")

    if geom_type:
        gdf = gdf[gdf.geometry.geom_type == geom_type]

    return gdf

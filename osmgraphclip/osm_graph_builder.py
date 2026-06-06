"""
OSM graph builder integration utilities for OSMGraphCLIP.
"""

import os
import threading
from typing import Dict, Tuple

import geopandas as gpd
from torch_geometric.data import HeteroData

from .graph_builder_base import BaseOSMGraphBuilder
from .osm_to_graph import OSM2Graph
from .osm_tags import (
    OSM_TAG_COLUMNS,
    _is_missing_value,
    _label_from_row,
    _ensure_label_column,
    load_geojson_to_gdf,
)

_BUILDER_REGISTRY: Dict[str, type] = {
    'geolink': OSM2Graph,
}

_BUILDER_CACHE: Dict[Tuple, BaseOSMGraphBuilder] = {}
_BUILDER_CACHE_LOCK = threading.Lock()


def _get_builder(method: str, tagw_path: str, device: str,
                 embedding_backend: str = 'clip',
                 embedding_model: str = None,
                 embedding_cache=None) -> BaseOSMGraphBuilder:
    cache_key = (method, os.path.abspath(tagw_path), device,
                 embedding_backend, embedding_model or '')
    if cache_key not in _BUILDER_CACHE:
        with _BUILDER_CACHE_LOCK:
            if cache_key not in _BUILDER_CACHE:
                cls = _BUILDER_REGISTRY[method]
                _BUILDER_CACHE[cache_key] = cls(
                    tagw_path, device, embedding_backend,
                    embedding_model, embedding_cache=embedding_cache,
                )
    return _BUILDER_CACHE[cache_key]


def build_zero_graph(node_embedding_dim: int):
    """Return a HeteroData graph with one all-zero node per type and no edges.

    Used as a stand-in when fine-grain OSM data is unavailable so the training
    pipeline can still use the band features for the location.
    """
    import torch
    from torch_geometric.data import HeteroData
    data = HeteroData()
    data["point"].x   = torch.zeros(1, node_embedding_dim + 2)
    data["line"].x    = torch.zeros(1, node_embedding_dim + 6)
    data["polygon"].x = torch.zeros(1, node_embedding_dim + 8)
    data.is_zero_graph = torch.tensor([True])
    return data


def calculate_bounds(polygon_gdf: gpd.GeoDataFrame = None,
                     line_gdf: gpd.GeoDataFrame = None,
                     point_gdf: gpd.GeoDataFrame = None) -> tuple:
    """Calculate bounding box from GeoDataFrames.

    Returns:
        Tuple (north, south, east, west)
    """
    all_geoms = []
    if polygon_gdf is not None and len(polygon_gdf) > 0:
        all_geoms.extend(polygon_gdf.geometry)
    if line_gdf is not None and len(line_gdf) > 0:
        all_geoms.extend(line_gdf.geometry)
    if point_gdf is not None and len(point_gdf) > 0:
        all_geoms.extend(point_gdf.geometry)

    if all_geoms:
        bounds_geom = gpd.GeoSeries(all_geoms).total_bounds
        west, south, east, north = bounds_geom
        return (north, south, east, west)
    else:
        return (1, 0, 1, 0)


def build_osm_graph(polygon_gdf: gpd.GeoDataFrame = None,
                    line_gdf: gpd.GeoDataFrame = None,
                    point_gdf: gpd.GeoDataFrame = None,
                    tagw_path: str = None,
                    device: str = 'cuda',
                    bounds: tuple = None,
                    embedding_backend: str = 'clip',
                    embedding_model: str = None,
                    embedding_cache=None,
                    method: str = 'geolink') -> HeteroData:
    """Build an OSM heterogeneous graph using the selected construction method.

    Args:
        polygon_gdf: GeoDataFrame with polygons
        line_gdf: GeoDataFrame with lines
        point_gdf: GeoDataFrame with points
        tagw_path: Path to tag weights JSON file (required)
        device: 'cuda' or 'cpu'
        bounds: Optional tuple (north, south, east, west). Computed if not given.
        embedding_backend: 'clip' or 'sbert'
        embedding_model: Optional model name/path override.
        embedding_cache: Optional persistent embedding cache instance.
        method: Graph construction method — 'geolink' (default).

    Returns:
        HeteroData graph
    """
    if tagw_path is None:
        raise ValueError("tagw_path is required for graph construction")

    if method not in _BUILDER_REGISTRY:
        raise ValueError(
            f"Unknown graph method {method!r}. Choose from: {sorted(_BUILDER_REGISTRY)}"
        )

    polygon_gdf = _ensure_label_column(polygon_gdf)
    line_gdf = _ensure_label_column(line_gdf)
    point_gdf = _ensure_label_column(point_gdf)

    if bounds is None:
        bounds = calculate_bounds(polygon_gdf, line_gdf, point_gdf)

    north, south, east, west = bounds

    builder = _get_builder(method, tagw_path, device, embedding_backend,
                           embedding_model, embedding_cache=embedding_cache)
    return builder.process(polygon_gdf, line_gdf, point_gdf, north, south, east, west)

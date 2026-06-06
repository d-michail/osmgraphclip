from .utils import *
import logging
import torch
from scipy.spatial import Delaunay
from torch_geometric.data import HeteroData
from sklearn.preprocessing import OneHotEncoder
from .graph_builder_base import BaseOSMGraphBuilder, CLIP_DEFAULT_MODEL, SBERT_DEFAULT_MODEL

logger = logging.getLogger(__name__)


def _make_dense_onehot_encoder():
    try:
        return OneHotEncoder(sparse_output=False)
    except TypeError:
        return OneHotEncoder(sparse=False)


def _normalize_polygon_geometry(geometry):
    if geometry is None:
        return None

    geom_type = getattr(geometry, "geom_type", "")
    if geom_type == "Polygon":
        return geometry

    if geom_type == "MultiPolygon":
        polygons = [poly for poly in geometry.geoms if getattr(poly, "geom_type", "") == "Polygon"]
        if polygons:
            return max(polygons, key=lambda poly: poly.area)

    if hasattr(geometry, "convex_hull"):
        hull = geometry.convex_hull
        if getattr(hull, "geom_type", "") == "Polygon":
            return hull

    return None


def _normalize_line_geometry(geometry):
    if geometry is None:
        return None
    geom_type = getattr(geometry, "geom_type", "")
    if geom_type == "LineString":
        return geometry
    if geom_type == "MultiLineString":
        lines = [g for g in geometry.geoms if getattr(g, "geom_type", "") == "LineString"]
        if lines:
            return max(lines, key=lambda g: g.length)
    return None


def _normalize_point_geometry(geometry):
    if geometry is None:
        return None
    geom_type = getattr(geometry, "geom_type", "")
    if geom_type == "Point":
        return geometry
    if geom_type == "MultiPoint":
        pts = list(geometry.geoms)
        if pts:
            return pts[0]
    return None


class OSM2Graph(BaseOSMGraphBuilder):
    def __init__(self, tagw_path, device='cuda', embedding_backend='clip', embedding_model=None,
                 embedding_cache=None):
        super().__init__(tagw_path, device, embedding_backend, embedding_model, embedding_cache)

        # polygon to polygon: touches 2, overlaps 3, covers 4, coverby 5, equals 6
        self.p2p_onehot = _make_dense_onehot_encoder().fit([[0], [2], [3], [4], [5], [6]])
        # line to line: touches 2, overlaps 3, covers 4, coverby 5, equals 6
        self.l2l_onehot = _make_dense_onehot_encoder().fit([[0], [2], [3], [4], [5], [6]])
        # line to polygon: touches 2, cross 3, coverby 4
        self.l2p_onehot = _make_dense_onehot_encoder().fit([[2], [3], [4]])
        # polygon to line: touches 2, cross 3, cover 4
        self.p2l_onehot = _make_dense_onehot_encoder().fit([[2], [3], [4]])
        # point to other: touches 2, within 3
        self.po2o_onehot = _make_dense_onehot_encoder().fit([[2], [3]])
        # other to point: touches 2, contain 3
        self.o2po_onehot = _make_dense_onehot_encoder().fit([[2], [3]])


    def process(self, polygon_file, line_file, point_file, north, south, east, west):
        if polygon_file is None and line_file is None and point_file is None:
            print('no valid osm data')
            return None

        width = east - west
        height = north - south
        # Prevent division-by-zero when all features share the same lon or lat
        if width == 0:
            width = 1e-8
        if height == 0:
            height = 1e-8
        data = HeteroData()

        if polygon_file is not None and polygon_file.shape[0] > 0:

            sentences = np.array(list(polygon_file['label']))
            semantic_embeddings = self._build_semantic_embeddings(sentences) # n*512

            polygons = np.array(list(polygon_file['geometry']))
            polygon_centers = []
            for p in polygons:
                polygon = _normalize_polygon_geometry(p)
                if polygon is None:
                    center = p.centroid
                    sampled_centers = [center, center, center, center]
                else:
                    sampled_centers = generate_points_around_center(polygon, num_points=3)
                sampled_centers = [[(c.coords[0][0] - west) / width,
                                    1 - (c.coords[0][1] - south) / height] for c in sampled_centers]
                polygon_centers.append(sampled_centers)
            polygon_centers = torch.from_numpy(np.array(polygon_centers).reshape(-1, 8)) # n*8

            node_features = torch.cat(
                [semantic_embeddings, polygon_centers], dim=1).float()
            data['polygon'].x = node_features # n*524

            # self 0, disjoint 1, toches 2, overlaps 3, covers 4, coverby 5, equals 6, other 7
            r_m = get_spatial_relation_samedim(polygons)
            # 获得边的特征
            edge_o, edge_d, edge_w = [], [], []
            for i in range(r_m.shape[0]):
                for j in range(r_m.shape[1]):
                    if r_m[i, j] != 7 and r_m[i, j] != 1:
                        edge_o.append(i)
                        edge_d.append(j)
                        edge_w.append([r_m[i, j]])
            edge = torch.tensor([edge_o, edge_d]).long()
            if len(edge_w) > 0:
                edge_w = torch.from_numpy(self.p2p_onehot.transform(edge_w)).float()
            else:
                edge_w = torch.empty(0, 6).float()
            data['polygon', 'to', 'polygon'].edge_index = edge
            data['polygon', 'to', 'polygon'].edge_attr = edge_w


        else:
            data['polygon'].x = torch.empty(0, self.embedding_dim + 8, dtype=torch.float)
            data['polygon', 'to', 'polygon'].edge_index = torch.empty(2, 0, dtype=torch.long)
            data['polygon', 'to', 'polygon'].edge_attr = torch.empty(0, 6, dtype=torch.float)
            data['polygon', 'to', 'point'].edge_index = torch.empty(2, 0, dtype=torch.long)
            data['polygon', 'to', 'point'].edge_attr = torch.empty(0, 2, dtype=torch.float)
            data['point', 'to', 'polygon'].edge_index = torch.empty(2, 0, dtype=torch.long)
            data['point', 'to', 'polygon'].edge_attr = torch.empty(0, 2, dtype=torch.float)
            data['line', 'to', 'polygon'].edge_index = torch.empty(2, 0, dtype=torch.long)
            data['line', 'to', 'polygon'].edge_attr = torch.empty(0, 3, dtype=torch.float)
            data['polygon', 'to', 'line'].edge_index = torch.empty(2, 0, dtype=torch.long)
            data['polygon', 'to', 'line'].edge_attr = torch.empty(0, 3, dtype=torch.float)

        if line_file is not None and line_file.shape[0] > 0:
            sentences = np.array(list(line_file['label']))
            semantic_embeddings = self._build_semantic_embeddings(sentences)  # n*512

            lines = np.array(list(line_file['geometry']))
            line_centers = []
            for raw_p in lines:
                p = _normalize_line_geometry(raw_p)
                centroid = raw_p.centroid
                if p is not None:
                    start = p.coords[0]
                    end = p.coords[-1]
                else:
                    start = centroid.coords[0]
                    end = centroid.coords[0]
                centers = [
                    (centroid.coords[0][0] - west) / width,
                    1 - (centroid.coords[0][1] - south) / height,
                    (start[0] - west) / width,
                    1 - (start[1] - south) / height,
                    (end[0] - west) / width,
                    1 - (end[1] - south) / height,
                ]
                line_centers.append(centers)
            line_centers = torch.from_numpy(np.array(line_centers).reshape(-1, 6))  # n*6

            # print(semantic_embeddings.shape, line_centers.shape, line_envs.shape, max_tag_labels.shape)
            node_features = torch.cat(
                [semantic_embeddings, line_centers], dim=1).float() # n*522
            data['line'].x = node_features

            # line to line
            # self 0, disjoint 1, toches 2, overlaps 3, covers 4, coverby 5, equals 6, other 7
            r_m = get_spatial_relation_samedim(lines)
            # 获得边的特征
            edge_o, edge_d, edge_w = [], [], []
            for i in range(r_m.shape[0]):
                for j in range(r_m.shape[1]):
                    if r_m[i, j] != 7 and r_m[i, j] != 1:
                        edge_o.append(i)
                        edge_d.append(j)
                        edge_w.append([r_m[i, j]])
            edge = torch.tensor([edge_o, edge_d]).long()
            if len(edge_w) > 0:
                edge_w = torch.from_numpy(self.l2l_onehot.transform(edge_w)).float()
            else:
                edge_w = torch.empty(0, 6).float()
            data['line', 'to', 'line'].edge_index = edge
            data['line', 'to', 'line'].edge_attr = edge_w

            if polygon_file is not None and polygon_file.shape[0] > 0:
                r_m = get_spatial_relation_line2polygon(lines, polygons)
                # disjoint 1, toches 2, cross 3, coverby 4, other 0
                edge_o, edge_d, edge_w = [], [], []
                for i in range(r_m.shape[0]):
                    for j in range(r_m.shape[1]):
                        if r_m[i, j] != 1 and r_m[i, j] != 0:
                            edge_o.append(i)
                            edge_d.append(j)
                            edge_w.append([r_m[i, j]])
                edge = torch.tensor([edge_o, edge_d]).long()
                if len(edge_w) > 0:
                    edge_w = torch.from_numpy(self.l2p_onehot.transform(edge_w)).float()
                else:
                    edge_w = torch.empty(0, 3).float()
                data['line', 'to', 'polygon'].edge_index = edge
                data['line', 'to', 'polygon'].edge_attr = edge_w

                # disjoint 1, toches 2, cross 3, cover 4, other 0
                edge_o, edge_d, edge_w = [], [], []
                for j in range(r_m.shape[1]):
                    for i in range(r_m.shape[0]):
                        if r_m[i, j] != 1 and r_m[i, j] != 0:
                            edge_o.append(j)
                            edge_d.append(i)
                            edge_w.append([r_m[i, j]])
                edge = torch.tensor([edge_o, edge_d]).long()
                if len(edge_w) > 0:
                    edge_w = torch.from_numpy(self.p2l_onehot.transform(edge_w)).float()
                else:
                    edge_w = torch.empty(0, 3).float()
                data['polygon', 'to', 'line'].edge_index = edge
                data['polygon', 'to', 'line'].edge_attr = edge_w

        else:
            data['line'].x = torch.empty(0, self.embedding_dim + 6, dtype=torch.float)
            data['line', 'to', 'line'].edge_index = torch.empty(2, 0, dtype=torch.long)
            data['line', 'to', 'line'].edge_attr = torch.empty(0, 6, dtype=torch.float)
            data['line', 'to', 'point'].edge_index = torch.empty(2, 0, dtype=torch.long)
            data['point', 'to', 'line'].edge_index = torch.empty(2, 0, dtype=torch.long)
            data['point', 'to', 'line'].edge_attr = torch.empty(0, 2, dtype=torch.float)
            data['line', 'to', 'point'].edge_attr = torch.empty(0, 2, dtype=torch.float)
            data['line', 'to', 'polygon'].edge_index = torch.empty(2, 0, dtype=torch.long)
            data['line', 'to', 'polygon'].edge_attr = torch.empty(0, 3, dtype=torch.float)
            data['polygon', 'to', 'line'].edge_index = torch.empty(2, 0, dtype=torch.long)
            data['polygon', 'to', 'line'].edge_attr = torch.empty(0, 3, dtype=torch.float)

        if point_file is not None and point_file.shape[0] > 0:

            sentences = np.array(list(point_file['label']))
            semantic_embeddings = self._build_semantic_embeddings(sentences)  # n*512

            points = np.array(list(point_file['geometry']))
            point_centers = []
            for raw_p in points:
                p = _normalize_point_geometry(raw_p)
                coord = p.coords[0] if p is not None else raw_p.centroid.coords[0]
                centers = [(coord[0] - west) / width,
                           1 - (coord[1] - south) / height]
                point_centers.append(centers)
            point_centers = torch.from_numpy(np.array(point_centers).reshape(-1, 2))  # n*2

            node_features = torch.cat(
                [semantic_embeddings, point_centers], dim=1).float()  # n*514
            data['point'].x = node_features

            # point to point
            edge = []
            edge_w = []
            if points.shape[0] > 3:
                try:
                    delaunay = Delaunay([
                        (_normalize_point_geometry(points[i]) or points[i].centroid).coords[0]
                        for i in range(len(points))
                    ]).simplices.copy()
                    for tri in delaunay:
                        edge.append([tri[0], tri[1]])
                        edge.append([tri[1], tri[2]])
                        edge.append([tri[0], tri[2]])
                        edge.append([tri[1], tri[0]])
                        edge.append([tri[2], tri[1]])
                        edge.append([tri[2], tri[0]])
                    edge = np.unique(edge, axis=0).tolist()
                    for i in range(len(edge)):
                        edge_w.append([0, 1])
                except:
                    print('cannot delaunay, point num: ', points.shape[0])

                for i in range(points.shape[0]):
                    edge.append([i, i])
                    edge_w.append([1, 0])
                edge = np.array(edge)
                edge = edge.transpose()
                edge = torch.tensor(edge, dtype=torch.long)
                edge_w = torch.tensor(edge_w).float()
            else:
                for i in range(points.shape[0]):
                    edge.append([i, i])
                    edge_w.append([1, 0])

                edge = np.array(edge)
                edge = edge.transpose()
                edge = torch.tensor(edge, dtype=torch.long)
                edge_w = torch.tensor(edge_w).float()
            data['point', 'to', 'point'].edge_index = edge
            data['point', 'to', 'point'].edge_attr = edge_w

            # point to line and line to point
            if line_file is not None and line_file.shape[0] > 0:
                r_m = get_spatial_relation_point2other(points, lines)
                # disjoint 1, toches 2,  within 3, other 0
                edge_o, edge_d, edge_w = [], [], []
                for i in range(r_m.shape[0]):
                    for j in range(r_m.shape[1]):
                        if r_m[i, j] != 1 and r_m[i, j] != 0:
                            edge_o.append(i)
                            edge_d.append(j)
                            edge_w.append([r_m[i, j]])
                edge = torch.tensor([edge_o, edge_d]).long()
                if len(edge_w) > 0:
                    edge_w = torch.from_numpy(self.po2o_onehot.transform(edge_w)).float()
                else:
                    edge_w = torch.empty(0, 2).float()
                data['point', 'to', 'line'].edge_index = edge
                data['point', 'to', 'line'].edge_attr = edge_w

                # disjoint 1, toches 2,  contain 3, other 0
                edge_o, edge_d, edge_w = [], [], []
                for j in range(r_m.shape[1]):
                    for i in range(r_m.shape[0]):
                        if r_m[i, j] != 1 and r_m[i, j] != 0:
                            edge_o.append(j)
                            edge_d.append(i)
                            edge_w.append([r_m[i, j]])
                edge = torch.tensor([edge_o, edge_d]).long()
                if len(edge_w) > 0:
                    edge_w = torch.from_numpy(self.o2po_onehot.transform(edge_w)).float()
                else:
                    edge_w = torch.empty(0, 2).float()
                data['line', 'to', 'point'].edge_index = edge
                data['line', 'to', 'point'].edge_attr = edge_w

            # point to polygon and polygon to point
            if polygon_file is not None and polygon_file.shape[0] > 0:
                r_m = get_spatial_relation_point2other(points, polygons)
                # disjoint 1, toches 2,  within 3, other 0
                edge_o, edge_d, edge_w = [], [], []
                for i in range(r_m.shape[0]):
                    for j in range(r_m.shape[1]):
                        if r_m[i, j] != 1 and r_m[i, j] != 0:
                            edge_o.append(i)
                            edge_d.append(j)
                            edge_w.append([r_m[i, j]])
                edge = torch.tensor([edge_o, edge_d]).long()
                if len(edge_w) > 0:
                    edge_w = torch.from_numpy(self.po2o_onehot.transform(edge_w)).float()
                else:
                    edge_w = torch.empty(0, 2).float()
                data['point', 'to', 'polygon'].edge_index = edge
                data['point', 'to', 'polygon'].edge_attr = edge_w

                # disjoint 1, toches 2,  contain 3, other 0
                edge_o, edge_d, edge_w = [], [], []
                for j in range(r_m.shape[1]):
                    for i in range(r_m.shape[0]):
                        if r_m[i, j] != 1 and r_m[i, j] != 0:
                            edge_o.append(j)
                            edge_d.append(i)
                            edge_w.append([r_m[i, j]])
                edge = torch.tensor([edge_o, edge_d]).long()
                if len(edge_w) > 0:
                    edge_w = torch.from_numpy(self.o2po_onehot.transform(edge_w)).float()
                else:
                    edge_w = torch.empty(0, 2).float()
                data['polygon', 'to', 'point'].edge_index = edge
                data['polygon', 'to', 'point'].edge_attr = edge_w

        else:
            data['point'].x = torch.empty(0, self.embedding_dim + 2, dtype=torch.float)
            data['point', 'to', 'point'].edge_index = torch.empty(2, 0, dtype=torch.long)
            data['point', 'to', 'point'].edge_attr = torch.empty(0, 2, dtype=torch.float)
            data['line', 'to', 'point'].edge_index = torch.empty(2, 0, dtype=torch.long)
            data['point', 'to', 'line'].edge_index = torch.empty(2, 0, dtype=torch.long)
            data['point', 'to', 'line'].edge_attr = torch.empty(0, 2, dtype=torch.float)
            data['line', 'to', 'point'].edge_attr = torch.empty(0, 2, dtype=torch.float)
            data['polygon', 'to', 'point'].edge_index = torch.empty(2, 0, dtype=torch.long)
            data['polygon', 'to', 'point'].edge_attr = torch.empty(0, 2, dtype=torch.float)
            data['point', 'to', 'polygon'].edge_index = torch.empty(2, 0, dtype=torch.long)
            data['point', 'to', 'polygon'].edge_attr = torch.empty(0, 2, dtype=torch.float)

        return data

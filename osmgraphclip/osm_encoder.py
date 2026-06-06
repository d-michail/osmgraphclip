import torch
import torch.nn.functional as F

from torch_geometric.nn import HeteroConv, GATConv


class OSMHeteroGAT(torch.nn.Module):
    def __init__(self, out_chans=128, embedding_dim=512):
        super().__init__()

        # Node feature dims = semantic embedding dim + geometry features per type:
        #   point: 2 (x, y)
        #   line:  6 (centroid_x, centroid_y, start_x, start_y, end_x, end_y)
        #   polygon: 8 (4 sampled interior points × 2 coords)
        point_in_chans = embedding_dim + 2
        line_in_chans = embedding_dim + 6
        polygon_in_chans = embedding_dim + 8

        self.convs = torch.nn.ModuleList()
        conv1 = HeteroConv({
            ('point', 'to', 'point'): GATConv((point_in_chans, point_in_chans), out_chans, add_self_loops=False),
            ('point', 'to', 'line'): GATConv((point_in_chans, line_in_chans), out_chans, add_self_loops=False),
            ('point', 'to', 'polygon'): GATConv((point_in_chans, polygon_in_chans), out_chans, add_self_loops=False),
            ('line', 'to', 'line'): GATConv((line_in_chans, line_in_chans), out_chans, add_self_loops=False),
            ('line', 'to', 'point'): GATConv((line_in_chans, point_in_chans), out_chans, add_self_loops=False),
            ('line', 'to', 'polygon'): GATConv((line_in_chans, polygon_in_chans), out_chans, add_self_loops=False),
            ('polygon', 'to', 'polygon'): GATConv((polygon_in_chans, polygon_in_chans), out_chans, add_self_loops=False),
            ('polygon', 'to', 'point'): GATConv((polygon_in_chans, point_in_chans), out_chans, add_self_loops=False),
            ('polygon', 'to', 'line'): GATConv((polygon_in_chans, line_in_chans), out_chans, add_self_loops=False),
        }, aggr='mean')
        self.convs.append(conv1)

    def forward(self, data):
        x_dict = data.x_dict
        try:
            edge_index_dict = data.edge_index_dict
        except KeyError:
            edge_index_dict = {}
        try:
            edge_attr_dict = data.edge_attr_dict
        except KeyError:
            edge_attr_dict = None

        for conv in self.convs:
            if edge_attr_dict is None:
                x_dict = conv(x_dict, edge_index_dict)
            else:
                x_dict = conv(x_dict, edge_index_dict, edge_attr_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
        return x_dict

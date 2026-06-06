import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import Set2Set

from .osm_encoder import OSMHeteroGAT
from .location_encoder import (
    LocationEncoder,
    get_neural_network,
    get_positional_encoding,
)


class OSMGraphCLIP(nn.Module):
    """
    OSMGraphCLIP: A location encoder combined with an OSM graph encoder.
    
    This model follows the SatCLIP architecture but replaces the visual encoder with
    a heterogeneous OSM graph encoder. It learns joint embeddings of OSM graph data
    and geographic coordinates through contrastive learning.
    
    Args:
        embed_dim: Embedding dimension for both encoders
        graph_out_chans: Output channels of the graph encoder
        graph_aggr_embed_dim: Aggregation embedding dimension for graph outputs
        le_type: Location encoding type (e.g., 'grid', 'fourier')
        pe_type: Positional encoding neural network type (e.g., 'siren', 'mlp')
        frequency_num: Number of frequency bands for positional encoding
        max_radius: Maximum radius for positional encoding
        min_radius: Minimum radius for positional encoding
        harmonics_calculation: Type of harmonics calculation
        legendre_polys: Number of Legendre polynomials
        num_hidden_layers: Number of hidden layers in location encoder MLP
        capacity: Hidden dimension capacity of location encoder
        temperature: Temperature parameter for logit scaling (default: 0.07)
    """
    
    def __init__(self,
                 embed_dim: int,
                 # graph
                 graph_out_chans: int,
                 graph_aggr_embed_dim: int = 128,
                 node_embedding_dim: int = 512,
                 # location
                 le_type: str = "grid",
                 pe_type: str = "siren",
                 frequency_num: int = 16,
                 max_radius: int = 260,
                 min_radius: int = 1,
                 harmonics_calculation: str = "analytic",
                 legendre_polys: int = 16,
                 num_hidden_layers: int = 2,
                 capacity: int = 256,
                 temperature: float = 0.07,
                 *args,
                 **kwargs
                 ):
        super().__init__()

        # OSM Graph Encoder
        self.graph = OSMHeteroGAT(out_chans=graph_out_chans, embedding_dim=node_embedding_dim)
        
        # Graph aggregation: aggregate heterogeneous node embeddings per sample
        # Using Set2Set for each node type, then combining via attention
        self.osm_types = ['point', 'line', 'polygon']
        self.graph_out_chans = graph_out_chans
        self.graph_aggr_embed_dim = graph_aggr_embed_dim
        
        # Set2Set aggregation for each OSM type
        self.osm_aggregation = nn.ModuleDict({
            t: Set2Set(graph_out_chans, processing_steps=5)
            for t in self.osm_types
        })
        
        # MLP to project aggregated outputs to embedding dimension
        self.osm_aggr_mlp = nn.ModuleDict({
            t: nn.Linear(graph_out_chans * 2, graph_aggr_embed_dim)
            for t in self.osm_types
        })
        
        # Attention-based combination of different OSM types
        self.type_attention = nn.Linear(graph_aggr_embed_dim, 1)
        
        # Final projection from aggregated graph to embed_dim
        self.graph_proj = nn.Linear(graph_aggr_embed_dim, embed_dim)
        
        # Location Encoder
        self.posenc = get_positional_encoding(
            name=le_type,
            harmonics_calculation=harmonics_calculation,
            legendre_polys=legendre_polys,
            min_radius=min_radius,
            max_radius=max_radius,
            frequency_num=frequency_num
        ).double()
        
        self.nnet = get_neural_network(
            name=pe_type,
            input_dim=self.posenc.embedding_dim,
            num_classes=embed_dim,
            dim_hidden=capacity,
            num_layers=num_hidden_layers
        ).double()
        
        self.location = LocationEncoder(self.posenc, self.nnet).double()
        
        # Temperature parameter for scaling logits
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))

    def aggregate_graph_features(self, x_dict, osm_data):
        """
        Aggregate heterogeneous graph node embeddings into a single sample embedding.
        
        Args:
            x_dict: Dictionary with keys 'point', 'line', 'polygon' containing node embeddings
            
        Returns:
            Aggregated graph embedding of shape [batch_size, embed_dim]
        """
        aggregated = []
        total_graphs = getattr(osm_data, "num_graphs", None)
        if total_graphs is None:
            total_graphs = 1
        
        for osm_type in self.osm_types:
            if osm_type in x_dict:
                # Get node embeddings for this type
                nodes = x_dict[osm_type]
                node_store = osm_data[osm_type]
                node_batch = getattr(node_store, "batch", None)
                if node_batch is None:
                    node_batch = torch.zeros(nodes.size(0), dtype=torch.long, device=nodes.device)
                
                # Aggregate nodes using Set2Set (handles variable number of nodes)
                if len(nodes) > 0:
                    # node_batch may have sparse/non-contiguous indices (e.g. [0,3,7,...]).
                    # Remap to contiguous [0..k-1] so Set2Set returns exactly k rows,
                    # then scatter those k rows back to the correct positions in a
                    # full [total_graphs, dim] output tensor.
                    unique_graphs, remapped_batch = node_batch.unique(return_inverse=True)
                    agg = self.osm_aggregation[osm_type](nodes, remapped_batch)
                    # agg[i] is the embedding for unique_graphs[i]
                    full_agg = torch.zeros(
                        total_graphs,
                        agg.size(1),
                        dtype=agg.dtype,
                        device=agg.device,
                    )
                    valid = unique_graphs < total_graphs
                    full_agg[unique_graphs[valid]] = agg[valid]
                    agg = full_agg
                    
                    # Project to aggregation embedding dimension
                    agg = self.osm_aggr_mlp[osm_type](agg)
                    aggregated.append(agg)
        
        if not aggregated:
            # Batch contains graphs with no nodes at all (e.g. empty OSM tiles).
            # Return a zero tensor so the batch can still be processed; nan_to_num
            # in forward() will normalise these to near-zero embeddings.
            device = next(self.graph_proj.parameters()).device
            dtype = next(self.graph_proj.parameters()).dtype
            return torch.zeros(total_graphs, self.graph_proj.out_features, device=device, dtype=dtype)
        
        # Stack aggregations from different types: [num_types, total_graphs, graph_aggr_embed_dim]
        aggregated = torch.stack(aggregated, dim=0)
        
        # Compute attention weights for each type
        # aggregated shape: [num_types, total_graphs, graph_aggr_embed_dim]
        type_weights = self.type_attention(aggregated)  # [num_types, total_graphs, 1]
        type_weights = F.softmax(type_weights, dim=0)  # [num_types, total_graphs, 1]
        
        # Weighted combination
        combined = (aggregated * type_weights).sum(dim=0)  # [total_graphs, graph_aggr_embed_dim]
        
        # Project to final embedding dimension
        graph_embed = self.graph_proj(combined)  # [total_graphs, embed_dim]
        
        return graph_embed

    def encode_graph(self, osm_data):
        """
        Encode OSM graph data to a fixed-size embedding.
        
        Args:
            osm_data: PyTorch Geometric HeteroData object with node/edge information
            
        Returns:
            Graph embedding of shape [batch_size, embed_dim]
        """
        x_dict = self.graph(osm_data)
        graph_features = self.aggregate_graph_features(x_dict, osm_data)
        return graph_features

    def encode_location(self, coords):
        """
        Encode geographic coordinates to a fixed-size embedding.
        
        Args:
            coords: Coordinate tensor
            
        Returns:
            Location embedding
        """
        return self.location(coords.double())

    def forward(self, osm, coords):
        """
        Forward pass computing contrastive loss logits.
        
        Args:
            osm: PyTorch Geometric HeteroData object(s)
            coords: Coordinate tensor(s) of shape [batch_size, 2]
            
        Returns:
            logits_per_graph: Similarity logits [batch_size, batch_size]
            logits_per_location: Similarity logits [batch_size, batch_size]
        """
        # Encode OSM graph to fixed-size embedding
        graph_features = self.encode_graph(osm)
        
        # Encode geographic coordinates to fixed-size embedding
        location_features = self.encode_location(coords).float()
        
        # Replace any NaN/Inf that leaked from corrupted graph features (e.g. zero-area
        # bounding boxes causing division by zero in geometry features).
        graph_features = graph_features.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
        location_features = location_features.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)

        # L2 normalization
        graph_features = graph_features / (graph_features.norm(dim=1, keepdim=True) + 1e-8)
        location_features = location_features / (location_features.norm(dim=1, keepdim=True) + 1e-8)
        
        # Compute cosine similarity as logits
        # Clamp logit_scale to [0, log(100)] to prevent divergence (standard CLIP practice)
        logit_scale = self.logit_scale.clamp(max=4.6052).exp()
        logits_per_graph = logit_scale * graph_features @ location_features.t()
        logits_per_location = logits_per_graph.t()

        return logits_per_graph, logits_per_location


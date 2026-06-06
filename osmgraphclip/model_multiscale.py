"""OSMGraphCLIPMultiscale: GAT graph encoder + BandEncoder transformer fused together."""

import torch
import torch.nn as nn
from typing import Optional

from .model import OSMGraphCLIP
from .band_encoder import BandEncoder


class OSMGraphCLIPMultiscale(OSMGraphCLIP):
    """OSMGraphCLIP extended with a BandEncoder for multi-scale band features.

    When band_data is supplied the graph embedding and band embedding are
    concatenated and projected back to embed_dim.  When band_data is None the
    model falls back to the plain graph embedding (identical to OSMGraphCLIP).

    Extra Args:
        band_encoder_d_model: transformer hidden dim for BandEncoder
        band_encoder_n_heads: attention heads
        band_encoder_n_layers: transformer encoder layers
        sbert_dim: SBERT embedding dim used when building the band .npz files
        band_fusion: fusion strategy; only 'concat' is supported for now
    """

    def __init__(
        self,
        embed_dim: int,
        graph_out_chans: int,
        graph_aggr_embed_dim: int = 128,
        node_embedding_dim: int = 512,
        band_encoder_d_model: int = 256,
        band_encoder_n_heads: int = 4,
        band_encoder_n_layers: int = 2,
        sbert_dim: int = 384,
        band_fusion: str = "concat",
        **kwargs,
    ) -> None:
        super().__init__(
            embed_dim=embed_dim,
            graph_out_chans=graph_out_chans,
            graph_aggr_embed_dim=graph_aggr_embed_dim,
            node_embedding_dim=node_embedding_dim,
            **kwargs,
        )

        self.band_encoder = BandEncoder(
            sbert_dim=sbert_dim,
            d_model=band_encoder_d_model,
            n_heads=band_encoder_n_heads,
            n_layers=band_encoder_n_layers,
            embed_dim=embed_dim,
        )

        self.fusion_proj = nn.Linear(embed_dim * 2, embed_dim)
        self.band_fusion = band_fusion

    def encode_graph_and_bands(
        self,
        osm_data,
        band_data: Optional[dict],
    ) -> torch.Tensor:
        """Return fused [B, embed_dim] embedding from graph + optional bands."""
        graph_emb = self.encode_graph(osm_data)

        if band_data is None:
            return graph_emb

        band_emb = self.band_encoder(band_data)
        return self.fusion_proj(torch.cat([graph_emb, band_emb], dim=-1))

    def forward(self, osm, coords, band_data: Optional[dict] = None):
        graph_features = self.encode_graph_and_bands(osm, band_data)
        location_features = self.encode_location(coords).float()

        graph_features = graph_features.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
        location_features = location_features.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)

        graph_features = graph_features / (graph_features.norm(dim=1, keepdim=True) + 1e-8)
        location_features = location_features / (location_features.norm(dim=1, keepdim=True) + 1e-8)

        logit_scale = self.logit_scale.clamp(max=4.6052).exp()
        logits_per_graph = logit_scale * graph_features @ location_features.t()
        return logits_per_graph, logits_per_graph.t()

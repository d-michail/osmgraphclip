"""BandEncoder: transformer over multi-scale band tokens."""

import torch
import torch.nn as nn


class BandEncoder(nn.Module):
    """Encode multi-scale band features via a transformer with a CLS token.

    Token layout (per batch element):
      [CLS] + n_bands global tokens + n_bands*2 sub-bin tokens + n_bands*4 sector tokens
    Total: 1 + n_bands * 7

    Args:
        n_spatial_global: global spatial feature dim per band (default 47)
        n_spatial_subbin: sub-bin spatial feature dim per band (default 16)
        n_spatial_sector: sector spatial feature dim per band (default 11)
        sbert_dim: SBERT embedding dim (default 384)
        d_model: transformer hidden dim
        n_heads: attention heads
        n_layers: transformer encoder layers
        embed_dim: output projection dim
        max_bands: maximum number of bands supported (controls pos_embed table size)
    """

    def __init__(
        self,
        n_spatial_global: int = 47,
        n_spatial_subbin: int = 16,
        n_spatial_sector: int = 11,
        sbert_dim: int = 384,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        embed_dim: int = 256,
        max_bands: int = 8,
    ) -> None:
        super().__init__()

        self.global_proj = nn.Linear(n_spatial_global + sbert_dim, d_model)
        self.subbin_proj = nn.Linear(n_spatial_subbin + sbert_dim, d_model)
        self.sector_proj = nn.Linear(n_spatial_sector + sbert_dim, d_model)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # 1 CLS + max_bands global + max_bands*2 subbin + max_bands*4 sector
        n_pos = 1 + max_bands * 7
        self.pos_embed = nn.Embedding(n_pos, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.out_proj = nn.Linear(d_model, embed_dim)

        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, band_data: dict) -> torch.Tensor:
        """Encode band features to [B, embed_dim].

        Args:
            band_data: dict with float32 tensors:
                spatial_features   [B, n_bands, n_spatial_global]
                global_embeddings  [B, n_bands, sbert_dim]
                subbin_spatial     [B, n_bands, 2, n_spatial_subbin]
                subbin_embeddings  [B, n_bands, 2, sbert_dim]
                sector_spatial     [B, n_bands, 4, n_spatial_sector]
                sector_embeddings  [B, n_bands, 4, sbert_dim]
        """
        B = band_data["spatial_features"].shape[0]
        n_bands = band_data["spatial_features"].shape[1]
        device = band_data["spatial_features"].device

        # ── Global tokens [B, n_bands, d_model] ──────────────────────────────
        global_in = torch.cat(
            [band_data["spatial_features"], band_data["global_embeddings"]], dim=-1
        )
        global_tokens = self.global_proj(global_in)

        # ── Sub-bin tokens [B, n_bands*2, d_model] ───────────────────────────
        B_s, n_b, n_bins, _ = band_data["subbin_spatial"].shape
        subbin_in = torch.cat(
            [band_data["subbin_spatial"], band_data["subbin_embeddings"]], dim=-1
        ).view(B_s, n_b * n_bins, -1)
        subbin_tokens = self.subbin_proj(subbin_in)

        # ── Sector tokens [B, n_bands*4, d_model] ────────────────────────────
        B_sec, n_b2, n_sec, _ = band_data["sector_spatial"].shape
        sector_in = torch.cat(
            [band_data["sector_spatial"], band_data["sector_embeddings"]], dim=-1
        ).view(B_sec, n_b2 * n_sec, -1)
        sector_tokens = self.sector_proj(sector_in)

        # ── Assemble sequence [B, 1 + n_bands*7, d_model] ────────────────────
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, global_tokens, subbin_tokens, sector_tokens], dim=1)

        # ── Positional embeddings ─────────────────────────────────────────────
        n_tokens = tokens.shape[1]
        pos_ids = torch.arange(n_tokens, device=device)
        tokens = tokens + self.pos_embed(pos_ids).unsqueeze(0)

        # ── Transformer ───────────────────────────────────────────────────────
        out = self.transformer(tokens)

        # CLS output → projection
        return self.out_proj(out[:, 0])

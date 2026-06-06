"""Lightning module for OSMGraphCLIPMultiscale."""

import logging
from typing import Optional

import lightning.pytorch
import torch

from .loss import OSMGraphCLIPLoss
from .model_multiscale import OSMGraphCLIPMultiscale

logger = logging.getLogger(__name__)


class OSMGraphCLIPMultiscaleLightningModule(lightning.pytorch.LightningModule):
    """Lightning wrapper for OSMGraphCLIPMultiscale.

    Identical training loop to OSMGraphCLIPLightningModule but:
    - Instantiates OSMGraphCLIPMultiscale (graph + band encoder)
    - Extracts band_data from batch and moves tensors to device
    - Passes band_data to model.forward(); gracefully handles None
    """

    def __init__(
        self,
        embed_dim: int = 256,
        graph_out_chans: int = 256,
        graph_aggr_embed_dim: int = 256,
        node_embedding_dim: int = 512,
        le_type: str = "sphericalharmonics",
        pe_type: str = "siren",
        frequency_num: int = 16,
        max_radius: int = 260,
        min_radius: int = 1,
        legendre_polys: int = 10,
        harmonics_calculation: str = "analytic",
        num_hidden_layers: int = 2,
        capacity: int = 512,
        temperature: float = 0.07,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        band_encoder_d_model: int = 256,
        band_encoder_n_heads: int = 4,
        band_encoder_n_layers: int = 2,
        sbert_dim: int = 384,
        band_fusion: str = "concat",
    ) -> None:
        super().__init__()

        self.model = OSMGraphCLIPMultiscale(
            embed_dim=embed_dim,
            graph_out_chans=graph_out_chans,
            graph_aggr_embed_dim=graph_aggr_embed_dim,
            node_embedding_dim=node_embedding_dim,
            le_type=le_type,
            pe_type=pe_type,
            frequency_num=frequency_num,
            max_radius=max_radius,
            min_radius=min_radius,
            legendre_polys=legendre_polys,
            harmonics_calculation=harmonics_calculation,
            num_hidden_layers=num_hidden_layers,
            capacity=capacity,
            temperature=temperature,
            band_encoder_d_model=band_encoder_d_model,
            band_encoder_n_heads=band_encoder_n_heads,
            band_encoder_n_layers=band_encoder_n_layers,
            sbert_dim=sbert_dim,
            band_fusion=band_fusion,
        )

        self.loss_fun = OSMGraphCLIPLoss()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.save_hyperparameters()

    def _move_band_data(self, band_data: Optional[dict]) -> Optional[dict]:
        if band_data is None:
            return None
        device = self.device
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in band_data.items()}

    def common_step(self, batch, batch_idx):
        osm_data = batch["osm"]
        coords = batch["coords"].float()
        band_data = self._move_band_data(batch.get("bands"))

        logits_per_graph, logits_per_location = self.model(osm_data, coords, band_data)

        # Build valid mask: exclude samples where both graph and bands are zero.
        # A zero graph with non-zero bands still provides useful signal.
        is_zero_graph = getattr(osm_data, "is_zero_graph", None)
        if is_zero_graph is not None and is_zero_graph.any():
            is_zero_graph = is_zero_graph.bool().to(coords.device)
            if band_data is not None:
                # has_bands[i] = True when sample i has at least one non-zero band feature
                sf = band_data["spatial_features"]  # [B, n_bands, n_feats]
                has_bands = sf.abs().flatten(1).sum(dim=1) > 0
            else:
                has_bands = torch.zeros(coords.shape[0], dtype=torch.bool, device=coords.device)

            valid = ~is_zero_graph | has_bands
            if valid.sum() < 2:
                return torch.tensor(0.0, requires_grad=True, device=coords.device)
            logits_per_graph    = logits_per_graph[valid][:, valid]
            logits_per_location = logits_per_location[valid][:, valid]
            coords = coords[valid]

        pos_mask = (coords.unsqueeze(0) == coords.unsqueeze(1)).all(dim=-1)
        return self.loss_fun(logits_per_graph, logits_per_location, pos_mask=pos_mask)

    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        if not torch.isfinite(loss):
            logger.warning(
                "non-finite loss (%s) at batch %d — skipping optimizer step",
                loss.item(), batch_idx,
            )
            return None
        batch_size = int(batch["coords"].shape[0]) if "coords" in batch else 1
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True,
                 logger=True, batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        batch_size = int(batch["coords"].shape[0]) if "coords" in batch else 1
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True,
                 logger=True, batch_size=batch_size)
        return loss

    def configure_optimizers(self):
        exclude = (
            lambda n, p: p.ndim < 2
            or "bn" in n
            or "ln" in n
            or "bias" in n
            or "logit_scale" in n
        )
        include = lambda n, p: not exclude(n, p)

        named_parameters = list(self.model.named_parameters())
        gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
        rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]

        return torch.optim.AdamW(
            [
                {"params": gain_or_bias_params, "weight_decay": 0.0},
                {"params": rest_params, "weight_decay": self.weight_decay},
            ],
            lr=self.learning_rate,
        )

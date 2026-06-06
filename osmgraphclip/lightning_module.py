import logging

import lightning.pytorch
import torch

from .loss import OSMGraphCLIPLoss
from .model import OSMGraphCLIP

logger = logging.getLogger(__name__)

class OSMGraphCLIPLightningModule(lightning.pytorch.LightningModule):
    """
    PyTorch Lightning module for training OSMGraphCLIP.

    Handles training, validation, and optimization configuration with contrastive learning.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        graph_out_chans: int = 128,
        graph_aggr_embed_dim: int = 128,
        node_embedding_dim: int = 512,
        le_type: str = "grid",
        pe_type: str = "siren",
        frequency_num: int = 16,
        max_radius: int = 260,
        min_radius: int = 1,
        legendre_polys: int = 16,
        harmonics_calculation: str = "analytic",
        num_hidden_layers: int = 2,
        capacity: int = 256,
        temperature: float = 0.07,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
    ) -> None:
        super().__init__()

        self.model = OSMGraphCLIP(
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
        )

        self.loss_fun = OSMGraphCLIPLoss()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.save_hyperparameters()

    def forward(self, osm, coords):
        """Forward pass proxy used by Lightning for summary/inference."""
        return self.model(osm, coords)

    def common_step(self, batch, batch_idx):
        """Shared logic for training and validation steps."""
        osm_data = batch["osm"]
        coords = batch["coords"].float()
        logits_per_graph, logits_per_location = self.model(osm_data, coords)

        # Exclude zero-graph samples (no fine-grain OSM data) from the loss.
        is_zero = getattr(osm_data, "is_zero_graph", None)
        if is_zero is not None and is_zero.any():
            valid = ~is_zero.bool().to(coords.device)
            if valid.sum() < 2:
                return torch.tensor(0.0, requires_grad=True, device=coords.device)
            logits_per_graph    = logits_per_graph[valid][:, valid]
            logits_per_location = logits_per_location[valid][:, valid]
            coords = coords[valid]

        pos_mask = (coords.unsqueeze(0) == coords.unsqueeze(1)).all(dim=-1)
        return self.loss_fun(logits_per_graph, logits_per_location, pos_mask=pos_mask)

    def training_step(self, batch, batch_idx):
        """Training step."""
        loss = self.common_step(batch, batch_idx)
        if not torch.isfinite(loss):
            logger.warning(
                "non-finite loss (%s) at batch %d — skipping optimizer step",
                loss.item(), batch_idx,
            )
            return None
        batch_size = int(batch["coords"].shape[0]) if "coords" in batch else 1
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch_size,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        loss = self.common_step(batch, batch_idx)
        batch_size = int(batch["coords"].shape[0]) if "coords" in batch else 1
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch_size,
        )
        return loss

    def configure_optimizers(self):
        """Configure optimizer with weight decay exclusions."""
        exclude = (
            lambda n, p: p.ndim < 2
            or "bn" in n
            or "ln" in n
            or "bias" in n
            or "logit_scale" in n
        )
        include = lambda n, p: not exclude(n, p)

        named_parameters = list(self.model.named_parameters())
        gain_or_bias_params = [
            p for n, p in named_parameters if exclude(n, p) and p.requires_grad
        ]
        rest_params = [
            p for n, p in named_parameters if include(n, p) and p.requires_grad
        ]

        optimizer = torch.optim.AdamW(
            [
                {"params": gain_or_bias_params, "weight_decay": 0.0},
                {
                    "params": rest_params,
                    "weight_decay": self.weight_decay,
                },
            ],
            lr=self.learning_rate,
        )

        return optimizer


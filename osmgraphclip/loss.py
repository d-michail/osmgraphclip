import torch
import torch.nn.functional as F
import torch.nn as nn


class OSMGraphCLIPLoss(nn.Module):
    """Symmetric multi-positive contrastive loss for OSMGraphCLIP.

    When ``pos_mask`` is the identity (no duplicate coordinates in the batch),
    this reduces exactly to the standard CLIP diagonal cross-entropy loss.

    For multi-resolution datasets, several samples share the same (lat, lon).
    Passing a ``pos_mask`` that marks all same-location pairs as positives
    avoids penalising those pairs as negatives (false-negative problem).

    Loss for one direction::

        loss_i = -1/|P(i)| · Σ_{j ∈ P(i)} log_softmax(logits_i)[j]

    where P(i) = { j : pos_mask[i, j] is True }.
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def _loss_one_direction(logits: torch.Tensor, pos_mask: torch.Tensor) -> torch.Tensor:
        """Cross-entropy loss for one direction of the similarity matrix.

        Args:
            logits:   (N, N) temperature-scaled similarity matrix (already scaled).
            pos_mask: (N, N) boolean matrix; True where (i, j) is a positive pair.

        Returns:
            Scalar loss.
        """
        log_probs = F.log_softmax(logits, dim=1)            # (N, N)
        n_pos = pos_mask.sum(dim=1).clamp(min=1).float()    # (N,)
        per_sample = -(log_probs * pos_mask).sum(dim=1) / n_pos  # (N,)
        return per_sample.mean()

    def forward(
        self,
        logits_per_graph: torch.Tensor,
        logits_per_coord: torch.Tensor,
        pos_mask: torch.Tensor | None = None,
        output_dict: bool = False,
    ):
        """Compute symmetric contrastive loss.

        Args:
            logits_per_graph: (N, N) similarity matrix, rows = graphs.
            logits_per_coord: (N, N) similarity matrix, rows = coords.
            pos_mask: optional (N, N) boolean tensor of positive pairs.
                      Defaults to the identity (standard CLIP behaviour).
            output_dict: if True, return a dict instead of a scalar.

        Returns:
            Scalar loss (or dict with key ``"contrastive_loss"``).
        """
        N = logits_per_graph.shape[0]
        if pos_mask is None:
            pos_mask = torch.eye(N, device=logits_per_graph.device, dtype=torch.bool)

        total_loss = (
            self._loss_one_direction(logits_per_graph, pos_mask) +
            self._loss_one_direction(logits_per_coord, pos_mask)
        ) / 2

        return {"contrastive_loss": total_loss} if output_dict else total_loss

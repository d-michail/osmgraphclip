import logging

import lightning.pytorch as pl

logger = logging.getLogger(__name__)


class LoggingCallback(pl.Callback):
    """Logs epoch-level metrics via Python logging so they appear in SLURM output files."""

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        metrics = {k: v for k, v in trainer.callback_metrics.items() if "train" in k}
        logger.info(
            "Epoch %d/%d — %s",
            trainer.current_epoch + 1,
            trainer.max_epochs,
            "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()),
        )

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        metrics = {k: v for k, v in trainer.callback_metrics.items() if "val" in k}
        logger.info(
            "Epoch %d/%d — %s",
            trainer.current_epoch + 1,
            trainer.max_epochs,
            "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()),
        )

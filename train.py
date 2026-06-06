#!/usr/bin/env python3

"""CLI entrypoint for training OSMGraphCLIP with LightningCLI."""

import logging
import re
import sys
from datetime import datetime
from pathlib import Path
import torch
from lightning.pytorch.cli import LightningCLI

_log_level = logging.DEBUG if "--debug" in sys.argv else logging.INFO
logging.basicConfig(level=_log_level)
if "--debug" in sys.argv:
    sys.argv.remove("--debug")

torch.set_float32_matmul_precision('high')


class OSMGraphCLIPLightningCLI(LightningCLI):
    """Extended LightningCLI for additional configuration options."""

    def add_arguments_to_parser(self, parser):
        parser.add_argument("--watchmodel", action="store_true")
        parser.add_argument("--ckpt_path", type=str, default=None)


def cli_main(default_config_filename: str = "./configs/default.yaml"):
    """
    Main training entry point using LightningCLI.

    Args:
        default_config_filename: Path to default configuration YAML file
    """
    save_config_fn = default_config_filename.replace(".yaml", "-latest.yaml")

    cli = OSMGraphCLIPLightningCLI(
        save_config_kwargs=dict(
            config_filename=save_config_fn,
            overwrite=True,
        ),
        parser_kwargs={"default_config_files": [default_config_filename]},
        seed_everything_default=0,
        run=False,
    )

    ckpt_path = getattr(cli.config, "ckpt_path", None)

    # When resuming from a checkpoint, pin each CSVLogger to the same version
    # directory so ModelCheckpoint's dirpath doesn't change and epoch/best-model
    # state is fully restored.
    if ckpt_path is not None:
        m = re.search(r'version_(\d+)', str(ckpt_path))
        if m:
            resume_version = int(m.group(1))
            loggers = cli.trainer.loggers if hasattr(cli.trainer, 'loggers') else []
            if cli.trainer.logger is not None:
                loggers = loggers or [cli.trainer.logger]
            from lightning.pytorch.loggers import CSVLogger
            for logger in loggers:
                if isinstance(logger, CSVLogger) and logger.version != resume_version:
                    logger._version = resume_version

    ts = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    run_name = f"OsmGraphCLIP_{ts}"
    if cli.trainer.logger is not None:
        cli.trainer.logger.experiment.name = run_name
        cli.trainer.logger.log_hyperparams(cli.datamodule.hparams)

    dirname_cfg = Path(default_config_filename).parent
    dir_log_cfg = Path(cli.trainer.log_dir) / dirname_cfg
    dir_log_cfg.mkdir(parents=True, exist_ok=True)

    cli.trainer.fit(
        model=cli.model,
        datamodule=cli.datamodule,
        ckpt_path=ckpt_path,
    )


if __name__ == "__main__":
    config_fn = "./configs/default.yaml"

    if torch.cuda.is_available() and torch.cuda.get_device_name(device=0) == 'NVIDIA A100 80GB PCIe':
        torch.backends.cuda.matmul.allow_tf32 = True
        print('A100 detected: Enabling TF32 for speed! 🚀')
    else:
        torch.backends.cuda.matmul.allow_tf32 = False

    cli_main(config_fn)

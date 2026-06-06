import inspect

import torch

from .lightning_module import OSMGraphCLIPLightningModule
from .lightning_module_multiscale import OSMGraphCLIPMultiscaleLightningModule

HF_ORG = "d-michail"


def _pick_lightning_module_class(hp: dict):
    """Return the correct Lightning module class based on saved hyperparameters."""
    if 'band_encoder_d_model' in hp:
        return OSMGraphCLIPMultiscaleLightningModule
    return OSMGraphCLIPLightningModule


def get_osmgraphclip(ckpt_path, device, return_all=False):
    """Load an OSMGraphCLIP checkpoint similarly to SatCLIP's ``get_satclip`` helper.

    Automatically detects whether the checkpoint is from ``OSMGraphCLIPLightningModule``
    or ``OSMGraphCLIPMultiscaleLightningModule`` based on saved hyperparameters.

    Args:
        ckpt_path: Path to a Lightning checkpoint (.ckpt).
        device: Torch device (e.g. ``"cpu"`` or ``"cuda"``).
        return_all: If True, return full model; otherwise return only the location encoder.

    Returns:
        Either the full model or a ``LocationEncoder``.
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    hp = dict(ckpt["hyper_parameters"])

    module_cls = _pick_lightning_module_class(hp)

    # Keep only constructor arguments supported by the detected Lightning module.
    valid_keys = set(inspect.signature(module_cls.__init__).parameters)
    valid_keys.discard("self")
    hp = {k: v for k, v in hp.items() if k in valid_keys}

    lightning_model = module_cls(**hp).to(device)
    lightning_model.load_state_dict(ckpt["state_dict"])
    lightning_model.eval()

    geo_model = lightning_model.model
    if return_all:
        return geo_model
    return geo_model.location


def get_osmgraphclip_from_hf(model_name: str, device, return_all: bool = False):
    """Load an OSMGraphCLIP checkpoint from HuggingFace Hub.

    Args:
        model_name: HF model name, e.g. ``"osmgraphclip-a-l10"``.
        device: Torch device (e.g. ``"cpu"`` or ``"cuda"``).
        return_all: If True, return full model; otherwise return only the location encoder.

    Returns:
        Either the full model or a ``LocationEncoder``.
    """
    from huggingface_hub import hf_hub_download

    name = model_name.lower()
    ckpt_path = hf_hub_download(repo_id=f"{HF_ORG}/{name}", filename=f"{name}.ckpt")
    return get_osmgraphclip(ckpt_path, device, return_all=return_all)

import torch

from .location_encoder import LocationEncoder, get_neural_network, get_positional_encoding

HF_ORG = "d-michail"


def _extract_nnet_state_dict(state_dict):
    prefixes = (
        "model.nnet.",
        "nnet.",
        "model.location.nnet.",
        "location.nnet.",
    )

    for prefix in prefixes:
        extracted = {
            f"nnet.{k[len(prefix):]}": v
            for k, v in state_dict.items()
            if k.startswith(prefix)
        }
        if extracted:
            return extracted

    # Fallback: any key that contains an ``nnet.`` segment.
    extracted = {}
    for key, value in state_dict.items():
        marker = "nnet."
        idx = key.find(marker)
        if idx >= 0:
            extracted[key[idx:]] = value
    return extracted


def get_osmgraphclip_loc_encoder(ckpt_path, device):
    """Load only location encoder components from an OSMGraphCLIP checkpoint.

    This mirrors SatCLIP's lightweight loader by recreating positional encoding +
    location MLP and loading only ``nnet`` weights from the checkpoint.
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    hp = ckpt["hyper_parameters"]

    posenc = get_positional_encoding(
        hp["le_type"],
        hp["legendre_polys"],
        hp["harmonics_calculation"],
        hp["min_radius"],
        hp["max_radius"],
        hp["frequency_num"],
    )

    nnet = get_neural_network(
        hp["pe_type"],
        posenc.embedding_dim,
        hp["embed_dim"],
        hp["capacity"],
        hp["num_hidden_layers"],
    )

    nnet_state_dict = _extract_nnet_state_dict(ckpt["state_dict"])
    if not nnet_state_dict:
        raise KeyError("Could not find location network (nnet) weights in checkpoint state_dict")

    loc_encoder = LocationEncoder(posenc, nnet).double().to(device)
    loc_encoder.load_state_dict(nnet_state_dict, strict=False)
    loc_encoder.eval()

    return loc_encoder


def get_osmgraphclip_loc_encoder_from_hf(model_name: str, device):
    """Load only the location encoder from HuggingFace Hub.

    Args:
        model_name: HF model name, e.g. ``"osmgraphclip-a-l10"``.
        device: Torch device (e.g. ``"cpu"`` or ``"cuda"``).

    Returns:
        A ``LocationEncoder`` with weights loaded from the checkpoint.
    """
    from huggingface_hub import hf_hub_download

    name = model_name.lower()
    ckpt_path = hf_hub_download(repo_id=f"{HF_ORG}/{name}", filename=f"{name}.ckpt")
    return get_osmgraphclip_loc_encoder(ckpt_path, device)

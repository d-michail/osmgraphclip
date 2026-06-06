#!/usr/bin/env python3
"""Inference script for the OSMGraphCLIP location encoder.

Usage examples:
    # From a local checkpoint
    python infer.py --ckpt path/to/checkpoint.ckpt --lat 52.52 --lon 13.40

    # From HuggingFace Hub
    python infer.py --hf-model osmgraphclip-a-l10 --lat 52.52 --lon 13.40

    # Lightweight loader, output as JSON
    python infer.py --hf-model osmgraphclip-a-l10 --lat 52.52 --lon 13.40 \
        --lightweight --format json

    # Save numpy array to file
    python infer.py --ckpt path/to/checkpoint.ckpt --lat 52.52 --lon 13.40 \
        --format numpy --output embedding.npy

    # Save JSON to file
    python infer.py --ckpt path/to/checkpoint.ckpt --lat 52.52 --lon 13.40 \
        --format json --output embedding.json
"""

import argparse
import json
import sys

import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with the OSMGraphCLIP location encoder.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--ckpt", help="Path to a local Lightning checkpoint (.ckpt).")
    source.add_argument(
        "--hf-model",
        metavar="MODEL_NAME",
        help="HuggingFace model name to download, e.g. 'osmgraphclip-a-l10'.",
    )
    parser.add_argument("--lat", type=float, required=True, help="Latitude in degrees.")
    parser.add_argument("--lon", type=float, required=True, help="Longitude in degrees.")
    parser.add_argument(
        "--lightweight",
        action="store_true",
        help="Use the lightweight loader (reconstructs encoder from hyper-parameters) "
             "instead of the full Lightning loader.",
    )
    parser.add_argument(
        "--format",
        choices=["print", "json", "numpy"],
        default="print",
        help="Output format: 'print' (default), 'json', or 'numpy'.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="File path to save the embedding. If omitted, output is written to stdout.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device override (e.g. 'cpu', 'cuda'). Auto-detected if not set.",
    )
    return parser.parse_args()


def resolve_device(device_arg):
    if device_arg is not None:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_encoder(args, device):
    if args.hf_model:
        if args.lightweight:
            from osmgraphclip.load_lightweight import get_osmgraphclip_loc_encoder_from_hf
            return get_osmgraphclip_loc_encoder_from_hf(args.hf_model, device)
        else:
            from osmgraphclip.load import get_osmgraphclip_from_hf
            return get_osmgraphclip_from_hf(args.hf_model, device).to(device)
    else:
        if args.lightweight:
            from osmgraphclip.load_lightweight import get_osmgraphclip_loc_encoder
            return get_osmgraphclip_loc_encoder(args.ckpt, device)
        else:
            from osmgraphclip.load import get_osmgraphclip
            return get_osmgraphclip(args.ckpt, device).to(device)


def run_inference(encoder, lat, lon, device):
    # Coords are expected as (lon, lat), shape [1, 2], double precision.
    coords = torch.tensor([[lon, lat]], dtype=torch.float64, device=device)
    with torch.no_grad():
        embedding = encoder(coords)
    return embedding.squeeze(0).cpu().numpy()


def output_result(embedding, fmt, output_path):
    if fmt == "numpy":
        if output_path:
            np.save(output_path, embedding)
            print(f"Saved numpy array to {output_path}", file=sys.stderr)
        else:
            np.save(sys.stdout.buffer, embedding)
    elif fmt == "json":
        data = json.dumps(embedding.tolist())
        if output_path:
            with open(output_path, "w") as f:
                f.write(data)
            print(f"Saved JSON to {output_path}", file=sys.stderr)
        else:
            print(data)
    else:  # print
        if output_path:
            with open(output_path, "w") as f:
                f.write(str(embedding.tolist()))
            print(f"Saved to {output_path}", file=sys.stderr)
        else:
            print(embedding.tolist())


def main():
    args = parse_args()
    device = resolve_device(args.device)
    print(f"Using device: {device}", file=sys.stderr)

    encoder = load_encoder(args, device)
    encoder.eval()

    embedding = run_inference(encoder, args.lat, args.lon, device)
    output_result(embedding, args.format, args.output)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Convert an existing OSM graph pickle dataset to LitData streaming format.

Produces two splits (train/ and val/) under --output-dir, ready for
OsmLitDataModule.  Tensor cloning is done here so __getitem__ at training
time is just an unpickle.

Usage:
    python create_litdata_dataset.py \
        --input-dir /path/to/satclip_osm_dataset_sbert384 \
        --output-dir /path/to/satclip_litdata \
        --workers 16
"""

import argparse
import logging
import os
import pickle
import re

import numpy as np
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_BAND_KEYS = (
    "spatial_features",
    "global_embeddings",
    "subbin_spatial",
    "subbin_embeddings",
    "sector_spatial",
    "sector_embeddings",
)

_PICKLE_INDEX_RE = re.compile(r"osm_(\d+)(?:_L\d+)?_graph\.pkl")


def _bands_path_for_pickle(filepath: str) -> str | None:
    """Derive the bands .npz path from a graph pickle path, or None if not found."""
    name = os.path.basename(filepath)
    m = _PICKLE_INDEX_RE.match(name)
    if m is None:
        return None
    idx = m.group(1)
    input_dir = os.path.dirname(os.path.dirname(filepath))  # graphs/../
    bands_path = os.path.join(input_dir, "bands", f"osm_{idx}_bands.npz")
    return bands_path if os.path.exists(bands_path) else None


def process_fn(item):
    """Load one pickle (+ optional band .npz), return serialisable dict."""
    filepath, lon, lat, bands_path = item
    try:
        with open(filepath, "rb") as f:
            graph = pickle.load(f)
    except Exception as exc:
        logger.warning("Skipping %s: %s", filepath, exc)
        return None

    for store in graph.stores:
        for key in list(store.keys()):
            val = store[key]
            if isinstance(val, torch.Tensor):
                store[key] = val.clone()

    out = {
        "coords": np.array([lon, lat], dtype=np.float64),
        "graph":  pickle.dumps(graph),
    }

    if bands_path is not None:
        try:
            npz = np.load(bands_path)
            out["bands"] = {k: npz[k].astype(np.float32) for k in _BAND_KEYS}
        except Exception as exc:
            logger.warning("Could not load bands %s: %s", bands_path, exc)

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir", dest="input_dirs", required=True, nargs="+",
        metavar="DIR",
        help="One or more dataset directories (each must contain dataset.db). "
             "All are merged into a single output dataset.",
    )
    parser.add_argument("--output-dir", required=True, help="LitData output root")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed",        type=int,   default=17)
    parser.add_argument("--workers",     type=int,   default=8)
    parser.add_argument("--chunk-size",  type=int,   default=128, help="Graphs per chunk file")
    parser.add_argument("--resolution-mode", choices=["all", "finest"], default="finest")
    parser.add_argument(
        "--include-bands", action="store_true", default=False,
        help="Pack per-location band .npz files into the LitData chunks (for multiscale training).",
    )
    args = parser.parse_args()

    from litdata import optimize
    from osmgraphclip.dataset_db import DatasetDB
    import pandas as pd

    CHECK_MIN_FILESIZE = 2000

    # ── Collect items from all input directories ──────────────────────────────
    all_dfs = []
    for input_dir in args.input_dirs:
        with DatasetDB(os.path.join(input_dir, "dataset.db")) as db:
            df = db.graphs_as_dataframe()
        df["_input_dir"] = input_dir
        all_dfs.append(df)
        logger.info("Loaded %d rows from %s", len(df), input_dir)

    df = pd.concat(all_dfs, ignore_index=True)
    logger.info("Total rows across all datasets: %d", len(df))

    if args.resolution_mode == "finest" and "level" in df.columns:
        sort_cols = ["level"] + (["bbox_size"] if "bbox_size" in df.columns else [])
        df = (
            df.sort_values(sort_cols)
              .groupby(["lat", "lon"], sort=False)
              .first()
              .reset_index()
        )
        logger.info("resolution_mode='finest': kept %d rows after deduplication", len(df))

    items = []
    for _, row in df.iterrows():
        filepath = os.path.join(row["_input_dir"], "graphs", row["graph_pickle"])
        if not os.path.exists(filepath) or os.path.getsize(filepath) < CHECK_MIN_FILESIZE:
            continue
        bands_path = _bands_path_for_pickle(filepath) if args.include_bands else None
        items.append((filepath, float(row["lon"]), float(row["lat"]), bands_path))
    logger.info("Valid samples: %d", len(items))

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(items))
    n_val = int(len(items) * args.val_fraction)
    train_items = [items[i] for i in perm[n_val:]]
    val_items   = [items[i] for i in perm[:n_val]]
    logger.info("Split — train: %d  val: %d", len(train_items), len(val_items))

    for split, split_items in [("train", train_items), ("val", val_items)]:
        out = os.path.join(args.output_dir, split)
        logger.info("Writing %s → %s", split, out)
        optimize(
            fn=process_fn,
            inputs=split_items,
            output_dir=out,
            num_workers=args.workers,
            chunk_size=args.chunk_size,
        )

    logger.info("Done.")


if __name__ == "__main__":
    main()

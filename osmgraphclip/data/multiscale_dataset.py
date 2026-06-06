"""MultiscaleOsmDataset and MultiscaleOsmGeoDataModule.

Extends OsmDataset / OsmGeoDataModule to load per-location band .npz files
produced by create_multiscale_dataset.py.  Band data is returned under the
"bands" key; if a .npz is missing the key is absent from the sample and the
collate function zero-fills that slot so batch["bands"] is always a complete
dict (or None if no sample in the batch has bands).
"""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Callable, Dict, Optional

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

from .osm_dataset import OsmDataset, OsmGeoDataModule, _LOG_EVERY_N_BATCHES

logger = logging.getLogger(__name__)

_BAND_KEYS = (
    "spatial_features",
    "global_embeddings",
    "subbin_spatial",
    "subbin_embeddings",
    "sector_spatial",
    "sector_embeddings",
)

_collate_batch_count = 0


def _load_band_npz(path: str) -> Optional[dict]:
    """Load band .npz → dict of float32 tensors, or None on failure."""
    try:
        data = np.load(path)
        return {k: torch.from_numpy(data[k].astype(np.float32)) for k in _BAND_KEYS}
    except Exception as exc:
        logger.debug("Could not load band file %s: %s", path, exc)
        return None


def _infer_index_from_pickle(pickle_path: str) -> Optional[int]:
    """Extract the integer i from paths like .../graphs/osm_{i}_graph.pkl."""
    name = os.path.basename(pickle_path)
    m = re.match(r"osm_(\d+)(?:_L\d+)?_graph\.pkl", name)
    return int(m.group(1)) if m else None


class MultiscaleOsmDataset(OsmDataset):
    """OsmDataset that also loads band .npz files from <root>/bands/."""

    def __init__(
        self,
        root: str,
        transform: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
        mode: Optional[str] = "both",
        resolution_mode: str = "all",
    ) -> None:
        super().__init__(root=root, transform=transform, mode=mode, resolution_mode=resolution_mode)
        self.bands_dir = os.path.join(root, "bands")
        has_bands = os.path.isdir(self.bands_dir)
        logger.info(
            "MultiscaleOsmDataset: bands_dir=%s exists=%s", self.bands_dir, has_bands
        )

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        sample = super().__getitem__(index)

        # Derive band .npz path from the graph pickle filename
        idx = _infer_index_from_pickle(self.graph_filenames[index])
        if idx is not None and os.path.isdir(self.bands_dir):
            npz_path = os.path.join(self.bands_dir, f"osm_{idx}_bands.npz")
            band_data = _load_band_npz(npz_path)
            if band_data is not None:
                sample["bands"] = band_data

        return sample


def _collate_multiscale(batch):
    """Collate samples; stacks band tensors or returns None if none present."""
    global _collate_batch_count

    t0 = time.perf_counter()
    result = {}

    if "coords" in batch[0]:
        result["coords"] = torch.cat([s["coords"].view(1, -1) for s in batch], dim=0)

    if "osm" in batch[0]:
        import torch.utils.data as _tud
        _orig_gwi = _tud.get_worker_info
        _tud.get_worker_info = lambda: None
        try:
            result["osm"] = Batch.from_data_list([s["osm"] for s in batch])
        finally:
            _tud.get_worker_info = _orig_gwi

    # ── Band collation ────────────────────────────────────────────────────────
    samples_with_bands = [s for s in batch if "bands" in s]
    if samples_with_bands:
        # Determine shapes from first available sample
        ref = samples_with_bands[0]["bands"]
        band_shapes = {k: ref[k].shape for k in _BAND_KEYS}

        stacked: Dict[str, list] = {k: [] for k in _BAND_KEYS}
        for s in batch:
            bd = s.get("bands")
            for k in _BAND_KEYS:
                if bd is not None and k in bd:
                    stacked[k].append(bd[k])
                else:
                    stacked[k].append(torch.zeros(band_shapes[k], dtype=torch.float32))

        result["bands"] = {k: torch.stack(stacked[k], dim=0) for k in _BAND_KEYS}

    elapsed = time.perf_counter() - t0
    _collate_batch_count += 1
    if _collate_batch_count == 1 or _collate_batch_count % _LOG_EVERY_N_BATCHES == 0:
        logger.info(
            "multiscale collate batch=%d size=%d took=%.3fs bands=%s",
            _collate_batch_count, len(batch), elapsed,
            "yes" if "bands" in result else "no",
        )

    return result


class MultiscaleOsmGeoDataModule(OsmGeoDataModule):
    """OsmGeoDataModule that uses MultiscaleOsmDataset and the band-aware collate."""

    def setup(self, stage="fit"):
        logger.info("MultiscaleOsmGeoDataModule.setup() start, stage=%s", stage)
        dataset = MultiscaleOsmDataset(
            root=self.data_dir,
            transform=self.train_transform,
            mode=self.mode,
            resolution_mode=self.resolution_mode,
        )
        logger.info("MultiscaleOsmDataset created, len=%d", len(dataset))

        import torch.utils.data
        N_val = int(len(dataset) * self.val_random_split_fraction)
        N_train = len(dataset) - N_val
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            dataset, [N_train, N_val]
        )
        logger.info("setup() done — train=%d, val=%d", N_train, N_val)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=_collate_multiscale,
            persistent_workers=self.num_workers > 0,
            pin_memory=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=_collate_multiscale,
            persistent_workers=self.num_workers > 0,
            pin_memory=self.num_workers > 0,
        )

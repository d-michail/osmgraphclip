import logging
import multiprocessing
import os
import pickle
import time
from typing import Optional

import numpy as np
import torch
import lightning.pytorch as pl
from litdata import StreamingDataset, StreamingDataLoader
from torch_geometric.data import Batch

from .transforms import get_train_transform

_BAND_KEYS = (
    "spatial_features",
    "global_embeddings",
    "subbin_spatial",
    "subbin_embeddings",
    "sector_spatial",
    "sector_embeddings",
)

logger = logging.getLogger(__name__)

_LOG_EVERY_N_ITEMS = 5000
_LOG_EVERY_N_BATCHES = 5
_litdata_collate_batch_count = multiprocessing.Value("i", 0)


def _litdata_collate(batch):
    """Collate for the litdata path.

    Graphs come from ``pickle.loads`` so their tensors are already plain CPU
    tensors (not numpy-backed / shared-memory).  We therefore do NOT need the
    ``get_worker_info`` monkey-patch used by ``OsmGeoDataModule._collate_batch``
    — patching that function interferes with ``StreamingDataLoader``'s internal
    chunk-distribution logic and prevents batches from being delivered after the
    first round.
    """
    t0 = time.perf_counter()

    collated = {}
    coords = [s["coords"].view(1, -1) for s in batch]
    collated["coords"] = torch.cat(coords, dim=0)

    if "osm" in batch[0]:
        collated["osm"] = Batch.from_data_list([s["osm"] for s in batch])

    elapsed = time.perf_counter() - t0
    with _litdata_collate_batch_count.get_lock():
        _litdata_collate_batch_count.value += 1
        n = _litdata_collate_batch_count.value
    if n == 1 or n % _LOG_EVERY_N_BATCHES == 0:
        logger.info("collate_fn batch=%d, size=%d, took=%.3fs", n, len(batch), elapsed)
    return collated


class OsmStreamingDataset(StreamingDataset):
    def __init__(self, input_dir: str, shuffle: bool = False, transform=None):
        # litdata calls self.transform(item) unconditionally in __getitem__,
        # so give it an identity function and store ours under a different name.
        super().__init__(input_dir=input_dir, shuffle=shuffle)
        self.transform = lambda x: x
        self._osm_transform = transform
        self._items_loaded = 0
        self._input_dir_name = os.path.basename(input_dir.rstrip("/"))

    def __getitem__(self, index):
        item = super().__getitem__(index)
        graph = pickle.loads(item["graph"])
        for ntype in graph.node_types:
            if graph[ntype].x is not None:
                graph[ntype].x = graph[ntype].x.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
        if not hasattr(graph, "is_zero_graph"):
            graph.is_zero_graph = torch.tensor([False])
        coords = torch.tensor(item["coords"], dtype=torch.float32)
        sample = {"coords": coords, "osm": graph}
        if self._osm_transform is not None:
            sample = self._osm_transform(sample)
        self._items_loaded += 1
        if self._items_loaded % _LOG_EVERY_N_ITEMS == 0:
            logger.debug(
                "StreamingDataset[%s] loaded item %d (index=%s)",
                self._input_dir_name,
                self._items_loaded,
                index,
            )
        return sample



class OsmLitDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 256,
        num_workers: int = 4,
        transform: str = "default",
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        if transform == "default":
            self.train_transform = get_train_transform()
        else:
            self.train_transform = transform
        self.save_hyperparameters()

    def setup(self, stage="fit"):
        logger.info("setup() start, stage=%s", stage)
        self.train_dataset = OsmStreamingDataset(
            os.path.join(self.data_dir, "train"),
            shuffle=True,
            transform=self.train_transform,
        )
        self.val_dataset = OsmStreamingDataset(
            os.path.join(self.data_dir, "val"),
            shuffle=False,
        )
        logger.info(
            "setup() done — train=%d, val=%d",
            len(self.train_dataset),
            len(self.val_dataset),
        )

    def train_dataloader(self):
        logger.info(
            "train_dataloader() called, num_workers=%d, batch_size=%d, persistent_workers=%s",
            self.num_workers, self.batch_size, self.num_workers > 0,
        )
        dl = StreamingDataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=_litdata_collate,
            pin_memory=False,
            persistent_workers=False,
        )
        logger.info("train_dataloader length (batches/epoch): %d", len(dl))
        return dl

    def val_dataloader(self):
        logger.info(
            "val_dataloader() called, num_workers=%d, batch_size=%d, persistent_workers=%s",
            self.num_workers, self.batch_size, self.num_workers > 0,
        )
        return StreamingDataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=_litdata_collate,
            pin_memory=False,
            persistent_workers=False,
        )


# ── Multiscale (band-aware) streaming classes ─────────────────────────────────

class MultiscaleStreamingDataset(OsmStreamingDataset):
    """OsmStreamingDataset that also decodes band feature arrays when present."""

    def __getitem__(self, index):
        raw = StreamingDataset.__getitem__(self, index)

        graph = pickle.loads(raw["graph"])
        for ntype in graph.node_types:
            if graph[ntype].x is not None:
                graph[ntype].x = graph[ntype].x.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
        if not hasattr(graph, "is_zero_graph"):
            graph.is_zero_graph = torch.tensor([False])

        coords = torch.tensor(raw["coords"], dtype=torch.float32)
        sample = {"coords": coords, "osm": graph}

        if "bands" in raw:
            sample["bands"] = {
                k: torch.from_numpy(np.array(raw["bands"][k], dtype=np.float32))
                for k in _BAND_KEYS
                if k in raw["bands"]
            }

        if self._osm_transform is not None:
            sample = self._osm_transform(sample)

        self._items_loaded += 1
        if self._items_loaded % _LOG_EVERY_N_ITEMS == 0:
            logger.debug(
                "MultiscaleStreamingDataset[%s] loaded item %d",
                self._input_dir_name, self._items_loaded,
            )
        return sample


_ms_collate_count = multiprocessing.Value("i", 0)


def _litdata_multiscale_collate(batch):
    """Collate for MultiscaleStreamingDataset; handles optional band tensors."""
    t0 = time.perf_counter()

    collated = {
        "coords": torch.cat([s["coords"].view(1, -1) for s in batch], dim=0),
        "osm":    Batch.from_data_list([s["osm"] for s in batch]),
    }

    samples_with_bands = [s for s in batch if "bands" in s]
    if samples_with_bands:
        ref = samples_with_bands[0]["bands"]
        band_shapes = {k: ref[k].shape for k in _BAND_KEYS if k in ref}
        stacked = {k: [] for k in band_shapes}
        for s in batch:
            bd = s.get("bands", {})
            for k, shape in band_shapes.items():
                stacked[k].append(bd[k] if k in bd else torch.zeros(shape, dtype=torch.float32))
        collated["bands"] = {k: torch.stack(v, dim=0) for k, v in stacked.items()}

    elapsed = time.perf_counter() - t0
    with _ms_collate_count.get_lock():
        _ms_collate_count.value += 1
        n = _ms_collate_count.value
    if n == 1 or n % _LOG_EVERY_N_BATCHES == 0:
        logger.info(
            "ms_collate batch=%d size=%d took=%.3fs bands=%s",
            n, len(batch), elapsed, "yes" if "bands" in collated else "no",
        )
    return collated


class MultiscaleLitDataModule(OsmLitDataModule):
    """OsmLitDataModule that uses MultiscaleStreamingDataset and band-aware collate."""

    def setup(self, stage="fit"):
        logger.info("MultiscaleLitDataModule.setup() start, stage=%s", stage)
        self.train_dataset = MultiscaleStreamingDataset(
            os.path.join(self.data_dir, "train"),
            shuffle=True,
            transform=self.train_transform,
        )
        self.val_dataset = MultiscaleStreamingDataset(
            os.path.join(self.data_dir, "val"),
            shuffle=False,
        )
        logger.info(
            "setup() done — train=%d, val=%d",
            len(self.train_dataset), len(self.val_dataset),
        )

    def train_dataloader(self):
        return StreamingDataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=_litdata_multiscale_collate,
            pin_memory=False,
            persistent_workers=False,
        )

    def val_dataloader(self):
        return StreamingDataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=_litdata_multiscale_collate,
            pin_memory=False,
            persistent_workers=False,
        )

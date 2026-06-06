import os
from typing import Any, Callable, Dict, Optional
import json
import logging
import pickle
import time

import pandas as pd
from torch import Tensor
from torchgeo.datasets.geo import NonGeoDataset
import numpy as np
import torch

import lightning.pytorch as pl
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

from .transforms import get_train_transform

logger = logging.getLogger(__name__)

CHECK_MIN_FILESIZE = 2000 # 2kb
_LOG_EVERY_N_BATCHES = 50
_collate_batch_count = 0

class OsmGeoDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "toy",
        batch_size: int = 64,
        num_workers: int = 6,
        val_random_split_fraction: float = 0.1,
        transform: str = 'default',
        mode: str = "both",
        resolution_mode: str = "all",
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        if transform=='default':
            self.train_transform = get_train_transform()
        else:
            self.train_transform = transform

        self.val_random_split_fraction = val_random_split_fraction
        self.mode = mode
        self.resolution_mode = resolution_mode
        self.save_hyperparameters()

    def prepare_data(self) -> None:
        if not os.path.exists(self.data_dir):
            logger.warning("No dataset found.")

    def setup(self, stage="fit"):
        logger.info("setup() start, stage=%s", stage)
        dataset = OsmDataset(root=self.data_dir, transform=self.train_transform, mode=self.mode, resolution_mode=self.resolution_mode)
        logger.info("OsmDataset created, len=%d", len(dataset))

        N_val = int(len(dataset) * self.val_random_split_fraction)
        N_train = len(dataset) - N_val
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(dataset, [N_train, N_val])
        logger.info("setup() done — train=%d, val=%d", N_train, N_val)

    def train_dataloader(self):
        logger.info("train_dataloader() called, num_workers=%d", self.num_workers)
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=self._collate_with_logging,
            persistent_workers=self.num_workers > 0,
            pin_memory=self.num_workers > 0,
        )

    def val_dataloader(self):
        logger.info("val_dataloader() called, num_workers=%d", self.num_workers)
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=self._collate_with_logging,
            persistent_workers=self.num_workers > 0,
            pin_memory=self.num_workers > 0,
        )

    def test_dataloader(self):
        raise NotImplementedError

    @staticmethod
    def _collate_with_logging(batch):
        global _collate_batch_count
        t0 = time.perf_counter()
        result = OsmGeoDataModule._collate_batch(batch)
        elapsed = time.perf_counter() - t0
        _collate_batch_count += 1
        if _collate_batch_count == 1 or _collate_batch_count % _LOG_EVERY_N_BATCHES == 0:
            logger.info("collate_fn batch=%d, size=%d, took=%.3fs", _collate_batch_count, len(batch), elapsed)
        return result

    @staticmethod
    def _collate_batch(batch):
        collated = {}

        if "coords" in batch[0]:
            coords = [sample["coords"].view(1, -1) for sample in batch]
            collated["coords"] = torch.cat(coords, dim=0)

        if "osm" in batch[0]:
            # PyG's Batch.from_data_list checks get_worker_info() and, when inside
            # a worker, pre-allocates output tensors via _new_shared() into shared
            # memory.  With the 'file_system' sharing strategy the resulting storage
            # is a fixed-size mmap and not resizable, so the subsequent resize_()
            # call raises RuntimeError.  Masking get_worker_info makes PyG skip that
            # path and allocate plain CPU tensors instead; PyTorch then handles the
            # worker→main transfer normally.
            import torch.utils.data as _tud
            _orig_gwi = _tud.get_worker_info
            _tud.get_worker_info = lambda: None
            try:
                collated["osm"] = Batch.from_data_list([sample["osm"] for sample in batch])
            finally:
                _tud.get_worker_info = _orig_gwi

        return collated

class OsmDataset(NonGeoDataset):
    """Dataset.
    """

    validation_filenames = [
        "dataset.db",
        "metadata.json",
        "graphs/",
    ]

    def __init__(
        self,
        root: str,
        transform: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
        mode: Optional[str] = "both",
        resolution_mode: str = "all",
    ) -> None:
        """Initialize a new dataset instance.
        Args:
            root: root directory of pre-sampled dataset
            transform: torch transform to apply to a sample
            mode: which data to return (options are "both" or "points"), useful for embedding locations without loading graphs
            resolution_mode: how to handle multi-resolution datasets:
                "all" — use every (location, level) row as a separate sample (default)
                "finest" — keep only the finest (smallest bbox) level per location
        """
        assert mode in ["both", "points"]
        assert resolution_mode in ["all", "finest"]
        self.root = root
        self.transform = transform
        self.mode = mode
        self.resolution_mode = resolution_mode
        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted.")

        # Load metadata
        metadata_path = os.path.join(self.root, "metadata.json")
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)
        self.bbox_half_width_m = self.metadata.get("bbox_half_width_m")

        from osmgraphclip.dataset_db import DatasetDB
        with DatasetDB(os.path.join(self.root, "dataset.db")) as _db:
            df = _db.graphs_as_dataframe()

        if self.resolution_mode == "finest" and "level" in df.columns:
            # Keep only the row with the smallest level (finest resolution) per location.
            # Ties are broken by smallest bbox_size when available.
            sort_cols = ["level"] + (["bbox_size"] if "bbox_size" in df.columns else [])
            df = (
                df.sort_values(sort_cols)
                  .groupby(["lat", "lon"], sort=False)
                  .first()
                  .reset_index()
            )
            logger.info(
                f"resolution_mode='finest': kept {len(df)} finest-level rows "
                f"(one per unique location)"
            )

        self.graph_filenames = []
        self.points = []

        # Per-type offsets added on top of the embedding dim:
        #   polygon: embedding_dim + 8, line: embedding_dim + 6, point: embedding_dim + 2
        _NTYPE_OFFSETS = {"polygon": 8, "line": 6, "point": 2}

        # Infer expected embedding dim from metadata to avoid loading every pickle.
        # Fall back to loading the first valid graph only if the backend is unknown.
        _BACKEND_DEFAULT_DIMS = {"clip": 512, "sbert": 384}
        _embedding_backend = self.metadata.get("embedding_backend", "clip")
        _base_dim = _BACKEND_DEFAULT_DIMS.get(_embedding_backend)
        if _base_dim is not None:
            expected_dims = {t: _base_dim + o for t, o in _NTYPE_OFFSETS.items()}
            logger.info(
                "Inferred expected node dims from metadata (backend=%s): %s",
                _embedding_backend, expected_dims,
            )
        else:
            logger.info(
                "Unknown embedding backend '%s'; will infer expected node dims from first valid graph",
                _embedding_backend,
            )
            expected_dims = {}  # will be populated from first valid graph below

        n_skipped_files = 0
        _first_graph_loaded = _base_dim is not None  # skip first-graph probe if already known

        for i in range(df.shape[0]):
            filename = os.path.join(self.root, "graphs", df.iloc[i]["graph_pickle"])

            if not os.path.exists(filename):
                logger.warning("Skipping graph %s because file does not exist", filename)
                n_skipped_files += 1
                continue

            if os.path.getsize(filename) < CHECK_MIN_FILESIZE:
                n_skipped_files += 1
                continue

            # If we still need to determine expected dims, load the first valid graph.
            if not _first_graph_loaded:
                logger.debug("Probing first valid graph to determine expected node dims: %s", filename)
                with open(filename, "rb") as f:
                    _g = pickle.load(f)
                for _ntype, _offset in _NTYPE_OFFSETS.items():
                    _x = getattr(_g[_ntype], "x", None)
                    if _x is not None and _x.numel() > 0:
                        _embedding_dim = _x.shape[-1] - _offset
                        expected_dims = {t: _embedding_dim + o for t, o in _NTYPE_OFFSETS.items()}
                        logger.info(
                            "Inferred expected node dims from first graph (ntype=%s, embedding_dim=%d): %s",
                            _ntype, _embedding_dim, expected_dims,
                        )
                        break
                if not expected_dims:
                    logger.warning("Could not infer expected node dims from %s (all node types empty)", filename)
                _first_graph_loaded = True

            self.graph_filenames.append(filename)
            self.points.append(
                (df.iloc[i]["lon"], df.iloc[i]["lat"])
            )

        logger.info(f"Skipped {n_skipped_files}/{len(df)} graphs because they were smaller "
                   f"than {CHECK_MIN_FILESIZE} bytes")

    def _calculate_bbox(self, lat: float, lon: float) -> tuple:
        """Calculate bounding box coordinates from a center point and half-width distance in meters.
        
        Args:
            lat: latitude of center point
            lon: longitude of center point
            
        Returns:
            tuple of (left, bottom, right, top) in degrees
        """
        EARTH_RADIUS_M = 6_371_009  # meters
        dist = self.bbox_half_width_m
        delta_lat = np.rad2deg(dist / EARTH_RADIUS_M)
        delta_lon = np.rad2deg(dist / EARTH_RADIUS_M) / np.cos(np.deg2rad(lat))
        top = lat + delta_lat
        bottom = lat - delta_lat
        right = lon + delta_lon
        left = lon - delta_lon
        return left, bottom, right, top

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Return an index within the dataset.
        Args:
            index: index to return
        Returns:
            dictionary with "osm" and "coords" keys where coords is in (lon, lat) format
        """
        logger.debug("__getitem__ index=%d", index)
        lon, lat = self.points[index]
        coords = torch.tensor(self.points[index], dtype=torch.float32)
        sample = {"coords": coords}

        if self.mode == "both":
            with open(self.graph_filenames[index], "rb") as f:
                osm_graph = pickle.load(f)
            # Clone every tensor so each has its own resizable storage.
            # torch.from_numpy produces contiguous but non-resizable (numpy-backed)
            # storage; PyG's Batch.from_data_list calls resize_() internally and
            # raises RuntimeError on non-resizable storage even when contiguous.
            for store in osm_graph.stores:
                for key in list(store.keys()):
                    val = store[key]
                    if isinstance(val, torch.Tensor):
                        store[key] = val.clone()
            # Mark non-zero graphs so Batch.from_data_list always finds this
            # attribute; zero graphs have it set to True by build_zero_graph().
            if not hasattr(osm_graph, "is_zero_graph"):
                osm_graph.is_zero_graph = torch.tensor([False])
            sample["osm"] = osm_graph
            
        if self.transform is not None:
            sample = self.transform(sample)
            
        return sample

    def __len__(self) -> int:
        """Return the number of datapoints in the dataset.
        Returns:
            length of dataset
        """
        return len(self.graph_filenames)
        

    def _check_integrity(self) -> bool:
        """Checks the integrity of the dataset structure.
        Returns:
            True if the dataset directories and split files are found, else False
        """
        
        for filename in self.validation_filenames:
            filepath = os.path.join(self.root, filename)
            if not os.path.exists(filepath):
                logger.error(f"{filepath} missing")
                return False
        return True

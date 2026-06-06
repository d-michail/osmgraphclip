"""Custom Lightning loggers for SLURM singleton job resumption."""

import csv
import os

from lightning.fabric.utilities.rank_zero import rank_zero_warn
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.loggers.csv_logs import ExperimentWriter


class AppendExperimentWriter(ExperimentWriter):
    """ExperimentWriter that appends to an existing metrics.csv instead of overwriting it."""

    def __init__(self, log_dir: str) -> None:
        # Read the existing CSV header *before* super().__init__() would delete
        # it via _check_log_dir_exists().
        metrics_path = os.path.join(log_dir, self.NAME_METRICS_FILE)
        existing_keys: list[str] = []

        if os.path.isfile(metrics_path):
            with open(metrics_path, newline="") as f:
                reader = csv.DictReader(f)
                existing_keys = list(reader.fieldnames or [])

        # base __init__ resets metrics_keys=[] then calls _check_log_dir_exists();
        # our override below skips the deletion.
        super().__init__(log_dir)

        # Pre-populate metrics_keys from the existing header so _record_new_keys()
        # returns an empty set on the first save() call, keeping Lightning in
        # append mode and preventing a duplicate header row.
        for key in existing_keys:
            if key not in self.metrics_keys:
                self.metrics_keys.append(key)
        self.metrics_keys.sort()  # match Lightning's sort in _record_new_keys

    def _check_log_dir_exists(self) -> None:
        """Suppress the metrics.csv deletion performed by the base class."""
        if self._fs.exists(self.log_dir) and self._fs.listdir(self.log_dir):
            rank_zero_warn(
                f"Experiment logs directory {self.log_dir} exists and is not empty. "
                "AppendCSVLogger will append to the existing metrics.csv."
            )
            # Intentionally do NOT call self._fs.rm_file(self.metrics_file_path)


class AppendCSVLogger(CSVLogger):
    """CSVLogger that preserves and appends to metrics.csv across SLURM restarts."""

    @property
    def experiment(self):
        if self._experiment is not None:
            return self._experiment
        self._fs.makedirs(self.root_dir, exist_ok=True)
        self._experiment = AppendExperimentWriter(log_dir=self.log_dir)
        return self._experiment

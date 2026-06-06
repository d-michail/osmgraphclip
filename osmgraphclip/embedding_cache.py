import json
import sqlite3
import threading
from typing import Optional

import numpy as np


class EmbeddingCache:
    """Persistent, thread-safe word-embedding cache backed by SQLite.

    Stores embeddings as BLOBs keyed by (model_name, word). An in-memory dict
    layer provides O(1) lookups after the first hit within a process run.
    """

    _DDL = """
    PRAGMA journal_mode = WAL;
    PRAGMA synchronous  = NORMAL;
    CREATE TABLE IF NOT EXISTS embeddings (
        model_name TEXT NOT NULL,
        word       TEXT NOT NULL,
        dtype      TEXT NOT NULL,
        shape      TEXT NOT NULL,
        data       BLOB NOT NULL,
        PRIMARY KEY (model_name, word)
    );
    """

    def __init__(self, db_path: str, model_name: str) -> None:
        self._model_name = model_name
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.executescript(self._DDL)
        self._conn.commit()
        self._write_lock = threading.Lock()
        self._mem_cache: dict[str, np.ndarray] = {}

    def get(self, word: str) -> Optional[np.ndarray]:
        if word in self._mem_cache:
            return self._mem_cache[word]
        row = self._conn.execute(
            "SELECT dtype, shape, data FROM embeddings WHERE model_name=? AND word=?",
            (self._model_name, word),
        ).fetchone()
        if row is None:
            return None
        dtype, shape_str, data = row
        arr = np.frombuffer(data, dtype=dtype).reshape(json.loads(shape_str)).copy()
        self._mem_cache[word] = arr
        return arr

    def put(self, word: str, embedding: np.ndarray) -> None:
        with self._write_lock:
            self._mem_cache[word] = embedding
            self._conn.execute(
                "INSERT OR IGNORE INTO embeddings (model_name, word, dtype, shape, data)"
                " VALUES (?, ?, ?, ?, ?)",
                (
                    self._model_name,
                    word,
                    str(embedding.dtype),
                    json.dumps(list(embedding.shape)),
                    embedding.tobytes(),
                ),
            )
            self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "EmbeddingCache":
        return self

    def __exit__(self, *_) -> None:
        self.close()

"""Abstract base class for OSM graph builders.

Shared embedding infrastructure (CLIP / SBERT) lives here so that concrete
builder subclasses (OSM2Graph, ...) do not duplicate it.
"""

from abc import ABC, abstractmethod
import json
import logging
from typing import Dict

import numpy as np
import torch
from torch_geometric.data import HeteroData

logger = logging.getLogger(__name__)

CLIP_DEFAULT_MODEL = 'openai/clip-vit-base-patch16'
SBERT_DEFAULT_MODEL = 'all-MiniLM-L6-v2'
_EMBED_BATCH_SIZE = 64


class BaseOSMGraphBuilder(ABC):
    """Shared embedding infrastructure + abstract process() contract."""

    def __init__(self, tagw_path: str, device: str,
                 embedding_backend: str = 'clip',
                 embedding_model=None,
                 embedding_cache=None):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.embedding_backend = embedding_backend

        if embedding_backend == 'clip':
            from transformers import CLIPTokenizer, CLIPTextModel
            model_name = embedding_model or CLIP_DEFAULT_MODEL
            self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
            self.text_model = CLIPTextModel.from_pretrained(model_name).to(self.device)
            self.text_model.eval()
            self.embedding_dim = self.text_model.config.hidden_size
        elif embedding_backend == 'sbert':
            from sentence_transformers import SentenceTransformer
            model_name = embedding_model or SBERT_DEFAULT_MODEL
            self.sbert_model = SentenceTransformer(model_name, device=str(self.device))
            self.embedding_dim = self.sbert_model.get_sentence_embedding_dimension()
        else:
            raise ValueError(
                f"Unknown embedding_backend: {embedding_backend!r}. Choose 'clip' or 'sbert'."
            )

        self.tag_w = json.load(open(tagw_path, 'r'))
        self._word_embedding_cache: Dict[str, np.ndarray] = {}
        self._sentence_embedding_cache: Dict[str, np.ndarray] = {}
        self._persistent_cache = embedding_cache

    def _get_word_embedding(self, word: str) -> np.ndarray:
        cache_key = str(word).strip()
        if cache_key in self._word_embedding_cache:
            return self._word_embedding_cache[cache_key]

        if self._persistent_cache is not None:
            cached = self._persistent_cache.get(cache_key)
            if cached is not None:
                logger.debug("Embedding cache hit for %r", cache_key)
                self._word_embedding_cache[cache_key] = cached
                return cached

        logger.debug("Computing embedding for %r", cache_key)
        if self.embedding_backend == 'clip':
            inputs = self.tokenizer(cache_key, return_tensors="pt", padding=False, truncation=False).to(self.device)
            with torch.no_grad():
                outputs = self.text_model(**inputs)
            embedding = outputs.last_hidden_state[:, 1:, ].mean(dim=1).cpu().squeeze().numpy()
        else:  # sbert
            embedding = self.sbert_model.encode(cache_key, convert_to_numpy=True)

        self._word_embedding_cache[cache_key] = embedding
        if self._persistent_cache is not None:
            self._persistent_cache.put(cache_key, embedding)
        return embedding

    def _batch_embed_words(self, words: list) -> None:
        """Pre-embed a list of words in one batched model call, populating both caches."""
        keys = list(dict.fromkeys(str(w).strip() for w in words))

        uncached = [k for k in keys if k not in self._word_embedding_cache]

        still_uncached = []
        for k in uncached:
            if self._persistent_cache is not None:
                cached = self._persistent_cache.get(k)
                if cached is not None:
                    self._word_embedding_cache[k] = cached
                    continue
            still_uncached.append(k)

        if not still_uncached:
            return

        logger.debug("Batch-computing embeddings for %d words", len(still_uncached))

        if self.embedding_backend == 'clip':
            inputs = self.tokenizer(
                still_uncached,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77,
            ).to(self.device)
            all_ids = inputs["input_ids"]
            all_mask = inputs["attention_mask"]
            result_parts = []
            for start in range(0, len(still_uncached), _EMBED_BATCH_SIZE):
                end = start + _EMBED_BATCH_SIZE
                with torch.no_grad():
                    chunk_out = self.text_model(
                        input_ids=all_ids[start:end],
                        attention_mask=all_mask[start:end],
                    )
                chunk_emb = chunk_out.last_hidden_state[:, 1:, ].mean(dim=1).cpu().numpy()
                result_parts.append(chunk_emb)
            embeddings = np.concatenate(result_parts, axis=0)
        else:  # sbert
            embeddings = self.sbert_model.encode(
                still_uncached,
                batch_size=_EMBED_BATCH_SIZE,
                convert_to_numpy=True,
                show_progress_bar=False,
            )

        for i, k in enumerate(still_uncached):
            emb = embeddings[i]
            self._word_embedding_cache[k] = emb
            if self._persistent_cache is not None:
                self._persistent_cache.put(k, emb)

    def _get_sentence_embedding(self, sentence: str) -> np.ndarray:
        cache_key = str(sentence)
        if cache_key in self._sentence_embedding_cache:
            return self._sentence_embedding_cache[cache_key]

        if self.embedding_backend == 'sbert':
            sentence_embed = self._get_word_embedding(cache_key.replace(';', ' '))
        else:
            words = cache_key.split(';')
            word_embeds = []
            word_ws = []
            for w in words:
                tag = w.split(':')[0]
                try:
                    word_ws.append(self.tag_w[tag])
                except Exception:
                    logger.debug("Tag %r not found in weights file, using weight 1", tag)
                    word_ws.append(1)
                word_embeds.append(self._get_word_embedding(w))

            if len(word_embeds) == 0:
                sentence_embed = np.zeros(self.embedding_dim, dtype=np.float32)
            else:
                word_embeds = np.array(word_embeds)
                word_ws = np.array(word_ws)
                word_ws = np.log(np.maximum(word_ws, 1e-8)).reshape(-1, 1)
                if word_ws.sum() != 0:
                    sentence_embed = (word_embeds * word_ws).sum(axis=0) / word_ws.sum()
                else:
                    sentence_embed = word_embeds.mean(axis=0)

        self._sentence_embedding_cache[cache_key] = sentence_embed
        return sentence_embed

    def _build_semantic_embeddings(self, sentences) -> torch.Tensor:
        if self.embedding_backend == 'sbert':
            self._batch_embed_words([str(s).replace(';', ' ') for s in sentences])
        else:
            all_words = []
            for sentence in sentences:
                all_words.extend(str(sentence).split(';'))
            self._batch_embed_words(all_words)

        semantic_embeddings = [self._get_sentence_embedding(sentence) for sentence in sentences]
        if len(semantic_embeddings) == 0:
            return torch.empty(0, self.embedding_dim)
        return torch.from_numpy(np.array(semantic_embeddings))

    @abstractmethod
    def process(self, polygon_file, line_file, point_file,
                north: float, south: float, east: float, west: float) -> HeteroData:
        """Build a HeteroData graph from GeoDataFrames and bounding box coordinates."""
        ...

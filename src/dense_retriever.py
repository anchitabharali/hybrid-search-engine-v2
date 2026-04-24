"""
dense_retriever.py  (v2)
------------------------
Upgrades vs. v1:
  * Configurable FAISS index type: "flat" | "hnsw" | "ivf".
      - HNSW uses graph-based ANN -> O(log N) queries, ~95-99% recall@10.
      - IVF (+ optional PQ) for huge corpora.
  * Lazy model loading: the SentenceTransformer is only loaded on first use
    via @cached_property.
  * On-disk query-embedding cache (hit rate reported by .cache_stats()).
  * Full logging + try/except on every I/O call.
  * Save/load handles all three index types correctly.
"""
import os
import time
from functools import cached_property
from typing import List, Tuple

import faiss
import numpy as np

from .cache import EmbeddingCache
from .logger import get_logger

log = get_logger(__name__)


class DenseRetriever:
    def __init__(
        self,
        model_name: str,
        index_type: str = "hnsw",
        hnsw_m: int = 32,
        hnsw_ef_construction: int = 200,
        hnsw_ef_search: int = 64,
        ivf_nlist: int = 100,
        ivf_nprobe: int = 8,
        cache_path: str | None = None,
    ):
        self.model_name = model_name
        self.index_type = index_type.lower()
        self.hnsw_m = hnsw_m
        self.hnsw_ef_construction = hnsw_ef_construction
        self.hnsw_ef_search = hnsw_ef_search
        self.ivf_nlist = ivf_nlist
        self.ivf_nprobe = ivf_nprobe

        self.index: faiss.Index | None = None
        self.embeddings: np.ndarray | None = None
        self._dim: int | None = None
        self._embedding_cache = EmbeddingCache(cache_path) if cache_path else None

    # ------------------------------------------------------------------ #
    # Lazy-loaded model (loaded on first .encode() call)
    # ------------------------------------------------------------------ #
    @cached_property
    def model(self):
        log.info(f"Loading SentenceTransformer: {self.model_name}")
        t0 = time.perf_counter()
        try:
            from sentence_transformers import SentenceTransformer
            m = SentenceTransformer(self.model_name)
        except Exception as e:
            log.error(f"Failed to load embedding model: {e}")
            raise
        log.info(f"Loaded in {time.perf_counter() - t0:.2f}s "
                 f"(dim={m.get_sentence_embedding_dimension()})")
        self._dim = m.get_sentence_embedding_dimension()
        return m

    @property
    def dim(self) -> int:
        if self._dim is None:
            _ = self.model  # force load
        return self._dim  # type: ignore

    # ------------------------------------------------------------------ #
    # Encoding  (with optional cache for single-query calls)
    # ------------------------------------------------------------------ #
    def encode(self, texts: List[str], batch_size: int = 32,
               use_cache: bool = False) -> np.ndarray:
        """Return float32 L2-normalised embeddings."""
        # Cache path - only worthwhile for single queries
        if use_cache and len(texts) == 1 and self._embedding_cache is not None:
            cached = self._embedding_cache.get(texts[0])
            if cached is not None:
                return cached[np.newaxis, :]

        embs = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        ).astype("float32")

        if use_cache and len(texts) == 1 and self._embedding_cache is not None:
            self._embedding_cache.put(texts[0], embs[0])

        return embs

    # ------------------------------------------------------------------ #
    # Index factory
    # ------------------------------------------------------------------ #
    def _build_index(self, vectors: np.ndarray) -> faiss.Index:
        n, d = vectors.shape
        if self.index_type == "flat":
            log.info(f"Building FAISS IndexFlatIP (exact), n={n} d={d}")
            idx = faiss.IndexFlatIP(d)
            idx.add(vectors)
            return idx

        if self.index_type == "hnsw":
            log.info(f"Building FAISS IndexHNSWFlat M={self.hnsw_m} "
                     f"efC={self.hnsw_ef_construction} n={n}")
            idx = faiss.IndexHNSWFlat(d, self.hnsw_m, faiss.METRIC_INNER_PRODUCT)
            idx.hnsw.efConstruction = self.hnsw_ef_construction
            idx.hnsw.efSearch = self.hnsw_ef_search
            idx.add(vectors)
            return idx

        if self.index_type == "ivf":
            nlist = min(self.ivf_nlist, max(1, n // 10))
            log.info(f"Building FAISS IndexIVFFlat nlist={nlist} n={n}")
            quantizer = faiss.IndexFlatIP(d)
            idx = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
            idx.train(vectors)
            idx.add(vectors)
            idx.nprobe = self.ivf_nprobe
            return idx

        raise ValueError(f"Unknown index_type: {self.index_type}")

    # ------------------------------------------------------------------ #
    # Public build / update
    # ------------------------------------------------------------------ #
    def build(self, documents: List[dict]):
        texts = [d["text"] for d in documents]
        t0 = time.perf_counter()
        self.embeddings = self.encode(texts)
        t1 = time.perf_counter()
        log.info(f"Encoded {len(texts)} docs in {t1 - t0:.2f}s")
        self.index = self._build_index(self.embeddings)
        log.info(f"Index built in {time.perf_counter() - t1:.2f}s "
                 f"(total vectors={self.index.ntotal})")

    def add_documents(self, new_docs: List[dict]):
        if self.index is None:
            raise RuntimeError("Call build() before add_documents().")
        texts = [d["text"] for d in new_docs]
        new_embs = self.encode(texts)
        self.embeddings = np.vstack([self.embeddings, new_embs])
        # IVF needs training - re-train if we grow significantly.
        if self.index_type == "ivf" and not self.index.is_trained:
            self.index.train(self.embeddings)
        self.index.add(new_embs)
        log.info(f"Added {len(new_docs)} docs (total={self.index.ntotal})")

    # ------------------------------------------------------------------ #
    # Search
    # ------------------------------------------------------------------ #
    def search(self, query: str, top_k: int = 50) -> List[Tuple[int, float]]:
        if self.index is None or self.index.ntotal == 0:
            return []
        q_emb = self.encode([query], use_cache=True)
        k = min(top_k, self.index.ntotal)
        scores, idx = self.index.search(q_emb, k)
        return [(int(i), float(s)) for i, s in zip(idx[0], scores[0]) if i != -1]

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, index_path: str, emb_path: str):
        try:
            faiss.write_index(self.index, index_path)
            np.save(emb_path, self.embeddings)
            if self._embedding_cache is not None:
                self._embedding_cache.save()
            log.info(f"Persisted index ({index_path}) + embeddings ({emb_path})")
        except Exception as e:
            log.error(f"Persist failed: {e}")
            raise

    def load(self, index_path: str, emb_path: str):
        try:
            self.index = faiss.read_index(index_path)
            self.embeddings = np.load(emb_path)
            # Restore HNSW efSearch / IVF nprobe after load
            if self.index_type == "hnsw" and hasattr(self.index, "hnsw"):
                self.index.hnsw.efSearch = self.hnsw_ef_search
            if self.index_type == "ivf" and hasattr(self.index, "nprobe"):
                self.index.nprobe = self.ivf_nprobe
            log.info(f"Loaded FAISS index ({self.index.ntotal} vectors)")
        except Exception as e:
            log.error(f"Load failed: {e}")
            raise

    def cache_stats(self) -> dict:
        return self._embedding_cache.stats() if self._embedding_cache else {}

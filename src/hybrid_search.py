"""
hybrid_search.py  (v2)
----------------------
Orchestrator upgrades vs. v1:
  * Optional LightGBM LTR ranker replaces manual weighted fusion
    (auto-falls back to weights if no trained model is present).
  * Per-stage timings collected on every query.
  * Full logging + exception safety.
  * Query processor is wired with the dense model so embedding-similarity
    expansion filtering works out of the box.
"""
import os
import pickle
import time
from typing import List, Optional

import numpy as np

from .query_processor import QueryProcessor
from .sparse_retriever import SparseRetriever
from .dense_retriever import DenseRetriever
from .reranker import CrossEncoderReranker
from .ltr import LTRRanker, extract_features
from .logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def min_max_norm(values):
    if not values:
        return []
    arr = np.asarray(values, dtype=float)
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-9:
        return np.ones_like(arr).tolist()
    return ((arr - lo) / (hi - lo)).tolist()


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
class HybridSearchEngine:
    def __init__(self, config):
        self.cfg = config
        self.documents: List[dict] = []

        # Dense retriever first - query processor reuses its model for expansion filter.
        self.dense = DenseRetriever(
            model_name=config.EMBEDDING_MODEL,
            index_type=config.INDEX_TYPE,
            hnsw_m=config.HNSW_M,
            hnsw_ef_construction=config.HNSW_EF_CONSTRUCTION,
            hnsw_ef_search=config.HNSW_EF_SEARCH,
            ivf_nlist=config.IVF_NLIST,
            ivf_nprobe=config.IVF_NPROBE,
            cache_path=config.QUERY_EMBED_CACHE_PATH,
        )

        self.processor = QueryProcessor(
            use_expansion=config.EXPANSION_ENABLED,
            pos_filter=config.EXPANSION_POS_FILTER,
            max_synonyms=config.EXPANSION_MAX_PER_TOKEN,
            similarity_threshold=config.EXPANSION_SIMILARITY_THRESHOLD,
            dense_model=(self.dense.model
                         if config.EXPANSION_USE_EMBEDDING_FILTER
                         else None),
        )

        self.sparse = SparseRetriever(self.processor)
        self.reranker = CrossEncoderReranker(config.CROSS_ENCODER_MODEL)
        self.ltr: Optional[LTRRanker] = None

    # ------------------------------------------------------------------ #
    # Fit / cache loading
    # ------------------------------------------------------------------ #
    def fit(self, documents: List[dict], use_cache: bool = True):
        self.documents = documents
        t0 = time.perf_counter()

        cache_valid = (
            use_cache
            and all(os.path.exists(p) for p in
                    (self.cfg.FAISS_INDEX_PATH, self.cfg.EMBEDDINGS_PATH,
                     self.cfg.BM25_PATH, self.cfg.DOC_META_PATH))
        )
        if cache_valid:
            try:
                with open(self.cfg.DOC_META_PATH, "rb") as f:
                    meta = pickle.load(f)
                if (meta.get("n_docs") == len(documents)
                        and meta.get("embedding_model") == self.cfg.EMBEDDING_MODEL
                        and meta.get("index_type") == self.cfg.INDEX_TYPE):
                    log.info("Loading caches (matching fingerprint)")
                    self.sparse.load(self.cfg.BM25_PATH)
                    self.dense.load(self.cfg.FAISS_INDEX_PATH, self.cfg.EMBEDDINGS_PATH)
                else:
                    log.info("Cache fingerprint mismatch - rebuilding")
                    self._build_fresh(documents)
            except Exception as e:
                log.warning(f"Cache load failed ({e}); rebuilding")
                self._build_fresh(documents)
        else:
            self._build_fresh(documents)

        # Try to load LTR if enabled
        if self.cfg.USE_LTR:
            self._try_load_ltr()

        log.info(f"Engine ready in {time.perf_counter() - t0:.2f}s "
                 f"(LTR={'on' if self.ltr and self.ltr.is_trained else 'off'})")

    def _build_fresh(self, documents):
        log.info(f"Building indices from scratch for {len(documents)} docs")
        self.sparse.build(documents)
        self.dense.build(documents)
        self._persist()

    def _persist(self):
        self.sparse.save(self.cfg.BM25_PATH)
        self.dense.save(self.cfg.FAISS_INDEX_PATH, self.cfg.EMBEDDINGS_PATH)
        with open(self.cfg.DOC_META_PATH, "wb") as f:
            pickle.dump(
                {"n_docs": len(self.documents),
                 "embedding_model": self.cfg.EMBEDDING_MODEL,
                 "index_type": self.cfg.INDEX_TYPE},
                f,
            )

    def _try_load_ltr(self):
        try:
            if os.path.exists(self.cfg.LTR_MODEL_PATH):
                self.ltr = LTRRanker(self.cfg.LTR_FEATURES, self.cfg.LTR_PARAMS)
                self.ltr.load(self.cfg.LTR_MODEL_PATH)
            else:
                log.info("No LTR model at %s — falling back to weighted fusion.",
                         self.cfg.LTR_MODEL_PATH)
        except Exception as e:
            log.warning(f"LTR load failed, using weighted fusion: {e}")
            self.ltr = None

    # ------------------------------------------------------------------ #
    # Dynamic doc insertion
    # ------------------------------------------------------------------ #
    def add_documents(self, new_docs: List[dict]) -> List[int]:
        start = len(self.documents)
        self.documents.extend(new_docs)
        self.sparse.build(self.documents)
        self.dense.add_documents(new_docs)
        self._persist()
        return list(range(start, len(self.documents)))

    # ------------------------------------------------------------------ #
    # Candidate generation (shared between search() and LTR trainer)
    # ------------------------------------------------------------------ #
    def generate_candidates(self, query: str):
        """Return everything the ranker needs. Used internally AND by the
        LTR trainer so features are identical at train and inference time."""
        t = {}
        t0 = time.perf_counter()
        q = self.processor.process(query)
        t["query_proc_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        sparse_hits = self.sparse.search(q["expanded_tokens"],
                                         top_k=self.cfg.BM25_TOP_K)
        t["bm25_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        dense_hits = self.dense.search(q["original"], top_k=self.cfg.DENSE_TOP_K)
        t["dense_ms"] = (time.perf_counter() - t0) * 1000

        sparse_map, dense_map = dict(sparse_hits), dict(dense_hits)
        candidate_ids = list(set(sparse_map) | set(dense_map))
        if not candidate_ids:
            return q, [], sparse_map, dense_map, {}, t

        t0 = time.perf_counter()
        cand_texts = [(cid, self.documents[cid]["text"]) for cid in candidate_ids]
        ce_ranked = self.reranker.rerank(q["original"], cand_texts)
        t["cross_ms"] = (time.perf_counter() - t0) * 1000
        ce_map = dict(ce_ranked)

        return q, candidate_ids, sparse_map, dense_map, ce_map, t

    # ------------------------------------------------------------------ #
    # Main search
    # ------------------------------------------------------------------ #
    def search(self, query: str, top_k: Optional[int] = None, explain: bool = True):
        top_k = top_k or self.cfg.FINAL_TOP_K
        if not query or not query.strip():
            return []

        t_total = time.perf_counter()
        try:
            q, ids, sparse_map, dense_map, ce_map, stage_ms = \
                self.generate_candidates(query)
        except Exception as e:
            log.exception(f"Candidate generation failed: {e}")
            return []

        if not ids:
            log.info("No candidates for query: %r", query)
            return []

        # Raw per-candidate feature arrays
        query_length = len(q["core_tokens"])
        feature_rows = []
        bm25_raw = [sparse_map.get(cid, 0.0) for cid in ids]
        dense_raw = [dense_map.get(cid, 0.0) for cid in ids]
        ce_raw = [ce_map.get(cid, 0.0) for cid in ids]

        overlaps = []
        doc_lengths = []
        for i, cid in enumerate(ids):
            matching = self.sparse.matching_keywords(q["expanded_tokens"], cid)
            overlap_count = len(matching)
            overlaps.append(matching)
            doc_lengths.append(self.sparse.doc_length(cid))
            feature_rows.append(
                extract_features(
                    bm25_raw=bm25_raw[i],
                    dense_raw=dense_raw[i],
                    cross_raw=ce_raw[i],
                    doc_length=doc_lengths[i],
                    query_length=query_length,
                    overlap_count=overlap_count,
                )
            )

        # ---- Final scoring: LTR if available, else normalised weighted sum - #
        t0 = time.perf_counter()
        if self.ltr is not None and self.ltr.is_trained:
            final_scores = self.ltr.predict(feature_rows).tolist()
            scorer_used = "ltr"
        else:
            bm25_n = min_max_norm(bm25_raw)
            dense_n = min_max_norm(dense_raw)
            ce_n = min_max_norm(ce_raw)
            w = self.cfg
            final_scores = [
                w.W_BM25 * bm25_n[i] + w.W_DENSE * dense_n[i] + w.W_CROSS * ce_n[i]
                for i in range(len(ids))
            ]
            scorer_used = "weighted"
        stage_ms["fusion_ms"] = (time.perf_counter() - t0) * 1000

        # For explanation we also compute the normalised values
        bm25_n = min_max_norm(bm25_raw)
        dense_n = min_max_norm(dense_raw)
        ce_n = min_max_norm(ce_raw)

        results = []
        for i, cid in enumerate(ids):
            doc = self.documents[cid]
            res = {
                "doc_id": cid,
                "title": doc.get("title", f"Document {cid}"),
                "text": doc["text"],
                "final_score": float(final_scores[i]),
            }
            if explain:
                res["explanation"] = {
                    "bm25_raw": bm25_raw[i], "bm25_norm": bm25_n[i],
                    "dense_raw": dense_raw[i], "dense_norm": dense_n[i],
                    "cross_raw": ce_raw[i], "cross_norm": ce_n[i],
                    "doc_length": doc_lengths[i],
                    "query_length": query_length,
                    "overlap_count": len(overlaps[i]),
                    "matching_keywords": overlaps[i],
                    "core_query_tokens": q["core_tokens"],
                    "expanded_query_tokens": q["expanded_tokens"],
                    "scorer": scorer_used,
                    "weights": {"bm25": self.cfg.W_BM25,
                                "dense": self.cfg.W_DENSE,
                                "cross": self.cfg.W_CROSS},
                }
            results.append(res)

        results.sort(key=lambda r: r["final_score"], reverse=True)
        results = results[:top_k]

        stage_ms["total_ms"] = (time.perf_counter() - t_total) * 1000
        for r in results:
            r["timings"] = stage_ms  # same dict on every row is fine
        log.info(
            "query=%r  cands=%d  scorer=%s  bm25=%.1fms  dense=%.1fms  "
            "cross=%.1fms  fusion=%.2fms  total=%.1fms",
            query, len(ids), scorer_used,
            stage_ms.get("bm25_ms", 0), stage_ms.get("dense_ms", 0),
            stage_ms.get("cross_ms", 0), stage_ms.get("fusion_ms", 0),
            stage_ms.get("total_ms", 0),
        )
        return results

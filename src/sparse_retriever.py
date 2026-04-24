"""
sparse_retriever.py  (v2)
-------------------------
Same BM25 + inverted-index logic as v1, plus:
  * Timing of build() and search()
  * Proper logging
  * Safer save/load with try/except
"""
import pickle
import time
from collections import defaultdict
from typing import List, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

from .logger import get_logger

log = get_logger(__name__)


class SparseRetriever:
    def __init__(self, processor):
        self.processor = processor
        self.bm25: BM25Okapi | None = None
        self.tokenized_corpus: List[List[str]] = []
        self.inverted_index: dict = defaultdict(set)

    def build(self, documents: List[dict]):
        t0 = time.perf_counter()
        self.tokenized_corpus = [
            self.processor.process_document(d["text"]) for d in documents
        ]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        self.inverted_index = defaultdict(set)
        for doc_id, toks in enumerate(self.tokenized_corpus):
            for t in toks:
                self.inverted_index[t].add(doc_id)
        log.info(f"BM25 built over {len(documents)} docs "
                 f"(vocab={len(self.inverted_index)}) in "
                 f"{time.perf_counter() - t0:.2f}s")

    def search(self, query_tokens: List[str], top_k: int = 50
               ) -> List[Tuple[int, float]]:
        if not self.bm25 or not query_tokens:
            return []
        scores = self.bm25.get_scores(query_tokens)
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_idx if scores[i] > 0.0]

    def matching_keywords(self, query_tokens: List[str], doc_id: int) -> List[str]:
        doc_tokens = set(self.tokenized_corpus[doc_id])
        seen, out = set(), []
        for t in query_tokens:
            if t in doc_tokens and t not in seen:
                out.append(t)
                seen.add(t)
        return out

    def doc_length(self, doc_id: int) -> int:
        """Number of tokens in a doc - used as an LTR feature."""
        return len(self.tokenized_corpus[doc_id])

    def save(self, path: str):
        try:
            with open(path, "wb") as f:
                pickle.dump(
                    {"tokenized_corpus": self.tokenized_corpus,
                     "inverted_index": {k: list(v)
                                        for k, v in self.inverted_index.items()}},
                    f,
                )
            log.info(f"BM25 saved -> {path}")
        except Exception as e:
            log.error(f"BM25 save failed: {e}")
            raise

    def load(self, path: str):
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            self.tokenized_corpus = data["tokenized_corpus"]
            self.inverted_index = defaultdict(
                set, {k: set(v) for k, v in data["inverted_index"].items()}
            )
            self.bm25 = BM25Okapi(self.tokenized_corpus)
            log.info(f"BM25 loaded ({len(self.tokenized_corpus)} docs)")
        except Exception as e:
            log.error(f"BM25 load failed: {e}")
            raise

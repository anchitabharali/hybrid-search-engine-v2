"""
reranker.py  (v2)
-----------------
Upgrades vs. v1:
  * Lazy model loading via @cached_property (no HF download until first use).
  * Logging of model load time + rerank latency.
  * Graceful error propagation.
"""
import time
from functools import cached_property
from typing import List, Tuple

from .logger import get_logger

log = get_logger(__name__)


class CrossEncoderReranker:
    def __init__(self, model_name: str):
        self.model_name = model_name

    @cached_property
    def model(self):
        log.info(f"Loading CrossEncoder: {self.model_name}")
        t0 = time.perf_counter()
        try:
            from sentence_transformers import CrossEncoder
            m = CrossEncoder(self.model_name)
        except Exception as e:
            log.error(f"Failed to load cross-encoder: {e}")
            raise
        log.info(f"CrossEncoder ready in {time.perf_counter() - t0:.2f}s")
        return m

    def rerank(self, query: str, candidates: List[Tuple[int, str]]
               ) -> List[Tuple[int, float]]:
        if not candidates:
            return []
        t0 = time.perf_counter()
        pairs = [(query, text) for _, text in candidates]
        scores = self.model.predict(pairs, show_progress_bar=False)
        log.debug(f"CE scored {len(pairs)} pairs in "
                  f"{time.perf_counter() - t0:.3f}s")
        scored = [(candidates[i][0], float(scores[i])) for i in range(len(candidates))]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

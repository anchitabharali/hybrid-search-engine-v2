"""
evaluator.py
------------
Standard IR evaluation metrics:

  * Precision@K :  fraction of top-K results that are relevant.
  * Recall@K    :  fraction of the known relevant docs recovered in top-K.
  * NDCG@K      :  rank-aware metric - rewards relevant hits near the top.

These are computed against a small hand-crafted gold-standard file
(tests/test_queries.json) so you can see how the hybrid engine behaves
against each stage individually.
"""
import math
from typing import List, Iterable


# ---------------------------------------------------------------------------
# Per-query metrics
# ---------------------------------------------------------------------------
def precision_at_k(retrieved: List[int], relevant: Iterable[int], k: int) -> float:
    if k <= 0:
        return 0.0
    retrieved_k = retrieved[:k]
    relevant_set = set(relevant)
    if not retrieved_k:
        return 0.0
    hits = sum(1 for d in retrieved_k if d in relevant_set)
    return hits / k


def recall_at_k(retrieved: List[int], relevant: Iterable[int], k: int) -> float:
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    retrieved_k = retrieved[:k]
    hits = sum(1 for d in retrieved_k if d in relevant_set)
    return hits / len(relevant_set)


def dcg_at_k(retrieved: List[int], relevant: Iterable[int], k: int) -> float:
    """Binary-relevance DCG. Position i (0-indexed) contributes 1/log2(i+2)."""
    relevant_set = set(relevant)
    dcg = 0.0
    for i, doc in enumerate(retrieved[:k]):
        rel = 1.0 if doc in relevant_set else 0.0
        if rel > 0:
            dcg += rel / math.log2(i + 2)
    return dcg


def ndcg_at_k(retrieved: List[int], relevant: Iterable[int], k: int) -> float:
    ideal_hits = min(len(set(relevant)), k)
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg_at_k(retrieved, relevant, k) / idcg


# ---------------------------------------------------------------------------
# Aggregate evaluation over a test set
# ---------------------------------------------------------------------------
def evaluate(engine, test_cases: List[dict], k: int = 5) -> dict:
    """
    Each test case is a dict:
        {"query": "...", "relevant_doc_ids": [id1, id2, ...]}
    """
    agg = {"precision": 0.0, "recall": 0.0, "ndcg": 0.0}
    per_query = []

    for tc in test_cases:
        results = engine.search(tc["query"], top_k=k, explain=False)
        retrieved_ids = [r["doc_id"] for r in results]
        rel = tc["relevant_doc_ids"]

        p = precision_at_k(retrieved_ids, rel, k)
        r = recall_at_k(retrieved_ids, rel, k)
        n = ndcg_at_k(retrieved_ids, rel, k)

        per_query.append(
            {
                "query": tc["query"],
                "precision": p,
                "recall": r,
                "ndcg": n,
                "retrieved": retrieved_ids,
                "relevant": rel,
            }
        )
        agg["precision"] += p
        agg["recall"] += r
        agg["ndcg"] += n

    n_cases = max(len(test_cases), 1)
    for key in agg:
        agg[key] /= n_cases

    return {"mean": agg, "per_query": per_query, "k": k}

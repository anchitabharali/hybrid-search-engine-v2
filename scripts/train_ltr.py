"""
scripts/train_ltr.py  (NEW)
---------------------------
Train the LightGBM LambdaRank model on top of the running hybrid engine.

Two labeling modes
------------------
--mode manual    (default)
    Uses tests/train_queries.json, where each entry is
        {"query": "...", "relevant_doc_ids": [id, id, ...]}
    Doc IDs must be valid IDs in the current corpus. This is the cleanest
    way to train if you have human relevance judgments.

--mode pseudo
    No labels needed. For each query in train_queries.json we run the full
    hybrid pipeline and TREAT THE TOP-K CROSS-ENCODER HITS AS POSITIVES
    (weak supervision / teacher-student distillation). Useful when you
    swap in a new corpus (e.g. 20 Newsgroups) and your hand-labelled IDs
    no longer map.

Usage
-----
    python scripts/train_ltr.py                    # manual labels
    python scripts/train_ltr.py --mode pseudo --pos_k 5
    python scripts/train_ltr.py --val-ratio 0.2
"""
import argparse
import json
import os
import random
import sys

# Make `import config` / `from src...` work when run as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from src.hybrid_search import HybridSearchEngine  # noqa: E402
from src.ltr import LTRRanker, extract_features  # noqa: E402
from src.evaluator import evaluate  # noqa: E402
from src.logger import get_logger  # noqa: E402

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Feature-set builder for a single query
# ---------------------------------------------------------------------------
def build_query_features(engine: HybridSearchEngine, query: str, relevant_ids: set):
    """
    Run the full candidate-generation pipeline and return:
        features  : list[dict]
        labels    : list[int]  (1 if doc_id in relevant_ids else 0)
        doc_ids   : list[int]
    The group size for LightGBM is len(features).
    """
    q, candidate_ids, sparse_map, dense_map, ce_map, _ = \
        engine.generate_candidates(query)

    features, labels, doc_ids = [], [], []
    query_length = len(q["core_tokens"])

    for cid in candidate_ids:
        matching = engine.sparse.matching_keywords(q["expanded_tokens"], cid)
        f = extract_features(
            bm25_raw=sparse_map.get(cid, 0.0),
            dense_raw=dense_map.get(cid, 0.0),
            cross_raw=ce_map.get(cid, 0.0),
            doc_length=engine.sparse.doc_length(cid),
            query_length=query_length,
            overlap_count=len(matching),
        )
        features.append(f)
        labels.append(1 if cid in relevant_ids else 0)
        doc_ids.append(cid)
    return features, labels, doc_ids, ce_map


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["manual", "pseudo"], default="manual")
    p.add_argument("--pos_k", type=int, default=5,
                   help="Top-K cross-encoder hits treated as positives in pseudo mode.")
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    # Load documents + engine
    with open(config.DOCUMENTS_PATH, "r", encoding="utf-8") as f:
        documents = json.load(f)
    log.info(f"Corpus size: {len(documents)}")

    # Disable LTR during training so candidate generation uses the untrained engine
    prev_use_ltr = config.USE_LTR
    config.USE_LTR = False
    engine = HybridSearchEngine(config)
    engine.fit(documents, use_cache=True)
    config.USE_LTR = prev_use_ltr

    # Load training queries
    if not os.path.exists(config.TRAIN_QUERIES_PATH):
        log.error(f"No training queries at {config.TRAIN_QUERIES_PATH}. "
                  "See tests/train_queries.json.")
        sys.exit(1)
    with open(config.TRAIN_QUERIES_PATH, "r", encoding="utf-8") as f:
        train_cases = json.load(f)
    log.info(f"Training queries: {len(train_cases)}")

    random.seed(args.seed)
    random.shuffle(train_cases)
    n_val = max(1, int(len(train_cases) * args.val_ratio))
    val_cases = train_cases[:n_val]
    tr_cases = train_cases[n_val:]
    log.info(f"Split: train={len(tr_cases)}  val={len(val_cases)}")

    def collect(cases):
        feats_all, labels_all, groups = [], [], []
        for tc in cases:
            q = tc["query"]
            if args.mode == "manual":
                relevant_ids = set(tc.get("relevant_doc_ids", []))
                if not relevant_ids:
                    continue
                feats, labels, _, _ = build_query_features(engine, q, relevant_ids)
            else:  # pseudo
                feats, labels, doc_ids, ce_map = \
                    build_query_features(engine, q, set())
                # Take top pos_k cross-encoder scores as positives
                top_ce = sorted(ce_map.items(), key=lambda x: x[1], reverse=True)
                positives = {did for did, _ in top_ce[: args.pos_k]}
                labels = [1 if did in positives else 0 for did in doc_ids]
            if not feats or sum(labels) == 0:
                continue
            feats_all.extend(feats)
            labels_all.extend(labels)
            groups.append(len(feats))
        return feats_all, labels_all, groups

    log.info("Collecting training features...")
    tr_feats, tr_labels, tr_groups = collect(tr_cases)
    log.info("Collecting validation features...")
    va_feats, va_labels, va_groups = collect(val_cases)

    if not tr_groups:
        log.error("No usable training queries - check your labels / corpus.")
        sys.exit(2)

    # Train LambdaRank
    ranker = LTRRanker(config.LTR_FEATURES, config.LTR_PARAMS)
    ranker.train(
        tr_feats, tr_labels, tr_groups,
        val_features=va_feats, val_labels=va_labels, val_groups=va_groups,
        num_rounds=config.LTR_NUM_ROUNDS,
        early_stopping=config.LTR_EARLY_STOPPING,
    )
    ranker.save(config.LTR_MODEL_PATH)

    print("\n=== Feature importance (gain) ===")
    for name, imp in sorted(ranker.feature_importance().items(),
                            key=lambda x: -x[1]):
        print(f"  {name:15s}  {imp:.1f}")

    # Held-out evaluation (weighted vs. LTR)
    if os.path.exists(config.TEST_QUERIES_PATH) and args.mode == "manual":
        with open(config.TEST_QUERIES_PATH, "r", encoding="utf-8") as f:
            test_cases = json.load(f)

        print("\n=== Evaluation on test set (k=5) ===")
        # before (weighted) - temporarily disable LTR
        engine.ltr = None
        before = evaluate(engine, test_cases, k=5)["mean"]
        # after (LTR)
        engine.ltr = ranker
        after = evaluate(engine, test_cases, k=5)["mean"]

        print(f"                  P@5     R@5    NDCG@5")
        print(f"  weighted:     {before['precision']:.3f}  "
              f"{before['recall']:.3f}  {before['ndcg']:.3f}")
        print(f"  LTR      :    {after['precision']:.3f}  "
              f"{after['recall']:.3f}  {after['ndcg']:.3f}")


if __name__ == "__main__":
    main()

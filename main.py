"""
main.py (v2)
------------
CLI driver. New in v2:
  * `--train-ltr` trains the LightGBM LambdaRank model in-process.
  * Per-query response includes a timing breakdown.
  * Uses src/data_loader for CSV/JSON/TXT corpora via `--corpus`.
  * Full logging through src/logger.

Usage
-----
    python main.py -q "how does BM25 work"
    python main.py --eval
    python main.py --rebuild
    python main.py --add "Title::Text" "Title2::Text2"
    python main.py --train-ltr                     # manual labels
    python main.py --train-ltr --ltr-mode pseudo   # weak supervision
    python main.py --corpus data/my_corpus.csv -q "machine learning"
"""
import argparse
import json
import os
import subprocess
import sys

import config
from src.hybrid_search import HybridSearchEngine
from src.evaluator import evaluate
from src.data_loader import load_documents
from src.logger import get_logger

log = get_logger(__name__)


def save_documents(path, docs):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(docs, f, indent=2, ensure_ascii=False)


def print_result(rank, r):
    print(f"\n[{rank}] doc_id={r['doc_id']}  final_score={r['final_score']:.4f}")
    print(f"    Title : {r['title']}")
    snippet = r["text"][:180].replace("\n", " ")
    print(f"    Text  : {snippet}{'...' if len(r['text']) > 180 else ''}")
    if "explanation" in r:
        e = r["explanation"]
        print(f"    Scorer: {e['scorer']}")
        print(f"    BM25  raw={e['bm25_raw']:.3f}  norm={e['bm25_norm']:.3f}")
        print(f"    Dense raw={e['dense_raw']:.3f}  norm={e['dense_norm']:.3f}")
        print(f"    Cross raw={e['cross_raw']:.3f}  norm={e['cross_norm']:.3f}")
        print(f"    doc_len={e['doc_length']}  q_len={e['query_length']}  "
              f"overlap={e['overlap_count']}")
        print(f"    Matching keywords: {e['matching_keywords']}")


def print_timings(timings):
    print("\n--- Timing breakdown ---")
    for k in ("query_proc_ms", "bm25_ms", "dense_ms", "cross_ms",
              "fusion_ms", "total_ms"):
        if k in timings:
            print(f"  {k:18s} {timings[k]:8.2f} ms")


def main():
    parser = argparse.ArgumentParser(description="Hybrid Search Engine v2 CLI")
    parser.add_argument("--query", "-q", type=str, help="Run a single query.")
    parser.add_argument("--top_k", "-k", type=int, default=config.FINAL_TOP_K)
    parser.add_argument("--eval", action="store_true",
                        help="Run evaluation on tests/test_queries.json.")
    parser.add_argument("--rebuild", action="store_true",
                        help="Ignore caches and rebuild from scratch.")
    parser.add_argument("--add", nargs="+",
                        help="Add documents with 'Title::Text' syntax.")
    parser.add_argument("--corpus", type=str, default=None,
                        help="Path to CSV/JSON/TXT corpus (default documents.json).")
    parser.add_argument("--text-col", type=str, default="text")
    parser.add_argument("--title-col", type=str, default="title")
    parser.add_argument("--train-ltr", action="store_true",
                        help="Train the LightGBM LambdaRank model.")
    parser.add_argument("--ltr-mode", choices=["manual", "pseudo"], default="manual")
    parser.add_argument("--no-ltr", action="store_true",
                        help="Disable LTR scoring for this run (use weighted fusion).")
    args = parser.parse_args()

    if args.train_ltr:
        # delegate to the training script so logs / feature importance print cleanly
        cmd = [sys.executable, "scripts/train_ltr.py", "--mode", args.ltr_mode]
        log.info(f"Running: {' '.join(cmd)}")
        sys.exit(subprocess.call(cmd))

    if args.rebuild:
        for p in (config.FAISS_INDEX_PATH, config.EMBEDDINGS_PATH,
                  config.BM25_PATH, config.DOC_META_PATH):
            if os.path.exists(p):
                os.remove(p)
        log.info("Caches cleared.")

    # Load corpus -------------------------------------------------------- #
    corpus_path = args.corpus or config.DOCUMENTS_PATH
    if not os.path.exists(corpus_path):
        log.error(f"Corpus not found: {corpus_path}. "
                  "Run: python scripts/prepare_dataset.py")
        sys.exit(1)
    documents = load_documents(corpus_path, text_col=args.text_col,
                               title_col=args.title_col)

    if args.no_ltr:
        config.USE_LTR = False

    engine = HybridSearchEngine(config)
    engine.fit(documents, use_cache=not args.rebuild)

    # Dynamic add -------------------------------------------------------- #
    if args.add:
        new_docs = []
        start_id = len(documents)
        for i, entry in enumerate(args.add):
            title, text = (entry.split("::", 1) if "::" in entry
                           else (f"New Doc {start_id + i}", entry))
            new_docs.append({"id": start_id + i,
                             "title": title.strip(),
                             "text": text.strip()})
        engine.add_documents(new_docs)
        documents.extend(new_docs)
        # persist only if we're editing the default JSON corpus
        if corpus_path == config.DOCUMENTS_PATH:
            save_documents(corpus_path, documents)
        log.info(f"Added {len(new_docs)} new documents")

    # Evaluation --------------------------------------------------------- #
    if args.eval:
        with open(config.TEST_QUERIES_PATH, "r", encoding="utf-8") as f:
            test_cases = json.load(f)
        report = evaluate(engine, test_cases, k=5)
        m = report["mean"]
        print(f"\n=== Evaluation (k=5) ===")
        print(f"  Precision@5 : {m['precision']:.4f}")
        print(f"  Recall@5    : {m['recall']:.4f}")
        print(f"  NDCG@5      : {m['ndcg']:.4f}")
        scorer = ("LTR" if engine.ltr and engine.ltr.is_trained else "weighted")
        print(f"  Scorer used : {scorer}")
        print("\nPer-query:")
        for pq in report["per_query"]:
            print(f"  {pq['query'][:55]:55s}  "
                  f"P={pq['precision']:.2f} R={pq['recall']:.2f} "
                  f"NDCG={pq['ndcg']:.2f}")

    # Single query ------------------------------------------------------- #
    if args.query:
        print(f"\n[query] {args.query}")
        results = engine.search(args.query, top_k=args.top_k, explain=True)
        if not results:
            print("  (no results)")
            return
        for i, r in enumerate(results, 1):
            print_result(i, r)
        print_timings(results[0].get("timings", {}))
        print(f"\nEmbedding cache: {engine.dense.cache_stats()}")


if __name__ == "__main__":
    main()

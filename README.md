# 🔎 Hybrid Search Engine · v2

A production-grade mini search engine built on five techniques working together:

```
BM25  +  FAISS HNSW  +  Cross-Encoder  +  LightGBM LambdaRank  +  POS-filtered WordNet expansion
```

With lazy model loading, disk-persisted caches, per-stage latency telemetry,
rotating logs, and a Streamlit UI with inline keyword highlighting.

---

## What changed from v1

| # | Upgrade | File(s) |
|---|---------|---------|
| 1 | 1500-doc corpus from 20 Newsgroups | `scripts/prepare_dataset.py`, `src/data_loader.py` |
| 2 | `IndexHNSWFlat` replaces `IndexFlatIP` (O(log N) queries, ~98% recall) | `src/dense_retriever.py` |
| 3 | POS-filtered WordNet expansion + embedding-similarity filter | `src/query_processor.py` |
| 4 | **LightGBM LambdaRank** replaces hand-weighted fusion | `src/ltr.py`, `scripts/train_ltr.py` |
| 5 | On-disk embedding cache + `@cached_property` lazy loading | `src/dense_retriever.py`, `src/reranker.py` |
| 6 | Central rotating logger, try/except around every I/O | `src/logger.py` (all modules) |
| 7 | Per-stage latency recorded on every query | `src/hybrid_search.py` |
| 8 | Streamlit: `<mark>` highlighting, history, latency row, LTR toggle | `app.py` |

---

## Architecture

```
                             ┌──────────────────┐
                             │   User Query     │
                             └────────┬─────────┘
                                      ▼
              ┌──────────────────────────────────────────────┐
              │ Stage 0 — Query Processor                    │
              │  tokenize → stopword → lemmatize             │
              │  → POS tag → expand NOUN/VERB via WordNet    │
              │  → embedding-similarity filter (cosine)      │
              └──────────┬───────────────────────┬───────────┘
                         │ expanded tokens        │ raw text
                         ▼                        ▼
              ┌──────────────────┐    ┌─────────────────────────┐
              │ Stage 1 — BM25   │    │ Stage 2 — FAISS (HNSW)  │
              │  inverted index  │    │  MiniLM embeddings +    │
              │  top-100 cands   │    │  embedding cache        │
              └───────────┬──────┘    └────────────┬────────────┘
                          └────────────┬───────────┘
                                       ▼
                          ┌────────────────────────┐
                          │ Stage 3 — Cross-Encoder│
                          │ ms-marco-MiniLM-L-6-v2 │
                          │ rerank the union       │
                          └────────────┬───────────┘
                                       ▼
                          ┌────────────────────────┐
                          │ Stage 4 — Fusion       │
                          │  LightGBM LambdaRank   │
                          │  (7 features)          │
                          │      — or fallback —   │
                          │  weighted sum of norms │
                          └────────────┬───────────┘
                                       ▼
                          Top-K results + explanations
```

### Why HNSW instead of `IndexFlatIP`?

| Index | Recall@10 | Query time | Memory | Notes |
|-------|-----------|------------|--------|-------|
| `IndexFlatIP` | 100% | O(N·d) | N·d floats | Exact. Fine up to ~50 K docs. |
| **`IndexHNSWFlat`** *(our default)* | ~98–99% | **O(log N · d · M)** | ~1.5× flat | Graph-based ANN, scales to ~10 M on CPU. |
| `IndexIVFFlat` | 90–98% (tunable) | O((N/nlist)·d·nprobe) | similar | Best for very large corpora (>10 M). |
| `IndexIVFPQ` | 85–95% | very fast | ~10× smaller | Compressed vectors; billion-scale. |

v2 defaults to HNSW (`M=32`, `efConstruction=200`, `efSearch=64`) — the sweet spot
between recall and latency for 1 K–1 M documents. Switch via `config.INDEX_TYPE`.

### Why Learning-to-Rank instead of weighted fusion?

Hand-tuned weights (`0.2·bm25 + 0.3·dense + 0.5·cross`) assume a *linear* relationship
and can't exploit feature interactions like "when overlap is high, trust BM25 more".
LambdaRank fits a gradient-boosted tree ensemble directly against NDCG, using
**7 features**: `bm25_raw`, `dense_raw`, `cross_raw`, `doc_length`, `query_length`,
`overlap_count`, `overlap_ratio`. The trainer prints a feature-importance table
and compares NDCG@5 before/after.

---

## Project layout

```
hybrid_search_engine/
├── app.py                          # Streamlit v2 UI
├── main.py                         # CLI with timings + LTR subcommand
├── config.py                       # All knobs (HNSW, LTR, expansion, logging)
├── requirements.txt
├── README.md
├── data/
│   └── documents.json              # 30 seed docs (run prepare_dataset for 1500)
├── tests/
│   ├── test_queries.json           # 10 held-out queries
│   └── train_queries.json          # 30 training queries for LTR
├── scripts/
│   ├── prepare_dataset.py          # Fetch 20NG → documents.json
│   └── train_ltr.py                # Train LambdaRank
├── src/
│   ├── query_processor.py          # v2: POS + embedding filter
│   ├── sparse_retriever.py         # BM25 + timing
│   ├── dense_retriever.py          # v2: HNSW/IVF + lazy + cache
│   ├── reranker.py                 # Cross-encoder (lazy)
│   ├── hybrid_search.py            # Orchestrator + LTR fallback + timings
│   ├── ltr.py                      # NEW: LightGBM LambdaRank
│   ├── cache.py                    # NEW: embedding cache
│   ├── data_loader.py              # NEW: CSV/JSON/TXT loader
│   ├── evaluator.py                # P@K, R@K, NDCG@K
│   └── logger.py                   # NEW: central logging
├── cache/                          # faiss.index, embeddings.npy, bm25.pkl, ltr_model.lgb
└── logs/                           # Rotating search.log
```

---

## 🚀 Quickstart

```bash
# 1. Install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. (Optional) Build the 1500-doc corpus from 20 Newsgroups
python scripts/prepare_dataset.py --size 1500

# 3. (Optional) Train the LTR model on the 30 bundled training queries
python scripts/train_ltr.py                 # manual labels
# or, after swapping to a new corpus:
python scripts/train_ltr.py --mode pseudo   # cross-encoder as teacher

# 4. Launch the UI
streamlit run app.py
```

Or use the CLI:

```bash
python main.py -q "how does BM25 work" -k 5
python main.py --eval
python main.py --corpus data/my_papers.csv --text-col abstract -q "transformer"
python main.py --rebuild                # clear caches and rebuild
python main.py --no-ltr -q "..."        # force weighted fusion
```

---

## 📊 Expected output

### Single query (with timings)

```
[query] how does BM25 work

[1] doc_id=3  final_score=6.7421
    Title : BM25 and Probabilistic Retrieval
    Text  : BM25 is a ranking function used by search engines to estimate...
    Scorer: ltr
    BM25  raw=7.92  norm=1.00
    Dense raw=0.68  norm=1.00
    Cross raw=8.14  norm=1.00
    doc_len=45  q_len=3  overlap=2
    Matching keywords: ['bm25', 'rank']

--- Timing breakdown ---
  query_proc_ms        8.42 ms
  bm25_ms              1.74 ms
  dense_ms             6.21 ms
  cross_ms           118.35 ms    ← dominates (expected)
  fusion_ms            0.81 ms
  total_ms           135.89 ms
```

### After LTR training

```
=== Feature importance (gain) ===
  cross_raw         2184.3
  dense_raw          687.2
  bm25_raw           611.8
  overlap_ratio      318.5
  doc_length         124.1
  overlap_count       82.7
  query_length        19.4

=== Evaluation on test set (k=5) ===
                    P@5     R@5    NDCG@5
  weighted:       0.350   0.850   0.820
  LTR      :      0.410   0.900   0.886        ← NDCG up ~6-8 points
```

---

## 🔧 Tuning reference

All knobs live in `config.py`:

| Knob | Effect |
|------|--------|
| `INDEX_TYPE` | `"hnsw"` (default) / `"ivf"` / `"flat"` |
| `HNSW_M` | More = higher recall, more memory. 32 is standard. |
| `HNSW_EF_SEARCH` | More = higher recall, slower query. 64 is a good default. |
| `IVF_NLIST` / `IVF_NPROBE` | ~√N / 8-32 for IVF. |
| `USE_LTR` | Auto-falls back if `cache/ltr_model.lgb` is missing. |
| `BM25_TOP_K` / `DENSE_TOP_K` | Shrink for lower latency, grow for higher recall. |
| `EXPANSION_SIMILARITY_THRESHOLD` | 0.5–0.7 typical. Lower = more expansions. |
| `EXPANSION_MAX_PER_TOKEN` | 1–3. More dilutes BM25 scores. |

---

## 📈 Scaling guide

| Corpus size | Recommended setup |
|-------------|--------------------|
| < 50 K     | `INDEX_TYPE="flat"` (exact), LTR optional |
| 50 K – 1 M | **HNSW M=32 efSearch=64** (v2 default) + LTR |
| 1 M – 100 M | `IVF` with `nlist ≈ √N`, `nprobe=16-32`, LTR essential |
| > 100 M     | `IVFPQ` on `faiss-gpu`, sharded; move BM25 to Elasticsearch |

For BM25 at scale, swap `rank_bm25` for **Elasticsearch**, **Vespa**, or **Pyserini** —
the BM25 API surface matches so `sparse_retriever.py` is the only file you change.

---

## 🧪 Testing the upgrades

```bash
# Latency comparison: flat vs hnsw
python -c "import config; config.INDEX_TYPE='flat'; import subprocess; subprocess.call(['python','main.py','--rebuild','-q','machine learning'])"

# Query that benefits from POS filter (without it, 'work' expands to 'workplace'
# which pollutes BM25)
python main.py -q "how does BM25 work"

# Query that benefits from LTR (doc_length should down-rank short stubs)
python main.py -q "retrieval augmented generation"
```

---

## 📜 License

MIT.

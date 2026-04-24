"""
config.py  (v2)
---------------
Adds: FAISS index type selection, LTR paths, logging, caching, POS/expansion knobs.
All v1 knobs preserved.
"""
import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CACHE_DIR = os.path.join(BASE_DIR, "cache")
TESTS_DIR = os.path.join(BASE_DIR, "tests")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

DOCUMENTS_PATH = os.path.join(DATA_DIR, "documents.json")
TEST_QUERIES_PATH = os.path.join(TESTS_DIR, "test_queries.json")
TRAIN_QUERIES_PATH = os.path.join(TESTS_DIR, "train_queries.json")  # NEW

# Cache artefacts
FAISS_INDEX_PATH = os.path.join(CACHE_DIR, "faiss.index")
EMBEDDINGS_PATH = os.path.join(CACHE_DIR, "embeddings.npy")
BM25_PATH = os.path.join(CACHE_DIR, "bm25.pkl")
DOC_META_PATH = os.path.join(CACHE_DIR, "doc_meta.pkl")
QUERY_EMBED_CACHE_PATH = os.path.join(CACHE_DIR, "query_embed_cache.pkl")  # NEW
LTR_MODEL_PATH = os.path.join(CACHE_DIR, "ltr_model.lgb")                  # NEW

# Logs
LOG_FILE = os.path.join(LOGS_DIR, "search.log")
LOG_LEVEL = "INFO"

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ---------------------------------------------------------------------------
# Retrieval top-K
# ---------------------------------------------------------------------------
BM25_TOP_K = 100
DENSE_TOP_K = 100
FINAL_TOP_K = 10

# ---------------------------------------------------------------------------
# FAISS index configuration  (NEW)
# ---------------------------------------------------------------------------
# "flat" -> exact, slow for large N
# "hnsw" -> graph-based ANN, fast + high recall (our default for 1k-1M docs)
# "ivf"  -> inverted file + flat / PQ, best for >1M docs
INDEX_TYPE = "hnsw"

HNSW_M = 32
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 64

IVF_NLIST = 100
IVF_NPROBE = 8

# ---------------------------------------------------------------------------
# Hybrid fusion weights (fallback when LTR is disabled/untrained)
# ---------------------------------------------------------------------------
W_BM25 = 0.20
W_DENSE = 0.30
W_CROSS = 0.50

# ---------------------------------------------------------------------------
# Learning-to-Rank  (NEW)
# ---------------------------------------------------------------------------
USE_LTR = True

LTR_FEATURES = [
    "bm25_raw", "dense_raw", "cross_raw",
    "doc_length", "query_length",
    "overlap_count", "overlap_ratio",
]

LTR_PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [5, 10],
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 5,
    "feature_fraction": 0.9,
    "verbose": -1,
}
LTR_NUM_ROUNDS = 300
LTR_EARLY_STOPPING = 30

# ---------------------------------------------------------------------------
# Query expansion (NEW controls)
# ---------------------------------------------------------------------------
EXPANSION_ENABLED = True
EXPANSION_POS_FILTER = True
EXPANSION_SIMILARITY_THRESHOLD = 0.55
EXPANSION_MAX_PER_TOKEN = 2
EXPANSION_USE_EMBEDDING_FILTER = True

for d in (CACHE_DIR, LOGS_DIR, DATA_DIR, TESTS_DIR):
    os.makedirs(d, exist_ok=True)

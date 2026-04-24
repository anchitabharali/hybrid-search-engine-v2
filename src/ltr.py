"""
ltr.py (NEW)
------------
LightGBM LambdaRank learning-to-rank model.

Replaces the hand-tuned linear fusion (w1*bm25 + w2*dense + w3*cross) with a
non-linear ranker trained from query-document relevance data.

Features (order must match config.LTR_FEATURES):
    bm25_raw        : BM25 score (raw, not normalised)
    dense_raw       : FAISS inner-product / cosine similarity
    cross_raw       : cross-encoder logit
    doc_length      : token count of the candidate document
    query_length    : token count of the query (post-processing)
    overlap_count   : |query_tokens ∩ doc_tokens|
    overlap_ratio   : overlap_count / max(query_length, 1)

The trainer (scripts/train_ltr.py) builds the training set by:
    1. Running every stage of the hybrid engine on each training query.
    2. Taking the union of top-K BM25 ∪ top-K dense as candidates.
    3. Labeling each candidate: 1 if in relevant_doc_ids else 0.
    4. Fitting LambdaRank with NDCG@k early-stopping on a held-out split.
"""
import os
from typing import Dict, List, Optional

import numpy as np

from .logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def extract_features(
    *,
    bm25_raw: float,
    dense_raw: float,
    cross_raw: float,
    doc_length: int,
    query_length: int,
    overlap_count: int,
) -> Dict[str, float]:
    """Build one feature vector as a dict keyed by config.LTR_FEATURES."""
    overlap_ratio = overlap_count / max(query_length, 1)
    return {
        "bm25_raw": float(bm25_raw),
        "dense_raw": float(dense_raw),
        "cross_raw": float(cross_raw),
        "doc_length": float(doc_length),
        "query_length": float(query_length),
        "overlap_count": float(overlap_count),
        "overlap_ratio": float(overlap_ratio),
    }


def features_to_matrix(feature_dicts: List[dict], feature_names: List[str]
                       ) -> np.ndarray:
    """Convert a list of feature dicts into an (N, F) float32 matrix."""
    return np.asarray(
        [[fd.get(f, 0.0) for f in feature_names] for fd in feature_dicts],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# LTR wrapper
# ---------------------------------------------------------------------------
class LTRRanker:
    """Thin, save/load-friendly wrapper around a LightGBM LambdaRank model."""

    def __init__(self, feature_names: List[str], params: dict):
        self.feature_names = feature_names
        self.params = params
        self.model = None

    # ---- training ----------------------------------------------------- #
    def train(
        self,
        train_features: List[dict],
        train_labels: List[int],
        train_groups: List[int],
        val_features: Optional[List[dict]] = None,
        val_labels: Optional[List[int]] = None,
        val_groups: Optional[List[int]] = None,
        num_rounds: int = 300,
        early_stopping: int = 30,
    ):
        try:
            import lightgbm as lgb
        except ImportError as e:
            raise ImportError(
                "LightGBM is required to train the LTR model. "
                "pip install lightgbm"
            ) from e

        X_tr = features_to_matrix(train_features, self.feature_names)
        y_tr = np.asarray(train_labels, dtype=np.int32)
        train_set = lgb.Dataset(X_tr, label=y_tr, group=train_groups,
                                feature_name=self.feature_names)

        valid_sets, valid_names = [train_set], ["train"]
        if val_features and val_groups:
            X_va = features_to_matrix(val_features, self.feature_names)
            y_va = np.asarray(val_labels, dtype=np.int32)
            valid_sets.append(lgb.Dataset(X_va, label=y_va, group=val_groups,
                                          feature_name=self.feature_names))
            valid_names.append("val")

        callbacks = []
        if len(valid_sets) > 1 and early_stopping:
            callbacks.append(lgb.early_stopping(early_stopping, verbose=False))
        callbacks.append(lgb.log_evaluation(period=50))

        log.info(f"Training LambdaRank on {len(y_tr)} pairs "
                 f"across {len(train_groups)} queries, "
                 f"{X_tr.shape[1]} features.")
        self.model = lgb.train(
            self.params,
            train_set,
            num_boost_round=num_rounds,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )
        log.info("LTR training complete.")
        return self

    # ---- inference ---------------------------------------------------- #
    def predict(self, feature_dicts: List[dict]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("LTR model not loaded - call train() or load().")
        X = features_to_matrix(feature_dicts, self.feature_names)
        return np.asarray(self.model.predict(X), dtype=np.float32)

    # ---- feature importance ------------------------------------------ #
    def feature_importance(self) -> Dict[str, float]:
        if self.model is None:
            return {}
        imp = self.model.feature_importance(importance_type="gain")
        return dict(zip(self.feature_names, imp.tolist()))

    # ---- persistence -------------------------------------------------- #
    def save(self, path: str):
        if self.model is None:
            raise RuntimeError("Cannot save - model not trained.")
        self.model.save_model(path)
        log.info(f"LTR model saved -> {path}")

    def load(self, path: str):
        try:
            import lightgbm as lgb
        except ImportError as e:
            raise ImportError("pip install lightgbm") from e
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        self.model = lgb.Booster(model_file=path)
        log.info(f"LTR model loaded <- {path}")
        return self

    @property
    def is_trained(self) -> bool:
        return self.model is not None

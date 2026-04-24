"""
query_processor.py  (v2)
------------------------
Improvements vs. v1:
  * POS-filtered WordNet expansion - only expand NOUNS and VERBS
    (skipping stopword-adjacent POS tags like DT, IN, CC, etc.)
  * Expansion candidates are scored by cosine similarity to the original token's
    embedding; only those above EXPANSION_SIMILARITY_THRESHOLD survive.
  * Top-N meaningful expansions per token (EXPANSION_MAX_PER_TOKEN).
  * Graceful fallback: if the dense model is not supplied or embedding filter
    is disabled, falls back to pure POS-filtered expansion.
"""
import re
from typing import List, Optional

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from .logger import get_logger

log = get_logger(__name__)

# Map NLTK Penn-Treebank POS prefixes -> WordNet POS.
_POS_MAP = {"N": wordnet.NOUN, "V": wordnet.VERB,
            "J": wordnet.ADJ, "R": wordnet.ADV}


def _ensure_nltk():
    """Download required NLTK data if missing. Safe to call multiple times."""
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
    ]
    for path, pkg in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except Exception as e:
                log.warning(f"NLTK download '{pkg}' failed: {e}")


_ensure_nltk()


class QueryProcessor:
    """Clean → tokenize → stopword → lemmatize → POS-aware WordNet expansion."""

    def __init__(
        self,
        use_expansion: bool = True,
        pos_filter: bool = True,
        max_synonyms: int = 2,
        similarity_threshold: float = 0.55,
        dense_model=None,
    ):
        self.stopwords = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()
        self.use_expansion = use_expansion
        self.pos_filter = pos_filter
        self.max_synonyms = max_synonyms
        self.similarity_threshold = similarity_threshold
        # If supplied, we use it to filter synonyms by cosine similarity.
        self.dense_model = dense_model

    # ------------------------------------------------------------------ #
    # Low-level
    # ------------------------------------------------------------------ #
    def clean(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def tokenize(self, text: str):
        return word_tokenize(text)

    def remove_stopwords(self, tokens):
        return [t for t in tokens if t not in self.stopwords and len(t) > 1]

    def lemmatize(self, tokens):
        return [self.lemmatizer.lemmatize(t) for t in tokens]

    # ------------------------------------------------------------------ #
    # Expansion
    # ------------------------------------------------------------------ #
    def _penn_to_wordnet(self, penn: str):
        return _POS_MAP.get(penn[0].upper()) if penn else None

    def _filter_by_embedding(self, token: str, candidates: List[str]) -> List[str]:
        """Keep candidates whose embedding cosine with `token` >= threshold."""
        if not candidates or self.dense_model is None:
            return candidates
        try:
            embs = self.dense_model.encode([token] + candidates)  # L2-normalised
        except Exception as e:
            log.warning(f"Embedding filter unavailable ({e}); skipping.")
            return candidates
        base = embs[0]
        scored = []
        for cand, emb in zip(candidates, embs[1:]):
            cos = float((base * emb).sum())  # already L2-normalised
            if cos >= self.similarity_threshold:
                scored.append((cand, cos))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [c for c, _ in scored]

    def expand_with_wordnet(self, tokens: List[str]) -> List[str]:
        """POS-filtered, embedding-filtered, top-N synonym expansion."""
        expanded = list(tokens)
        seen = set(tokens)

        # POS-tag once for the whole token list.
        try:
            tagged = nltk.pos_tag(tokens)
        except Exception as e:
            log.warning(f"pos_tag failed ({e}); skipping POS filter.")
            tagged = [(t, "NN") for t in tokens]

        for tok, penn in tagged:
            wn_pos = self._penn_to_wordnet(penn)

            # Only expand nouns / verbs when POS filter is on.
            if self.pos_filter and wn_pos not in (wordnet.NOUN, wordnet.VERB):
                continue

            # Collect candidate synonyms restricted to same POS.
            candidates = []
            synsets = wordnet.synsets(tok, pos=wn_pos) if wn_pos else wordnet.synsets(tok)
            for syn in synsets:
                for lemma in syn.lemmas():
                    word = lemma.name().lower().replace("_", " ")
                    if word != tok and word not in seen and " " not in word and word.isalpha():
                        candidates.append(word)
                        seen.add(word)

            if not candidates:
                continue

            # Embedding-similarity filter (optional).
            if self.dense_model is not None:
                candidates = self._filter_by_embedding(tok, candidates)

            expanded.extend(candidates[: self.max_synonyms])

        return expanded

    # ------------------------------------------------------------------ #
    # Public pipelines
    # ------------------------------------------------------------------ #
    def process(self, query: str) -> dict:
        cleaned = self.clean(query)
        tokens = self.tokenize(cleaned)
        tokens = self.remove_stopwords(tokens)
        tokens = self.lemmatize(tokens)
        core = list(tokens)
        expanded = self.expand_with_wordnet(tokens) if self.use_expansion else list(core)
        return {
            "original": query,
            "cleaned": cleaned,
            "core_tokens": core,
            "expanded_tokens": expanded,
        }

    def process_document(self, text: str):
        cleaned = self.clean(text)
        tokens = self.tokenize(cleaned)
        tokens = self.remove_stopwords(tokens)
        return self.lemmatize(tokens)

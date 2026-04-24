"""
data_loader.py (NEW)
--------------------
Read a corpus from CSV, JSON, or plain-text (one doc per line).

Unified output: list of {"id": int, "title": str, "text": str}.

Examples
--------
>>> load_documents("data/documents.json")
>>> load_documents("/path/to/papers.csv", text_col="abstract", title_col="title")
>>> load_documents("/path/to/corpus.txt")
"""
import csv
import json
import os
from typing import List

from .logger import get_logger

log = get_logger(__name__)

_SUPPORTED = {".json", ".csv", ".tsv", ".txt"}


def load_documents(
    path: str,
    text_col: str = "text",
    title_col: str = "title",
    id_col: str = "id",
    max_docs: int | None = None,
) -> List[dict]:
    """Load documents from CSV / JSON / TXT. Raises FileNotFoundError on missing."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    ext = os.path.splitext(path)[1].lower()
    if ext not in _SUPPORTED:
        raise ValueError(f"Unsupported extension: {ext} (expected one of {_SUPPORTED})")

    log.info(f"Loading documents from {path} (ext={ext})")
    try:
        if ext == ".json":
            docs = _load_json(path, text_col, title_col, id_col)
        elif ext in (".csv", ".tsv"):
            docs = _load_csv(path, text_col, title_col, id_col,
                             delim="\t" if ext == ".tsv" else ",")
        else:  # .txt
            docs = _load_txt(path)
    except Exception as e:
        log.error(f"Failed to load {path}: {e}")
        raise

    if max_docs is not None:
        docs = docs[:max_docs]

    # Ensure contiguous IDs
    for i, d in enumerate(docs):
        d["id"] = i
        d.setdefault("title", f"Document {i}")
        d["text"] = (d.get("text") or "").strip()
    docs = [d for d in docs if d["text"]]
    log.info(f"Loaded {len(docs)} non-empty documents")
    return docs


# ---------------------------------------------------------------------------
# Format-specific readers
# ---------------------------------------------------------------------------
def _load_json(path, text_col, title_col, id_col):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError("JSON corpus must be a list of objects.")
    docs = []
    for row in raw:
        docs.append({
            "id": row.get(id_col),
            "title": row.get(title_col, ""),
            "text": row.get(text_col, ""),
        })
    return docs


def _load_csv(path, text_col, title_col, id_col, delim=","):
    docs = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delim)
        if text_col not in reader.fieldnames:
            raise ValueError(f"CSV must have column '{text_col}'. "
                             f"Found: {reader.fieldnames}")
        for row in reader:
            docs.append({
                "id": row.get(id_col),
                "title": row.get(title_col, ""),
                "text": row.get(text_col, ""),
            })
    return docs


def _load_txt(path):
    with open(path, "r", encoding="utf-8") as f:
        return [{"id": i, "title": f"Line {i}", "text": line.strip()}
                for i, line in enumerate(f) if line.strip()]

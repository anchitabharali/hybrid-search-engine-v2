"""
scripts/prepare_dataset.py  (NEW)
---------------------------------
Build a 1000+ document corpus from sklearn's 20 Newsgroups dataset.

Usage
-----
    python scripts/prepare_dataset.py              # default 1500 docs
    python scripts/prepare_dataset.py --size 3000
    python scripts/prepare_dataset.py --source my.csv --text-col body

Output: data/documents.json, overwriting any existing file.
"""
import argparse
import json
import os
import sys

# Allow `python scripts/prepare_dataset.py` from the project root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from src.data_loader import load_documents  # noqa: E402
from src.logger import get_logger  # noqa: E402

log = get_logger(__name__)


def from_20newsgroups(n: int):
    try:
        from sklearn.datasets import fetch_20newsgroups
    except ImportError as e:
        raise ImportError("pip install scikit-learn") from e

    log.info(f"Fetching 20 Newsgroups ({n} docs)... "
             "(first run downloads ~14 MB to ~/scikit_learn_data)")
    data = fetch_20newsgroups(
        subset="all",
        remove=("headers", "footers", "quotes"),
        random_state=42,
    )
    docs = []
    for i, (text, target) in enumerate(zip(data.data, data.target)):
        text = " ".join(text.split())  # collapse whitespace
        if len(text) < 120:            # drop near-empty posts
            continue
        # Keep a reasonable length - cross-encoder truncates to 512 tokens anyway.
        text = text[:1200]
        title = f"[{data.target_names[target]}] " + text[:80].rsplit(" ", 1)[0] + "..."
        docs.append({"id": len(docs), "title": title, "text": text})
        if len(docs) >= n:
            break
    return docs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=1500,
                        help="Number of documents to keep (default 1500).")
    parser.add_argument("--source", type=str, default=None,
                        help="Optional CSV/JSON/TXT to use instead of 20NG.")
    parser.add_argument("--text-col", type=str, default="text")
    parser.add_argument("--title-col", type=str, default="title")
    parser.add_argument("--output", type=str, default=config.DOCUMENTS_PATH)
    args = parser.parse_args()

    if args.source:
        docs = load_documents(args.source,
                              text_col=args.text_col,
                              title_col=args.title_col,
                              max_docs=args.size)
    else:
        docs = from_20newsgroups(args.size)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)
    log.info(f"Wrote {len(docs)} documents -> {args.output}")


if __name__ == "__main__":
    main()

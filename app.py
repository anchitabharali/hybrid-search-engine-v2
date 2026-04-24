"""
app.py (v2 Streamlit UI)
------------------------
New in v2:
  * Inline keyword highlighting (<mark>) of matching query tokens.
  * Per-stage latency row (query-proc / BM25 / dense / cross-encoder / total).
  * Persistent search history sidebar (click to re-run).
  * LTR on/off toggle with live status indicator.
  * Weight sliders kept as a fallback when LTR is disabled.

Run:
    streamlit run app.py
"""
import html
import json
import re

import pandas as pd
import streamlit as st

import config
from src.hybrid_search import HybridSearchEngine
from src.evaluator import evaluate
from src.data_loader import load_documents

st.set_page_config(page_title="Hybrid Search Engine v2",
                   page_icon="🔎", layout="wide")


# ---------------------------------------------------------------------------
# Engine cache (built once per Streamlit session)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading models and building indices...")
def get_engine():
    docs = load_documents(config.DOCUMENTS_PATH)
    engine = HybridSearchEngine(config)
    engine.fit(docs, use_cache=True)
    return engine


# ---------------------------------------------------------------------------
# Keyword highlighting
# ---------------------------------------------------------------------------
_MARK_STYLE = ("background:#FFE58A; padding:0 2px; border-radius:3px; "
               "color:#000; font-weight:600;")

def highlight_keywords(text: str, keywords: list[str]) -> str:
    """Wrap matched keywords in <mark>. Case-insensitive, word-boundary aware."""
    safe = html.escape(text)
    if not keywords:
        return safe
    # Longer tokens first so substrings don't shadow them.
    for kw in sorted(set(keywords), key=len, reverse=True):
        if not kw.strip():
            continue
        pattern = re.compile(rf"\b({re.escape(kw)})\b", flags=re.IGNORECASE)
        safe = pattern.sub(
            rf'<mark style="{_MARK_STYLE}">\1</mark>', safe
        )
    return safe


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []
if "active_query" not in st.session_state:
    st.session_state.active_query = ""

engine = get_engine()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("⚙️ Settings")

top_k = st.sidebar.slider("Number of results", 1, 20, 5)

ltr_trained = engine.ltr is not None and engine.ltr.is_trained
ltr_on = st.sidebar.toggle(
    f"Use LTR model ({'trained' if ltr_trained else 'not trained'})",
    value=ltr_trained,
    disabled=not ltr_trained,
    help="Train with:  python scripts/train_ltr.py",
)
# Live-toggle: stash/restore the LTR object
if ltr_on and ltr_trained:
    if engine.ltr is None:
        engine._try_load_ltr()
else:
    engine.ltr = None

st.sidebar.markdown("### Fallback fusion weights")
st.sidebar.caption("Used when LTR is off.")
w_bm25  = st.sidebar.slider("BM25 weight",          0.0, 1.0, config.W_BM25,  0.05)
w_dense = st.sidebar.slider("Dense weight",         0.0, 1.0, config.W_DENSE, 0.05)
w_cross = st.sidebar.slider("Cross-encoder weight", 0.0, 1.0, config.W_CROSS, 0.05)
config.W_BM25, config.W_DENSE, config.W_CROSS = w_bm25, w_dense, w_cross

st.sidebar.markdown("---")
st.sidebar.markdown("### 📜 Search history")
if st.session_state.history:
    for i, h in enumerate(st.session_state.history[:10]):
        if st.sidebar.button(h, key=f"hist_{i}", use_container_width=True):
            st.session_state.active_query = h
else:
    st.sidebar.caption("(no queries yet)")

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Corpus size:** {len(engine.documents):,} docs")
st.sidebar.markdown(f"**FAISS index:** `{config.INDEX_TYPE}`")
st.sidebar.markdown(f"**Embedding model:** `{config.EMBEDDING_MODEL}`")
cs = engine.dense.cache_stats()
if cs:
    st.sidebar.markdown(
        f"**Embedding cache:** {cs['hits']} hits / "
        f"{cs['misses']} misses ({cs['hit_rate']:.0%})"
    )

# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------
st.title("🔎 Hybrid Search Engine — v2")
st.caption(
    "BM25 + FAISS (HNSW) + Cross-Encoder re-rank + "
    "LightGBM Learning-to-Rank, with POS-filtered WordNet expansion and "
    "per-stage latency telemetry."
)

tab_search, tab_add, tab_eval = st.tabs(
    ["🔍 Search", "➕ Add Document", "📊 Evaluate"]
)

# ======================================================================= #
# SEARCH
# ======================================================================= #
with tab_search:
    query = st.text_input(
        "Enter your query",
        value=st.session_state.active_query,
        placeholder="e.g. how does BM25 ranking work?",
    )
    go = st.button("Search", type="primary")

    if go and query.strip():
        with st.spinner("Searching..."):
            results = engine.search(query, top_k=top_k, explain=True)
        if query not in st.session_state.history:
            st.session_state.history.insert(0, query)
        st.session_state.active_query = query

        if not results:
            st.warning("No results found.")
        else:
            # -------- Latency row -------------------------------------- #
            timings = results[0].get("timings", {})
            cols = st.columns(5)
            cols[0].metric("Query-proc", f"{timings.get('query_proc_ms', 0):.1f} ms")
            cols[1].metric("BM25",       f"{timings.get('bm25_ms', 0):.1f} ms")
            cols[2].metric("Dense",      f"{timings.get('dense_ms', 0):.1f} ms")
            cols[3].metric("Cross-enc",  f"{timings.get('cross_ms', 0):.1f} ms")
            cols[4].metric("Total",      f"{timings.get('total_ms', 0):.1f} ms")

            # -------- Query processing panel --------------------------- #
            first_exp = results[0]["explanation"]
            with st.expander("🔬 Query processing & scorer", expanded=False):
                st.markdown(f"**Core tokens:** `{first_exp['core_query_tokens']}`")
                st.markdown(
                    f"**Expanded (POS-filtered + embedding-filtered):** "
                    f"`{first_exp['expanded_query_tokens']}`"
                )
                st.markdown(f"**Scorer used:** `{first_exp['scorer']}`")
                if first_exp["scorer"] == "weighted":
                    w = first_exp["weights"]
                    st.markdown(
                        f"**Weights:** BM25={w['bm25']:.2f} · "
                        f"Dense={w['dense']:.2f} · Cross={w['cross']:.2f}"
                    )

            st.markdown(f"### Top {len(results)} results")

            # -------- Results ------------------------------------------ #
            for rank, r in enumerate(results, 1):
                with st.container(border=True):
                    hdr = st.columns([6, 1])
                    hdr[0].markdown(
                        f"**#{rank} · {html.escape(r['title'])}**  "
                        f"<span style='color:#888'>"
                        f"(doc_id={r['doc_id']})</span>",
                        unsafe_allow_html=True,
                    )
                    hdr[1].metric("Final", f"{r['final_score']:.3f}")

                    e = r["explanation"]
                    highlighted = highlight_keywords(r["text"], e["matching_keywords"])
                    st.markdown(
                        f"<div style='line-height:1.5;'>{highlighted}</div>",
                        unsafe_allow_html=True,
                    )

                    m = st.columns(3)
                    m[0].metric("BM25  (raw)",    f"{e['bm25_raw']:.2f}",
                                f"norm {e['bm25_norm']:.2f}")
                    m[1].metric("Dense (raw)",    f"{e['dense_raw']:.2f}",
                                f"norm {e['dense_norm']:.2f}")
                    m[2].metric("CrossEnc (raw)", f"{e['cross_raw']:.2f}",
                                f"norm {e['cross_norm']:.2f}")

                    meta = st.columns(3)
                    meta[0].caption(f"doc_length: {e['doc_length']} tokens")
                    meta[1].caption(f"query_length: {e['query_length']} tokens")
                    meta[2].caption(f"overlap: {e['overlap_count']} terms")

                    if e["matching_keywords"]:
                        st.markdown("**Matching keywords:** " +
                                    " ".join(f"`{k}`" for k in e["matching_keywords"]))
                    else:
                        st.markdown(
                            "*No direct keyword matches — retrieved semantically.*"
                        )


# ======================================================================= #
# ADD DOCUMENT
# ======================================================================= #
with tab_add:
    st.markdown("Append a new document. BM25 is rebuilt globally, FAISS "
                "extends in place, caches are refreshed.")
    new_title = st.text_input("Title", key="new_title")
    new_text = st.text_area("Text", key="new_text", height=160)
    if st.button("Add document"):
        if not new_text.strip():
            st.error("Text cannot be empty.")
        else:
            new_id = len(engine.documents)
            new_doc = {"id": new_id,
                       "title": new_title.strip() or f"Untitled {new_id}",
                       "text": new_text.strip()}
            engine.add_documents([new_doc])
            with open(config.DOCUMENTS_PATH, "w", encoding="utf-8") as f:
                json.dump(engine.documents, f, indent=2, ensure_ascii=False)
            st.success(f"Added doc_id={new_id} · corpus size: {len(engine.documents)}")
            st.cache_resource.clear()


# ======================================================================= #
# EVALUATE
# ======================================================================= #
with tab_eval:
    st.markdown("Runs the gold queries in `tests/test_queries.json`.")
    k_eval = st.slider("k", 1, 20, 5)
    compare = st.checkbox("Compare Weighted vs. LTR", value=ltr_trained,
                          disabled=not ltr_trained)
    if st.button("Run evaluation"):
        with open(config.TEST_QUERIES_PATH, "r", encoding="utf-8") as f:
            cases = json.load(f)

        if compare and ltr_trained:
            saved_ltr = engine.ltr
            engine.ltr = None
            with st.spinner("Evaluating weighted..."):
                rw = evaluate(engine, cases, k=k_eval)
            engine.ltr = saved_ltr
            with st.spinner("Evaluating LTR..."):
                rl = evaluate(engine, cases, k=k_eval)

            df = pd.DataFrame([
                {"scorer": "weighted", **rw["mean"]},
                {"scorer": "LTR",      **rl["mean"]},
            ]).set_index("scorer")
            st.dataframe(df.style.format("{:.3f}"), use_container_width=True)
        else:
            with st.spinner("Evaluating..."):
                report = evaluate(engine, cases, k=k_eval)
            m = report["mean"]
            c1, c2, c3 = st.columns(3)
            c1.metric(f"Precision@{k_eval}", f"{m['precision']:.3f}")
            c2.metric(f"Recall@{k_eval}",    f"{m['recall']:.3f}")
            c3.metric(f"NDCG@{k_eval}",      f"{m['ndcg']:.3f}")
            df = pd.DataFrame([
                {"query": pq["query"],
                 f"P@{k_eval}":    round(pq["precision"], 3),
                 f"R@{k_eval}":    round(pq["recall"], 3),
                 f"NDCG@{k_eval}": round(pq["ndcg"], 3),
                 "retrieved":      pq["retrieved"],
                 "relevant":       pq["relevant"]}
                for pq in report["per_query"]
            ])
            st.dataframe(df, use_container_width=True)

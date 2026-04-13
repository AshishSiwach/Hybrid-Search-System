"""
streamlit_app.py — QueryLens
Intelligent search powered by hybrid retrieval and semantic reranking.

Run: python -m streamlit run streamlit_app.py
"""

import time, json, os, sys, urllib.request
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st

st.set_page_config(
    page_title="QueryLens",
    page_icon="🔎",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Instrument+Serif:ital@0;1&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: "DM Sans", sans-serif; }
#MainMenu, footer, header { visibility: hidden; }

.block-container {
    padding-top: 0 !important;
    padding-bottom: 3rem !important;
    max-width: 760px !important;
}

[data-testid="stForm"] {
    border: none !important;
    padding: 0 !important;
    background: transparent !important;
}

.stTextInput > div > div > input {
    font-size: 16px !important;
    padding: 14px 18px !important;
    border-radius: 12px !important;
    border: 1.5px solid #e2e2e2 !important;
    background: #fafafa !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04) !important;
}
.stTextInput > div > div > input:focus {
    border-color: #1a1a1a !important;
    background: #fff !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08) !important;
}

.stButton > button {
    background: #1a1a1a !important;
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 500 !important;
    font-size: 15px !important;
    padding: 12px 0 !important;
    box-shadow: 0 2px 6px rgba(0,0,0,0.15) !important;
}
.stButton > button:hover {
    background: #333 !important;
}

[data-testid="stExpander"] {
    border: 1px solid #ebebeb !important;
    border-radius: 10px !important;
    background: #fff !important;
    margin-bottom: 8px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
}
[data-testid="stExpander"]:hover {
    border-color: #d0d0d0 !important;
}

hr { border-color: #f0f0f0 !important; margin: 24px 0 !important; }
</style>
""", unsafe_allow_html=True)

ANTHROPIC_API_KEY = st.secrets.get("ANTHROPIC_API_KEY", os.environ.get("ANTHROPIC_API_KEY", ""))

@st.cache_resource(show_spinner="Loading search indexes...")
def load_pipeline():
    try:
        import yaml
        from data_loader import MSMarcoLoader, load_embeddings, load_bm25_index
        from retrievers import BM25Retriever, DenseRetriever, HybridRetriever
        from reranker import CrossEncoderReranker, RetrievalPipeline
    except ImportError as e:
        return None, None, None, f"Import error: {e}. Run: pip install rank-bm25 sentence-transformers faiss-cpu pyyaml"

    if not Path("configs/config.yaml").exists():
        return None, None, None, "configs/config.yaml not found. Run from your project root."

    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    if not Path(config["data"]["ms_marco_passages_path"]).exists():
        return None, None, None, "MS MARCO data not found. Run: python dataset_builder.py"

    loader = MSMarcoLoader(config)
    passages, passage_ids, metadata, queries = loader.load()

    bm25 = BM25Retriever(config)
    ip = config["data"]["index_path"]
    if Path(ip).exists():
        bm25.index = load_bm25_index(ip)
        bm25.tokenized_corpus = [p.lower().split() for p in passages]
    else:
        bm25.build_index(passages)

    dense = DenseRetriever(config)
    fp = config["data"]["faiss_index_path"]
    ep = config["data"]["embeddings_path"]
    if Path(fp).exists():
        dense.load_faiss_index(fp)
    elif Path(ep).exists():
        dense.load_embeddings(load_embeddings(ep))
        dense.save_faiss_index(fp)
    else:
        return None, None, None, "Embeddings not found. Run: python main.py once to build indexes."

    hybrid = HybridRetriever(bm25, dense, config)
    reranker = CrossEncoderReranker(config)

    pipelines = {
        "BM25 only":         RetrievalPipeline(bm25,   reranker, passages),
        "Dense only":        RetrievalPipeline(dense,  reranker, passages),
        "Hybrid (RRF)":      RetrievalPipeline(hybrid, reranker, passages),
        "Hybrid + Reranker": RetrievalPipeline(hybrid, reranker, passages),
    }
    for m in ["BM25 only", "Dense only", "Hybrid (RRF)"]:
        orig = pipelines[m].search
        pipelines[m].search = lambda q, _o=orig: _o(q, rerank=False)

    return pipelines, passages, metadata, None


def generate_answer(query, top_passages, passages, metadata, api_key):
    if not api_key:
        return None
    top3 = top_passages[:3]
    block = "\n\n".join(
        f"[{i+1}] {passages[int(idx)]}\nSource: {metadata[int(idx)]['url']}"
        for i, (idx, _) in enumerate(top3)
    )
    payload = json.dumps({
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 500,
        "system": (
            "You are a helpful search assistant. Answer the query clearly using "
            "ONLY the provided passages. Structure your answer as numbered points "
            "with bold headings. Cite each fact with [1], [2], or [3] after the "
            "relevant sentence. End with a References section listing sources "
            "as [1], [2], [3] with their URLs. Do not add external knowledge."
        ),
        "messages": [{"role": "user", "content": f"Query: {query}\n\nPassages:\n{block}"}],
    }).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            d = json.loads(r.read())
            return {
                "text": d["content"][0]["text"],
                "urls": [metadata[int(idx)]["url"] for idx, _ in top3],
            }
    except Exception:
        return None


# ---------------------------------------------------------------
# Load pipeline
# ---------------------------------------------------------------
pipelines, passages, metadata, error = load_pipeline()

# ---------------------------------------------------------------
# Header
# ---------------------------------------------------------------
st.markdown("""
<div style='padding: 3.5rem 0 2rem; text-align: center;'>
  <div style='font-family: "Instrument Serif", serif; font-size: 3rem;
              font-weight: 400; letter-spacing: -1px; color: #0f0f0f;
              margin-bottom: 10px; line-height: 1;'>
    QueryLens
  </div>
  <div style='font-size: 13px; color: #aaa; font-weight: 300; letter-spacing: 0.03em;'>
    Hybrid retrieval &nbsp;&middot;&nbsp; Semantic reranking &nbsp;&middot;&nbsp; AI-generated answers
  </div>
</div>
""", unsafe_allow_html=True)

if error:
    st.error(f"**Setup required:** {error}")
    st.stop()

# ---------------------------------------------------------------
# Search form
# ---------------------------------------------------------------
with st.form("search_form", clear_on_submit=False):
    query = st.text_input(
        "",
        placeholder="Ask anything — e.g. what causes global warming, how do vaccines work...",
        label_visibility="collapsed",
    )
    submitted = st.form_submit_button("Search", use_container_width=True, type="primary")

# ---------------------------------------------------------------
# Run search
# ---------------------------------------------------------------
if submitted and query and query.strip():
    q = query.strip()

    with st.spinner("Searching..."):
        t_results = {}
        for method, pipeline in pipelines.items():
            t0 = time.perf_counter()
            indices, scores = pipeline.search(q)
            t_results[method] = {
                "passages": list(zip(indices[:5], scores[:5])),
                "latency":  (time.perf_counter() - t0) * 1000,
            }

    best = t_results["Hybrid + Reranker"]["passages"]

    # AI Answer
    if ANTHROPIC_API_KEY:
        with st.spinner("Generating answer..."):
            answer = generate_answer(q, best, passages, metadata, ANTHROPIC_API_KEY)

        if answer:
            st.markdown("""
<div style='font-family:"DM Mono",monospace;font-size:10px;color:#aaa;
            letter-spacing:0.12em;text-transform:uppercase;margin:28px 0 16px;
            display:flex;align-items:center;gap:10px;'>
  <span>Answer</span>
  <span style='flex:1;height:1px;background:#ececec;display:inline-block;'></span>
</div>""", unsafe_allow_html=True)

            import re
            # Remove markdown headings so answer renders as normal body text
            clean = re.sub(r"(?m)^#{1,3} +", "", answer["text"])
            # Split body from references
            body = re.split(r"References:?", clean)[0].strip()
            st.markdown(body)

            st.markdown(
                "<div style='font-family:DM Mono,monospace;font-size:10px;color:#aaa;"
                "letter-spacing:0.1em;text-transform:uppercase;margin:20px 0 10px;"
                "display:flex;align-items:center;gap:10px;'>"
                "<span>References</span>"
                "<span style='flex:1;height:1px;background:#ececec;display:inline-block;'></span>"
                "</div>",
                unsafe_allow_html=True,
            )
            for i, url in enumerate(answer["urls"], 1):
                st.markdown(
                    f"<span style='font-family:DM Mono,monospace;font-size:12px;color:#888;'>"
                    f"[{i}]&nbsp;&nbsp;"
                    f"<a href='{url}' target='_blank' style='color:#555;text-decoration:none;'>"
                    f"{url}</a></span>",
                    unsafe_allow_html=True,
                )
            st.divider()

    # Top results
    st.markdown("""
<div style='font-family:"DM Mono",monospace;font-size:11px;color:#bbb;
            letter-spacing:0.06em;text-transform:uppercase;margin:8px 0 14px;'>
  Top results &nbsp;&middot;&nbsp; Hybrid + Reranker
</div>""", unsafe_allow_html=True)

    for rank, (idx, score) in enumerate(best, 1):
        idx  = int(idx)
        text = passages[idx]
        url  = metadata[idx]["url"]
        rel  = metadata[idx].get("is_selected", 0)
        tag  = "  [relevant]" if rel else ""

        with st.expander(f"#{rank} — {text[:90]}...{tag}", expanded=(rank <= 2)):
            st.write(text)
            st.markdown(f"[{url}]({url})")

    # Technical comparison — hidden
    st.divider()
    with st.expander("Method comparison", expanded=False):
        st.caption("BM25  ·  Dense  ·  Hybrid RRF  ·  Hybrid + Reranker")
        colors = {
            "BM25 only":         "#4a9eff",
            "Dense only":        "#2dc77a",
            "Hybrid (RRF)":      "#e3a020",
            "Hybrid + Reranker": "#9b6dff",
        }
        for method, data in t_results.items():
            color  = colors[method]
            winner = method == "Hybrid + Reranker"
            st.markdown(
                f"<span style='font-family:DM Mono,monospace;font-size:12px;"
                f"color:{color};font-weight:{'600' if winner else '400'};'>"
                f"{'★ ' if winner else ''}{method}&nbsp;&nbsp;"
                f"<span style='color:#bbb;font-size:11px;'>{data['latency']:.0f}ms</span>"
                f"</span>",
                unsafe_allow_html=True,
            )
            for rank, (idx, score) in enumerate(data["passages"], 1):
                idx = int(idx)
                rel = metadata[idx].get("is_selected", 0)
                st.markdown(
                    f"&nbsp;&nbsp;`#{rank}` {passages[idx][:100]}...{'  [relevant]' if rel else ''}",
                    unsafe_allow_html=True,
                )
            st.markdown("---")